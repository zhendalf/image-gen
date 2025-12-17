import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import OpenAI from "openai";
import { GoogleGenAI } from "@google/genai";
import path from "node:path";

// Constants
const MIME_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
};

// Response helpers
const textContent = (text: string) => ({ content: [{ type: "text" as const, text }] });
const errorResponse = (message: string) => textContent(`Error: ${message}`);
const successResponse = (data: object) => textContent(JSON.stringify(data, null, 2));

// File utilities
function getMimeType(filePath: string): string {
  return MIME_TYPES[path.extname(filePath).toLowerCase()] || "image/png";
}

async function readImageFile(imagePath: string): Promise<{ data: ArrayBuffer; name: string } | { error: string }> {
  const file = Bun.file(imagePath);
  if (!(await file.exists())) {
    return { error: `Input image not found: ${imagePath}` };
  }
  return { data: await file.arrayBuffer(), name: path.basename(imagePath) };
}

async function saveImage(imageData: string, outputPath: string): Promise<{ path: string; bytes: number }> {
  const buffer = Buffer.from(imageData, "base64");
  const resolvedPath = path.resolve(outputPath);
  await Bun.write(resolvedPath, buffer);
  return { path: resolvedPath, bytes: buffer.length };
}

// Lazy-loaded API clients
let openaiClient: OpenAI | null = null;
let googleClient: GoogleGenAI | null = null;

function getOpenAI(): OpenAI {
  if (!openaiClient) {
    openaiClient = new OpenAI();
  }
  return openaiClient;
}

function getGoogle(): GoogleGenAI {
  if (!googleClient) {
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    if (!apiKey) {
      throw new Error("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable");
    }
    googleClient = new GoogleGenAI({ apiKey });
  }
  return googleClient;
}

// Server setup
const server = new McpServer({
  name: "images-mcp",
  version: "1.0.0",
});

// OpenAI image generation tool
server.registerTool(
  "openai_generate_image",
  {
    title: "OpenAI Image Generator",
    description: "Generate an image using OpenAI and save it to a file. Can accept input images for editing.",
    inputSchema: {
      prompt: z.string().describe("Description of the image to generate, or editing instructions if input_images provided"),
      output_path: z.string().describe("Path where the image should be saved (e.g., /path/to/image.png)"),
      model: z.enum(["gpt-image-1.5"]).default("gpt-image-1.5").describe("Model: gpt-image-1.5"),
      input_images: z.array(z.string()).optional().describe("Optional array of image file paths for editing/reference"),
      size: z.enum(["auto", "1024x1024", "1536x1024", "1024x1536"]).default("auto").describe("Image size"),
      quality: z.enum(["auto", "high", "medium", "low"]).default("auto").describe("Image quality"),
      background: z.enum(["auto", "transparent", "opaque"]).default("auto").describe("Background type"),
    },
  },
  async ({ prompt, output_path, model, input_images, size, quality, background }) => {
    try {
      let imageData: string | undefined;

      if (input_images?.length) {
        const imageFiles: File[] = [];
        for (const imagePath of input_images) {
          const result = await readImageFile(imagePath);
          if ("error" in result) return errorResponse(result.error);
          imageFiles.push(new File([result.data], result.name, { type: "image/png" }));
        }

        const response = await getOpenAI().images.edit({
          model,
          prompt,
          image: imageFiles.length === 1 ? imageFiles[0] : imageFiles,
          size: size === "auto" ? undefined : size,
        } as Parameters<OpenAI["images"]["edit"]>[0]);

        imageData = (response as OpenAI.ImagesResponse).data?.[0]?.b64_json;
      } else {
        const response = await getOpenAI().images.generate({
          model,
          prompt,
          n: 1,
          size: size === "auto" ? undefined : size,
          quality,
          background,
          output_format: "png",
        } as Parameters<OpenAI["images"]["generate"]>[0]);

        imageData = (response as OpenAI.ImagesResponse).data?.[0]?.b64_json;
      }

      if (!imageData) {
        return errorResponse("No image data received from OpenAI");
      }

      const saved = await saveImage(imageData, output_path);
      return successResponse({
        success: true,
        ...saved,
        model,
        size,
        quality,
        input_images_count: input_images?.length ?? 0,
      });
    } catch (error) {
      return errorResponse(error instanceof Error ? error.message : String(error));
    }
  }
);

// Gemini image generation tool
server.registerTool(
  "gemini_generate_image",
  {
    title: "Gemini Image Generator",
    description: "Generate or edit an image using Google Gemini and save it to a file. Can accept input images for editing.",
    inputSchema: {
      prompt: z.string().describe("Description of the image to generate, or editing instructions if input_images provided"),
      output_path: z.string().describe("Path where the image should be saved (e.g., /path/to/image.png)"),
      model: z.enum(["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]).default("gemini-2.5-flash-image").describe("Model"),
      input_images: z.array(z.string()).optional().describe("Optional array of image file paths for editing/reference"),
      aspect_ratio: z.enum(["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]).optional().describe("Aspect ratio"),
      image_size: z.enum(["1K", "2K", "4K"]).optional().describe("Image size"),
    },
  },
  async ({ prompt, output_path, model, input_images, aspect_ratio, image_size }) => {
    try {
      const contents: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [{ text: prompt }];

      if (input_images?.length) {
        for (const imagePath of input_images) {
          const result = await readImageFile(imagePath);
          if ("error" in result) return errorResponse(result.error);
          contents.push({
            inlineData: {
              mimeType: getMimeType(imagePath),
              data: Buffer.from(result.data).toString("base64"),
            },
          });
        }
      }

      const config: { responseModalities: string[]; imageConfig?: { aspectRatio?: string; imageSize?: string } } = {
        responseModalities: ["IMAGE", "TEXT"],
      };

      if (aspect_ratio || image_size) {
        config.imageConfig = {
          ...(aspect_ratio && { aspectRatio: aspect_ratio }),
          ...(image_size && { imageSize: image_size }),
        };
      }

      const response = await getGoogle().models.generateContent({ model, contents, config });
      const parts = response.candidates?.[0]?.content?.parts;
      const imagePart = parts?.find((part) => part.inlineData?.data);

      if (!imagePart?.inlineData?.data) {
        const textPart = parts?.find((part) => part.text);
        return errorResponse(textPart?.text || "No image data received from Google");
      }

      const saved = await saveImage(imagePart.inlineData.data, output_path);
      return successResponse({
        success: true,
        ...saved,
        model,
        aspect_ratio,
        image_size,
        input_images_count: input_images?.length ?? 0,
      });
    } catch (error) {
      return errorResponse(error instanceof Error ? error.message : String(error));
    }
  }
);

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Images MCP server running on stdio");
