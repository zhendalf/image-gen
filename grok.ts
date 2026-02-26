import OpenAI from "openai";
import { getGrokApiKey } from "./config.ts";
import type { GrokParams } from "./schemas.ts";
import {
  type GenerateResult,
  type GrokResult,
  loadInputImages,
  resolveOutputFormat,
  saveImage,
  toOpenAIUploadFiles,
} from "./core-utils.ts";

export type GrokClient = {
  images: {
    edit: (params: Parameters<OpenAI["images"]["edit"]>[0]) => Promise<{ data?: Array<{ b64_json?: string }> }>;
    generate: (params: Parameters<OpenAI["images"]["generate"]>[0]) => Promise<{ data?: Array<{ b64_json?: string }> }>;
  };
};

let grokClient: GrokClient | null = null;

export function setGrokClientForTests(client: GrokClient | null) {
  grokClient = client;
}

function getGrok(): GrokClient {
  if (!grokClient) {
    const apiKey = getGrokApiKey();
    if (!apiKey) {
      throw new Error("Missing XAI_API_KEY environment variable or xai_api_key in config");
    }
    grokClient = new OpenAI({ apiKey, baseURL: "https://api.x.ai/v1" }) as unknown as GrokClient;
  }
  return grokClient;
}

export async function generateGrokImage({
  prompt,
  output_path,
  model,
  input_images,
  aspect_ratio,
  resolution,
}: GrokParams): Promise<GenerateResult<GrokResult>> {
  const outputFormat = resolveOutputFormat("grok", output_path);
  if (!outputFormat.ok) {
    return { ok: false, error: outputFormat.error };
  }

  const loadedImages = await loadInputImages(input_images);
  if (!loadedImages.ok) {
    return loadedImages;
  }

  let imageData: string | undefined;

  if (loadedImages.data.length > 0) {
    const imageFiles = toOpenAIUploadFiles(loadedImages.data);
    const imagePayload = imageFiles.length === 1 ? imageFiles[0]! : imageFiles;
    const response = await getGrok().images.edit({
      model,
      prompt,
      image: imagePayload,
      response_format: "b64_json",
    });
    imageData = response.data?.[0]?.b64_json;
  } else {
    const response = await getGrok().images.generate({
      model,
      prompt,
      n: 1,
      response_format: "b64_json",
      ...(aspect_ratio && { aspect_ratio }),
      ...(resolution && { resolution }),
    } as Parameters<OpenAI["images"]["generate"]>[0]);
    imageData = response.data?.[0]?.b64_json;
  }

  if (!imageData) {
    return { ok: false, error: "No image data received from xAI" };
  }

  const saved = await saveImage(imageData, output_path);
  return {
    ok: true,
    data: {
      success: true,
      ...saved,
      model,
      aspect_ratio,
      resolution,
      input_images_count: input_images?.length ?? 0,
    },
  };
}
