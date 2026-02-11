#!/usr/bin/env bun
import { Cli, Command, Option } from "clipanion";
import {
  generateGeminiImage,
  generateOpenAIImage,
} from "./core.ts";
import {
  geminiAspectRatioSchema,
  geminiImageSizeSchema,
  geminiModelSchema,
  geminiParamsSchema,
  openAIBackgroundSchema,
  openAIModelSchema,
  openAIParamsSchema,
  openAIQualitySchema,
  openAISizeSchema,
  type GeminiParams,
  type OpenAIParams,
} from "./schemas.ts";

export type ParsedArgs =
  | { mode: "openai"; params: OpenAIParams }
  | { mode: "gemini"; params: GeminiParams }
  | { mode: "help"; message?: string };

function printUsage(message?: string) {
  if (message) {
    console.error(message);
  }
  const write = message ? console.error : console.log;
  write(`images-mcp (CLI)

Usage:
  images-mcp openai  [args]  Generate/edit via OpenAI
  images-mcp gemini  [args]  Generate/edit via Gemini

Common args:
  --prompt        Text prompt (required)
  --output        Output file path (required)
  --input         Input image path (repeatable)

OpenAI args:
  --model         gpt-image-1.5 (default)
  --size          auto | 1024x1024 | 1536x1024 | 1024x1536
  --quality       auto | high | medium | low
  --background    auto | transparent | opaque

Gemini args:
  --model         gemini-3-pro-image-preview (default) | gemini-2.5-flash-image
  --aspect-ratio  1:1 | 2:3 | 3:2 | 3:4 | 4:3 | 4:5 | 5:4 | 9:16 | 16:9 | 21:9
  --image-size    1K | 2K | 4K
`);
}

const formatEnumError = (flag: string, value: string, allowed: readonly string[]) =>
  `Invalid value for --${flag}: "${value}". Allowed values: ${allowed.join(", ")}`;
const isAllowedEnumValue = (value: string, allowed: readonly string[]) => allowed.includes(value);

const formatUnknownFlagsError = (command: "openai" | "gemini", flags: string[]) =>
  `Unknown flag(s) for ${command}: ${flags.map((flag) => `--${flag}`).join(", ")}`;

const uniqueInOrder = (values: string[]) => [...new Set(values)];

const parseInputImages = (input: string[] | undefined, inputs: string[] | undefined) => {
  const all = [...(input ?? []), ...(inputs ?? [])];
  return all.length > 0 ? all : undefined;
};

class OpenAICommand extends Command {
  static override paths = [["openai"]];

  prompt = Option.String("--prompt", { required: false });
  output = Option.String("--output", { required: false });
  input = Option.Array("--input", [], { arity: 1 });
  inputs = Option.Array("--inputs", [], { arity: 1 });
  model = Option.String("--model", { required: false });
  size = Option.String("--size", { required: false });
  quality = Option.String("--quality", { required: false });
  background = Option.String("--background", { required: false });

  override async execute() {
    return 0;
  }
}

class GeminiCommand extends Command {
  static override paths = [["gemini"]];

  prompt = Option.String("--prompt", { required: false });
  output = Option.String("--output", { required: false });
  input = Option.Array("--input", [], { arity: 1 });
  inputs = Option.Array("--inputs", [], { arity: 1 });
  model = Option.String("--model", { required: false });
  aspectRatio = Option.String("--aspect-ratio", { required: false });
  imageSize = Option.String("--image-size", { required: false });

  override async execute() {
    return 0;
  }
}

function createCli() {
  const cli = new Cli({
    binaryName: "images-mcp",
    binaryLabel: "images-mcp",
  });

  cli.register(OpenAICommand);
  cli.register(GeminiCommand);

  return cli;
}

function structuralErrorForArgs(args: string[]): string | undefined {
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (!token) break;
    if (token === "--help" || token === "-h") return undefined;
    if (!token.startsWith("--")) return `Unexpected argument: ${token}`;

    const key = token.slice(2);
    const value = args[i + 1];
    if (!value || value.startsWith("--")) return `Missing value for --${key}`;
    i += 1;
  }
}

function findUnknownFlags(command: "openai" | "gemini", args: string[]): string[] {
  const known = command === "openai"
    ? new Set(["prompt", "output", "input", "inputs", "model", "size", "quality", "background"])
    : new Set(["prompt", "output", "input", "inputs", "model", "aspect-ratio", "image-size"]);

  const unknownFlags: string[] = [];
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (!token?.startsWith("--")) {
      continue;
    }

    const key = token.slice(2);
    if (!known.has(key)) {
      unknownFlags.push(key);
    }

    i += 1;
  }

  return uniqueInOrder(unknownFlags);
}

function parseCommand(command: "openai" | "gemini", args: string[]) {
  const cli = createCli();

  try {
    const parsed = cli.process([command, ...args]);
    if (parsed instanceof OpenAICommand || parsed instanceof GeminiCommand) {
      return parsed;
    }
    return { mode: "help", message: `Unknown command: ${command}` } as const;
  } catch (error) {
    return {
      mode: "help",
      message: error instanceof Error ? error.message.split("\n")[0] : String(error),
    } as const;
  }
}

function parseOpenAICommand(args: string[]) {
  const parsed = parseCommand("openai", args);
  if ("mode" in parsed) return parsed;
  if (!(parsed instanceof OpenAICommand)) {
    return { mode: "help", message: "Unknown command: openai" } as const;
  }
  return parsed;
}

function parseGeminiCommand(args: string[]) {
  const parsed = parseCommand("gemini", args);
  if ("mode" in parsed) return parsed;
  if (!(parsed instanceof GeminiCommand)) {
    return { mode: "help", message: "Unknown command: gemini" } as const;
  }
  return parsed;
}

export function parseArgs(argv: string[]): ParsedArgs {
  if (argv.length === 0) return { mode: "help" };

  const command = argv[0];
  const rest = argv.slice(1);

  if (!command) return { mode: "help" };
  if (command === "--help" || command === "-h") return { mode: "help" };
  if (command !== "openai" && command !== "gemini") {
    return { mode: "help", message: `Unknown command: ${command}` };
  }

  if (rest.includes("--help") || rest.includes("-h")) return { mode: "help" };

  const structuralError = structuralErrorForArgs(rest);
  if (structuralError) {
    return { mode: "help", message: structuralError };
  }

  const unknownFlags = findUnknownFlags(command, rest);
  if (unknownFlags.length > 0) {
    return { mode: "help", message: formatUnknownFlagsError(command, unknownFlags) };
  }

  if (command === "openai") {
    const parsed = parseOpenAICommand(rest);
    if ("mode" in parsed) return parsed;

    const prompt = parsed.prompt;
    const output_path = parsed.output;
    const input_images = parseInputImages(parsed.input, parsed.inputs);
    if (!prompt || !output_path) {
      return { mode: "help", message: "Missing required --prompt or --output" };
    }

    const model = parsed.model ?? "gpt-image-1.5";
    const size = parsed.size ?? "auto";
    const quality = parsed.quality ?? "auto";
    const background = parsed.background ?? "auto";

    if (!isAllowedEnumValue(model, openAIModelSchema.options)) {
      return { mode: "help", message: formatEnumError("model", model, openAIModelSchema.options) };
    }
    if (!isAllowedEnumValue(size, openAISizeSchema.options)) {
      return { mode: "help", message: formatEnumError("size", size, openAISizeSchema.options) };
    }
    if (!isAllowedEnumValue(quality, openAIQualitySchema.options)) {
      return { mode: "help", message: formatEnumError("quality", quality, openAIQualitySchema.options) };
    }
    if (!isAllowedEnumValue(background, openAIBackgroundSchema.options)) {
      return { mode: "help", message: formatEnumError("background", background, openAIBackgroundSchema.options) };
    }

    const validated = openAIParamsSchema.safeParse({
      prompt,
      output_path,
      model,
      input_images,
      size,
      quality,
      background,
    });
    if (!validated.success) {
      return { mode: "help", message: `Invalid OpenAI parameters: ${validated.error.issues[0]?.message ?? "Unknown error"}` };
    }

    const params: OpenAIParams = {
      ...validated.data,
    };
    return { mode: "openai", params };
  }

  const parsed = parseGeminiCommand(rest);
  if ("mode" in parsed) return parsed;

  const prompt = parsed.prompt;
  const output_path = parsed.output;
  const input_images = parseInputImages(parsed.input, parsed.inputs);
  if (!prompt || !output_path) {
    return { mode: "help", message: "Missing required --prompt or --output" };
  }

  const model = parsed.model ?? "gemini-3-pro-image-preview";
  const aspectRatio = parsed.aspectRatio;
  const imageSize = parsed.imageSize;
  if (!isAllowedEnumValue(model, geminiModelSchema.options)) {
    return { mode: "help", message: formatEnumError("model", model, geminiModelSchema.options) };
  }
  if (aspectRatio && !isAllowedEnumValue(aspectRatio, geminiAspectRatioSchema.options)) {
    return { mode: "help", message: formatEnumError("aspect-ratio", aspectRatio, geminiAspectRatioSchema.options) };
  }
  if (imageSize && !isAllowedEnumValue(imageSize, geminiImageSizeSchema.options)) {
    return { mode: "help", message: formatEnumError("image-size", imageSize, geminiImageSizeSchema.options) };
  }

  const validated = geminiParamsSchema.safeParse({
    prompt,
    output_path,
    model,
    input_images,
    aspect_ratio: aspectRatio,
    image_size: imageSize,
  });
  if (!validated.success) {
    return { mode: "help", message: `Invalid Gemini parameters: ${validated.error.issues[0]?.message ?? "Unknown error"}` };
  }

  const params: GeminiParams = {
    ...validated.data,
  };
  return { mode: "gemini", params };
}

async function run() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.mode === "help") {
    printUsage(parsed.message);
    process.exit(parsed.message ? 1 : 0);
  }

  try {
    const result =
      parsed.mode === "openai"
        ? await generateOpenAIImage(parsed.params)
        : await generateGeminiImage(parsed.params);

    if (!result.ok) {
      console.error(`Error: ${result.error}`);
      process.exit(1);
    }

    console.log(JSON.stringify(result.data, null, 2));
  } catch (error) {
    console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}

if (import.meta.main) {
  await run();
}
