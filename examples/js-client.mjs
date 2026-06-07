#!/usr/bin/env node

const baseUrl = process.env.LAYERRUN_BASE_URL ?? "http://127.0.0.1:8080";
const defaultModel = process.env.LAYERRUN_MODEL;

const [command = "chat", ...args] = process.argv.slice(2);

function usage() {
  console.log(`Usage:
  node examples/js-client.mjs models
  node examples/js-client.mjs tags
  node examples/js-client.mjs show <model>
  node examples/js-client.mjs complete <model> <prompt>
  node examples/js-client.mjs chat <model> <prompt>

Environment:
  LAYERRUN_BASE_URL   Server URL. Defaults to http://127.0.0.1:8080
  LAYERRUN_MODEL      Default model when <model> is omitted for complete/chat

Examples:
  node examples/js-client.mjs models
  node examples/js-client.mjs chat gemma-4-E4B-it-qat-mobile-transformers "Write a short greeting"
  LAYERRUN_MODEL=gemma-4-E4B-it-qat-mobile-transformers node examples/js-client.mjs complete "Hello"`);
}

async function request(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      "content-type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  const text = await response.text();
  const body = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message = body?.error?.message ?? response.statusText;
    throw new Error(`${response.status} ${message}`);
  }

  return body;
}

function parseModelAndPrompt(args) {
  if (args.length === 0) {
    throw new Error("missing prompt");
  }

  if (defaultModel && args.length === 1) {
    return { model: defaultModel, prompt: args[0] };
  }

  if (args.length < 2) {
    throw new Error("missing model or LAYERRUN_MODEL");
  }

  return { model: args[0], prompt: args.slice(1).join(" ") };
}

async function main() {
  switch (command) {
    case "models": {
      const body = await request("/v1/models");
      console.log(JSON.stringify(body, null, 2));
      break;
    }

    case "tags": {
      const body = await request("/api/tags");
      console.log(JSON.stringify(body, null, 2));
      break;
    }

    case "show": {
      const model = args[0] ?? defaultModel;
      if (!model) {
        throw new Error("missing model or LAYERRUN_MODEL");
      }
      const body = await request("/api/show", {
        method: "POST",
        body: JSON.stringify({ model }),
      });
      console.log(JSON.stringify(body, null, 2));
      break;
    }

    case "complete": {
      const { model, prompt } = parseModelAndPrompt(args);
      const body = await request("/v1/completions", {
        method: "POST",
        body: JSON.stringify({
          model,
          prompt,
          max_tokens: 32,
        }),
      });
      console.log(body.choices[0]?.text ?? "");
      break;
    }

    case "chat": {
      const { model, prompt } = parseModelAndPrompt(args);
      const body = await request("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 32,
        }),
      });
      console.log(body.choices[0]?.message?.content ?? "");
      break;
    }

    case "help":
    case "--help":
    case "-h":
      usage();
      break;

    default:
      throw new Error(`unknown command: ${command}`);
  }
}

main().catch((error) => {
  console.error(error.message);
  usage();
  process.exitCode = 1;
});
