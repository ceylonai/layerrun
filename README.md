# LayerRun

LayerRun is a Rust workspace for experimenting with raw safetensors-level LLM loading, tokenizer plumbing, model probing, generation, and conversion into a per-layer model layout.

The workspace contains:

- `layerrun-core`: shared model, tokenizer, safetensors, optimizer, and Hugging Face helpers.
- `layerrun-cli`: command-line tools for inspecting, probing, optimizing, and generating.
- `layerrun-server`: OpenAI-compatible server with an Ollama-style local model catalog.

## Requirements

- Rust toolchain with Cargo.
- Local model files, or network access to Hugging Face for `--hf-repo` commands.
- `HF_TOKEN` or `--hf-token` for gated/private Hugging Face repositories.
- Optional MLX backend: Apple Silicon macOS with the `mlx` Cargo feature enabled, `cmake`, and full Xcode selected so `xcrun -find metal` succeeds.

## Workspace Commands

Build everything:

```sh
cargo build
```

Run tests:

```sh
cargo test
```

Run the CLI help:

```sh
cargo run -p layerrun-cli -- --help
```

Build the CLI with the optional MLX backend:

```sh
cargo build -p layerrun-cli --features mlx
```

The current `mlx-rs` native build requires the Metal shader compiler even when only the top-level MLX feature is selected. Command Line Tools alone are not enough; install full Xcode and select it with `xcode-select`.

```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcrun -find metal
```

Run the OpenAI-compatible server:

```sh
cargo run -p layerrun-server
```

## CLI Overview

All CLI commands are run through the `layerrun-cli` package:

```sh
cargo run -p layerrun-cli -- <command> [options]
```

Available commands:

- `inspect`: inspect tensor names, dtypes, and shapes from a `.safetensors` file.
- `tokenize`: tokenize text with a `tokenizer.json`.
- `probe-model`: load model config and basic tensors.
- `generate`: run the raw model generation path.
- `optimize`: create a LayerRun directory with embeddings, per-layer files, and final weights.
- `generate-layered`: generate from a LayerRun per-layer model directory.

## Server Mode

By default, the server scans `models/` for local model directories and exposes them through both OpenAI-compatible and Ollama-style model-list endpoints. Models are loaded lazily the first time a generation request uses them.

Serve every discoverable model under `models/`:

```sh
cargo run -p layerrun-server
```

Serve models from a different local catalog directory:

```sh
cargo run -p layerrun-server -- \
  --models-dir /path/to/models
```

Also register a specific local single-file safetensors model:

```sh
cargo run -p layerrun-server -- \
  --model-dir models/qwen \
  --model-id qwen
```

Also register a specific LayerRun optimized per-layer model:

```sh
cargo run -p layerrun-server -- \
  --model-dir models/qwen-layered \
  --layered \
  --model-id qwen-layered
```

Serve a Hugging Face model after resolving it into the LayerRun cache:

```sh
cargo run -p layerrun-server -- \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --model-id llama-3.2-1b-instruct
```

The server listens on `127.0.0.1:8080` by default. Override this with `--host` and `--port`.

Available endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/ps`

`/api/tags`, `/api/show`, and `/api/ps` follow Ollama's model-management shape closely enough for local catalog inspection. Generation still goes through the OpenAI-compatible endpoints. When multiple models are available, requests must include a `model` field.

Example completion request:

```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-E4B-it-qat-mobile-transformers",
    "prompt": "Write a short greeting",
    "max_tokens": 8,
    "temperature": 0.7,
    "top_k": 40
  }'
```

Example chat completion request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-E4B-it-qat-mobile-transformers",
    "messages": [
      { "role": "system", "content": "You are concise." },
      { "role": "user", "content": "Write a short greeting" }
    ],
    "max_tokens": 8,
    "temperature": 0.7,
    "top_k": 40
  }'
```

Chat requests are rendered with the loaded model's chat template before tokenization. LayerRun reads template metadata from tokenizer/config files when available, has explicit Gemma 4 instruct formatting support, and only uses the plain fallback format for base or unknown models. Gemma 4 chat prompts use the model's turn tokens, map assistant messages to Gemma's `model` role, and preserve `system`, `user`, `assistant`, and `developer` roles.

OpenAI SDK-compatible clients can use `http://127.0.0.1:8080/v1` as the base URL. Streaming responses are not supported yet; send non-streaming requests.

`temperature`, `top_k`, and `top_p` are supported on `/v1/completions` and `/v1/chat/completions`. The default temperature is `0`, which keeps deterministic greedy generation. When `temperature` is above `0`, the server defaults to `top_k: 40` to avoid sampling low-quality tail tokens. You can override it explicitly.

Completion logs are printed to stderr. Every generation request emits live progress lines for request start, model load, tokenization, generation, decoding, and request completion, plus a summary with endpoint, model, token counts, finish reason, and elapsed time. To include the full prompt and generated output in the server log, start the server with:

```sh
cargo run -p layerrun-server -- --log-completions
```

For faster generation, use a release build and preload layered model files so they are not paged from disk during each generated token:

```sh
cargo run --release -p layerrun-server -- --preload-layers
```

For lower memory usage, preload only the first N layers:

```sh
cargo run --release -p layerrun-server -- --preload-layer-count 8
```

Inspect the local catalog:

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/api/tags
curl http://127.0.0.1:8080/api/ps
```

Show model metadata:

```sh
curl http://127.0.0.1:8080/api/show \
  -H 'Content-Type: application/json' \
  -d '{ "model": "gemma-4-E4B-it-qat-mobile-transformers" }'
```

JavaScript client example:

```sh
node examples/js-client.mjs models
node examples/js-client.mjs tags
node examples/js-client.mjs chat gemma-4-E4B-it-qat-mobile-transformers "Write a short greeting"
LAYERRUN_MODEL=gemma-4-E4B-it-qat-mobile-transformers LAYERRUN_TEMPERATURE=0.7 LAYERRUN_TOP_K=40 node examples/js-client.mjs complete "Hello"
```

Server options:

- `--host <HOST>`: bind address. Defaults to `127.0.0.1`.
- `--port <PORT>`: bind port. Defaults to `8080`.
- `--models-dir <MODELS_DIR>`: directory containing local model directories to discover. Defaults to `models`.
- `--model-id <MODEL_ID>`: public model id returned by `/v1/models`.
- `--model-dir <MODEL_DIR>`: additional local model directory to register.
- `--hf-repo <HF_REPO>`: Hugging Face repo id.
- `--hf-revision <HF_REVISION>`: Hugging Face branch, tag, or commit. Defaults to `main`.
- `--hf-token <HF_TOKEN>`: Hugging Face token. If omitted, `HF_TOKEN` is used when present.
- `--hf-cache-dir <HF_CACHE_DIR>`: directory for cached Hugging Face files.
- `--weights <WEIGHTS>`: safetensors filename inside the model directory for non-layered models. Defaults to `model.safetensors`.
- `--layered`: load a LayerRun optimized per-layer model directory.
- `--preload-layers`: load all per-layer weights before serving. Requires `--layered`.
- `--preload-layer-count <N>`: load only the first `N` per-layer weights before serving. Requires `--layered`.
- `--debug`: print model execution details during requests.
- `--log-completions`: print full prompts and generated text for completion requests.
- `--backend <BACKEND>`: runtime backend, either `cpu` or `mlx`. Defaults to `cpu`.

## inspect

Inspect a safetensors file:

```sh
cargo run -p layerrun-cli -- inspect \
  --file models/qwen/model.safetensors
```

Options:

- `--file <FILE>`: path to the `.safetensors` file.

## tokenize

Tokenize and decode text using a tokenizer file:

```sh
cargo run -p layerrun-cli -- tokenize \
  --tokenizer models/qwen/tokenizer.json \
  --text "Hello from LayerRun"
```

Options:

- `--tokenizer <TOKENIZER>`: path to `tokenizer.json`.
- `--text <TEXT>`: text to tokenize.

## probe-model

Probe a local model directory:

```sh
cargo run -p layerrun-cli -- probe-model \
  --model-dir models/qwen
```

Probe with an explicit weights filename:

```sh
cargo run -p layerrun-cli -- probe-model \
  --model-dir models/qwen \
  --weights model.safetensors
```

Probe a Hugging Face repository:

```sh
cargo run -p layerrun-cli -- probe-model \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct
```

Options:

- `--model-dir <MODEL_DIR>`: local model directory. Required unless `--hf-repo` is set.
- `--hf-repo <HF_REPO>`: Hugging Face repo id.
- `--hf-revision <HF_REVISION>`: Hugging Face branch, tag, or commit. Defaults to `main`.
- `--hf-token <HF_TOKEN>`: Hugging Face token. If omitted, `HF_TOKEN` is used when present.
- `--hf-cache-dir <HF_CACHE_DIR>`: directory for cached Hugging Face files.
- `--weights <WEIGHTS>`: safetensors filename inside the model directory. Defaults to `model.safetensors`.

Pass either `--model-dir` or `--hf-repo`, not both.

## generate

Generate from a local single-file safetensors model:

```sh
cargo run -p layerrun-cli -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --max-new-tokens 8
```

Select a backend explicitly:

```sh
cargo run -p layerrun-cli -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --backend cpu
```

Use MLX when the CLI was built with `--features mlx`:

```sh
cargo run -p layerrun-cli --features mlx -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --backend mlx
```

Generate with debug output:

```sh
cargo run -p layerrun-cli -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --max-new-tokens 8 \
  --debug
```

Generate from Hugging Face:

```sh
cargo run -p layerrun-cli -- generate \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --prompt "Write a short greeting"
```

Options:

- `--model-dir <MODEL_DIR>`: local model directory. Required unless `--hf-repo` is set.
- `--hf-repo <HF_REPO>`: Hugging Face repo id.
- `--hf-revision <HF_REVISION>`: Hugging Face branch, tag, or commit. Defaults to `main`.
- `--hf-token <HF_TOKEN>`: Hugging Face token. If omitted, `HF_TOKEN` is used when present.
- `--hf-cache-dir <HF_CACHE_DIR>`: directory for cached Hugging Face files.
- `--weights <WEIGHTS>`: safetensors filename inside the model directory. Defaults to `model.safetensors`.
- `--prompt <PROMPT>`: prompt text.
- `--max-new-tokens <MAX_NEW_TOKENS>`: number of new tokens to generate. Defaults to `8`.
- `--debug`: print additional model execution details.
- `--backend <BACKEND>`: runtime backend, either `cpu` or `mlx`. Defaults to `cpu`.

## optimize

Convert a local single-file safetensors model into the LayerRun per-layer layout:

```sh
cargo run -p layerrun-cli -- optimize \
  --input-model-dir models/qwen \
  --output-model-dir models/qwen-layered
```

Convert from Hugging Face:

```sh
cargo run -p layerrun-cli -- optimize \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --output-model-dir models/llama-3.2-1b-instruct-layered
```

Options:

- `--input-model-dir <INPUT_MODEL_DIR>`: local model directory. Required unless `--hf-repo` is set.
- `--hf-repo <HF_REPO>`: Hugging Face repo id.
- `--hf-revision <HF_REVISION>`: Hugging Face branch, tag, or commit. Defaults to `main`.
- `--hf-token <HF_TOKEN>`: Hugging Face token. If omitted, `HF_TOKEN` is used when present.
- `--hf-cache-dir <HF_CACHE_DIR>`: directory for cached Hugging Face files.
- `--output-model-dir <OUTPUT_MODEL_DIR>`: destination LayerRun model directory.
- `--weights <WEIGHTS>`: safetensors filename inside the input model directory. Defaults to `model.safetensors`.

The output directory contains files such as `embeddings.safetensors`, `layer_000.safetensors`, `final.safetensors`, `config.json`, `tokenizer.json`, and `layerrun.json`.

## generate-layered

Generate from a LayerRun optimized model directory:

```sh
cargo run -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 8
```

Preload all layer files before generation:

```sh
cargo run -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 8 \
  --preload-layers
```

Preload only the first 8 layer files before generation:

```sh
cargo run -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 8 \
  --preload-layer-count 8
```

Run with debug output:

```sh
cargo run -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 8 \
  --preload-layers \
  --debug
```

Options:

- `--model-dir <MODEL_DIR>`: LayerRun model directory.
- `--prompt <PROMPT>`: prompt text.
- `--max-new-tokens <MAX_NEW_TOKENS>`: number of new tokens to generate. Defaults to `8`.
- `--debug`: print additional model execution details.
- `--preload-layers`: load all per-layer weights before generation.
- `--preload-layer-count <N>`: load only the first `N` per-layer weights before generation. Cannot be combined with `--preload-layers`.
- `--backend <BACKEND>`: runtime backend, either `cpu` or `mlx`. Defaults to `cpu`.

## validate

Validate LayerRun tokenizer output, first-token logits, and short greedy generation against reference fixtures generated from a trusted runtime such as Hugging Face Transformers:

```sh
cargo run -p layerrun-cli -- validate \
  --model-dir models/gemma-4-E4B-it-qat-mobile-transformers \
  --fixtures fixtures/gemma.json
```

`validate` auto-detects LayerRun layered model directories and also supports non-layered safetensors model directories with `--weights`.

Fixture files contain:

- reference runtime metadata
- fixed prompts
- expected tokenizer token ids
- expected first-token top-N logits
- expected greedy generated ids for `temperature: 0`
- optional explanations for known generated-id mismatches

Generate or refresh fixtures from a Transformers-readable reference model directory:

```sh
python3 scripts/generate_validation_fixture.py \
  --model-dir /path/to/transformers/model \
  --output fixtures/gemma.json \
  --prompt "Hello" \
  --prompt "The capital of France is" \
  --max-new-tokens 4 \
  --top-n 10
```

The fixture generator requires Python packages for `torch` and `transformers`. The checked-in `fixtures/gemma.json` pins prompt/token-id cases and should be regenerated with reference logits and generated ids before using it as a passing correctness gate.

Options:

- `--model-dir <MODEL_DIR>`: LayerRun or standard model directory.
- `--fixtures <FIXTURES>`: validation fixture JSON file.
- `--weights <WEIGHTS>`: safetensors filename for non-layered model directories. Defaults to `model.safetensors`.
- `--backend <BACKEND>`: runtime backend, either `cpu` or `mlx`. Defaults to `cpu`.
- `--preload-layers`: load all per-layer weights before validation.
- `--preload-layer-count <N>`: load only the first `N` per-layer weights before validation. Cannot be combined with `--preload-layers`.

## Hugging Face Cache

When using `--hf-repo`, LayerRun downloads `config.json`, `tokenizer.json`, and safetensors weights or shards into a local cache.

Default cache location:

```text
$XDG_CACHE_HOME/layerrun/huggingface
```

If `XDG_CACHE_HOME` is not set:

```text
$HOME/.cache/layerrun/huggingface
```

Override it with:

```sh
cargo run -p layerrun-cli -- probe-model \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --hf-cache-dir .cache/huggingface
```

## Installed Example Models

This repository currently includes example model directories under `models/`, including single-file and LayerRun optimized layouts:

- `models/qwen`
- `models/qwen-layered`
- `models/llama-3.2-1b-instruct`
- `models/llama-3.2-1b-instruct-layered`
- `models/mistral`

Use the single-file directories with `inspect`, `probe-model`, `generate`, and `optimize`. Use the `*-layered` directories with `generate-layered`.
