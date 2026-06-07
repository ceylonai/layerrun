# LayerRun

LayerRun is a Rust workspace for experimenting with raw safetensors-level LLM loading, tokenizer plumbing, model probing, generation, and conversion into a per-layer model layout.

The workspace contains:

- `layerrun-core`: shared model, tokenizer, safetensors, optimizer, and Hugging Face helpers.
- `layerrun-cli`: command-line tools for inspecting, probing, optimizing, and generating.
- `layerrun-server`: current server crate stub.

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

Run the server stub:

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
