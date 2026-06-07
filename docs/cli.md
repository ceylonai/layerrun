# LayerRun CLI Reference Guide

The `layerrun-cli` package provides command-line utilities to inspect raw model files, tokenize text, probe configs, perform optimization (splitting checkpoints into a per-layer layout), execute greedy generation, and run correctness validation fixtures.

## Invocation

All CLI commands are executed through the `layerrun-cli` workspace package:

```sh
cargo run -p layerrun-cli -- <command> [options]
```

To run with optimization enabled (strongly recommended for generation and validation commands):

```sh
cargo run --release -p layerrun-cli -- <command> [options]
```

---

## Commands

### 1. `inspect`

Inspects a `.safetensors` file, printing tensor names, data types (dtypes), and shapes.

#### Usage
```sh
cargo run -p layerrun-cli -- inspect --file <path-to-file>
```

#### Options
* `--file <FILE>`: The absolute or relative path to the target `.safetensors` file.

#### Example
```sh
cargo run -p layerrun-cli -- inspect --file models/qwen/model.safetensors
```

---

### 2. `tokenize`

Tokenizes input text using a specified `tokenizer.json` and prints both the token IDs and the decoded text.

#### Usage
```sh
cargo run -p layerrun-cli -- tokenize --tokenizer <path-to-tokenizer> --text "<text-to-tokenize>"
```

#### Options
* `--tokenizer <TOKENIZER>`: Path to the `tokenizer.json` file.
* `--text <TEXT>`: The text string to tokenize.

#### Example
```sh
cargo run -p layerrun-cli -- tokenize \
  --tokenizer models/qwen/tokenizer.json \
  --text "Hello from LayerRun"
```

---

### 3. `probe-model`

Loads model configuration and major tensors (embeddings, final norm, LM head) to inspect shape and model type compatibility.

#### Usage
```sh
cargo run -p layerrun-cli -- probe-model [options]
```

#### Options
* `--model-dir <MODEL_DIR>`: Path to a local model directory containing `config.json` and weight files. *Required unless `--hf-repo` is set.*
* `--hf-repo <HF_REPO>`: Hugging Face repository ID (e.g., `meta-llama/Llama-3.2-1B-Instruct`).
* `--hf-revision <HF_REVISION>`: Hugging Face branch, tag, or commit hash (default: `main`).
* `--hf-token <HF_TOKEN>`: Hugging Face API token for gated or private repositories. (If omitted, uses the `HF_TOKEN` environment variable if present).
* `--hf-cache-dir <HF_CACHE_DIR>`: Target directory for caching downloaded Hugging Face files.
* `--weights <WEIGHTS>`: The name of the safetensors file inside the model directory (default: `model.safetensors`).

> [!WARNING]
> Do not supply both `--model-dir` and `--hf-repo` in the same command.

#### Example
```sh
cargo run -p layerrun-cli -- probe-model --model-dir models/qwen
```

---

### 4. `generate`

Executes greedy generation from a single-file or sharded model layout. Useful for testing baseline math and configurations before conversion.

#### Usage
```sh
cargo run -p layerrun-cli -- generate [options]
```

#### Options
* `--model-dir <MODEL_DIR>`: Local model directory. *Required unless `--hf-repo` is set.*
* `--hf-repo <HF_REPO>`: Hugging Face repository ID.
* `--hf-revision <HF_REVISION>`: Hugging Face revision (default: `main`).
* `--hf-token <HF_TOKEN>`: Hugging Face API token.
* `--hf-cache-dir <HF_CACHE_DIR>`: Hugging Face cache directory.
* `--weights <WEIGHTS>`: The name of the safetensors file inside the model directory (default: `model.safetensors`).
* `--prompt <PROMPT>`: Prompt text to generate from.
* `--max-new-tokens <MAX_NEW_TOKENS>`: Number of new tokens to generate (default: `8`).
* `--debug`: If set, prints additional model execution steps, tensor shape calculations, and elapsed timings.
* `--stream`: Prints generated text incrementally as each token is produced.
* `--backend <BACKEND>`: Compute backend, either `cpu` or `mlx` (default: `cpu`).

> [!NOTE]
> Selecting the `mlx` backend requires compiling with the optional MLX feature enabled:
> `cargo build -p layerrun-cli --features mlx`

#### Example
```sh
cargo run -p layerrun-cli -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --max-new-tokens 8
```

#### Streaming Example
```sh
cargo run -p layerrun-cli -- generate \
  --model-dir models/qwen \
  --prompt "Write a short greeting" \
  --max-new-tokens 32 \
  --stream
```

---

### 5. `optimize`

Converts a standard model layout into a LayerRun optimized per-layer directory. It separates the model weights into:
* `embeddings.safetensors`
* `layer_000.safetensors`, `layer_001.safetensors`, ...
* `final.safetensors`
* `layerrun.json` (LayerRun metadata)
* Copy of `config.json` and `tokenizer.json`

#### Usage
```sh
cargo run -p layerrun-cli -- optimize [options]
```

#### Options
* `--input-model-dir <INPUT_MODEL_DIR>`: Standard model directory to convert. *Required unless `--hf-repo` is set.*
* `--hf-repo <HF_REPO>`: Hugging Face repository ID to download and convert.
* `--hf-revision <HF_REVISION>`: Hugging Face revision (default: `main`).
* `--hf-token <HF_TOKEN>`: Hugging Face API token.
* `--hf-cache-dir <HF_CACHE_DIR>`: Hugging Face cache directory.
* `--output-model-dir <OUTPUT_MODEL_DIR>`: Destination directory for the optimized per-layer files.
* `--weights <WEIGHTS>`: The name of the safetensors file inside the input model directory (default: `model.safetensors`).

#### Example
```sh
cargo run -p layerrun-cli -- optimize \
  --input-model-dir models/qwen \
  --output-model-dir models/qwen-layered
```

---

### 6. `generate-layered`

Generates text from a LayerRun optimized per-layer model directory. This command supports preloading weights to minimize repeated disk access.

#### Usage
```sh
cargo run -p layerrun-cli -- generate-layered [options]
```

#### Options
* `--model-dir <MODEL_DIR>`: Path to the LayerRun optimized model directory.
* `--prompt <PROMPT>`: Prompt text.
* `--max-new-tokens <MAX_NEW_TOKENS>`: Number of new tokens to generate (default: `8`).
* `--debug`: Prints step-by-step model execution details and timing logs.
* `--stream`: Prints generated text incrementally as each token is produced.
* `--preload-layers`: Loads all per-layer weights into memory before generation starts.
* `--preload-layer-count <N>`: Loads only the first `N` layers into memory, streaming the remaining layers. *(Conflicts with `--preload-layers`)*.
* `--backend <BACKEND>`: Compute backend, either `cpu` or `mlx` (default: `cpu`).

> [!TIP]
> Running in release mode with `--preload-layers` yields the fastest generation times because it avoids reloading layer files from disk for every token generated:
> `cargo run --release -p layerrun-cli -- generate-layered --model-dir models/qwen-layered --prompt "Hello" --preload-layers`

#### Example
```sh
cargo run -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 8 \
  --preload-layers
```

#### Streaming Example
```sh
cargo run --release -p layerrun-cli -- generate-layered \
  --model-dir models/qwen-layered \
  --prompt "Write a short greeting" \
  --max-new-tokens 32 \
  --preload-layers \
  --stream
```

---

### 7. `serve`

Starts the OpenAI-compatible and Ollama-style HTTP server from the CLI package.

#### Usage
```sh
cargo run --release -p layerrun-cli -- serve [options]
```

#### Options
* `--host <HOST>`: IP address to bind (default: `127.0.0.1`).
* `--port <PORT>`: Port to bind (default: `8080`).
* `--models-dir <MODELS_DIR>`: Folder containing local models to automatically discover (default: `models`).
* `--model-id <MODEL_ID>`: Public model ID returned by `/v1/models` and used in request payloads.
* `--model-dir <MODEL_DIR>`: Path to an individual model directory to register.
* `--hf-repo <HF_REPO>`: Hugging Face repository ID.
* `--hf-revision <HF_REVISION>`: Hugging Face revision/commit hash.
* `--hf-token <HF_TOKEN>`: Hugging Face API token.
* `--hf-cache-dir <HF_CACHE_DIR>`: Custom Hugging Face cache directory.
* `--weights <WEIGHTS>`: Filename for weights inside the model directory (default: `model.safetensors`).
* `--layered`: Loads the model as an optimized per-layer directory structure.
* `--preload-layers`: Preload all per-layer weight files before serving.
* `--preload-layer-count <N>`: Preload only the first `N` layer weight files before serving. *(Conflicts with `--preload-layers`)*
* `--backend <BACKEND>`: Compute backend, either `cpu` or `mlx` (default: `cpu`).
* `--debug`: Prints step-by-step model execution information during inference.
* `--log-completions`: Outputs full prompt and generated output to `stderr`.

#### Example
```sh
cargo run --release -p layerrun-cli -- serve \
  --model-dir models/gemma-4-E4B-it-qat-mobile-transformers \
  --model-id gemma \
  --port 8080
```

#### Streaming Example
```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Write a short greeting"}],
    "max_tokens": 32,
    "stream": true
  }'
```

---

### 8. `validate`

Validates LayerRun's tokenizer outputs, first-token logits, and short greedy generations against a reference JSON fixture generated from a trusted runtime (like Hugging Face Transformers).

#### Usage
```sh
cargo run -p layerrun-cli -- validate [options]
```

#### Options
* `--model-dir <MODEL_DIR>`: Path to a LayerRun or standard model directory.
* `--fixtures <FIXTURES>`: Path to the validation fixture JSON file.
* `--weights <WEIGHTS>`: Safetensors weights filename for non-layered directories (default: `model.safetensors`).
* `--backend <BACKEND>`: Backend to use, either `cpu` or `mlx` (default: `cpu`).
* `--preload-layers`: Preload all layers before starting validation.
* `--preload-layer-count <N>`: Preload only the first `N` layers. *(Conflicts with `--preload-layers`)*.

#### Validation Fixtures
Validation fixtures are JSON files containing reference prompts, token IDs, top-N expected logits (token ID + logit values), and expected generated token IDs.

To generate a new reference fixture, use the provided Python script:
```sh
python3 scripts/generate_validation_fixture.py \
  --model-dir /path/to/transformers/model \
  --output fixtures/gemma.json \
  --prompt "Hello" \
  --prompt "The capital of France is" \
  --max-new-tokens 4 \
  --top-n 10
```
*(Requires Python with `torch` and `transformers` installed).*

#### Example
```sh
cargo run -p layerrun-cli -- validate \
  --model-dir models/gemma-4-E4B-it-qat-mobile-transformers \
  --fixtures fixtures/gemma.json
```

---

## Hugging Face Cache

When downloading repos using `--hf-repo`, files are written to a localized cache directory.

* Default location:
  * `$XDG_CACHE_HOME/layerrun/huggingface` (if `$XDG_CACHE_HOME` is set)
  * `$HOME/.cache/layerrun/huggingface` (if `$XDG_CACHE_HOME` is not set)
* Override the cache directory on any command using the `--hf-cache-dir` flag.
