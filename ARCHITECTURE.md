# LayerRun Architecture

LayerRun is a Rust workspace for loading, inspecting, reorganizing, validating, and serving causal language models directly from Hugging Face-style `safetensors` files. It intentionally avoids a higher-level inference framework: model config parsing, tensor loading, tokenizer integration, decoder execution, generation, and HTTP serving are implemented in this workspace.

## Workspace Layout

```text
LayerRun
├── crates/layerrun-core      Shared runtime, tensor loading, tokenizer, model execution
├── crates/layerrun-cli       CLI for config, inspection, generation, optimization, serving
└── crates/layerrun-server    Axum HTTP server with OpenAI-compatible and Ollama-style APIs
```

The core crate owns model behavior. The CLI crate handles user-facing commands, local config defaults, Hugging Face resolution, and command output. The server crate hosts one or more models over HTTP and calls into `layerrun-core` for tokenization, chat prompt rendering, and generation.

## Runtime Boundaries

```text
CLI / HTTP request
    │
    ├── resolve local or Hugging Face model source
    ├── load tokenizer and model config
    ├── construct RawLlm from standard or LayerRun layout
    ├── select backend: cpu or feature-gated mlx
    └── run token-by-token generation
```

LayerRun supports two physical model layouts:

```text
Hugging Face layout
├── config.json
├── tokenizer.json
├── model.safetensors
└── model.safetensors.index.json + shards, when sharded

LayerRun layered layout
├── config.json
├── tokenizer.json
├── embeddings.safetensors
├── layer_000.safetensors
├── layer_001.safetensors
├── ...
├── final.safetensors
└── layerrun.json
```

The standard layout is the compatibility path. The layered layout is the runtime-oriented path: embeddings, each decoder layer, and final weights can be loaded independently, allowing low-memory streaming or partial/full layer preloading.

## Main Components

### `layerrun-cli`

`crates/layerrun-cli/src/main.rs` exposes the user-facing commands:

- `init`: creates `$HOME/.layerrun-conf`, stores default model/cache paths, and can store a Hugging Face token.
- `inspect`: prints tensor names, dtypes, and shapes from a `.safetensors` file.
- `tokenize`: tokenizes and decodes text through `tokenizer.json`.
- `probe-model`: loads model config plus major tensors.
- `generate`: runs generation from a standard local or Hugging Face model layout.
- `optimize`: converts a standard model into the LayerRun per-layer layout.
- `generate-layered`: runs generation from an optimized per-layer layout.
- `validate`: checks tokenizer, logits, and generation output against JSON fixtures.
- `serve`: starts the HTTP server from the CLI package.

The CLI resolves config defaults, downloads Hugging Face files when requested, builds `RawLlm`, optionally preloads layer weights, and prints timings. Generation commands support debug logs and token streaming to stdout.

### `layerrun-server`

`crates/layerrun-server/src/main.rs` is an Axum server. It registers models from:

- a scanned `models_dir`
- a specific local `--model-dir`
- a Hugging Face `--hf-repo`

The server keeps a `ModelCatalog` of model specs and lazily loads each requested model into a cache keyed by public model id. Loaded models contain a tokenizer, chat template, and `RawLlm` protected by a mutex, so generation for a loaded model uses one mutable runtime instance at a time.

Implemented endpoints include:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/ps`

Completion and chat endpoints support non-streaming JSON responses and Server-Sent Events when `stream` is true. Sampling parameters are validated at the HTTP boundary and passed into core as `SamplingConfig`.

### `layerrun-core::config`

`ModelConfig` reads `config.json` and normalizes fields needed by execution:

- model family detection: Qwen, Llama, Mistral, Gemma4, or unknown
- hidden size, layer count, vocabulary size
- attention head and key/value head counts
- optional sliding-window attention fields
- RoPE theta, layer-specific RoPE parameters, and partial rotary factors
- activation type, final logit softcapping, and model-specific fields
- BOS/EOS/PAD token IDs, including single or multiple EOS values
- Gemma4-style per-layer input fields

This module is the bridge between Hugging Face config variants and the runtime's assumptions.

### `layerrun-core::safetensor_loader`

This module loads tensors from:

- a single `.safetensors` file
- a sharded model with `model.safetensors.index.json`

Single files are memory mapped with `memmap2`; metadata and tensor views come from `safetensors`. The loader converts supported formats into LayerRun tensors and handles alternate tensor prefixes such as `model.language_model.*`.

Supported tensor representations include:

- `F32`
- `F16`
- `BF16`
- Gemma QAT-style quantized tensors where supported

### `layerrun-core::tensor`

`Tensor` is the internal tensor representation. It stores shape plus backing data for dense floats, half/bfloat bits, and supported quantized formats. Runtime code reads values through helpers such as `value`, `row_f32`, and `dot_row`.

That interface keeps model code format-agnostic, but it also means some non-F32 paths use scalar conversion during math. The design currently favors compatibility and debuggability over maximum throughput.

### `layerrun-core::backend`

`BackendOps` defines the math operations used by model execution:

- linear projection
- RMSNorm
- elementwise add and multiply
- SiLU and GELU activation
- softmax

`CpuBackend` is the default and delegates to the `ops` kernels. `MlxBackend` is available behind the `mlx` Cargo feature and currently accelerates dense F32 linear projections, while unsupported operations and tensor formats fall back to CPU semantics when the feature is enabled. If LayerRun is built without the feature, requesting `--backend mlx` fails during model setup.

### `layerrun-core::weights`

`DecoderLayerWeights` is the per-layer weight bundle consumed by the forward pass. It loads and validates:

- attention norms
- Q/K/V/O projections and optional biases
- optional Q/K norms
- MLP gate/up/down projections and optional biases
- optional post-residual norms, layer scalars, and per-layer input adapters

Unsupported checkpoint layouts, such as GPTQ `qweight` tensors or Qwen linear-attention layers, are rejected explicitly during load.

### `layerrun-core::layer_store`

`LayerStore` abstracts the optimized per-layer directory. It resolves:

- `embeddings.safetensors`
- `layer_XXX.safetensors`
- `final.safetensors`

It also loads optional Gemma4 per-layer input embeddings and projection weights from the embeddings file. `RawLlm` uses `LayerStore` only for layered models; standard models load layer weights from `SafeTensorSource`.

### `layerrun-core::model`

`RawLlm` is the runtime boundary. It owns:

- `ModelConfig`
- selected backend
- embedding tensors and optional per-layer input tensors
- final norm and LM head
- optional `LayerStore`
- optional full-model `SafeTensorSource`
- optional preloaded `DecoderLayerWeights`

There are two load paths:

```text
RawLlm::load()
  standard Hugging Face layout
  config.json + tokenizer.json + model.safetensors or shards

RawLlm::load_layered()
  LayerRun optimized layout
  config.json + tokenizer.json + embeddings.safetensors + layer_XXX.safetensors + final.safetensors
```

Generation is token by token. Greedy generation is `temperature = 0`; sampled generation uses temperature, optional top-k, and top-p filtering.

```text
input token IDs
    │
    ├── prefill all prompt tokens except the last token
    │
    ▼
current token
    │
    ▼
embed token
    │
    ▼
for each decoder layer
    ├── load preloaded layer, layered file, or source tensor slice
    ├── RMSNorm
    ├── attention with KV cache
    ├── residual add
    ├── RMSNorm / optional post-residual norms
    ├── MLP
    ├── optional per-layer input adapter
    └── residual add
    │
    ▼
final RMSNorm
    │
    ▼
LM head projection
    │
    ▼
argmax or sampled next token
```

`KvCache` stores prior keys and values per layer. Each generated token appends one K/V entry per layer. Some Gemma4-style shared-attention configurations can reuse cache state across configured layers.

### `layerrun-core::chat_template`

`ChatTemplate` renders server chat messages into model prompts. It detects templates from tokenizer/config metadata where possible, falls back to model-family rules for Gemma4, and otherwise uses a plain role-prefixed prompt.

The server uses this path only for `/v1/chat/completions`; raw `/v1/completions` prompts are passed through without chat wrapping.

### `layerrun-core::ops`

This module contains the CPU kernels used by the default backend:

- matvec and linear projection
- RMSNorm
- SiLU and GELU activation
- vector add/multiply
- RoPE variants
- softmax and argmax
- optional logit softcapping

Large row-wise matvec operations use Rayon parallel iterators. There is no BLAS dependency.

### `layerrun-core::optimizer`

The optimizer converts a standard model layout into the LayerRun per-layer layout. It is a file-layout optimization, not a numerical optimization pass.

```text
input model directory
├── config.json
├── tokenizer.json
└── model.safetensors or sharded safetensors

optimized LayerRun directory
├── config.json
├── tokenizer.json
├── embeddings.safetensors
├── layer_000.safetensors
├── layer_001.safetensors
├── ...
├── final.safetensors
└── layerrun.json
```

The goal is to align files with decoder execution so layers can be paged, partially preloaded, or fully preloaded.

### `layerrun-core::huggingface`

`HuggingFaceSource` downloads required files from Hugging Face:

- `config.json`
- `tokenizer.json`
- requested weights, usually `model.safetensors`
- shard index and shard files when needed

Files are cached under `~/.cache/layerrun/huggingface` by default, or under the configured/CLI-provided cache directory. Tokens can come from CLI arguments, saved config, or `HF_TOKEN`.

## Runtime Data Flow

### Direct Generation

```text
CLI generate / server standard model
    │
    ├── resolve local or Hugging Face directory
    ├── load tokenizer and config
    ├── RawLlm::load()
    │     ├── open single or sharded safetensors source
    │     ├── load embeddings
    │     ├── load final norm
    │     └── load lm_head or tie to embeddings
    │
    └── generate_with_sampling...
          └── layer weights are loaded from source unless preloaded state exists
```

This path works directly against Hugging Face-style directories. It is the simplest compatibility mode, but repeated layer loading can dominate generation time.

### Optimized Layered Generation

```text
CLI optimize
    │
    └── split model weights into embeddings, per-layer files, and final weights

CLI generate-layered / server --layered
    │
    ├── RawLlm::load_layered()
    ├── optionally preload all or the first N layer files
    └── generate_with_sampling...
```

The layered layout gives the runtime explicit control over I/O and memory. `--preload-layers` loads every layer once. `--preload-layer-count N` keeps the first `N` layers resident and streams the rest.

### HTTP Serving

```text
HTTP request
    │
    ├── resolve model id from request or single-model catalog
    ├── lazily load model into ModelCatalog cache
    ├── render chat template for chat completions
    ├── tokenize prompt
    ├── run blocking generation on a worker thread
    ├── decode generated token IDs
    └── return JSON or SSE chunks
```

The server exposes OpenAI-compatible shapes for completions and chat completions, plus a small Ollama-style compatibility surface for model discovery and status.

### Validation

`validate` loads a model and fixture file, then checks:

- tokenizer output
- first next-token logits
- generated token IDs

This provides a regression path for model math and tokenizer behavior without requiring a running server.

## Performance Model

The current runtime is CPU-oriented and token-by-token. The most important speed factors are:

- Build mode: `cargo run --release` is required for meaningful performance.
- Layer loading: repeated layer loading is expensive; layered generation with preloading avoids most repeated I/O.
- Prompt length: prefill currently runs one token at a time.
- Matvec cost: projections dominate runtime and use Rayon row parallelism rather than BLAS or specialized SIMD kernels.
- Tensor format: F16/BF16 and quantized tensors may use scalar conversion during math.
- Backend: MLX currently helps dense F32 projections only when built with the `mlx` feature.

High-impact future improvements include batched prompt prefill, fused attention loops, optimized matvec kernels or BLAS integration, precomputed RoPE tables, flatter KV cache storage, memory budget controls for layer preloading, and broader MLX coverage.

## Current Limitations

- Prompt prefill is token-by-token.
- Streaming emits decoded token chunks, not a fully incremental tokenizer state machine.
- The server serializes generation per loaded model instance with a mutex.
- MLX support is feature-gated and partial.
- Some checkpoint formats are explicitly rejected, including GPTQ-style `qweight` tensors.
- Qwen linear-attention checkpoints are detected but not implemented.
- The runtime is an experimental implementation and is not a drop-in replacement for optimized inference engines.

## Design Intent

LayerRun is structured as a low-level experimental runtime. The architecture favors visibility into model files, tensor names, layer boundaries, and execution behavior over hiding those details behind a large inference framework.

The per-layer layout is the central design idea: model weights can be reorganized around decoder layer execution, letting the runtime choose between low-memory streaming behavior and faster preloaded behavior while keeping the original safetensors format understandable.
