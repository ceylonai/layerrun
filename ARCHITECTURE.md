# LayerRun Architecture

LayerRun is a Rust workspace for loading, inspecting, reorganizing, and running causal language models directly from Hugging Face-style `safetensors` files. The project avoids a higher-level inference framework and instead implements the model plumbing, tensor loading, layer execution, and generation loop inside `layerrun-core`.

## Workspace Layout

```text
LayerRun
├── crates/layerrun-core      Shared model runtime, tensor loading, tokenizer, optimizer
├── crates/layerrun-cli       Command-line interface for inspection, conversion, generation
└── crates/layerrun-server    Server crate stub
```

The core crate owns almost all behavior. The CLI crate parses commands and calls into `layerrun-core`. The server crate currently exists as a placeholder for a future service interface.

## Main Components

### `layerrun-cli`

`crates/layerrun-cli/src/main.rs` exposes the user-facing commands:

- `inspect`: prints tensor names, dtypes, and shapes from a `.safetensors` file.
- `tokenize`: tokenizes and decodes text through `tokenizer.json`.
- `probe-model`: loads model config and major tensors.
- `generate`: runs generation from a normal single-file or sharded Hugging Face model layout.
- `optimize`: converts a model into the LayerRun per-layer layout.
- `generate-layered`: runs generation from the optimized per-layer layout.

The CLI is intentionally thin. It resolves model paths, downloads Hugging Face files when requested, starts timers, and delegates model work to `RawLlm`.

### `layerrun-core::config`

`ModelConfig` reads `config.json` and normalizes fields needed by the runtime:

- model family detection: Qwen, Llama, Mistral, Gemma4, or unknown
- hidden size, layer count, vocabulary size
- attention head counts and key/value head counts
- optional sliding-window attention
- RoPE settings, including layer-type-specific theta values
- activation type and model-specific fields
- BOS/EOS/PAD token IDs

This layer is the bridge between Hugging Face config formats and the runtime's execution assumptions.

### `layerrun-core::safetensor_loader`

This module loads tensor data from either:

- a single `.safetensors` file
- a sharded model with `model.safetensors.index.json`

It memory maps single files with `memmap2`, uses `safetensors` for metadata and views, and converts supported dtypes into LayerRun tensors:

- `F32`
- `F16`
- `BF16`
- Gemma QAT-style quantized tensors where supported

It also handles alternate tensor prefixes such as `model.language_model.*`, which lets the runtime support checkpoints with slightly different naming conventions.

### `layerrun-core::tensor`

`Tensor` is the internal tensor representation. It stores shape plus one of several backing formats:

- dense `Vec<f32>`
- `F16` or `BF16` bits
- Gemma QAT `u8` or `i8` data with scales

The runtime accesses tensor values through helpers like `value`, `row_f32`, and `dot_row`. This keeps model code simple, but it also means some formats are dequantized or converted during scalar access. That design favors correctness and compatibility over maximum speed.

### `layerrun-core::weights`

`DecoderLayerWeights` is the per-layer weight bundle used by the forward pass. It loads:

- attention norms
- Q/K/V/O projections and optional biases
- optional Q/K norms
- MLP gate/up/down projections and optional biases
- optional model-specific norms and per-layer input projections

This module validates expected tensor shapes from `ModelConfig`. It also detects some unsupported checkpoint layouts, such as GPTQ `qweight` tensors or Qwen linear-attention layers, and returns explicit errors instead of failing later in model execution.

### `layerrun-core::model`

`RawLlm` is the runtime boundary. It owns:

- `ModelConfig`
- embedding tensors
- final norm
- LM head
- optional `LayerStore`
- optional full-model `SafeTensorSource`
- optional preloaded layer weights

There are two load paths:

```text
RawLlm::load()
  normal Hugging Face layout
  config.json + tokenizer.json + model.safetensors or shards

RawLlm::load_layered()
  LayerRun optimized layout
  config.json + tokenizer.json + embeddings.safetensors + layer_XXX.safetensors + final.safetensors
```

Generation is greedy. The current flow processes one token at a time:

```text
prompt token IDs
    │
    ▼
embed token
    │
    ▼
for each decoder layer
    ├── load or reuse layer weights
    ├── RMSNorm
    ├── attention with KV cache
    ├── residual add
    ├── RMSNorm
    ├── MLP
    └── residual add
    │
    ▼
final RMSNorm
    │
    ▼
LM head projection
    │
    ▼
argmax next token
```

`KvCache` stores prior keys and values per layer as vectors of token states. During generation, each new token appends one K/V entry per layer.

### `layerrun-core::ops`

This module contains the math kernels used by the runtime:

- matvec and linear projection
- RMSNorm
- SiLU and GELU activations
- vector add/multiply
- RoPE and Llama-style RoPE
- softmax and argmax

Large row-wise matvec operations use Rayon parallel iterators. This gives basic CPU parallelism without introducing a BLAS dependency.

### `layerrun-core::optimizer`

The optimizer converts a normal model layout into a LayerRun per-layer layout:

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

This is not a numerical optimization pass. It reorganizes tensors by runtime access pattern. The goal is to make it possible to load layer files independently, and optionally preload some or all layers into memory before generation.

### `layerrun-core::huggingface`

`HuggingFaceSource` downloads required files from Hugging Face:

- `config.json`
- `tokenizer.json`
- a requested weights file, usually `model.safetensors`
- shard index and shard files when needed

Files are cached under `~/.cache/layerrun/huggingface` by default, or under `--hf-cache-dir` when provided.

## Runtime Data Flow

### Direct Generation

```text
CLI generate
    │
    ├── resolve local or Hugging Face model directory
    ├── load tokenizer
    ├── RawLlm::load()
    │     ├── read config.json
    │     ├── open single or sharded safetensors source
    │     ├── load embeddings
    │     ├── load final norm
    │     └── load lm_head or tie to embeddings
    │
    └── generate_greedy()
          └── each token loads each layer from the source unless already preloaded
```

This path is simple and works directly from Hugging Face-style model directories. It can be slower because layer weights may be repeatedly loaded while generating.

### Optimized Layered Generation

```text
CLI optimize
    │
    └── split model weights into embeddings, per-layer files, and final weights

CLI generate-layered
    │
    ├── RawLlm::load_layered()
    ├── optionally preload layer files
    └── generate_greedy()
```

The layered layout gives the runtime more control over memory and I/O. With `--preload-layers`, layer weights are loaded once and reused across tokens.

## Performance Model

The current runtime is CPU-oriented and scalar-heavy. The most important speed factors are:

- Build mode: `cargo run --release` is required for meaningful performance.
- Layer loading: repeated layer loading is expensive; `generate-layered --preload-layers` avoids most repeated I/O.
- Prompt length: prefill currently runs token by token, so long prompts do more repeated per-token work than a batched prefill implementation would.
- Matvec cost: projections dominate runtime and currently use Rayon row parallelism rather than BLAS, SIMD-specialized kernels, or GPU kernels.
- Tensor format: F16/BF16 and quantized tensors are often converted through scalar access during math, which is flexible but not optimal.

The highest-impact future improvements are:

- batched prompt prefill
- fused attention loops that reduce allocations
- optimized matvec kernels or BLAS integration
- precomputed RoPE sin/cos tables
- flatter KV cache storage to improve locality
- optional memory budget controls for partial layer preloading

## Current Limitations

- Generation is greedy only.
- Prompt prefill is token-by-token.
- No GPU backend is present.
- Some checkpoint formats are explicitly rejected, including GPTQ-style `qweight` tensors.
- Qwen linear-attention checkpoints are detected but not implemented.
- The server crate is currently a stub.

## Design Intent

LayerRun is structured as a low-level experimental runtime. The architecture favors visibility into model files, tensor names, layer boundaries, and execution behavior over hiding those details behind a large inference framework.

The per-layer layout is the central design idea: model weights can be reorganized around decoder layer execution, letting the runtime choose between low-memory streaming behavior and faster preloaded behavior.
