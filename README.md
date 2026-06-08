# LayerRun

LayerRun is a Rust workspace designed for experimenting with raw `safetensors`-level Large Language Model (LLM) loading, custom tokenizer plumbing, model probing, greedy/sampled text generation, and optimization into a per-layer model layout.

Unlike high-level inference frameworks, LayerRun implements model execution, tensor deserialization, attention logic, and server interfaces directly from scratch.

---

## Workspace Layout

The repository is organized as a Cargo workspace containing the following crates:

* **`layerrun-core`**: The shared runtime covering configurations, memory-mapped tensor loading, tokenization, compute backend abstraction, and mathematical kernels.
* **`layerrun-cli`**: Command-line interface for inspection, tokenizer execution, validation fixtures, and offline layout optimizations.
* **`layerrun-server`**: OpenAI-compatible and Ollama-style HTTP server providing local model serving.

---

## Documentation Index

Detailed guides are available in the following documents:

* 📖 **[CLI Reference Guide](file:///Users/dewmal/Projects/layer_run/docs/cli.md)**: How to run inspections, tokenize text, generate text, convert checkpoints, and run validation.
* 🖥️ **[Server API Guide](file:///Users/dewmal/Projects/layer_run/docs/server.md)**: Endpoints reference, sampling options, model-specific chat templates, and integration examples.
* 📐 **[Architecture Overview](file:///Users/dewmal/Projects/layer_run/ARCHITECTURE.md)**: Deep dive into the workspace crates, tensor representation, CPU/MLX math kernels, and data flow.
* 🎯 **[Stabilization & Features Plan](file:///Users/dewmal/Projects/layer_run/PLAN.md)**: Current roadmap, correctness validation strategy, and upcoming feature enhancements.

---

## Requirements

* **Rust Toolchain**: Stable compiler with `cargo`.
* **Local Model Files**: PyTorch/Hugging Face standard `.safetensors` files or optimized per-layer LayerRun directories.
* **Hugging Face Hub Access**: Gated/private models require a valid `HF_TOKEN` environment variable or `--hf-token` argument.
* **MLX Backend (Optional)**: Apple Silicon macOS with `cmake` and full Xcode installed (ensure `xcrun -find metal` succeeds).

---

## Common Commands

### 1. Build the Workspace
```sh
cargo build --release
```

### 2. Initialize Local Config
```sh
cargo run -p layerrun-cli -- init --models-dir models
```

This writes `$HOME/.layerrun-conf`, creates the models directory, and can save a Hugging Face token for later `--hf-repo` commands.

### 3. Run Workspace Tests
```sh
cargo test
```

### 4. Run the CLI Help
```sh
cargo run -p layerrun-cli -- --help
```

### 5. Build with MLX Acceleration
```sh
cargo build -p layerrun-cli --features mlx
```
*Note: Ensure full Xcode is selected if compiling MLX native dependencies:*
```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

### 6. Launch the Server
```sh
cargo run --release -p layerrun-cli -- serve
```

---

## Installed Example Models

The workspace comes with local models under the `models/` directory:
* `models/qwen`: Local standard single-file safetensors model.
* `models/qwen-layered`: LayerRun optimized per-layer model directory.
* `models/llama-3.2-1b-instruct`: Local standard sharded safetensors model.
* `models/llama-3.2-1b-instruct-layered`: LayerRun optimized per-layer directory.
* `models/mistral`: Local model files configuration.
