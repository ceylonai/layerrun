# LayerRun Server API & Mode Reference

`layerrun-server` is an OpenAI-compatible and Ollama-style HTTP server that hosts models directly from local folders or resolves them from Hugging Face caches on the fly. It supports lazy loading, preloading, parameter-controlled sampling, and automated chat-template formatting.

## Running the Server

Start the server using the workspace package:

```sh
cargo run -p layerrun-server [options]
```

To run in production-optimized release mode:

```sh
cargo run --release -p layerrun-server [options]
```

### Discovery & Registration Examples

#### 1. Scan Model Directory (Default)
By default, the server scans the `models/` directory for subdirectories containing a valid `config.json` and `tokenizer.json` to expose in its catalog.
```sh
cargo run -p layerrun-server
```

#### 2. Scan Custom Directory
```sh
cargo run -p layerrun-server -- --models-dir /path/to/my/models
```

#### 3. Register Specific Local Single-File Model
```sh
cargo run -p layerrun-server -- \
  --model-dir models/qwen \
  --model-id qwen-base
```

#### 4. Register Specific LayerRun Optimized Per-Layer Model
```sh
cargo run -p layerrun-server -- \
  --model-dir models/qwen-layered \
  --layered \
  --model-id qwen-layered
```

#### 5. Serve directly from Hugging Face (downloads to local cache)
```sh
cargo run -p layerrun-server -- \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --model-id llama-1b
```

---

## Server Options

* `--host <HOST>`: IP address to bind (default: `127.0.0.1`).
* `--port <PORT>`: Port to bind (default: `8080`).
* `--models-dir <MODELS_DIR>`: Folder containing local models to automatically discover (default: `models`).
* `--model-id <MODEL_ID>`: Public model ID returned by `/v1/models` and used in request payloads.
* `--model-dir <MODEL_DIR>`: Path to an individual model directory to register.
* `--hf-repo <HF_REPO>`: Hugging Face repository ID.
* `--hf-revision <HF_REVISION>`: Hugging Face revision/commit hash (default: `main`).
* `--hf-token <HF_TOKEN>`: Hugging Face API Token.
* `--hf-cache-dir <HF_CACHE_DIR>`: Custom Hugging Face cache directory.
* `--weights <WEIGHTS>`: Filename for the weights inside the model directory (default: `model.safetensors`).
* `--layered`: Loads the model as an optimized per-layer directory structure.
* `--preload-layers`: Preload all per-layer weight files into memory at startup. *Requires `--layered`*.
* `--preload-layer-count <N>`: Preload only the first `N` layer weight files, streaming the rest from disk. *Requires `--layered`; conflicts with `--preload-layers`*.
* `--backend <BACKEND>`: Compute backend, either `cpu` or `mlx` (default: `cpu`).
* `--debug`: Prints step-by-step model execution information during inference.
* `--log-completions`: Outputs the full prompt text and completed output to `stderr` on completion.

---

## API Endpoints

### 1. Health Check
`GET /health`

#### Response
```json
{
  "status": "ok"
}
```

---

### 2. List Models (OpenAI-compatible)
`GET /v1/models`

Use one of the returned `data[].id` values as the `model` in completion requests.

#### Curl Example
```sh
curl http://127.0.0.1:8080/v1/models
export LAYERRUN_MODEL=gemma-4-E4B-it-qat-mobile-transformers
```

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-layered",
      "object": "model",
      "created": 1717789422,
      "owned_by": "layerrun"
    }
  ]
}
```

---

### 3. Create Completion (OpenAI-compatible)
`POST /v1/completions`

Generates completions for the provided prompt.

For instruction-tuned/chat models such as Gemma, prefer `/v1/chat/completions`.
Raw completion prompts are not wrapped in the model's chat template and can produce poor
continuations even when streaming is working correctly.

#### Payload
```json
{
  "model": "qwen-layered",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "temperature": 0.0,
  "top_k": 40,
  "top_p": 1.0,
  "stream": false
}
```

Set `"stream": true` to receive Server-Sent Events. Each event contains a JSON chunk in its
`data:` field, and the stream ends with `data: [DONE]`.

#### Curl Example
```sh
curl http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"$LAYERRUN_MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0.0
  }"
```

#### Response
```json
{
  "id": "cmpl-1717789430-1",
  "object": "text_completion",
  "created": 1717789430,
  "model": "qwen-layered",
  "choices": [
    {
      "text": " Paris.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 2,
    "total_tokens": 7
  }
}
```

---

#### Streaming Response
```text
data: {"id":"cmpl-1717789430-1","object":"text_completion","created":1717789430,"model":"qwen-layered","choices":[{"text":" Paris","index":0,"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"cmpl-1717789430-1","object":"text_completion","created":1717789430,"model":"qwen-layered","choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}

data: [DONE]
```

Example:
```sh
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"$LAYERRUN_MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"stream\": true
  }"
```

---

### 4. Create Chat Completion (OpenAI-compatible)
`POST /v1/chat/completions`

Generates a chat response using structural model-specific template formatting.

#### Payload
```json
{
  "model": "qwen-layered",
  "messages": [
    { "role": "system", "content": "You are a helpful coding assistant." },
    { "role": "user", "content": "Write a hello world program in Rust" }
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "stream": false
}
```

#### Curl Example
```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"$LAYERRUN_MODEL\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful coding assistant.\"},
      {\"role\": \"user\", \"content\": \"Write a hello world program in Rust\"}
    ],
    \"max_tokens\": 64,
    \"temperature\": 0.7
  }"
```

#### Response
```json
{
  "id": "chatcmpl-1717789445-2",
  "object": "chat.completion",
  "created": 1717789445,
  "model": "qwen-layered",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is hello world in Rust:\n\n```rust\nfn main() {\n    println!(\"Hello, World!\");\n}\n```"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 28,
    "total_tokens": 52
  }
}
```

---

#### Streaming Response
When `"stream": true`, chat completions use OpenAI-style chat completion chunks:

```text
data: {"id":"chatcmpl-1717789445-2","object":"chat.completion.chunk","created":1717789445,"model":"qwen-layered","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-1717789445-2","object":"chat.completion.chunk","created":1717789445,"model":"qwen-layered","choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-1717789445-2","object":"chat.completion.chunk","created":1717789445,"model":"qwen-layered","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":1,"total_tokens":25}}

data: [DONE]
```

Example:
```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"$LAYERRUN_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write a short greeting\"}],
    \"max_tokens\": 32,
    \"stream\": true
  }"
```

---

### 5. Ollama Catalog Tags
`GET /api/tags`

Retrieves the catalog in Ollama format.

#### Response
```json
{
  "models": [
    {
      "name": "qwen-layered",
      "model": "qwen-layered",
      "modified_at": 1717789422,
      "size": 1256038912,
      "details": {
        "family": "layerrun",
        "format": "layerrun",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

---

### 6. Ollama Show Model
`POST /api/show`

Inspect details of a registered model directory.

#### Payload
```json
{
  "name": "qwen-layered"
}
```

#### Response
```json
{
  "modelfile": "FROM models/qwen-layered\nPARAMETER backend cpu\nPARAMETER layered true\nPARAMETER weights model.safetensors",
  "parameters": "",
  "template": "{{ .Prompt }}",
  "details": {
    "family": "layerrun",
    "format": "layerrun",
    "parameter_size": "",
    "quantization_level": ""
  },
  "model_info": {
    "layerrun.model": "qwen-layered",
    "layerrun.backend": "cpu",
    "layerrun.layered": true
  }
}
```

---

### 7. Ollama Loaded Models
`GET /api/ps`

Shows which models are currently loaded in-memory and their sizes.

#### Response
```json
{
  "models": [
    {
      "name": "qwen-layered",
      "model": "qwen-layered",
      "size": 1256038912,
      "expires_at": null,
      "size_vram": 0,
      "details": {
        "family": "layerrun",
        "format": "layerrun",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

---

## Sampling Behavior

The server supports sampling parameters: `temperature`, `top_k`, and `top_p`.
* **Greedy Mode**: Default when `temperature` is `0.0` or omitted. This ensures deterministic generation using the highest probability token (argmax).
* **Sampling Mode**: Triggered when `temperature` is greater than `0.0`.
  * If a `temperature > 0.0` is specified without an explicit `top_k`, the server defaults `top_k` to `40`. This prevents tail-token noise from degrading the generated response.
  * Validation rules enforce that `temperature` must be non-negative, and `top_p` must be in the `(0, 1.0]` range.

---

## Chat Templates

When using `/v1/chat/completions`, the server parses standard roles (`system`, `user`, `assistant`, `developer`). It automatically detects and formats model-specific chat templates from configuration metadata.
* Supports specialized formats for **Gemma 4** model roles and turn sequences.
* Default fallback template (for base or unrecognized model configurations) uses:
  ```text
  user: <content>
  assistant: <content>
  ```

---

## Logging Details

Generation telemetry is written to `stderr` and includes detailed phase-based durations:
1. `request_start`: Logs request metadata.
2. `load_start` / `load_done`: Time elapsed loading weights from disk.
3. `tokenize_start` / `tokenize_done`: Tokenizer execution timing.
4. `generation_start` / `generation_done`: Model forward pass iterations.
5. `decode_start` / `decode_done`: Token ID to string decoding time.
6. `request_done`: Cumulative completion summary.

To log full prompts and generated text, launch the server with the `--log-completions` option.

---

## JavaScript Client Example

Using `fetch` to query the OpenAI-compatible endpoint:

```javascript
async function generateGreeting() {
  const response = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "qwen-layered",
      messages: [
        { "role": "system", "content": "You are concise." },
        { "role": "user", "content": "Write a short greeting" }
      ],
      max_tokens: 16,
      temperature: 0.7
    })
  });

  const data = await response.json();
  console.log(data.choices[0].message.content);
}

generateGreeting();
```
