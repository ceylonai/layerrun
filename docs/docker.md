# LayerRun Docker Usage Guide

This guide explains how to build and run the LayerRun server with Docker. The
container runs `layerrun-cli serve`, downloads a default Qwen model from
Hugging Face on first start, and exposes the OpenAI-compatible HTTP API on port
`8080`.

## Files

Docker deployment files live under `deploy/`:

* `deploy/Dockerfile`: Multi-stage image build for the release CLI binary.
* `deploy/docker-compose.yml`: Local service definition with a persistent cache
  volume.
* `deploy/Dockerfile.dockerignore`: Suggested ignore list for Docker builds.

## Prerequisites

* Docker Engine or Docker Desktop.
* Enough disk and memory for the model you plan to serve.
* A Hugging Face token only if you want to use gated or private models.

The default container model is `Qwen/Qwen3-0.6B` with the public model id
`qwen-0.6b`.

## Build the Image

From the repository root:

```sh
docker build -f deploy/Dockerfile -t layerrun:local .
```

The image builds the `layerrun-cli` release binary with:

```sh
cargo build --release --locked -p layerrun-cli
```

## Run with Docker

Run the default model download flow:

```sh
docker run --rm \
  -p 8080:8080 \
  -v layerrun-cache:/root/.cache/layerrun \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  layerrun:local
```

The default container command is equivalent to:

```sh
layerrun-cli serve \
  --host 0.0.0.0 \
  --port 8080 \
  --hf-repo Qwen/Qwen3-0.6B \
  --model-id qwen-0.6b
```

## Run with Docker Compose

Start the service:

```sh
docker compose -f deploy/docker-compose.yml up --build
```

Run it in the background:

```sh
docker compose -f deploy/docker-compose.yml up --build -d
```

View logs:

```sh
docker compose -f deploy/docker-compose.yml logs -f layerrun
```

Stop the service:

```sh
docker compose -f deploy/docker-compose.yml down
```

Remove the named Hugging Face/cache volume too:

```sh
docker compose -f deploy/docker-compose.yml down -v
```

## Hugging Face Tokens

For gated or private Hugging Face models, export `HF_TOKEN` before starting the
container:

```sh
export HF_TOKEN=hf_your_token_here
docker compose -f deploy/docker-compose.yml up --build
```

Compose passes `HF_TOKEN` into the service. The Docker run example does the same
with `-e HF_TOKEN="${HF_TOKEN:-}"`.

## Serving a Specific Model

Override the default command to register one local model explicitly:

```sh
docker run --rm \
  -p 8080:8080 \
  -v /var/lib/layerrun/models:/app/models \
  layerrun:local \
  serve \
  --host 0.0.0.0 \
  --port 8080 \
  --model-dir /app/models/qwen-layered \
  --layered \
  --model-id qwen-layered
```

For a different Hugging Face repository:

```sh
docker run --rm \
  -p 8080:8080 \
  -v layerrun-cache:/root/.cache/layerrun \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  layerrun:local \
  serve \
  --host 0.0.0.0 \
  --port 8080 \
  --hf-repo meta-llama/Llama-3.2-1B-Instruct \
  --model-id llama-1b
```

## Verify the Server

Health check:

```sh
curl http://127.0.0.1:8080/health
```

List registered models:

```sh
curl http://127.0.0.1:8080/v1/models
```

Chat completion:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen-layered",
    "messages": [
      { "role": "user", "content": "Write a short greeting." }
    ],
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

## Operational Notes

* The published container port is `8080`.
* The container health check calls `GET /health`.
* The default image entrypoint downloads `Qwen/Qwen3-0.6B` into the cache volume
  on first boot.
* Model files are not copied into the image. Mount them into `/app/models` only
  if you want to serve local files instead of Hugging Face downloads.
* Hugging Face downloads should use a persistent cache volume to avoid repeated
  downloads. The examples use `layerrun-cache` for that purpose.
* The Docker image builds the CPU backend. MLX acceleration is intended for
  native Apple Silicon builds, not this Linux container image.
