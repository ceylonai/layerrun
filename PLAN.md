# LayerRun Server Stabilization Plan

## Goal

Make LayerRun produce trustworthy model output through an OpenAI-compatible server before expanding server features. The priority order is correctness, prompt formatting, performance, then broader API compatibility.

## 1. Validate Model Correctness

Current risk: bad output may come from model math, not sampling or server code.

Tasks:

- Add a reference validation harness that compares LayerRun against a trusted implementation such as Hugging Face Transformers for the same model.
- Validate tokenizer output for fixed prompts.
- Validate first-token logits against the reference runtime.
- Validate short greedy generations with `temperature: 0`.
- Store small fixtures for prompts, token ids, expected top logits, and expected generated ids.
- Add a command such as:

```sh
cargo run -p layerrun-cli -- validate \
  --model-dir models/gemma-4-E4B-it-qat-mobile-transformers \
  --fixtures fixtures/gemma.json
```

Acceptance criteria:

- Token ids match reference output.
- First-token top-N logits are close enough for the selected dtype/quantization path.
- Greedy generated token ids match for a short deterministic prompt, or every mismatch is explained.

## 2. Implement Model-Specific Chat Templates

Current risk: chat requests are rendered with a naive text format:

```text
user: ...
assistant:
```

That is usually wrong for instruct models.

Tasks:

- Load chat template metadata from tokenizer/config files when available.
- Add explicit template support for Gemma-style chat formatting.
- Keep a fallback plain prompt format only for base models or unknown models.
- Preserve message roles: `system`, `user`, `assistant`, and future `developer`.
- Add tests for rendered prompt strings and token ids.

Acceptance criteria:

- The chat prompt passed to the model matches the reference model's chat template.
- The same chat request produces the same input token ids as the reference tokenizer.

## 3. Improve Sampling Controls

Current status: `temperature`, `top_k`, and `top_p` exist, and `temperature > 0` defaults to `top_k: 40`.

Tasks:

- Add `repetition_penalty`.
- Add `frequency_penalty` and `presence_penalty` only if they map cleanly to the current generation loop.
- Add `stop` string support at the server layer.
- Add `seed` support for reproducible sampling.
- Return clear validation errors for invalid sampling parameters.

Acceptance criteria:

- `temperature: 0` remains deterministic.
- `temperature > 0` avoids full-vocabulary tail-token sampling by default.
- Stop strings can terminate output without returning unwanted suffix text.

## 4. Make Performance Predictable

Current risk: layered models are slow without preload; release builds are much faster than dev builds.

Tasks:

- Document release-mode serving as the default recommended path.
- Keep `--preload-layers` and `--preload-layer-count`.
- Add startup warnings when serving a layered model without preload.
- Add request log timings for model load, tokenization, generation, and decode.
- Add a simple benchmark command for tokens/sec.

Acceptance criteria:

- Users can see whether time is spent loading, tokenizing, generating, or decoding.
- Performance documentation gives clear commands for speed-first and memory-first modes.

## 5. Add Model Lifecycle Controls

Current risk: models load lazily and stay loaded forever.

Tasks:

- Add explicit load and unload endpoints.
- Add an idle keep-alive timeout.
- Add a max-loaded-models setting.
- Add `/api/ps` details that show loaded time, last-used time, and memory estimate.
- Avoid loading multiple copies of the same model concurrently.

Acceptance criteria:

- A user can list, load, unload, and inspect loaded models.
- The server can cap memory use in multi-model catalogs.

## 6. Improve Server Concurrency

Current risk: each loaded model is behind one mutex, so requests serialize.

Tasks:

- Keep one worker queue per loaded model for now.
- Make serialization explicit in logs.
- Later evaluate multiple model workers if memory allows.
- Consider request cancellation support.
- Consider streaming partial tokens after correctness is stable.

Acceptance criteria:

- Concurrent requests do not corrupt KV cache or model state.
- Logs make queueing visible.
- The server remains simple until correctness is proven.

## 7. Expand API Compatibility

Current status: basic OpenAI-compatible `/v1/models`, `/v1/completions`, and `/v1/chat/completions` exist. Basic Ollama-style catalog endpoints exist.

Tasks:

- Add OpenAI `stop` support.
- Add streaming responses.
- Add Ollama `/api/generate` and `/api/chat`.
- Add better error codes and parameter names.
- Add request/response examples for JS clients.

Acceptance criteria:

- Simple OpenAI SDK clients work without custom response handling.
- Simple Ollama-style local clients can list models and generate text.

## 8. Logging And Safety

Current status: completion phase logs are always printed; full prompt/output logs require `--log-completions`.

Tasks:

- Keep full prompt/output logging opt-in.
- Add structured JSON logs as an option.
- Add request ids to every error response.
- Add log redaction hooks before any non-local deployment.

Acceptance criteria:

- Debugging remains easy locally.
- Sensitive prompt/output text is not logged unless explicitly requested.

## Recommended Execution Order

1. Build the reference validation harness.
2. Fix any model math/tokenization mismatches.
3. Implement correct chat templates.
4. Re-test greedy generation with `temperature: 0`.
5. Tune sampling only after greedy output is sane.
6. Improve lifecycle/performance controls.
7. Add streaming and broader API compatibility.

## Immediate Next Step

Start with a deterministic greedy validation fixture for the Gemma model currently in `models/`. If greedy output is bad, focus on model math or prompt formatting. If greedy output is good, then tune sampling and server behavior.
