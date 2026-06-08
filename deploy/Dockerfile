FROM rust:1-bookworm AS builder

WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release --locked -p layerrun-cli

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /src/target/release/layerrun-cli /usr/local/bin/layerrun-cli

RUN mkdir -p /app/models /root/.cache/layerrun

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["layerrun-cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--hf-repo", "Qwen/Qwen3-0.6B", "--model-id", "qwen-0.6b"]
