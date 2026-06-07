use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use layerrun_core::{
    backend::BackendKind, huggingface::HuggingFaceSource, model::RawLlm,
    tokenizer_wrap::LayerTokenizer,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Parser, Debug)]
#[command(name = "layerrun-server")]
#[command(about = "Serve LayerRun models through an OpenAI-compatible HTTP API")]
struct Cli {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Directory containing local model directories to serve.
    #[arg(long, default_value = "models")]
    models_dir: String,

    /// Public model id returned by /v1/models and accepted in requests.
    #[arg(long)]
    model_id: Option<String>,

    /// Additional local model directory to register.
    #[arg(long)]
    model_dir: Option<String>,

    /// Hugging Face repo id, for example meta-llama/Llama-3.2-1B-Instruct.
    #[arg(long)]
    hf_repo: Option<String>,

    /// Hugging Face revision, branch, or commit.
    #[arg(long)]
    hf_revision: Option<String>,

    /// Hugging Face token. If omitted, HF_TOKEN is used when present.
    #[arg(long)]
    hf_token: Option<String>,

    /// Directory for cached Hugging Face files.
    #[arg(long)]
    hf_cache_dir: Option<PathBuf>,

    /// Name of the safetensors file inside model_dir for non-layered models.
    #[arg(long, default_value = "model.safetensors")]
    weights: String,

    /// Load a LayerRun optimized per-layer model directory.
    #[arg(long)]
    layered: bool,

    /// Preload all per-layer weight files before serving.
    #[arg(long)]
    preload_layers: bool,

    /// Preload the first N per-layer weight files before serving.
    #[arg(long, conflicts_with = "preload_layers")]
    preload_layer_count: Option<usize>,

    /// Runtime backend to use for generation.
    #[arg(long, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,

    /// Print model execution debug logs during requests.
    #[arg(long)]
    debug: bool,

    /// Log full prompts and generated text for completion requests.
    #[arg(long)]
    log_completions: bool,
}

#[derive(Clone)]
struct AppState {
    catalog: Arc<ModelCatalog>,
    debug: bool,
    log_completions: bool,
}

struct ModelCatalog {
    specs: Vec<ModelSpec>,
    loaded: Mutex<HashMap<String, Arc<LoadedModel>>>,
}

#[derive(Clone)]
struct ModelSpec {
    id: String,
    source: ModelSource,
    backend: BackendKind,
    preload_layer_count: Option<usize>,
}

#[derive(Clone)]
enum ModelSource {
    Local {
        model_dir: String,
        weights: String,
        layered: bool,
    },
    HuggingFace {
        repo: String,
        revision: Option<String>,
        token: Option<String>,
        cache_dir: Option<PathBuf>,
        weights: String,
    },
}

struct LoadedModel {
    tokenizer: LayerTokenizer,
    model: Mutex<RawLlm>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let catalog = build_catalog(&cli)?;
    if catalog.specs.is_empty() {
        anyhow::bail!(
            "no models found; pass --model-dir/--hf-repo or add model directories under {}",
            cli.models_dir
        );
    }
    eprintln!("available models:");
    for spec in &catalog.specs {
        eprintln!("  {}", spec.id);
    }

    let state = AppState {
        catalog: Arc::new(catalog),
        debug: cli.debug,
        log_completions: cli.log_completions,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(create_completion))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/ps", get(ollama_ps))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .with_context(|| format!("invalid bind address {}:{}", cli.host, cli.port))?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("serving OpenAI-compatible API at http://{addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": state.catalog.specs.iter().map(|spec| {
            json!({
            "id": spec.id,
            "object": "model",
            "created": now_ts(),
            "owned_by": "layerrun"
            })
        }).collect::<Vec<_>>()
    }))
}

async fn ollama_tags(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "models": state.catalog.specs.iter().map(|spec| {
            json!({
                "name": spec.id,
                "model": spec.id,
                "modified_at": now_ts(),
                "size": spec.size_bytes().unwrap_or(0),
                "details": spec.details()
            })
        }).collect::<Vec<_>>()
    }))
}

#[derive(Debug, Deserialize)]
struct OllamaShowRequest {
    name: Option<String>,
    model: Option<String>,
}

async fn ollama_show(
    State(state): State<AppState>,
    Json(request): Json<OllamaShowRequest>,
) -> Result<Json<Value>, ApiError> {
    let model_id = request
        .model
        .or(request.name)
        .ok_or_else(|| ApiError::bad_request("missing model name"))?;
    let spec = state.catalog.find(&model_id)?;
    Ok(Json(json!({
        "modelfile": spec.modelfile(),
        "parameters": "",
        "template": "{{ .Prompt }}",
        "details": spec.details(),
        "model_info": {
            "layerrun.model": spec.id,
            "layerrun.backend": spec.backend.as_str(),
            "layerrun.layered": spec.layered()
        }
    })))
}

async fn ollama_ps(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let loaded = state
        .catalog
        .loaded
        .lock()
        .map_err(|_| ApiError::internal(anyhow::anyhow!("model cache lock poisoned")))?;
    Ok(Json(json!({
        "models": loaded.keys().map(|id| {
            json!({
                "name": id,
                "model": id,
                "size": state.catalog.spec_size_bytes(id).unwrap_or(0),
                "expires_at": null,
                "size_vram": 0,
                "details": state.catalog.spec_details(id).unwrap_or_else(|| json!({}))
            })
        }).collect::<Vec<_>>()
    })))
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: Prompt,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Prompt {
    String(String),
    Strings(Vec<String>),
}

async fn create_completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<Value>, ApiError> {
    reject_streaming(request.stream)?;
    let model_id = state
        .catalog
        .resolve_request_model(request.model.as_deref())?;

    let prompt = match request.prompt {
        Prompt::String(prompt) => prompt,
        Prompt::Strings(prompts) => prompts.join("\n"),
    };
    let max_tokens = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(16);

    let generated = generate_text(&state, &model_id, &prompt, max_tokens).await?;
    log_completion("completion", &model_id, &prompt, &generated, &state);

    Ok(Json(json!({
        "id": next_id("cmpl"),
        "object": "text_completion",
        "created": now_ts(),
        "model": model_id,
        "choices": [{
            "text": generated.text,
            "index": 0,
            "logprobs": null,
            "finish_reason": generated.finish_reason
        }],
        "usage": {
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "total_tokens": generated.total_tokens
        }
    })))
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: ChatContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatContent {
    String(String),
    Parts(Vec<ChatContentPart>),
    Null(()),
}

#[derive(Debug, Deserialize)]
struct ChatContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

async fn create_chat_completion(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<Value>, ApiError> {
    reject_streaming(request.stream)?;
    let model_id = state
        .catalog
        .resolve_request_model(request.model.as_deref())?;
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let prompt = render_chat_prompt(&request.messages);
    let max_tokens = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(16);
    let generated = generate_text(&state, &model_id, &prompt, max_tokens).await?;
    log_completion("chat.completion", &model_id, &prompt, &generated, &state);

    Ok(Json(json!({
        "id": next_id("chatcmpl"),
        "object": "chat.completion",
        "created": now_ts(),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated.text
            },
            "finish_reason": generated.finish_reason
        }],
        "usage": {
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "total_tokens": generated.total_tokens
        }
    })))
}

struct GeneratedText {
    request_id: u64,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    finish_reason: &'static str,
    elapsed_ms: f64,
}

async fn generate_text(
    state: &AppState,
    model_id: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<GeneratedText, ApiError> {
    let state = state.clone();
    let model_id = model_id.to_string();
    let prompt = prompt.to_string();
    let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();

    eprintln!(
        "[completion:{request_id}] request_start model={} max_tokens={} prompt_chars={}",
        model_id,
        max_tokens,
        prompt.chars().count(),
    );

    let mut generated = tokio::task::spawn_blocking(move || {
        let phase_started = Instant::now();
        eprintln!("[completion:{request_id}] load_start model={model_id}");
        let loaded = state.catalog.load(&model_id, state.debug)?;
        eprintln!(
            "[completion:{request_id}] load_done elapsed_ms={:.3}",
            phase_started.elapsed().as_secs_f64() * 1000.0
        );

        let phase_started = Instant::now();
        eprintln!("[completion:{request_id}] tokenize_start");
        let input_ids = loaded
            .tokenizer
            .encode(&prompt)
            .map_err(ApiError::internal)?;
        eprintln!(
            "[completion:{request_id}] tokenize_done prompt_tokens={} elapsed_ms={:.3}",
            input_ids.len(),
            phase_started.elapsed().as_secs_f64() * 1000.0
        );

        let input_ids_usize: Vec<usize> = input_ids.iter().map(|id| *id as usize).collect();
        let output_ids = {
            let model = loaded
                .model
                .lock()
                .map_err(|_| ApiError::internal(anyhow::anyhow!("model lock poisoned")))?;
            let phase_started = Instant::now();
            eprintln!(
                "[completion:{request_id}] generation_start prompt_tokens={} max_tokens={}",
                input_ids_usize.len(),
                max_tokens,
            );
            model
                .generate_greedy_with_debug(&input_ids_usize, max_tokens, state.debug)
                .map_err(ApiError::internal)
                .inspect(|output_ids| {
                    eprintln!(
                        "[completion:{request_id}] generation_done output_tokens={} elapsed_ms={:.3}",
                        output_ids.len(),
                        phase_started.elapsed().as_secs_f64() * 1000.0
                    );
                })?
        };

        let generated_ids: Vec<u32> = output_ids
            .iter()
            .skip(input_ids.len())
            .map(|id| *id as u32)
            .collect();
        let phase_started = Instant::now();
        eprintln!(
            "[completion:{request_id}] decode_start completion_tokens={}",
            generated_ids.len()
        );
        let text = loaded
            .tokenizer
            .decode(&generated_ids)
            .map_err(ApiError::internal)?;
        eprintln!(
            "[completion:{request_id}] decode_done output_chars={} elapsed_ms={:.3}",
            text.chars().count(),
            phase_started.elapsed().as_secs_f64() * 1000.0
        );
        let completion_tokens = generated_ids.len();

        Ok(GeneratedText {
            request_id,
            text,
            prompt_tokens: input_ids.len(),
            completion_tokens,
            total_tokens: input_ids.len() + completion_tokens,
            finish_reason: if completion_tokens < max_tokens {
                "stop"
            } else {
                "length"
            },
            elapsed_ms: 0.0,
        })
    })
    .await
    .map_err(|err| ApiError::internal(anyhow::anyhow!("generation task failed: {err}")))??;

    generated.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[completion:{}] request_done elapsed_ms={:.3}",
        generated.request_id, generated.elapsed_ms
    );
    Ok(generated)
}

fn log_completion(
    endpoint: &str,
    model_id: &str,
    prompt: &str,
    generated: &GeneratedText,
    state: &AppState,
) {
    eprintln!(
        "[completion:{}] endpoint={} model={} prompt_tokens={} completion_tokens={} total_tokens={} finish_reason={} elapsed_ms={:.3}",
        generated.request_id,
        endpoint,
        model_id,
        generated.prompt_tokens,
        generated.completion_tokens,
        generated.total_tokens,
        generated.finish_reason,
        generated.elapsed_ms,
    );

    if state.log_completions {
        eprintln!("[completion:{}] prompt_start", generated.request_id);
        eprintln!("{prompt}");
        eprintln!("[completion:{}] prompt_end", generated.request_id);
        eprintln!("[completion:{}] output_start", generated.request_id);
        eprintln!("{}", generated.text);
        eprintln!("[completion:{}] output_end", generated.request_id);
    }
}

fn render_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let content = match &message.content {
            ChatContent::String(content) => content.clone(),
            ChatContent::Parts(parts) => parts
                .iter()
                .filter(|part| part.kind == "text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
            ChatContent::Null(_) => String::new(),
        };
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        prompt.push_str(&content);
        prompt.push('\n');
    }
    prompt.push_str("assistant: ");
    prompt
}

fn reject_streaming(stream: Option<bool>) -> Result<(), ApiError> {
    if stream.unwrap_or(false) {
        return Err(ApiError::bad_request(
            "streaming responses are not supported yet",
        ));
    }
    Ok(())
}

fn build_catalog(cli: &Cli) -> Result<ModelCatalog> {
    let mut specs = Vec::new();
    let catalog_preload_layer_count = catalog_preload_layer_count(cli);
    discover_local_models(
        &cli.models_dir,
        cli.backend,
        catalog_preload_layer_count,
        &mut specs,
    )?;

    match (&cli.model_dir, &cli.hf_repo) {
        (Some(model_dir), None) => {
            let id = cli
                .model_id
                .clone()
                .or_else(|| model_dir_name(model_dir))
                .unwrap_or_else(|| "layerrun".to_string());
            let preload_layer_count =
                preload_layer_count(cli.layered, cli.preload_layers, cli.preload_layer_count)?;
            push_unique(
                &mut specs,
                ModelSpec {
                    id,
                    source: ModelSource::Local {
                        model_dir: model_dir.clone(),
                        weights: cli.weights.clone(),
                        layered: cli.layered,
                    },
                    backend: cli.backend,
                    preload_layer_count,
                },
            );
        }
        (None, Some(hf_repo)) => {
            if cli.layered {
                anyhow::bail!("--layered is only supported with local --model-dir models");
            }
            let id = cli.model_id.clone().unwrap_or_else(|| hf_repo.clone());
            push_unique(
                &mut specs,
                ModelSpec {
                    id,
                    source: ModelSource::HuggingFace {
                        repo: hf_repo.clone(),
                        revision: cli.hf_revision.clone(),
                        token: cli.hf_token.clone(),
                        cache_dir: cli.hf_cache_dir.clone(),
                        weights: cli.weights.clone(),
                    },
                    backend: cli.backend,
                    preload_layer_count: None,
                },
            );
        }
        (Some(_), Some(_)) => anyhow::bail!("pass either --model-dir or --hf-repo, not both"),
        (None, None) => {}
    }

    Ok(ModelCatalog {
        specs,
        loaded: Mutex::new(HashMap::new()),
    })
}

fn discover_local_models(
    models_dir: &str,
    backend: BackendKind,
    preload_layer_count: Option<usize>,
    specs: &mut Vec<ModelSpec>,
) -> Result<()> {
    let models_path = Path::new(models_dir);
    if !models_path.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(models_path)
        .with_context(|| format!("failed to read models directory {models_dir}"))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(id) = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
        else {
            continue;
        };
        let Some(spec) = local_model_spec(id, path, backend, preload_layer_count) else {
            continue;
        };
        push_unique(specs, spec);
    }

    specs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(())
}

fn local_model_spec(
    id: String,
    model_dir: PathBuf,
    backend: BackendKind,
    preload_layer_count: Option<usize>,
) -> Option<ModelSpec> {
    if !model_dir.join("config.json").exists() || !model_dir.join("tokenizer.json").exists() {
        return None;
    }

    let layered = model_dir.join("layerrun.json").exists();
    let weights = if layered {
        "model.safetensors".to_string()
    } else if model_dir.join("model.safetensors").exists() {
        "model.safetensors".to_string()
    } else {
        let first_safetensors = fs::read_dir(&model_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|ext| ext == "safetensors"))?;
        first_safetensors.file_name()?.to_string_lossy().to_string()
    };

    Some(ModelSpec {
        id,
        source: ModelSource::Local {
            model_dir: model_dir.display().to_string(),
            weights,
            layered,
        },
        backend,
        preload_layer_count: if layered { preload_layer_count } else { None },
    })
}

fn push_unique(specs: &mut Vec<ModelSpec>, spec: ModelSpec) {
    if let Some(existing) = specs.iter_mut().find(|existing| existing.id == spec.id) {
        *existing = spec;
    } else {
        specs.push(spec);
    }
}

fn preload_layer_count(
    layered: bool,
    preload_layers: bool,
    preload_layer_count: Option<usize>,
) -> Result<Option<usize>> {
    if preload_layers || preload_layer_count.is_some() {
        if !layered {
            anyhow::bail!("--preload-layers and --preload-layer-count require --layered");
        }
        Ok(Some(preload_layer_count.unwrap_or(usize::MAX)))
    } else {
        Ok(None)
    }
}

fn catalog_preload_layer_count(cli: &Cli) -> Option<usize> {
    if cli.preload_layers || cli.preload_layer_count.is_some() {
        Some(cli.preload_layer_count.unwrap_or(usize::MAX))
    } else {
        None
    }
}

impl ModelCatalog {
    fn resolve_request_model(&self, request_model: Option<&str>) -> Result<String, ApiError> {
        match request_model {
            Some(model) => {
                self.find(model)?;
                Ok(model.to_string())
            }
            None if self.specs.len() == 1 => Ok(self.specs[0].id.clone()),
            None => Err(ApiError::bad_request(
                "model is required when multiple models are available",
            )),
        }
    }

    fn find(&self, model_id: &str) -> Result<&ModelSpec, ApiError> {
        self.specs
            .iter()
            .find(|spec| spec.id == model_id)
            .ok_or_else(|| ApiError::bad_request(format!("unknown model `{model_id}`")))
    }

    fn load(&self, model_id: &str, debug: bool) -> Result<Arc<LoadedModel>, ApiError> {
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| ApiError::internal(anyhow::anyhow!("model cache lock poisoned")))?;
        if let Some(model) = loaded.get(model_id) {
            return Ok(Arc::clone(model));
        }

        let spec = self.find(model_id)?.clone();
        let started = Instant::now();
        eprintln!("[model:{model_id}] load_start");
        let model = Arc::new(spec.load(debug).map_err(ApiError::internal)?);
        eprintln!(
            "[model:{model_id}] load_done elapsed_ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        loaded.insert(model_id.to_string(), Arc::clone(&model));
        Ok(model)
    }

    fn spec_size_bytes(&self, model_id: &str) -> Option<u64> {
        self.specs
            .iter()
            .find(|spec| spec.id == model_id)
            .and_then(|spec| spec.size_bytes())
    }

    fn spec_details(&self, model_id: &str) -> Option<Value> {
        self.specs
            .iter()
            .find(|spec| spec.id == model_id)
            .map(|spec| spec.details())
    }
}

impl ModelSpec {
    fn load(&self, debug: bool) -> Result<LoadedModel> {
        let total_started = Instant::now();
        eprintln!("[model:{}] resolve_start", self.id);
        let (model_dir, weights, layered) = self.resolve_model_dir()?;
        eprintln!(
            "[model:{}] resolve_done model_dir={} layered={} weights={} elapsed_ms={:.3}",
            self.id,
            model_dir,
            layered,
            weights,
            total_started.elapsed().as_secs_f64() * 1000.0
        );

        let phase_started = Instant::now();
        eprintln!("[model:{}] tokenizer_load_start", self.id);
        let tokenizer = LayerTokenizer::from_file(Path::new(&model_dir).join("tokenizer.json"))
            .with_context(|| format!("failed to load tokenizer from {model_dir}"))?;
        eprintln!(
            "[model:{}] tokenizer_load_done elapsed_ms={:.3}",
            self.id,
            phase_started.elapsed().as_secs_f64() * 1000.0
        );

        let phase_started = Instant::now();
        eprintln!(
            "[model:{}] weights_load_start source={} backend={}",
            self.id,
            if layered { "layered" } else { "safetensors" },
            self.backend
        );
        let mut model = if layered {
            RawLlm::load_layered(&model_dir)
                .with_context(|| format!("failed to load layered model from {model_dir}"))?
        } else {
            RawLlm::load(&model_dir, &weights)
                .with_context(|| format!("failed to load model from {model_dir}"))?
        }
        .with_backend(self.backend)?;
        eprintln!(
            "[model:{}] weights_load_done elapsed_ms={:.3}",
            self.id,
            phase_started.elapsed().as_secs_f64() * 1000.0
        );

        if let Some(count) = self.preload_layer_count {
            if !layered {
                anyhow::bail!("cannot preload layers for non-layered model {}", self.id);
            }
            let count = count.min(model.config.num_hidden_layers);
            let phase_started = Instant::now();
            eprintln!(
                "[model:{}] preload_start layers={count}/{}",
                self.id, model.config.num_hidden_layers
            );
            model.preload_layer_count_with_debug(count, debug)?;
            eprintln!(
                "[model:{}] preload_done elapsed_ms={:.3}",
                self.id,
                phase_started.elapsed().as_secs_f64() * 1000.0
            );
        }

        eprintln!(
            "[model:{}] ready elapsed_ms={:.3}",
            self.id,
            total_started.elapsed().as_secs_f64() * 1000.0
        );

        Ok(LoadedModel {
            tokenizer,
            model: Mutex::new(model),
        })
    }

    fn resolve_model_dir(&self) -> Result<(String, String, bool)> {
        match &self.source {
            ModelSource::Local {
                model_dir,
                weights,
                layered,
            } => Ok((model_dir.clone(), weights.clone(), *layered)),
            ModelSource::HuggingFace {
                repo,
                revision,
                token,
                cache_dir,
                weights,
            } => {
                let source = HuggingFaceSource::new(repo.clone())?
                    .with_revision(revision.clone())
                    .with_token(token.clone())
                    .with_cache_dir(cache_dir.clone());

                eprintln!(
                    "resolving Hugging Face repo {} at revision {}...",
                    source.repo_id(),
                    source.revision()
                );

                let model_dir = source.prepare_model_dir(weights)?;
                Ok((model_dir.display().to_string(), weights.clone(), false))
            }
        }
    }

    fn layered(&self) -> bool {
        matches!(&self.source, ModelSource::Local { layered: true, .. })
    }

    fn size_bytes(&self) -> Option<u64> {
        match &self.source {
            ModelSource::Local { model_dir, .. } => dir_size(Path::new(model_dir)).ok(),
            ModelSource::HuggingFace { .. } => None,
        }
    }

    fn details(&self) -> Value {
        json!({
            "family": "layerrun",
            "format": if self.layered() { "layerrun" } else { "safetensors" },
            "parameter_size": "",
            "quantization_level": "",
        })
    }

    fn modelfile(&self) -> String {
        match &self.source {
            ModelSource::Local {
                model_dir,
                weights,
                layered,
            } => format!(
                "FROM {}\nPARAMETER backend {}\nPARAMETER layered {}\nPARAMETER weights {}",
                model_dir, self.backend, layered, weights
            ),
            ModelSource::HuggingFace { repo, weights, .. } => format!(
                "FROM hf://{}\nPARAMETER backend {}\nPARAMETER weights {}",
                repo, self.backend, weights
            ),
        }
    }
}

fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn model_dir_name(model_dir: &str) -> Option<String> {
    Path::new(model_dir)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
}

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn next_id(prefix: &str) -> String {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{id}", now_ts())
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": if self.status == StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                "param": null,
                "code": null
            }
        }));
        (self.status, body).into_response()
    }
}
