use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use layerrun_core::backend::BackendKind;
use layerrun_core::config::ModelConfig;
use layerrun_core::huggingface::HuggingFaceSource;
use layerrun_core::model::{RawLlm, SamplingConfig};
use layerrun_core::safetensor_loader::SafeTensorFile;
use layerrun_core::tokenizer_wrap::LayerTokenizer;
use layerrun_server::ServeConfig;
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(name = "layerrun_raw")]
#[command(about = "Raw safetensors-level LLM runtime skeleton without Candle")]
struct Cli {
    /// Path to the LayerRun config file.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create the local LayerRun config and models directory.
    Init {
        /// Directory where local LayerRun model directories are stored.
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,

        /// Hugging Face token to save. If omitted, the CLI prompts for it.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files.
        #[arg(long)]
        hf_cache_dir: Option<PathBuf>,
    },

    /// Inspect tensor names, dtypes, and shapes from a .safetensors file.
    Inspect {
        #[arg(long)]
        file: String,
    },

    /// Tokenize text using tokenizer.json.
    Tokenize {
        #[arg(long)]
        tokenizer: String,

        #[arg(long)]
        text: String,
    },

    /// Load model config and basic tensors.
    ProbeModel {
        /// Local model directory. Required unless --hf-repo is set.
        #[arg(long)]
        model_dir: Option<String>,

        /// Hugging Face repo id, for example meta-llama/Llama-3.2-1B-Instruct.
        #[arg(long)]
        hf_repo: Option<String>,

        /// Hugging Face revision, branch, or commit.
        #[arg(long)]
        hf_revision: Option<String>,

        /// Hugging Face token. If omitted, saved config then HF_TOKEN are used.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files. If omitted, saved config then defaults are used.
        #[arg(long)]
        hf_cache_dir: Option<PathBuf>,

        /// Name of the safetensors file inside model_dir.
        #[arg(long, default_value = "model.safetensors")]
        weights: String,
    },

    /// Placeholder generation path. This validates tokenizer/config/weights plumbing.
    Generate {
        /// Local model directory. Required unless --hf-repo is set.
        #[arg(long)]
        model_dir: Option<String>,

        /// Hugging Face repo id, for example meta-llama/Llama-3.2-1B-Instruct.
        #[arg(long)]
        hf_repo: Option<String>,

        /// Hugging Face revision, branch, or commit.
        #[arg(long)]
        hf_revision: Option<String>,

        /// Hugging Face token. If omitted, saved config then HF_TOKEN are used.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files. If omitted, saved config then defaults are used.
        #[arg(long)]
        hf_cache_dir: Option<PathBuf>,

        /// Name of the safetensors file inside model_dir.
        #[arg(long, default_value = "model.safetensors")]
        weights: String,

        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,

        #[arg(long)]
        debug: bool,

        /// Print generated text incrementally as each token is produced.
        #[arg(long)]
        stream: bool,

        /// Runtime backend to use for generation.
        #[arg(long, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
    },

    /// Create a LayerRun directory with embeddings, per-layer files, and final weights.
    Optimize {
        /// Local model directory. Required unless --hf-repo is set.
        #[arg(long)]
        input_model_dir: Option<String>,

        /// Hugging Face repo id, for example meta-llama/Llama-3.2-1B-Instruct.
        #[arg(long)]
        hf_repo: Option<String>,

        /// Hugging Face revision, branch, or commit.
        #[arg(long)]
        hf_revision: Option<String>,

        /// Hugging Face token. If omitted, saved config then HF_TOKEN are used.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files. If omitted, saved config then defaults are used.
        #[arg(long)]
        hf_cache_dir: Option<PathBuf>,

        #[arg(long)]
        output_model_dir: String,

        #[arg(long, default_value = "model.safetensors")]
        weights: String,
    },

    /// Generate from a LayerRun model directory with per-layer safetensors files.
    GenerateLayered {
        #[arg(long)]
        model_dir: String,

        #[arg(long)]
        prompt: String,

        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,

        #[arg(long)]
        debug: bool,

        /// Print generated text incrementally as each token is produced.
        #[arg(long)]
        stream: bool,

        #[arg(long)]
        preload_layers: bool,

        /// Preload the first N per-layer weight files before generation.
        #[arg(long, conflicts_with = "preload_layers")]
        preload_layer_count: Option<usize>,

        /// Runtime backend to use for generation.
        #[arg(long, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,
    },

    /// Validate LayerRun tokenizer, logits, and greedy generation against reference fixtures.
    Validate {
        #[arg(long)]
        model_dir: String,

        #[arg(long)]
        fixtures: PathBuf,

        /// Name of the safetensors file for non-layered model directories.
        #[arg(long, default_value = "model.safetensors")]
        weights: String,

        /// Runtime backend to use for validation.
        #[arg(long, default_value_t = BackendKind::Cpu)]
        backend: BackendKind,

        #[arg(long)]
        preload_layers: bool,

        /// Preload the first N per-layer weight files before validation.
        #[arg(long, conflicts_with = "preload_layers")]
        preload_layer_count: Option<usize>,
    },

    /// Serve models through the OpenAI-compatible HTTP API.
    Serve {
        /// Address to bind.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind.
        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Directory containing local model directories to serve.
        #[arg(long)]
        models_dir: Option<String>,

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

        /// Hugging Face token. If omitted, saved config then HF_TOKEN are used.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files. If omitted, saved config then defaults are used.
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
    },
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct LayerRunConfig {
    models_dir: Option<PathBuf>,
    hf_token: Option<String>,
    hf_cache_dir: Option<PathBuf>,
}

impl LayerRunConfig {
    fn load_optional(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create config dir {}", parent.display()))?;
        }

        let text = serde_json::to_string_pretty(self)?;
        fs::write(path, format!("{text}\n"))
            .with_context(|| format!("failed to write config {}", path.display()))?;
        set_private_permissions(path)?;
        Ok(())
    }
}

fn config_path(config: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(config) = config {
        return Ok(config);
    }

    let home = env::var_os("HOME").context("HOME is not set; pass --config")?;
    Ok(Path::new(&home).join(".layerrun-conf"))
}

fn prompt_hf_token() -> Result<Option<String>> {
    eprint!("Hugging Face token (leave blank to skip): ");
    io::stderr().flush()?;

    let mut token = String::new();
    io::stdin()
        .read_line(&mut token)
        .context("failed to read Hugging Face token")?;
    let token = token.trim().to_string();

    if token.is_empty() {
        Ok(None)
    } else {
        Ok(Some(token))
    }
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let permissions = fs::Permissions::from_mode(0o600);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("failed to set private permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = config_path(cli.config)?;
    let saved_config = match &cli.command {
        Commands::Init { .. } => LayerRunConfig::default(),
        _ => LayerRunConfig::load_optional(&config_path)?,
    };

    match cli.command {
        Commands::Init {
            models_dir,
            hf_token,
            hf_cache_dir,
        } => {
            let hf_token = match hf_token {
                Some(token) => Some(token),
                None => prompt_hf_token()?,
            };
            let config = LayerRunConfig {
                models_dir: Some(models_dir.clone()),
                hf_token,
                hf_cache_dir: hf_cache_dir.clone(),
            };
            fs::create_dir_all(&models_dir)
                .with_context(|| format!("failed to create models dir {}", models_dir.display()))?;
            if let Some(hf_cache_dir) = &hf_cache_dir {
                fs::create_dir_all(hf_cache_dir).with_context(|| {
                    format!("failed to create HF cache dir {}", hf_cache_dir.display())
                })?;
            }
            config.save(&config_path)?;
            println!("created models dir: {}", models_dir.display());
            if let Some(hf_cache_dir) = &config.hf_cache_dir {
                println!("created HF cache dir: {}", hf_cache_dir.display());
            }
            println!("wrote config: {}", config_path.display());
        }

        Commands::Inspect { file } => {
            let st = SafeTensorFile::open(file)?;
            st.print_summary()?;
        }

        Commands::Tokenize { tokenizer, text } => {
            let tok = LayerTokenizer::from_file(tokenizer)?;
            let ids = tok.encode(&text)?;
            println!("ids: {:?}", ids);
            println!("decoded: {}", tok.decode(&ids)?);
        }

        Commands::ProbeModel {
            model_dir,
            hf_repo,
            hf_revision,
            hf_token,
            hf_cache_dir,
            weights,
        } => {
            let model_dir = resolve_model_dir(
                model_dir,
                hf_repo,
                hf_revision,
                hf_token,
                hf_cache_dir,
                &saved_config,
                &weights,
            )?;
            let model = RawLlm::load(&model_dir, &weights)?;
            println!("Loaded model");
            println!("model_type: {}", model.config.model_type);
            println!("layers: {}", model.config.num_hidden_layers);
            println!("hidden_size: {}", model.config.hidden_size);
            println!("vocab_size: {}", model.config.vocab_size);
            println!("embed_tokens shape: {:?}", model.embed_tokens.shape);
            println!("final_norm shape: {:?}", model.final_norm.shape);
            println!("lm_head shape: {:?}", model.lm_head.shape);
        }

        Commands::Generate {
            model_dir,
            hf_repo,
            hf_revision,
            hf_token,
            hf_cache_dir,
            weights,
            prompt,
            max_new_tokens,
            debug,
            stream,
            backend,
        } => {
            let total_started = Instant::now();
            let model_dir = resolve_model_dir(
                model_dir,
                hf_repo,
                hf_revision,
                hf_token,
                hf_cache_dir,
                &saved_config,
                &weights,
            )?;
            let tok = LayerTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
            let ids = tok.encode(&prompt)?;
            println!("prompt token ids: {:?}", ids);

            let load_started = Instant::now();
            let model = RawLlm::load(&model_dir, &weights)?.with_backend(backend)?;
            let model_load_elapsed = load_started.elapsed();
            eprintln!("using {} backend", model.backend());
            if debug && model.backend() == BackendKind::Mlx {
                eprintln!(
                    "[debug] mlx backend accelerates dense f32 projections; non-f32/quantized tensors use CPU fallback"
                );
            }

            // This is intentionally a skeleton. Full generation needs tested RoPE/GQA
            // model execution. This proves raw safetensors-level plumbing first.
            let input_ids: Vec<usize> = ids.iter().map(|x| *x as usize).collect();

            let generation_started = Instant::now();
            let (output_ids_usize, generated_only, generated_decoded) =
                generate_greedy_cli(&model, &tok, &input_ids, max_new_tokens, debug, stream)?;
            let generation_elapsed = generation_started.elapsed();

            let output_ids: Vec<u32> = output_ids_usize.iter().map(|x| *x as u32).collect();

            println!("output ids: {:?}", output_ids);
            println!("generated ids: {:?}", generated_only);
            println!("generated decoded: {}", generated_decoded);
            println!(
                "timing: model_load_ms={:.3} generation_ms={:.3} total_ms={:.3}",
                model_load_elapsed.as_secs_f64() * 1000.0,
                generation_elapsed.as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Commands::Optimize {
            input_model_dir,
            hf_repo,
            hf_revision,
            hf_token,
            hf_cache_dir,
            output_model_dir,
            weights,
        } => {
            let input_model_dir = resolve_model_dir(
                input_model_dir,
                hf_repo,
                hf_revision,
                hf_token,
                hf_cache_dir,
                &saved_config,
                &weights,
            )?;
            let cfg = ModelConfig::from_model_dir(&input_model_dir)?;

            layerrun_core::optimizer::optimize_single_safetensors(
                &input_model_dir,
                &output_model_dir,
                &weights,
                cfg.num_hidden_layers,
            )?;

            println!("Layered model created at: {}", output_model_dir);
        }

        Commands::GenerateLayered {
            model_dir,
            prompt,
            max_new_tokens,
            debug,
            stream,
            preload_layers,
            preload_layer_count,
            backend,
        } => {
            let total_started = Instant::now();
            let tok = LayerTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
            let ids = tok.encode(&prompt)?;

            println!("prompt token ids: {:?}", ids);

            let load_started = Instant::now();
            let mut model = RawLlm::load_layered(&model_dir)?.with_backend(backend)?;
            let model_load_elapsed = load_started.elapsed();
            eprintln!("using {} backend", model.backend());
            if debug && model.backend() == BackendKind::Mlx {
                eprintln!(
                    "[debug] mlx backend accelerates dense f32 projections; non-f32/quantized tensors use CPU fallback"
                );
            }

            let preload_started = Instant::now();
            if preload_layers || preload_layer_count.is_some() {
                let count = preload_layer_count.unwrap_or(model.config.num_hidden_layers);
                eprintln!(
                    "preloading {count}/{} layer(s)...",
                    model.config.num_hidden_layers
                );
                model.preload_layer_count_with_debug(count, debug)?;
                eprintln!(
                    "preloaded {count}/{} layer(s) in {:.3} ms",
                    model.config.num_hidden_layers,
                    preload_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let preload_elapsed = preload_started.elapsed();

            let input_ids: Vec<usize> = ids.iter().map(|x| *x as usize).collect();

            let generation_started = Instant::now();
            eprintln!(
                "generating {} token(s){}...",
                max_new_tokens,
                if preload_layers || preload_layer_count.is_some() {
                    " with partially/fully preloaded layers"
                } else {
                    ""
                }
            );
            let (output_ids_usize, generated_only, generated_decoded) =
                generate_greedy_cli(&model, &tok, &input_ids, max_new_tokens, debug, stream)?;
            let generation_elapsed = generation_started.elapsed();

            let output_ids: Vec<u32> = output_ids_usize.iter().map(|x| *x as u32).collect();

            println!("output ids: {:?}", output_ids);
            println!("generated ids: {:?}", generated_only);
            println!("generated decoded: {}", generated_decoded);
            println!(
                "timing: model_load_ms={:.3} preload_ms={:.3} generation_ms={:.3} total_ms={:.3}",
                model_load_elapsed.as_secs_f64() * 1000.0,
                preload_elapsed.as_secs_f64() * 1000.0,
                generation_elapsed.as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }

        Commands::Validate {
            model_dir,
            fixtures,
            weights,
            backend,
            preload_layers,
            preload_layer_count,
        } => {
            run_validate(
                &model_dir,
                &fixtures,
                &weights,
                backend,
                preload_layers,
                preload_layer_count,
            )?;
        }

        Commands::Serve {
            host,
            port,
            models_dir,
            model_id,
            model_dir,
            hf_repo,
            hf_revision,
            hf_token,
            hf_cache_dir,
            weights,
            layered,
            preload_layers,
            preload_layer_count,
            backend,
            debug,
            log_completions,
        } => {
            let models_dir = models_dir
                .or_else(|| {
                    saved_config
                        .models_dir
                        .as_ref()
                        .map(|path| path.display().to_string())
                })
                .unwrap_or_else(|| "models".to_string());
            layerrun_server::serve(ServeConfig {
                host,
                port,
                models_dir,
                model_id,
                model_dir,
                hf_repo,
                hf_revision,
                hf_token: hf_token.or_else(|| saved_config.hf_token.clone()),
                hf_cache_dir: hf_cache_dir.or_else(|| saved_config.hf_cache_dir.clone()),
                weights,
                layered,
                preload_layers,
                preload_layer_count,
                backend,
                debug,
                log_completions,
            })
            .await?;
        }
    }

    Ok(())
}

fn generate_greedy_cli(
    model: &RawLlm,
    tokenizer: &LayerTokenizer,
    input_ids: &[usize],
    max_new_tokens: usize,
    debug: bool,
    stream: bool,
) -> Result<(Vec<usize>, Vec<u32>, String)> {
    if !stream {
        let output_ids = model.generate_greedy_with_debug(input_ids, max_new_tokens, debug)?;
        let generated_ids: Vec<u32> = output_ids
            .iter()
            .skip(input_ids.len())
            .map(|id| *id as u32)
            .collect();
        let generated_decoded = tokenizer.decode(&generated_ids)?;
        return Ok((output_ids, generated_ids, generated_decoded));
    }

    let mut generated_ids = Vec::<u32>::new();
    let mut decoded_text = String::new();
    let mut stdout = std::io::stdout().lock();

    let output_ids = model.generate_with_sampling_stream_with_debug(
        input_ids,
        max_new_tokens,
        SamplingConfig::greedy(),
        debug,
        |_, token_id| {
            generated_ids.push(token_id as u32);
            let next_text = tokenizer.decode(&generated_ids)?;
            let delta = next_text
                .strip_prefix(&decoded_text)
                .map(str::to_string)
                .unwrap_or_else(|| tokenizer.decode(&[token_id as u32]).unwrap_or_default());
            decoded_text = next_text;

            if !delta.is_empty() {
                write!(stdout, "{delta}")?;
                stdout.flush()?;
            }

            Ok(())
        },
    )?;
    writeln!(stdout)?;

    Ok((output_ids, generated_ids, decoded_text))
}

#[derive(Debug, Deserialize)]
struct ValidationFixtures {
    reference: ReferenceRuntime,
    #[serde(default = "default_top_n")]
    top_n: usize,
    #[serde(default = "default_logit_atol")]
    logit_atol: f32,
    #[serde(default = "default_logit_rtol")]
    logit_rtol: f32,
    #[serde(default = "default_true")]
    require_top_token_ids: bool,
    cases: Vec<ValidationCase>,
}

#[derive(Debug, Deserialize)]
struct ReferenceRuntime {
    runtime: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    revision: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ValidationCase {
    name: String,
    prompt: String,
    token_ids: Vec<u32>,
    first_token_top_logits: Vec<ExpectedLogit>,
    greedy: GreedyFixture,
}

#[derive(Debug, Deserialize)]
struct ExpectedLogit {
    token_id: u32,
    logit: f32,
}

#[derive(Debug, Deserialize)]
struct GreedyFixture {
    max_new_tokens: usize,
    #[serde(default)]
    temperature: f32,
    generated_ids: Vec<u32>,
    #[serde(default)]
    mismatches: Vec<ExplainedMismatch>,
}

#[derive(Debug, Deserialize)]
struct ExplainedMismatch {
    position: usize,
    expected: u32,
    actual: u32,
    reason: String,
}

fn default_top_n() -> usize {
    10
}

fn default_logit_atol() -> f32 {
    0.5
}

fn default_logit_rtol() -> f32 {
    0.05
}

fn default_true() -> bool {
    true
}

fn run_validate(
    model_dir: &str,
    fixtures_path: &Path,
    weights: &str,
    backend: BackendKind,
    preload_layers: bool,
    preload_layer_count: Option<usize>,
) -> Result<()> {
    let fixture_text = fs::read_to_string(fixtures_path)
        .with_context(|| format!("failed to read fixtures {}", fixtures_path.display()))?;
    let fixtures: ValidationFixtures = serde_json::from_str(&fixture_text)
        .with_context(|| format!("failed to parse fixtures {}", fixtures_path.display()))?;

    if fixtures.cases.is_empty() {
        anyhow::bail!("fixtures must contain at least one validation case");
    }
    if fixtures.top_n == 0 {
        anyhow::bail!("top_n must be greater than zero");
    }
    validate_fixture_completeness(&fixtures)?;

    println!(
        "reference runtime: {} model={:?} dtype={:?} revision={:?}",
        fixtures.reference.runtime,
        fixtures.reference.model,
        fixtures.reference.dtype,
        fixtures.reference.revision
    );

    let tok = LayerTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
    let mut model = load_model_auto(model_dir, weights)?.with_backend(backend)?;
    eprintln!("using {} backend", model.backend());

    if model.layer_store.is_some() && (preload_layers || preload_layer_count.is_some()) {
        let count = preload_layer_count.unwrap_or(model.config.num_hidden_layers);
        eprintln!(
            "preloading {count}/{} layer(s)...",
            model.config.num_hidden_layers
        );
        model.preload_layer_count_with_debug(count, false)?;
    }

    let mut failures = Vec::new();

    for case in &fixtures.cases {
        println!("case: {}", case.name);

        let actual_token_ids = tok.encode(&case.prompt)?;
        if actual_token_ids != case.token_ids {
            failures.push(format!(
                "{}: tokenizer ids differ\n  expected: {:?}\n  actual:   {:?}",
                case.name, case.token_ids, actual_token_ids
            ));
            continue;
        }
        println!("  tokenizer: ok");

        let input_ids: Vec<usize> = case.token_ids.iter().map(|id| *id as usize).collect();
        let logits = model.first_next_token_logits(&input_ids)?;
        let actual_top = top_logits(&logits, fixtures.top_n);
        let expected_top = &case.first_token_top_logits[..fixtures.top_n];

        for (rank, expected) in expected_top.iter().enumerate() {
            let Some(actual_for_token) = logits.get(expected.token_id as usize) else {
                failures.push(format!(
                    "{}: expected logit token_id {} is outside vocab",
                    case.name, expected.token_id
                ));
                continue;
            };
            if !logit_close(
                *actual_for_token,
                expected.logit,
                fixtures.logit_atol,
                fixtures.logit_rtol,
            ) {
                failures.push(format!(
                    "{}: logit differs for token {} at expected rank {}\n  expected: {:.6}\n  actual:   {:.6}",
                    case.name, expected.token_id, rank, expected.logit, actual_for_token
                ));
            }
        }

        if fixtures.require_top_token_ids {
            for (rank, expected) in expected_top.iter().enumerate() {
                if actual_top[rank].token_id != expected.token_id {
                    failures.push(format!(
                        "{}: top logit token differs at rank {}\n  expected token: {}\n  actual token:   {}",
                        case.name, rank, expected.token_id, actual_top[rank].token_id
                    ));
                }
            }
        }
        println!("  first-token logits: checked top {}", fixtures.top_n);

        let output_ids = model.generate_greedy_reference(&input_ids, case.greedy.max_new_tokens)?;
        let actual_generated: Vec<u32> = output_ids
            .iter()
            .skip(input_ids.len())
            .map(|id| *id as u32)
            .collect();

        validate_generated_ids(case, &actual_generated, &mut failures);
        println!(
            "  greedy generation: checked {} token(s)",
            actual_generated.len()
        );
    }

    if !failures.is_empty() {
        anyhow::bail!(
            "validation failed with {} issue(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    println!("validation passed: {} case(s)", fixtures.cases.len());
    Ok(())
}

fn validate_fixture_completeness(fixtures: &ValidationFixtures) -> Result<()> {
    let mut failures = Vec::new();

    for case in &fixtures.cases {
        if case.token_ids.is_empty() {
            failures.push(format!("{}: token_ids fixture is empty", case.name));
        }
        if case.first_token_top_logits.len() < fixtures.top_n {
            failures.push(format!(
                "{}: first_token_top_logits has {} entries, expected at least top_n={}",
                case.name,
                case.first_token_top_logits.len(),
                fixtures.top_n
            ));
        }
        if case.greedy.temperature != 0.0 {
            failures.push(format!(
                "{}: greedy.temperature must be 0 for deterministic validation",
                case.name
            ));
        }
        if case.greedy.generated_ids.is_empty() && case.greedy.max_new_tokens > 0 {
            failures.push(format!(
                "{}: greedy.generated_ids fixture is empty",
                case.name
            ));
        }
    }

    if !failures.is_empty() {
        anyhow::bail!(
            "fixtures are incomplete with {} issue(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    Ok(())
}

fn load_model_auto(model_dir: &str, weights: &str) -> Result<RawLlm> {
    let layered_markers = [
        "layerrun.json",
        "embeddings.safetensors",
        "final.safetensors",
    ];
    let is_layered = layered_markers
        .iter()
        .all(|marker| Path::new(model_dir).join(marker).exists());

    if is_layered {
        RawLlm::load_layered(model_dir)
    } else {
        RawLlm::load(model_dir, weights)
    }
}

#[derive(Debug)]
struct ActualLogit {
    token_id: u32,
    logit: f32,
}

fn top_logits(logits: &[f32], top_n: usize) -> Vec<ActualLogit> {
    let mut top: Vec<ActualLogit> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| ActualLogit {
            token_id: token_id as u32,
            logit,
        })
        .collect();
    top.sort_by(|a, b| b.logit.total_cmp(&a.logit));
    top.truncate(top_n);
    top
}

fn logit_close(actual: f32, expected: f32, atol: f32, rtol: f32) -> bool {
    let diff = (actual - expected).abs();
    diff <= atol + rtol * expected.abs()
}

fn validate_generated_ids(
    case: &ValidationCase,
    actual_generated: &[u32],
    failures: &mut Vec<String>,
) {
    if actual_generated == case.greedy.generated_ids {
        return;
    }

    let max_len = actual_generated.len().max(case.greedy.generated_ids.len());
    for index in 0..max_len {
        let expected = case.greedy.generated_ids.get(index).copied();
        let actual = actual_generated.get(index).copied();
        if expected == actual {
            continue;
        }

        let explained = case.greedy.mismatches.iter().any(|mismatch| {
            mismatch.position == index
                && Some(mismatch.expected) == expected
                && Some(mismatch.actual) == actual
                && !mismatch.reason.trim().is_empty()
        });

        if !explained {
            failures.push(format!(
                "{}: generated token differs at position {}\n  expected: {:?}\n  actual:   {:?}",
                case.name, index, expected, actual
            ));
        }
    }
}

fn resolve_model_dir(
    model_dir: Option<String>,
    hf_repo: Option<String>,
    hf_revision: Option<String>,
    hf_token: Option<String>,
    hf_cache_dir: Option<PathBuf>,
    config: &LayerRunConfig,
    weights_filename: &str,
) -> Result<String> {
    match (model_dir, hf_repo) {
        (Some(model_dir), None) => Ok(model_dir),
        (None, Some(hf_repo)) => {
            let hf_token = hf_token.or_else(|| config.hf_token.clone());
            let hf_cache_dir = hf_cache_dir.or_else(|| config.hf_cache_dir.clone());
            let source = HuggingFaceSource::new(hf_repo)?
                .with_revision(hf_revision)
                .with_token(hf_token)
                .with_cache_dir(hf_cache_dir);

            eprintln!(
                "resolving Hugging Face repo {} at revision {}...",
                source.repo_id(),
                source.revision()
            );

            let model_dir = source.prepare_model_dir(weights_filename)?;
            Ok(model_dir.display().to_string())
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("pass either --model-dir/--input-model-dir or --hf-repo, not both")
        }
        (None, None) => anyhow::bail!("pass --model-dir/--input-model-dir or --hf-repo"),
    }
}
