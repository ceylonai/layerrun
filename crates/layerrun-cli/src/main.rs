<<<<<<< HEAD
fn main() {
    println!("Hello, world!");
=======
use anyhow::Result;
use clap::{Parser, Subcommand};
use layerrun_core::config::ModelConfig;
use layerrun_core::huggingface::HuggingFaceSource;
use layerrun_core::model::RawLlm;
use layerrun_core::safetensor_loader::SafeTensorFile;
use layerrun_core::tokenizer_wrap::LayerTokenizer;
use std::{path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(name = "layerrun_raw")]
#[command(about = "Raw safetensors-level LLM runtime skeleton without Candle")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
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

        /// Hugging Face token. If omitted, HF_TOKEN is used when present.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files.
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

        /// Hugging Face token. If omitted, HF_TOKEN is used when present.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files.
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

        /// Hugging Face token. If omitted, HF_TOKEN is used when present.
        #[arg(long)]
        hf_token: Option<String>,

        /// Directory for cached Hugging Face files.
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

        #[arg(long)]
        preload_layers: bool,

        /// Preload the first N per-layer weight files before generation.
        #[arg(long, conflicts_with = "preload_layers")]
        preload_layer_count: Option<usize>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
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
        } => {
            let total_started = Instant::now();
            let model_dir = resolve_model_dir(
                model_dir,
                hf_repo,
                hf_revision,
                hf_token,
                hf_cache_dir,
                &weights,
            )?;
            let tok = LayerTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
            let ids = tok.encode(&prompt)?;
            println!("prompt token ids: {:?}", ids);

            let load_started = Instant::now();
            let model = RawLlm::load(&model_dir, &weights)?;
            let model_load_elapsed = load_started.elapsed();

            // This is intentionally a skeleton. Full generation needs tested RoPE/GQA
            // model execution. This proves raw safetensors-level plumbing first.
            let input_ids: Vec<usize> = ids.iter().map(|x| *x as usize).collect();

            let generation_started = Instant::now();
            let output_ids_usize =
                model.generate_greedy_with_debug(&input_ids, max_new_tokens, debug)?;
            let generation_elapsed = generation_started.elapsed();

            let output_ids: Vec<u32> = output_ids_usize.iter().map(|x| *x as u32).collect();

            println!("output ids: {:?}", output_ids);

            let generated_only: Vec<u32> = output_ids.iter().skip(ids.len()).copied().collect();

            println!("generated ids: {:?}", generated_only);
            println!("generated decoded: {}", tok.decode(&generated_only)?);
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
            preload_layers,
            preload_layer_count,
        } => {
            let total_started = Instant::now();
            let tok = LayerTokenizer::from_file(format!("{model_dir}/tokenizer.json"))?;
            let ids = tok.encode(&prompt)?;

            println!("prompt token ids: {:?}", ids);

            let load_started = Instant::now();
            let mut model = RawLlm::load_layered(&model_dir)?;
            let model_load_elapsed = load_started.elapsed();

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
            let output_ids_usize =
                model.generate_greedy_with_debug(&input_ids, max_new_tokens, debug)?;
            let generation_elapsed = generation_started.elapsed();

            let output_ids: Vec<u32> = output_ids_usize.iter().map(|x| *x as u32).collect();

            let generated_only: Vec<u32> = output_ids.iter().skip(ids.len()).copied().collect();

            println!("output ids: {:?}", output_ids);
            println!("generated ids: {:?}", generated_only);
            println!("generated decoded: {}", tok.decode(&generated_only)?);
            println!(
                "timing: model_load_ms={:.3} preload_ms={:.3} generation_ms={:.3} total_ms={:.3}",
                model_load_elapsed.as_secs_f64() * 1000.0,
                preload_elapsed.as_secs_f64() * 1000.0,
                generation_elapsed.as_secs_f64() * 1000.0,
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    Ok(())
}

fn resolve_model_dir(
    model_dir: Option<String>,
    hf_repo: Option<String>,
    hf_revision: Option<String>,
    hf_token: Option<String>,
    hf_cache_dir: Option<PathBuf>,
    weights_filename: &str,
) -> Result<String> {
    match (model_dir, hf_repo) {
        (Some(model_dir), None) => Ok(model_dir),
        (None, Some(hf_repo)) => {
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
>>>>>>> 3df8dd5 (init project)
}
