use crate::{
    config::ModelConfig, safetensor_loader::SafeTensorFile, tensor::Tensor,
    weights::DecoderLayerWeights,
};
use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct LayerStore {
    model_dir: PathBuf,
}

impl LayerStore {
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    pub fn embeddings_path(&self) -> PathBuf {
        self.model_dir.join("embeddings.safetensors")
    }

    pub fn final_path(&self) -> PathBuf {
        self.model_dir.join("final.safetensors")
    }

    pub fn layer_path(&self, layer_id: usize) -> PathBuf {
        self.model_dir
            .join(format!("layer_{layer_id:03}.safetensors"))
    }

    pub fn load_embeddings(&self, cfg: &ModelConfig) -> Result<Tensor> {
        let file = SafeTensorFile::open(self.embeddings_path())?;

        file.tensor_f32_or_gemma_qat(
            "model.embed_tokens.weight",
            &[cfg.vocab_size, cfg.hidden_size],
        )
    }

    pub fn load_embeddings_per_layer(&self, cfg: &ModelConfig) -> Result<Option<Tensor>> {
        let Some(hidden_size_per_layer_input) = cfg.hidden_size_per_layer_input else {
            return Ok(None);
        };

        let file = SafeTensorFile::open(self.embeddings_path())?;
        let vocab_size = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);
        let packed_hidden = cfg.num_hidden_layers * hidden_size_per_layer_input;

        match file.tensor_f32_or_gemma_qat(
            "model.embed_tokens_per_layer.weight",
            &[vocab_size, packed_hidden],
        ) {
            Ok(tensor) => Ok(Some(tensor)),
            Err(_) => Ok(None),
        }
    }

    pub fn load_final_norm(&self) -> Result<Tensor> {
        let file = SafeTensorFile::open(self.final_path())?;
        file.tensor_f32("model.norm.weight")
    }

    pub fn load_lm_head(&self, cfg: &ModelConfig) -> Result<Tensor> {
        let file = SafeTensorFile::open(self.final_path())?;

        if file.has_tensor("lm_head.weight")? {
            match file.tensor_f32_or_gemma_qat("lm_head.weight", &[cfg.vocab_size, cfg.hidden_size])
            {
                Ok(t) => return Ok(t),
                Err(err) => {
                    anyhow::bail!(
                        "failed to load `lm_head.weight` from {:?}: {err}",
                        self.final_path()
                    );
                }
            }
        }

        self.load_embeddings(cfg)
    }

    pub fn load_layer(&self, cfg: &ModelConfig, layer_id: usize) -> Result<DecoderLayerWeights> {
        let file = SafeTensorFile::open(self.layer_path(layer_id))?;
        DecoderLayerWeights::load_from_layer_file(&file, cfg, layer_id)
    }
}
