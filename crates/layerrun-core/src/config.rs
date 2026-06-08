// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use anyhow::Result;
use serde::{Deserialize, Deserializer};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen,
    Llama,
    Mistral,
    Gemma4,
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenIdConfig {
    Single(usize),
    Multiple(Vec<usize>),
}

impl TokenIdConfig {
    pub fn as_vec(&self) -> Vec<usize> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,

    pub architectures: Vec<String>,

    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    pub num_key_value_heads: Option<usize>,

    pub head_dim: Option<usize>,

    pub global_head_dim: Option<usize>,

    pub vocab_size: usize,
    pub vocab_size_per_layer_input: Option<usize>,

    pub hidden_size_per_layer_input: Option<usize>,

    pub bos_token_id: Option<TokenIdConfig>,

    pub eos_token_id: Option<TokenIdConfig>,

    pub pad_token_id: Option<TokenIdConfig>,

    pub rms_norm_eps: Option<f32>,

    pub rope_theta: Option<f32>,
    pub rope_thetas_by_layer_type: HashMap<String, f32>,
    pub rope_partial_rotary_factors_by_layer_type: HashMap<String, f32>,

    pub max_position_embeddings: Option<usize>,

    pub sliding_window: Option<usize>,

    pub use_sliding_window: Option<bool>,

    pub layer_types: Vec<String>,

    pub num_global_key_value_heads: Option<usize>,

    pub num_kv_shared_layers: Option<usize>,

    pub hidden_activation: Option<String>,

    pub final_logit_softcapping: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawModelConfig {
    #[serde(default)]
    model_type: String,

    #[serde(default)]
    architectures: Vec<String>,

    #[serde(default)]
    text_config: Option<Box<RawModelConfig>>,

    #[serde(default)]
    hidden_size: Option<usize>,

    #[serde(default)]
    intermediate_size: Option<usize>,

    #[serde(default)]
    num_hidden_layers: Option<usize>,

    #[serde(default)]
    num_attention_heads: Option<usize>,

    #[serde(default)]
    num_key_value_heads: Option<usize>,

    #[serde(default)]
    head_dim: Option<usize>,

    #[serde(default)]
    global_head_dim: Option<usize>,

    #[serde(default)]
    vocab_size: Option<usize>,

    #[serde(default)]
    vocab_size_per_layer_input: Option<usize>,

    #[serde(default)]
    hidden_size_per_layer_input: Option<usize>,

    #[serde(default)]
    bos_token_id: Option<TokenIdConfig>,

    #[serde(default)]
    eos_token_id: Option<TokenIdConfig>,

    #[serde(default)]
    pad_token_id: Option<TokenIdConfig>,

    #[serde(default)]
    rms_norm_eps: Option<f32>,

    #[serde(default)]
    rope_theta: Option<f32>,

    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,

    #[serde(default)]
    max_position_embeddings: Option<usize>,

    #[serde(default)]
    sliding_window: Option<usize>,

    #[serde(default)]
    use_sliding_window: Option<bool>,

    #[serde(default)]
    layer_types: Vec<String>,

    #[serde(default)]
    num_global_key_value_heads: Option<usize>,

    #[serde(default)]
    num_kv_shared_layers: Option<usize>,

    #[serde(default)]
    hidden_activation: Option<String>,

    #[serde(default)]
    final_logit_softcapping: Option<f32>,
}

impl<'de> Deserialize<'de> for ModelConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawModelConfig::deserialize(deserializer)?;
        ModelConfig::try_from_raw(raw).map_err(serde::de::Error::custom)
    }
}

impl ModelConfig {
    fn try_from_raw(raw: RawModelConfig) -> Result<Self> {
        let RawModelConfig {
            model_type,
            architectures,
            text_config,
            ..
        } = raw;

        if let Some(text_config) = text_config {
            let mut cfg = Self::try_from_raw(*text_config)?;
            cfg.model_type = model_type;
            cfg.architectures = architectures;
            return Ok(cfg);
        }

        Ok(Self {
            model_type,
            architectures,
            hidden_size: raw
                .hidden_size
                .ok_or_else(|| anyhow::anyhow!("missing field `hidden_size`"))?,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw
                .num_hidden_layers
                .ok_or_else(|| anyhow::anyhow!("missing field `num_hidden_layers`"))?,
            num_attention_heads: raw
                .num_attention_heads
                .ok_or_else(|| anyhow::anyhow!("missing field `num_attention_heads`"))?,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            global_head_dim: raw.global_head_dim,
            vocab_size: raw
                .vocab_size
                .ok_or_else(|| anyhow::anyhow!("missing field `vocab_size`"))?,
            vocab_size_per_layer_input: raw.vocab_size_per_layer_input,
            hidden_size_per_layer_input: raw.hidden_size_per_layer_input,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
            pad_token_id: raw.pad_token_id,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw
                .rope_theta
                .or_else(|| raw.rope_parameters.as_ref().and_then(default_rope_theta)),
            rope_thetas_by_layer_type: raw
                .rope_parameters
                .as_ref()
                .map(rope_thetas_by_layer_type)
                .unwrap_or_default(),
            rope_partial_rotary_factors_by_layer_type: raw
                .rope_parameters
                .as_ref()
                .map(rope_partial_rotary_factors_by_layer_type)
                .unwrap_or_default(),
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            use_sliding_window: raw.use_sliding_window,
            layer_types: raw.layer_types,
            num_global_key_value_heads: raw.num_global_key_value_heads,
            num_kv_shared_layers: raw.num_kv_shared_layers,
            hidden_activation: raw.hidden_activation,
            final_logit_softcapping: raw.final_logit_softcapping,
        })
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&text)?;
        Ok(cfg)
    }

    pub fn family(&self) -> ModelFamily {
        let model_type = self.model_type.to_ascii_lowercase();

        if model_type.starts_with("qwen") {
            return ModelFamily::Qwen;
        }

        if model_type == "llama" {
            return ModelFamily::Llama;
        }

        if model_type == "mistral" {
            return ModelFamily::Mistral;
        }

        if model_type.starts_with("gemma4") {
            return ModelFamily::Gemma4;
        }

        for architecture in &self.architectures {
            let architecture = architecture.to_ascii_lowercase();

            if architecture.contains("qwen") {
                return ModelFamily::Qwen;
            }

            if architecture.contains("llama") {
                return ModelFamily::Llama;
            }

            if architecture.contains("mistral") {
                return ModelFamily::Mistral;
            }

            if architecture.contains("gemma4") {
                return ModelFamily::Gemma4;
            }
        }

        ModelFamily::Unknown
    }

    pub fn is_llama_like(&self) -> bool {
        matches!(self.family(), ModelFamily::Llama | ModelFamily::Mistral)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn layer_type(&self, layer_id: usize) -> Option<&str> {
        self.layer_types.get(layer_id).map(String::as_str)
    }

    pub fn is_gemma4_global_attention(&self, layer_id: usize) -> bool {
        matches!(self.family(), ModelFamily::Gemma4)
            && self.layer_type(layer_id) == Some("full_attention")
    }

    pub fn layer_head_dim(&self, layer_id: usize) -> usize {
        if self.is_gemma4_global_attention(layer_id) {
            return self.global_head_dim.unwrap_or_else(|| self.head_dim());
        }

        self.head_dim()
    }

    pub fn layer_kv_heads(&self, layer_id: usize) -> usize {
        if self.is_gemma4_global_attention(layer_id) {
            return self
                .num_global_key_value_heads
                .unwrap_or_else(|| self.kv_heads());
        }

        self.kv_heads()
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10000.0)
    }

    pub fn layer_rope_theta(&self, layer_id: usize) -> f32 {
        self.layer_type(layer_id)
            .and_then(|layer_type| self.rope_thetas_by_layer_type.get(layer_type))
            .copied()
            .unwrap_or_else(|| self.rope_theta())
    }

    pub fn layer_rotary_dim(&self, layer_id: usize) -> usize {
        let head_dim = self.layer_head_dim(layer_id);
        let Some(factor) = self.layer_type(layer_id).and_then(|layer_type| {
            self.rope_partial_rotary_factors_by_layer_type
                .get(layer_type)
        }) else {
            return head_dim;
        };

        let rotary_dim = (head_dim as f32 * factor).round() as usize;
        rotary_dim.clamp(2, head_dim)
    }

    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or(1e-6)
    }

    pub fn eos_token_ids(&self) -> Vec<usize> {
        self.eos_token_id
            .as_ref()
            .map(TokenIdConfig::as_vec)
            .unwrap_or_else(|| match self.family() {
                ModelFamily::Qwen => vec![151645, 151643],
                _ => Vec::new(),
            })
    }

    pub fn attention_window(&self) -> Option<usize> {
        match self.family() {
            ModelFamily::Mistral if self.use_sliding_window.unwrap_or(true) => self.sliding_window,
            _ if self.use_sliding_window.unwrap_or(false) => self.sliding_window,
            _ => None,
        }
    }

    pub fn layer_attention_window(&self, layer_id: usize) -> Option<usize> {
        match self.family() {
            ModelFamily::Gemma4 if self.layer_type(layer_id) == Some("sliding_attention") => {
                self.sliding_window
            }
            _ => self.attention_window(),
        }
    }

    pub fn gemma4_first_kv_shared_layer(&self) -> Option<usize> {
        if self.family() != ModelFamily::Gemma4 {
            return None;
        }

        let shared_layers = self.num_kv_shared_layers?;
        Some(self.num_hidden_layers.saturating_sub(shared_layers))
    }

    pub fn gemma4_shared_kv_source_layer(&self, layer_id: usize) -> Option<usize> {
        let first_shared = self.gemma4_first_kv_shared_layer()?;
        if layer_id < first_shared {
            return None;
        }

        let layer_type = self.layer_type(layer_id)?;
        self.layer_types[..first_shared]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, candidate_type)| candidate_type.as_str() == layer_type)
            .map(|(source_layer, _)| source_layer)
    }

    pub fn use_gelu_mlp(&self) -> bool {
        self.hidden_activation
            .as_deref()
            .map(|activation| activation.contains("gelu"))
            .unwrap_or(false)
    }
}

fn default_rope_theta(value: &serde_json::Value) -> Option<f32> {
    value
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
}

fn rope_thetas_by_layer_type(value: &serde_json::Value) -> HashMap<String, f32> {
    let mut out = HashMap::new();

    if let Some(object) = value.as_object() {
        for (key, value) in object {
            if let Some(theta) = default_rope_theta(value) {
                out.insert(key.clone(), theta);
            }
        }
    }

    out
}

fn rope_partial_rotary_factors_by_layer_type(value: &serde_json::Value) -> HashMap<String, f32> {
    let mut out = HashMap::new();

    if let Some(object) = value.as_object() {
        for (key, value) in object {
            if let Some(factor) = value
                .get("partial_rotary_factor")
                .and_then(serde_json::Value::as_f64)
            {
                out.insert(key.clone(), factor as f32);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{ModelConfig, ModelFamily};

    #[test]
    fn parses_llama_family_and_token_ids() {
        let cfg: ModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "vocab_size": 32000,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.family(), ModelFamily::Llama);
        assert!(cfg.is_llama_like());
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.eos_token_ids(), vec![2]);
    }

    #[test]
    fn parses_mistral_family_with_sliding_window() {
        let cfg: ModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["MistralForCausalLM"],
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "eos_token_id": [2, 32000],
                "sliding_window": 4096
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.family(), ModelFamily::Mistral);
        assert_eq!(cfg.kv_heads(), 8);
        assert_eq!(cfg.eos_token_ids(), vec![2, 32000]);
        assert_eq!(cfg.attention_window(), Some(4096));
    }

    #[test]
    fn preserves_qwen_default_stop_tokens() {
        let cfg: ModelConfig = serde_json::from_str(
            r#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 151936
            }"#,
        )
        .unwrap();

        assert_eq!(cfg.family(), ModelFamily::Qwen);
        assert_eq!(cfg.eos_token_ids(), vec![151645, 151643]);
    }
}
