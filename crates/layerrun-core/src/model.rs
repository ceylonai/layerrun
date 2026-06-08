// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use crate::{
    backend::{Backend, BackendKind, BackendOps},
    config::{ModelConfig, ModelFamily},
    kv_cache::KvCache,
    layer_store::LayerStore,
    ops::{
        apply_llama_rope_all_heads, apply_llama_rope_one_head, apply_rope_all_heads, argmax,
        embed_token, softcap_vec,
    },
    safetensor_loader::SafeTensorSource,
    tensor::Tensor,
    weights::DecoderLayerWeights,
};
use anyhow::Result;
use rand::random;
use std::{path::Path, time::Instant};

pub struct RawLlm {
    pub config: ModelConfig,
    backend: Backend,
    pub layer_store: Option<LayerStore>,
    safetensor_source: Option<SafeTensorSource>,
    preloaded_layers: Vec<Option<DecoderLayerWeights>>,

    pub embed_tokens: Tensor,
    pub embed_tokens_per_layer: Option<Tensor>,
    pub per_layer_model_projection: Option<Tensor>,
    pub per_layer_projection_norm: Option<Tensor>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: f32,
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: 1.0,
        }
    }

    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: if temperature > 0.0 { Some(40) } else { None },
            top_p: 1.0,
        }
    }
}

impl RawLlm {
    pub fn load(model_dir: impl AsRef<Path>, weights_filename: &str) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = ModelConfig::from_model_dir(model_dir)?;

        let safetensor_source = SafeTensorSource::open_model_weights(model_dir, weights_filename)?;

        let embed_tokens = safetensor_source.tensor_f32_or_gemma_qat(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let embed_tokens_per_layer =
            load_embeddings_per_layer_from_source(&safetensor_source, &config)?;
        let (per_layer_model_projection, per_layer_projection_norm) =
            load_per_layer_projection_from_source(&safetensor_source, &config)?;
        let final_norm = safetensor_source.tensor_f32("model.norm.weight")?;

        let lm_head = match safetensor_source
            .tensor_f32_or_gemma_qat("lm_head.weight", &[config.vocab_size, config.hidden_size])
        {
            Ok(t) => t,
            Err(_) => {
                // Many causal LMs tie lm_head to embeddings.
                embed_tokens.clone()
            }
        };

        Ok(Self {
            config,
            backend: Backend::new(BackendKind::default())?,
            layer_store: None,
            safetensor_source: Some(safetensor_source),
            preloaded_layers: Vec::new(),
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            final_norm,
            lm_head,
        })
    }

    pub fn load_layered(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let config = ModelConfig::from_model_dir(model_dir)?;
        let layer_store = LayerStore::new(model_dir);

        let embed_tokens = layer_store.load_embeddings(&config)?;
        let embed_tokens_per_layer = layer_store.load_embeddings_per_layer(&config)?;
        let (per_layer_model_projection, per_layer_projection_norm) =
            layer_store.load_per_layer_projection(&config)?;
        let final_norm = layer_store.load_final_norm()?;
        let lm_head = layer_store.load_lm_head(&config)?;

        Ok(Self {
            config,
            backend: Backend::new(BackendKind::default())?,
            layer_store: Some(layer_store),
            safetensor_source: None,
            preloaded_layers: Vec::new(),
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            final_norm,
            lm_head,
        })
    }

    pub fn with_backend(mut self, backend: BackendKind) -> Result<Self> {
        self.backend = Backend::new(backend)?;
        Ok(self)
    }

    pub fn backend(&self) -> BackendKind {
        self.backend.kind()
    }

    pub fn preload_layers_with_debug(&mut self, debug: bool) -> Result<()> {
        self.preload_layer_count_with_debug(self.config.num_hidden_layers, debug)
    }

    pub fn preload_layer_count_with_debug(&mut self, count: usize, debug: bool) -> Result<()> {
        if count > self.config.num_hidden_layers {
            anyhow::bail!(
                "cannot preload {count} layer(s): model only has {} layers",
                self.config.num_hidden_layers
            );
        }

        if self.preloaded_layers.len() != self.config.num_hidden_layers {
            self.preloaded_layers = vec![None; self.config.num_hidden_layers];
        }

        for layer_id in 0..count {
            if self.preloaded_layers[layer_id].is_some() {
                continue;
            }

            let started = Instant::now();
            let layer_weights = self.load_layer_weights_with_debug(layer_id, debug)?;
            let elapsed = started.elapsed();

            if debug {
                eprintln!(
                    "[debug] preload layer={layer_id:03} load_ms={:.3}",
                    elapsed.as_secs_f64() * 1000.0,
                );
            }

            self.preloaded_layers[layer_id] = Some(layer_weights);
        }

        Ok(())
    }

    /// Placeholder to verify the application pipeline.
    /// Replace this with generate_greedy() after validating model math against Python.
    pub fn generate_placeholder(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut out = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            // Intentionally repeat the last token. This keeps the binary runnable while
            // you build/verify exact attention/RoPE/GQA implementation.
            let next = *out.last().unwrap_or(&0);
            out.push(next);
        }

        Ok(out)
    }

    pub fn forward_one_token(
        &self,
        token_id: usize,
        layer_caches: &mut [KvCache],
    ) -> Result<Vec<f32>> {
        self.forward_one_token_with_debug(token_id, layer_caches, false)
    }

    fn forward_one_token_with_debug(
        &self,
        token_id: usize,
        layer_caches: &mut [KvCache],
        debug: bool,
    ) -> Result<Vec<f32>> {
        if debug {
            eprintln!("[debug] forward token_id={token_id}");
        }

        let mut hidden = embed_token(&self.embed_tokens, token_id)?;
        if self.config.family() == ModelFamily::Gemma4 {
            let scale = (self.config.hidden_size as f32).sqrt();
            for value in &mut hidden {
                *value *= scale;
            }
        }
        let per_layer_inputs = self.per_layer_inputs_for_token(token_id, &hidden)?;

        for layer_id in 0..self.config.num_hidden_layers {
            if let Some(Some(layer_weights)) = self.preloaded_layers.get(layer_id) {
                if debug {
                    eprintln!("[debug] layer={layer_id:03} source=preloaded");
                }

                let run_started = Instant::now();
                let shared_attention_cache =
                    shared_kv_cache(&self.config, layer_id, layer_caches).cloned();
                hidden = decoder_layer_forward(
                    &hidden,
                    layer_weights,
                    &self.config,
                    layer_id,
                    per_layer_inputs.as_deref(),
                    &mut layer_caches[layer_id],
                    shared_attention_cache.as_ref(),
                    &self.backend,
                )?;
                let run_elapsed = run_started.elapsed();

                if debug {
                    eprintln!(
                        "[debug] layer={layer_id:03} load_ms=0.000 run_ms={:.3} hidden_len={} cache_len={}",
                        run_elapsed.as_secs_f64() * 1000.0,
                        hidden.len(),
                        layer_caches[layer_id].len(),
                    );
                }
            } else {
                let load_started = Instant::now();
                let layer_weights = self.load_layer_weights_with_debug(layer_id, debug)?;
                let load_elapsed = load_started.elapsed();

                let run_started = Instant::now();
                let shared_attention_cache =
                    shared_kv_cache(&self.config, layer_id, layer_caches).cloned();
                hidden = decoder_layer_forward(
                    &hidden,
                    &layer_weights,
                    &self.config,
                    layer_id,
                    per_layer_inputs.as_deref(),
                    &mut layer_caches[layer_id],
                    shared_attention_cache.as_ref(),
                    &self.backend,
                )?;
                let run_elapsed = run_started.elapsed();

                if debug {
                    eprintln!(
                        "[debug] layer={layer_id:03} load_ms={:.3} run_ms={:.3} hidden_len={} cache_len={}",
                        load_elapsed.as_secs_f64() * 1000.0,
                        run_elapsed.as_secs_f64() * 1000.0,
                        hidden.len(),
                        layer_caches[layer_id].len(),
                    );
                }
            }

            // layer_weights is dropped here.
            // This is the layer-by-layer paging point.
        }

        hidden = model_rms_norm(&hidden, &self.final_norm, &self.config, &self.backend)?;

        let mut logits = self.backend.linear_out_in(&hidden, &self.lm_head, None)?;
        if let Some(cap) = self.config.final_logit_softcapping {
            softcap_vec(&mut logits, cap);
        }
        Ok(logits)
    }

    pub fn generate_greedy(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        self.generate_greedy_with_debug(input_ids, max_new_tokens, false)
    }

    pub fn first_next_token_logits(&self, input_ids: &[usize]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids is empty");
        }

        let mut layer_caches: Vec<KvCache> = (0..self.config.num_hidden_layers)
            .map(|_| KvCache::new())
            .collect();

        if input_ids.len() > 1 {
            for &token in &input_ids[..input_ids.len() - 1] {
                let _ = self.forward_one_token(token, &mut layer_caches)?;
            }
        }

        self.forward_one_token(*input_ids.last().unwrap(), &mut layer_caches)
    }

    pub fn generate_greedy_reference(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids is empty");
        }

        let mut out = input_ids.to_vec();
        let stop_ids = self.config.eos_token_ids();
        let mut layer_caches: Vec<KvCache> = (0..self.config.num_hidden_layers)
            .map(|_| KvCache::new())
            .collect();

        if input_ids.len() > 1 {
            for &token in &input_ids[..input_ids.len() - 1] {
                let _ = self.forward_one_token(token, &mut layer_caches)?;
            }
        }

        let mut current = *input_ids.last().unwrap();
        for _ in 0..max_new_tokens {
            let logits = self.forward_one_token(current, &mut layer_caches)?;
            let next = argmax(&logits);
            out.push(next);

            if stop_ids.contains(&next) {
                break;
            }

            current = next;
        }

        Ok(out)
    }

    pub fn generate_greedy_with_debug(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        debug: bool,
    ) -> Result<Vec<usize>> {
        self.generate_with_temperature_with_debug(input_ids, max_new_tokens, 0.0, debug)
    }

    pub fn generate_with_temperature(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        self.generate_with_sampling_with_debug(
            input_ids,
            max_new_tokens,
            SamplingConfig::with_temperature(temperature),
            false,
        )
    }

    pub fn generate_with_temperature_with_debug(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        debug: bool,
    ) -> Result<Vec<usize>> {
        self.generate_with_sampling_with_debug(
            input_ids,
            max_new_tokens,
            SamplingConfig::with_temperature(temperature),
            debug,
        )
    }

    pub fn generate_with_sampling_with_debug(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        sampling: SamplingConfig,
        debug: bool,
    ) -> Result<Vec<usize>> {
        self.generate_with_sampling_stream_with_debug(
            input_ids,
            max_new_tokens,
            sampling,
            debug,
            |_, _| Ok(()),
        )
    }

    pub fn generate_with_sampling_stream_with_debug<F>(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        sampling: SamplingConfig,
        debug: bool,
        mut on_token: F,
    ) -> Result<Vec<usize>>
    where
        F: FnMut(usize, usize) -> Result<()>,
    {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids is empty");
        }
        if sampling.temperature.is_nan() || sampling.temperature.is_sign_negative() {
            anyhow::bail!("temperature must be a non-negative number");
        }
        if sampling.top_p.is_nan() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
            anyhow::bail!("top_p must be greater than 0 and less than or equal to 1");
        }

        let mut out = input_ids.to_vec();

        let stop_ids = self.config.eos_token_ids();

        let mut layer_caches: Vec<KvCache> = (0..self.config.num_hidden_layers)
            .map(|_| KvCache::new())
            .collect();

        // Prefill all tokens except the last one.
        // The last token will be used to produce the first next-token logits.
        if input_ids.len() > 1 {
            for (index, &token) in input_ids[..input_ids.len() - 1].iter().enumerate() {
                if debug {
                    eprintln!("[debug] prefill index={index} token_id={token}");
                }

                let _ = self.forward_one_token_with_debug(token, &mut layer_caches, debug)?;
            }
        }

        let mut current = *input_ids.last().unwrap();

        for step in 0..max_new_tokens {
            if debug {
                eprintln!("[debug] next step={step} input_token={current}");
            }

            let step_started = Instant::now();
            let logits = self.forward_one_token_with_debug(current, &mut layer_caches, debug)?;

            let next = sample_next_token(&logits, sampling);
            let next_logit = logits[next];

            out.push(next);
            on_token(step, next)?;

            if debug {
                eprintln!(
                    "[debug] next step={step} selected_token={next} selected_logit={next_logit:.6} temperature={:.3} top_k={:?} top_p={:.3} output_len={} step_ms={:.3}",
                    sampling.temperature,
                    sampling.top_k,
                    sampling.top_p,
                    out.len(),
                    step_started.elapsed().as_secs_f64() * 1000.0,
                );
            }

            if stop_ids.contains(&next) {
                if debug {
                    eprintln!("[debug] stop token hit token_id={next}");
                }

                break;
            }

            current = next;
        }

        Ok(out)
    }

    fn load_layer_weights_with_debug(
        &self,
        layer_id: usize,
        debug: bool,
    ) -> Result<DecoderLayerWeights> {
        if let Some(layer_store) = &self.layer_store {
            if debug {
                eprintln!(
                    "[debug] layer={layer_id:03} source={}",
                    layer_store.layer_path(layer_id).display()
                );
            }

            return layer_store.load_layer(&self.config, layer_id);
        }

        if debug {
            eprintln!("[debug] layer={layer_id:03} source=single-safetensors");
        }

        let safetensor_source = self
            .safetensor_source
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no layer weight source configured"))?;

        DecoderLayerWeights::load_from_source(safetensor_source, &self.config, layer_id)
    }

    fn per_layer_inputs_for_token(
        &self,
        token_id: usize,
        input_embed: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let Some(hidden_size_per_layer_input) = self.config.hidden_size_per_layer_input else {
            return Ok(None);
        };

        let packed_hidden = self.config.num_hidden_layers * hidden_size_per_layer_input;
        let mut token_identity = match &self.embed_tokens_per_layer {
            Some(embed_tokens_per_layer) if token_id < embed_tokens_per_layer.rows() => {
                let mut row = embed_token(embed_tokens_per_layer, token_id)?;
                let scale = (hidden_size_per_layer_input as f32).sqrt();
                for value in &mut row {
                    *value *= scale;
                }
                Some(row)
            }
            _ => None,
        };

        let context_aware = match (
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(projection), Some(norm)) => {
                let mut projected = self.backend.linear_out_in(input_embed, projection, None)?;
                let scale = 1.0 / (self.config.hidden_size as f32).sqrt();
                for value in &mut projected {
                    *value *= scale;
                }

                if projected.len() != packed_hidden {
                    anyhow::bail!(
                        "per-layer projection output mismatch: got {}, expected {}",
                        projected.len(),
                        packed_hidden
                    );
                }

                let mut normalized = Vec::with_capacity(projected.len());
                for layer_id in 0..self.config.num_hidden_layers {
                    let start = layer_id * hidden_size_per_layer_input;
                    let end = start + hidden_size_per_layer_input;
                    let layer_normed =
                        model_rms_norm(&projected[start..end], norm, &self.config, &self.backend)?;
                    normalized.extend(layer_normed);
                }

                Some(normalized)
            }
            _ => None,
        };

        match (token_identity.as_mut(), context_aware) {
            (Some(token_identity), Some(context_aware)) => {
                let scale = std::f32::consts::FRAC_1_SQRT_2;
                for (token_value, context_value) in token_identity.iter_mut().zip(context_aware) {
                    *token_value = (*token_value + context_value) * scale;
                }
                Ok(Some(std::mem::take(token_identity)))
            }
            (Some(token_identity), None) => Ok(Some(std::mem::take(token_identity))),
            (None, Some(context_aware)) => Ok(Some(context_aware)),
            (None, None) => Ok(None),
        }
    }
}

fn sample_next_token(logits: &[f32], sampling: SamplingConfig) -> usize {
    if sampling.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    if let Some(top_k) = sampling.top_k {
        candidates.truncate(top_k.max(1));
    }

    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    let mut weighted = Vec::with_capacity(candidates.len());

    for (token_id, logit) in candidates {
        let weight = ((logit - max_logit) / sampling.temperature).exp();
        let weight = if weight.is_finite() { weight } else { 0.0 };
        total += weight;
        weighted.push((token_id, weight));
    }

    if total <= 0.0 || !total.is_finite() {
        return argmax(logits);
    }

    if sampling.top_p < 1.0 {
        weighted.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut cumulative = 0.0f32;
        let mut cutoff_len = weighted.len();
        for (index, (_, weight)) in weighted.iter().enumerate() {
            cumulative += *weight;
            if cumulative / total >= sampling.top_p {
                cutoff_len = index + 1;
                break;
            }
        }
        weighted.truncate(cutoff_len.max(1));
        total = weighted.iter().map(|(_, weight)| *weight).sum();
    }

    let mut threshold = random::<f32>() * total;
    for (token_id, weight) in weighted {
        threshold -= weight;
        if threshold <= 0.0 {
            return token_id;
        }
    }

    argmax(logits)
}

pub fn decoder_layer_forward(
    hidden: &[f32],
    weights: &DecoderLayerWeights,
    cfg: &ModelConfig,
    layer_id: usize,
    per_layer_inputs: Option<&[f32]>,
    cache: &mut KvCache,
    shared_attention_cache: Option<&KvCache>,
    backend: &dyn BackendOps,
) -> Result<Vec<f32>> {
    let normed = model_rms_norm(hidden, &weights.input_layernorm, cfg, backend)?;

    let attn = attention_forward(
        &normed,
        weights,
        cfg,
        layer_id,
        cache,
        shared_attention_cache,
        backend,
    )?;

    let uses_post_residual_norms =
        weights.pre_feedforward_layernorm.is_some() || weights.post_feedforward_layernorm.is_some();

    let attn = if uses_post_residual_norms {
        model_rms_norm(&attn, &weights.post_attention_layernorm, cfg, backend)?
    } else {
        attn
    };

    let hidden = backend.add_vec(hidden, &attn)?;

    let normed = if let Some(norm) = &weights.pre_feedforward_layernorm {
        model_rms_norm(&hidden, norm, cfg, backend)?
    } else {
        model_rms_norm(&hidden, &weights.post_attention_layernorm, cfg, backend)?
    };

    let mlp = mlp_forward(&normed, weights, cfg, backend)?;
    let mlp = if let Some(norm) = &weights.post_feedforward_layernorm {
        model_rms_norm(&mlp, norm, cfg, backend)?
    } else {
        mlp
    };

    let mut hidden = backend.add_vec(&hidden, &mlp)?;

    if let (
        Some(per_layer_input_gate),
        Some(per_layer_projection),
        Some(post_per_layer_input_norm),
        Some(hidden_size_per_layer_input),
        Some(per_layer_inputs),
    ) = (
        &weights.per_layer_input_gate,
        &weights.per_layer_projection,
        &weights.post_per_layer_input_norm,
        cfg.hidden_size_per_layer_input,
        per_layer_inputs,
    ) {
        let start = layer_id * hidden_size_per_layer_input;
        let end = start + hidden_size_per_layer_input;

        if end <= per_layer_inputs.len() {
            let residual = hidden.clone();
            let gate = backend.linear_out_in(&hidden, per_layer_input_gate, None)?;
            let gate = if cfg.use_gelu_mlp() {
                backend.gelu_pytorch_tanh_vec(&gate)
            } else {
                backend.silu_vec(&gate)
            };
            let mixed = backend.mul_vec(&gate, &per_layer_inputs[start..end])?;
            let projected = backend.linear_out_in(&mixed, per_layer_projection, None)?;
            let projected = model_rms_norm(&projected, post_per_layer_input_norm, cfg, backend)?;
            hidden = backend.add_vec(&residual, &projected)?;
        }
    }

    if let Some(scalar) = &weights.layer_scalar {
        if scalar.len() == 1 {
            let scalar = scalar.value(0);
            for value in &mut hidden {
                *value *= scalar;
            }
        }
    }

    Ok(hidden)
}

pub fn mlp_forward(
    x: &[f32],
    w: &DecoderLayerWeights,
    cfg: &ModelConfig,
    backend: &dyn BackendOps,
) -> Result<Vec<f32>> {
    let gate = backend.linear_out_in(x, &w.gate_proj, w.gate_proj_bias.as_ref())?;
    let up = backend.linear_out_in(x, &w.up_proj, w.up_proj_bias.as_ref())?;

    if gate.len() != up.len() {
        anyhow::bail!("MLP gate/up mismatch: gate {}, up {}", gate.len(), up.len());
    }

    let gate_act = if cfg.use_gelu_mlp() {
        backend.gelu_pytorch_tanh_vec(&gate)
    } else {
        backend.silu_vec(&gate)
    };
    let mixed = backend.mul_vec(&gate_act, &up)?;
    let out = backend.linear_out_in(&mixed, &w.down_proj, w.down_proj_bias.as_ref())?;

    if out.len() != cfg.hidden_size {
        anyhow::bail!(
            "MLP output mismatch: got {}, expected hidden_size {}",
            out.len(),
            cfg.hidden_size
        );
    }

    Ok(out)
}

pub fn attention_forward(
    x: &[f32],
    w: &DecoderLayerWeights,
    cfg: &ModelConfig,
    layer_id: usize,
    cache: &mut KvCache,
    shared_attention_cache: Option<&KvCache>,
    backend: &dyn BackendOps,
) -> Result<Vec<f32>> {
    let mut q = backend.linear_out_in(x, &w.q_proj, w.q_proj_bias.as_ref())?;
    let mut k = backend.linear_out_in(x, &w.k_proj, w.k_proj_bias.as_ref())?;
    let mut v = backend.linear_out_in(x, &w.v_proj, w.v_proj_bias.as_ref())?;

    let num_q_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.layer_kv_heads(layer_id);

    if q.len() % num_q_heads != 0 {
        anyhow::bail!(
            "q len {} not divisible by num_q_heads {}",
            q.len(),
            num_q_heads
        );
    }

    if k.len() % num_kv_heads != 0 {
        anyhow::bail!(
            "k len {} not divisible by num_kv_heads {}",
            k.len(),
            num_kv_heads
        );
    }

    let q_head_dim = q.len() / num_q_heads;
    let kv_head_dim = k.len() / num_kv_heads;
    if let Some(q_norm) = &w.q_norm {
        let mut q_normed = vec![0.0f32; q.len()];

        for h in 0..num_q_heads {
            let start = h * q_head_dim;
            let end = start + q_head_dim;

            let normed = model_rms_norm(&q[start..end], q_norm, cfg, backend)?;

            q_normed[start..end].copy_from_slice(&normed);
        }

        q = q_normed;
    }

    if let Some(k_norm) = &w.k_norm {
        let mut k_normed = vec![0.0f32; k.len()];

        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;

            let normed = model_rms_norm(&k[start..end], k_norm, cfg, backend)?;

            k_normed[start..end].copy_from_slice(&normed);
        }

        k = k_normed;
    }

    if cfg.family() == ModelFamily::Gemma4 {
        let mut v_normed = vec![0.0f32; v.len()];

        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;

            let normed = backend.rms_norm_no_weight(&v[start..end], cfg.rms_norm_eps());

            v_normed[start..end].copy_from_slice(&normed);
        }

        v = v_normed;
    }

    if q_head_dim != kv_head_dim {
        anyhow::bail!(
            "head dim mismatch: q_head_dim {}, kv_head_dim {}",
            q_head_dim,
            kv_head_dim
        );
    }

    let head_dim = q_head_dim;
    let configured_head_dim = cfg.layer_head_dim(layer_id);

    if head_dim != configured_head_dim {
        anyhow::bail!(
            "head dim mismatch: projected head_dim {}, config head_dim {}",
            head_dim,
            configured_head_dim
        );
    }

    let position = shared_attention_cache
        .map(|cache| cache.len().saturating_sub(1))
        .unwrap_or_else(|| cache.position());
    let theta = cfg.layer_rope_theta(layer_id);

    if cfg.family() == ModelFamily::Gemma4 {
        apply_gemma4_rope_all_heads(&mut q, position, num_q_heads, head_dim, cfg, layer_id);
        apply_gemma4_rope_all_heads(&mut k, position, num_kv_heads, head_dim, cfg, layer_id);
    } else if cfg.is_llama_like() {
        apply_llama_rope_all_heads(&mut q, position, num_q_heads, head_dim, theta);
        apply_llama_rope_all_heads(&mut k, position, num_kv_heads, head_dim, theta);
    } else {
        apply_rope_all_heads(&mut q, position, num_q_heads, head_dim, theta);
        apply_rope_all_heads(&mut k, position, num_kv_heads, head_dim, theta);
    }

    if shared_attention_cache.is_none() {
        cache.push(k, v);
    }

    let attention_cache = shared_attention_cache.unwrap_or(cache);

    let groups = num_q_heads / num_kv_heads;

    // IMPORTANT:
    // Context length should match q length, not always cfg.hidden_size.
    // Some models use q projection size larger than hidden_size.
    let mut context = vec![0.0f32; q.len()];

    for h in 0..num_q_heads {
        let kv_h = h / groups;

        let q_start = h * head_dim;
        let q_end = q_start + head_dim;
        let qh = &q[q_start..q_end];

        let mut scores = Vec::with_capacity(attention_cache.len());

        let attention_start = cfg
            .layer_attention_window(layer_id)
            .map(|window| attention_cache.len().saturating_sub(window))
            .unwrap_or(0);

        for past_k in &attention_cache.keys[attention_start..] {
            let k_start = kv_h * head_dim;
            let k_end = k_start + head_dim;
            let kh = &past_k[k_start..k_end];

            let dot = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum::<f32>();

            let scale = if cfg.family() == ModelFamily::Gemma4 {
                1.0
            } else {
                1.0 / (head_dim as f32).sqrt()
            };
            scores.push(dot * scale);
        }

        let probs = backend.softmax(&scores);
        let mut head_out = vec![0.0f32; head_dim];

        for (offset, p) in probs.iter().enumerate() {
            let t = attention_start + offset;
            let v_start = kv_h * head_dim;
            let v_end = v_start + head_dim;
            let vh = &attention_cache.values[t][v_start..v_end];

            for i in 0..head_dim {
                head_out[i] += p * vh[i];
            }
        }

        let out_start = h * head_dim;
        let out_end = out_start + head_dim;
        context[out_start..out_end].copy_from_slice(&head_out);
    }

    // o_proj must return hidden_size.
    let out = backend.linear_out_in(&context, &w.o_proj, w.o_proj_bias.as_ref())?;

    Ok(out)
}

fn shared_kv_cache<'a>(
    cfg: &ModelConfig,
    layer_id: usize,
    layer_caches: &'a [KvCache],
) -> Option<&'a KvCache> {
    let source_layer = cfg.gemma4_shared_kv_source_layer(layer_id)?;
    layer_caches.get(source_layer)
}

fn apply_gemma4_rope_all_heads(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    cfg: &ModelConfig,
    layer_id: usize,
) {
    let theta = cfg.layer_rope_theta(layer_id);
    let rotary_dim = cfg.layer_rotary_dim(layer_id);

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + rotary_dim;

        if end <= x.len() {
            apply_llama_rope_one_head(&mut x[start..end], position, rotary_dim, theta);
        }
    }
}

fn model_rms_norm(
    x: &[f32],
    weight: &Tensor,
    cfg: &ModelConfig,
    backend: &dyn BackendOps,
) -> Result<Vec<f32>> {
    backend.rms_norm(x, weight, cfg.rms_norm_eps())
}

fn load_embeddings_per_layer_from_source(
    source: &SafeTensorSource,
    config: &ModelConfig,
) -> Result<Option<Tensor>> {
    let Some(hidden_size_per_layer_input) = config.hidden_size_per_layer_input else {
        return Ok(None);
    };

    let vocab_size = config
        .vocab_size_per_layer_input
        .unwrap_or(config.vocab_size);
    let packed_hidden = config.num_hidden_layers * hidden_size_per_layer_input;

    match source.tensor_f32_or_gemma_qat(
        "model.embed_tokens_per_layer.weight",
        &[vocab_size, packed_hidden],
    ) {
        Ok(tensor) => Ok(Some(tensor)),
        Err(_) => Ok(None),
    }
}

fn load_per_layer_projection_from_source(
    source: &SafeTensorSource,
    config: &ModelConfig,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let Some(hidden_size_per_layer_input) = config.hidden_size_per_layer_input else {
        return Ok((None, None));
    };

    let packed_hidden = config.num_hidden_layers * hidden_size_per_layer_input;
    let projection = source
        .tensor_f32_or_gemma_qat(
            "model.per_layer_model_projection.weight",
            &[packed_hidden, config.hidden_size],
        )
        .ok();
    let norm = source
        .tensor_f32("model.per_layer_projection_norm.weight")
        .ok();

    Ok((projection, norm))
}
