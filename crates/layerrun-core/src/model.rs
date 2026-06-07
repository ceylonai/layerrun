use crate::{
    config::ModelConfig,
    kv_cache::KvCache,
    layer_store::LayerStore,
    ops::{
        add_vec, apply_llama_rope_all_heads, apply_rope_all_heads, argmax, embed_token,
        gelu_pytorch_tanh_vec, linear_out_in, mul_vec, rms_norm, rms_norm_no_weight, silu_vec,
        softcap_vec, softmax,
    },
    safetensor_loader::SafeTensorSource,
    tensor::Tensor,
    weights::DecoderLayerWeights,
};
use anyhow::Result;
use std::{path::Path, time::Instant};

pub struct RawLlm {
    pub config: ModelConfig,
    pub layer_store: Option<LayerStore>,
    safetensor_source: Option<SafeTensorSource>,
    preloaded_layers: Vec<Option<DecoderLayerWeights>>,

    pub embed_tokens: Tensor,
    pub embed_tokens_per_layer: Option<Tensor>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
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
            layer_store: None,
            safetensor_source: Some(safetensor_source),
            preloaded_layers: Vec::new(),
            embed_tokens,
            embed_tokens_per_layer,
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
        let final_norm = layer_store.load_final_norm()?;
        let lm_head = layer_store.load_lm_head(&config)?;

        Ok(Self {
            config,
            layer_store: Some(layer_store),
            safetensor_source: None,
            preloaded_layers: Vec::new(),
            embed_tokens,
            embed_tokens_per_layer,
            final_norm,
            lm_head,
        })
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
        if self.config.family() == crate::config::ModelFamily::Gemma4 {
            let scale = (self.config.hidden_size as f32).sqrt();
            for value in &mut hidden {
                *value *= scale;
            }
        }
        let per_layer_inputs = self.per_layer_inputs_for_token(token_id)?;

        for layer_id in 0..self.config.num_hidden_layers {
            if let Some(Some(layer_weights)) = self.preloaded_layers.get(layer_id) {
                if debug {
                    eprintln!("[debug] layer={layer_id:03} source=preloaded");
                }

                let run_started = Instant::now();
                hidden = decoder_layer_forward(
                    &hidden,
                    layer_weights,
                    &self.config,
                    layer_id,
                    per_layer_inputs.as_deref(),
                    &mut layer_caches[layer_id],
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
                hidden = decoder_layer_forward(
                    &hidden,
                    &layer_weights,
                    &self.config,
                    layer_id,
                    per_layer_inputs.as_deref(),
                    &mut layer_caches[layer_id],
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

        hidden = model_rms_norm(&hidden, &self.final_norm, &self.config)?;

        let mut logits = linear_out_in(&hidden, &self.lm_head, None)?;
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

    pub fn generate_greedy_with_debug(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        debug: bool,
    ) -> Result<Vec<usize>> {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids is empty");
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
            let mut logits =
                self.forward_one_token_with_debug(current, &mut layer_caches, debug)?;

            for &old_token in &out {
                if old_token < logits.len() {
                    logits[old_token] -= 1.2;
                }
            }

            let next = argmax(&logits);
            let next_logit = logits[next];

            out.push(next);

            if debug {
                eprintln!(
                    "[debug] next step={step} selected_token={next} selected_logit={next_logit:.6} output_len={} step_ms={:.3}",
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

    fn per_layer_inputs_for_token(&self, token_id: usize) -> Result<Option<Vec<f32>>> {
        let Some(embed_tokens_per_layer) = &self.embed_tokens_per_layer else {
            return Ok(None);
        };

        let mut row = embed_token(embed_tokens_per_layer, token_id)?;

        if let Some(hidden_size_per_layer_input) = self.config.hidden_size_per_layer_input {
            let scale = (hidden_size_per_layer_input as f32).sqrt();
            for value in &mut row {
                *value *= scale;
            }
        }

        Ok(Some(row))
    }
}

pub fn decoder_layer_forward(
    hidden: &[f32],
    weights: &DecoderLayerWeights,
    cfg: &ModelConfig,
    layer_id: usize,
    per_layer_inputs: Option<&[f32]>,
    cache: &mut KvCache,
) -> Result<Vec<f32>> {
    let normed = model_rms_norm(hidden, &weights.input_layernorm, cfg)?;

    let attn = attention_forward(&normed, weights, cfg, layer_id, cache)?;

    let uses_post_residual_norms =
        weights.pre_feedforward_layernorm.is_some() || weights.post_feedforward_layernorm.is_some();

    let attn = if uses_post_residual_norms {
        model_rms_norm(&attn, &weights.post_attention_layernorm, cfg)?
    } else {
        attn
    };

    let hidden = add_vec(hidden, &attn)?;

    let normed = if let Some(norm) = &weights.pre_feedforward_layernorm {
        model_rms_norm(&hidden, norm, cfg)?
    } else {
        model_rms_norm(&hidden, &weights.post_attention_layernorm, cfg)?
    };

    let mlp = mlp_forward(&normed, weights, cfg)?;
    let mlp = if let Some(norm) = &weights.post_feedforward_layernorm {
        model_rms_norm(&mlp, norm, cfg)?
    } else {
        mlp
    };

    let mut hidden = add_vec(&hidden, &mlp)?;

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
            let gate = linear_out_in(&hidden, per_layer_input_gate, None)?;
            let gate = if cfg.use_gelu_mlp() {
                gelu_pytorch_tanh_vec(&gate)
            } else {
                silu_vec(&gate)
            };
            let mixed = mul_vec(&gate, &per_layer_inputs[start..end])?;
            let projected = linear_out_in(&mixed, per_layer_projection, None)?;
            let projected = model_rms_norm(&projected, post_per_layer_input_norm, cfg)?;
            hidden = add_vec(&residual, &projected)?;
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

pub fn mlp_forward(x: &[f32], w: &DecoderLayerWeights, cfg: &ModelConfig) -> Result<Vec<f32>> {
    let gate = linear_out_in(x, &w.gate_proj, w.gate_proj_bias.as_ref())?;
    let up = linear_out_in(x, &w.up_proj, w.up_proj_bias.as_ref())?;

    if gate.len() != up.len() {
        anyhow::bail!("MLP gate/up mismatch: gate {}, up {}", gate.len(), up.len());
    }

    let gate_act = if cfg.use_gelu_mlp() {
        gelu_pytorch_tanh_vec(&gate)
    } else {
        silu_vec(&gate)
    };
    let mixed = mul_vec(&gate_act, &up)?;
    let out = linear_out_in(&mixed, &w.down_proj, w.down_proj_bias.as_ref())?;

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
) -> Result<Vec<f32>> {
    let mut q = linear_out_in(x, &w.q_proj, w.q_proj_bias.as_ref())?;
    let mut k = linear_out_in(x, &w.k_proj, w.k_proj_bias.as_ref())?;
    let mut v = linear_out_in(x, &w.v_proj, w.v_proj_bias.as_ref())?;

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

            let normed = model_rms_norm(&q[start..end], q_norm, cfg)?;

            q_normed[start..end].copy_from_slice(&normed);
        }

        q = q_normed;
    }

    if let Some(k_norm) = &w.k_norm {
        let mut k_normed = vec![0.0f32; k.len()];

        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;

            let normed = model_rms_norm(&k[start..end], k_norm, cfg)?;

            k_normed[start..end].copy_from_slice(&normed);
        }

        k = k_normed;
    }

    if cfg.family() == crate::config::ModelFamily::Gemma4 {
        let mut v_normed = vec![0.0f32; v.len()];

        for h in 0..num_kv_heads {
            let start = h * kv_head_dim;
            let end = start + kv_head_dim;

            let normed = rms_norm_no_weight(&v[start..end], cfg.rms_norm_eps());

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

    let position = cache.position();
    let theta = cfg.layer_rope_theta(layer_id);

    if cfg.is_llama_like() {
        apply_llama_rope_all_heads(&mut q, position, num_q_heads, head_dim, theta);
        apply_llama_rope_all_heads(&mut k, position, num_kv_heads, head_dim, theta);
    } else {
        apply_rope_all_heads(&mut q, position, num_q_heads, head_dim, theta);
        apply_rope_all_heads(&mut k, position, num_kv_heads, head_dim, theta);
    }

    cache.push(k, v);

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

        let mut scores = Vec::with_capacity(cache.len());

        let attention_start = cfg
            .attention_window()
            .map(|window| cache.len().saturating_sub(window))
            .unwrap_or(0);

        for past_k in &cache.keys[attention_start..] {
            let k_start = kv_h * head_dim;
            let k_end = k_start + head_dim;
            let kh = &past_k[k_start..k_end];

            let dot = qh.iter().zip(kh.iter()).map(|(a, b)| a * b).sum::<f32>();

            scores.push(dot / (head_dim as f32).sqrt());
        }

        let probs = softmax(&scores);
        let mut head_out = vec![0.0f32; head_dim];

        for (offset, p) in probs.iter().enumerate() {
            let t = attention_start + offset;
            let v_start = kv_h * head_dim;
            let v_end = v_start + head_dim;
            let vh = &cache.values[t][v_start..v_end];

            for i in 0..head_dim {
                head_out[i] += p * vh[i];
            }
        }

        let out_start = h * head_dim;
        let out_end = out_start + head_dim;
        context[out_start..out_end].copy_from_slice(&head_out);
    }

    // o_proj must return hidden_size.
    let out = linear_out_in(&context, &w.o_proj, w.o_proj_bias.as_ref())?;

    Ok(out)
}

fn model_rms_norm(x: &[f32], weight: &Tensor, cfg: &ModelConfig) -> Result<Vec<f32>> {
    rms_norm(x, weight, cfg.rms_norm_eps())
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
