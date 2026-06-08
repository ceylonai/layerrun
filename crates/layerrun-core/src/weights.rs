// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use crate::{
    config::ModelConfig,
    safetensor_loader::{SafeTensorFile, SafeTensorSource},
    tensor::Tensor,
};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct DecoderLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub pre_feedforward_layernorm: Option<Tensor>,
    pub post_feedforward_layernorm: Option<Tensor>,
    pub post_per_layer_input_norm: Option<Tensor>,
    pub layer_scalar: Option<Tensor>,

    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,

    pub q_proj_bias: Option<Tensor>,
    pub k_proj_bias: Option<Tensor>,
    pub v_proj_bias: Option<Tensor>,
    pub o_proj_bias: Option<Tensor>,

    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,

    pub per_layer_input_gate: Option<Tensor>,
    pub per_layer_projection: Option<Tensor>,

    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,

    pub gate_proj_bias: Option<Tensor>,
    pub up_proj_bias: Option<Tensor>,
    pub down_proj_bias: Option<Tensor>,
}

impl DecoderLayerWeights {
    pub fn load(file: &SafeTensorFile, cfg: &ModelConfig, layer_id: usize) -> Result<Self> {
        Self::load_from_layer_file(file, cfg, layer_id)
    }

    pub fn load_from_layer_file(
        file: &SafeTensorFile,
        cfg: &ModelConfig,
        layer_id: usize,
    ) -> Result<Self> {
        Self::load_with(
            cfg,
            layer_id,
            |name, expected_shape| file.tensor_f32_or_gemma_qat(name, expected_shape),
            |name| file.has_tensor(name).unwrap_or(false),
        )
    }

    pub fn load_from_source(
        source: &SafeTensorSource,
        cfg: &ModelConfig,
        layer_id: usize,
    ) -> Result<Self> {
        Self::load_with(
            cfg,
            layer_id,
            |name, expected_shape| source.tensor_f32_or_gemma_qat(name, expected_shape),
            |name| source.has_tensor(name).unwrap_or(false),
        )
    }

    fn load_with(
        cfg: &ModelConfig,
        layer_id: usize,
        tensor_f32: impl Fn(&str, &[usize]) -> Result<Tensor>,
        has_tensor: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size.ok_or_else(|| {
            anyhow::anyhow!("missing field `intermediate_size` for decoder layer weights")
        })?;
        let head_dim = cfg.layer_head_dim(layer_id);
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.layer_kv_heads(layer_id);
        let q_out = num_q_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;
        let projection_in = hidden_size;

        let get = |suffix: &str, expected_shape: &[usize]| -> Result<Tensor> {
            let name = format!("model.layers.{layer_id}.{suffix}");
            match tensor_f32(&name, expected_shape) {
                Ok(tensor) => Ok(tensor),
                Err(err) => {
                    if suffix.ends_with(".weight") {
                        if suffix.starts_with("self_attn.") {
                            let linear_attn_name =
                                format!("model.layers.{layer_id}.linear_attn.in_proj_qkv.weight");

                            if has_tensor(&linear_attn_name) {
                                anyhow::bail!(
                                    "layer {layer_id} uses `linear_attn`, but LayerRun currently supports dense decoder layers with `self_attn.q_proj/k_proj/v_proj/o_proj` weights. This checkpoint can be downloaded and inspected, but generation needs a Qwen3.5 linear-attention implementation."
                                );
                            }
                        }

                        let quantized_name = name.trim_end_matches(".weight").to_string();
                        let qweight_name = format!("{quantized_name}.qweight");

                        if has_tensor(&qweight_name) {
                            anyhow::bail!(
                                "tensor `{name}` not found because this checkpoint appears to be GPTQ-quantized (`{qweight_name}` exists). LayerRun currently supports dense F16/BF16/F32 safetensors, not GPTQ qweight/qzeros/scales tensors."
                            );
                        }
                    }

                    Err(err)
                }
            }
        };

        let get_optional = |suffix: &str, expected_shape: &[usize]| -> Option<Tensor> {
            let name = format!("model.layers.{layer_id}.{suffix}");
            tensor_f32(&name, expected_shape).ok()
        };

        Ok(Self {
            input_layernorm: get("input_layernorm.weight", &[hidden_size])?,
            post_attention_layernorm: get("post_attention_layernorm.weight", &[hidden_size])?,
            pre_feedforward_layernorm: get_optional(
                "pre_feedforward_layernorm.weight",
                &[hidden_size],
            ),
            post_feedforward_layernorm: get_optional(
                "post_feedforward_layernorm.weight",
                &[hidden_size],
            ),
            post_per_layer_input_norm: get_optional(
                "post_per_layer_input_norm.weight",
                &[hidden_size],
            ),
            layer_scalar: get_optional("layer_scalar", &[1]),

            q_proj: get("self_attn.q_proj.weight", &[q_out, projection_in])?,
            k_proj: get("self_attn.k_proj.weight", &[kv_out, projection_in])?,
            v_proj: get("self_attn.v_proj.weight", &[kv_out, projection_in])?,
            o_proj: get("self_attn.o_proj.weight", &[hidden_size, q_out])?,

            q_proj_bias: get_optional("self_attn.q_proj.bias", &[q_out]),
            k_proj_bias: get_optional("self_attn.k_proj.bias", &[kv_out]),
            v_proj_bias: get_optional("self_attn.v_proj.bias", &[kv_out]),
            o_proj_bias: get_optional("self_attn.o_proj.bias", &[hidden_size]),

            q_norm: get_optional("self_attn.q_norm.weight", &[head_dim]),
            k_norm: get_optional("self_attn.k_norm.weight", &[head_dim]),

            per_layer_input_gate: get_optional(
                "per_layer_input_gate.weight",
                &[
                    cfg.hidden_size_per_layer_input.unwrap_or(hidden_size),
                    hidden_size,
                ],
            ),
            per_layer_projection: get_optional(
                "per_layer_projection.weight",
                &[
                    hidden_size,
                    cfg.hidden_size_per_layer_input.unwrap_or(hidden_size),
                ],
            ),

            gate_proj: get("mlp.gate_proj.weight", &[intermediate_size, projection_in])?,
            up_proj: get("mlp.up_proj.weight", &[intermediate_size, projection_in])?,
            down_proj: get("mlp.down_proj.weight", &[hidden_size, intermediate_size])?,

            gate_proj_bias: get_optional("mlp.gate_proj.bias", &[intermediate_size]),
            up_proj_bias: get_optional("mlp.up_proj.bias", &[intermediate_size]),
            down_proj_bias: get_optional("mlp.down_proj.bias", &[hidden_size]),
        })
    }
}
