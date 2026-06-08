// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use crate::tensor::Tensor;
use anyhow::Result;
use memmap2::Mmap;
use safetensors::{SafeTensors, tensor::Dtype};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

pub struct SafeTensorFile {
    mmap: Mmap,
}

impl SafeTensorFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn with_tensors<R>(&self, f: impl FnOnce(SafeTensors<'_>) -> Result<R>) -> Result<R> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        f(tensors)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        self.with_tensors(|tensors| {
            tensor_from_safetensors(&tensors, name).or_else(|err| {
                if let Some(alternate_name) = alternate_tensor_name(name) {
                    tensor_from_safetensors(&tensors, &alternate_name)
                } else {
                    Err(err)
                }
            })
        })
    }

    pub fn tensor_f32_or_gemma_qat(&self, name: &str, expected_shape: &[usize]) -> Result<Tensor> {
        self.with_tensors(|tensors| {
            tensor_from_safetensors(&tensors, name)
                .or_else(|dense_err| {
                    if let Some(alternate_name) = alternate_tensor_name(name) {
                        tensor_from_safetensors(&tensors, &alternate_name).or_else(|_| {
                            tensor_from_gemma_qat(&tensors, &alternate_name, expected_shape)
                        })
                    } else {
                        Err(dense_err)
                    }
                })
                .or_else(|_| tensor_from_gemma_qat(&tensors, name, expected_shape))
        })
    }

    pub fn has_tensor(&self, name: &str) -> Result<bool> {
        self.with_tensors(|tensors| {
            if tensors.names().contains(&name) {
                return Ok(true);
            }

            if let Some(alternate_name) = alternate_tensor_name(name) {
                return Ok(tensors.names().contains(&alternate_name.as_str()));
            }

            Ok(false)
        })
    }

    pub fn tensor_names(&self) -> Result<Vec<String>> {
        self.with_tensors(|tensors| {
            let mut names: Vec<String> =
                tensors.names().into_iter().map(|s| s.to_string()).collect();
            names.sort();
            Ok(names)
        })
    }

    pub fn print_summary(&self) -> Result<()> {
        self.with_tensors(|tensors| {
            let mut names: Vec<String> =
                tensors.names().into_iter().map(|s| s.to_string()).collect();

            names.sort();

            for name in names {
                let view = tensors.tensor(&name)?;
                println!("{:<80} {:?} {:?}", name, view.dtype(), view.shape());
            }

            Ok(())
        })
    }
}

#[derive(Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

pub enum SafeTensorSource {
    Single(SafeTensorFile),
    Sharded(ShardedSafeTensors),
}

impl SafeTensorSource {
    pub fn open_model_weights(model_dir: impl AsRef<Path>, weights_filename: &str) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let weights_path = model_dir.join(weights_filename);

        if weights_path.exists() {
            return Ok(Self::Single(SafeTensorFile::open(weights_path)?));
        }

        let index_path = model_dir.join(format!("{weights_filename}.index.json"));
        if index_path.exists() {
            return Ok(Self::Sharded(ShardedSafeTensors::open(
                model_dir, index_path,
            )?));
        }

        Ok(Self::Single(SafeTensorFile::open(weights_path)?))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        match self {
            Self::Single(file) => file.tensor_f32(name),
            Self::Sharded(sharded) => sharded.tensor_f32(name).or_else(|err| {
                if let Some(alternate_name) = alternate_tensor_name(name) {
                    sharded.tensor_f32(&alternate_name)
                } else {
                    Err(err)
                }
            }),
        }
    }

    pub fn tensor_f32_or_gemma_qat(&self, name: &str, expected_shape: &[usize]) -> Result<Tensor> {
        match self {
            Self::Single(file) => file.tensor_f32_or_gemma_qat(name, expected_shape),
            Self::Sharded(sharded) => sharded.tensor_f32_or_gemma_qat(name, expected_shape),
        }
    }

    pub fn has_tensor(&self, name: &str) -> Result<bool> {
        match self {
            Self::Single(file) => {
                if file.has_tensor(name)? {
                    return Ok(true);
                }

                if let Some(alternate_name) = alternate_tensor_name(name) {
                    return file.has_tensor(&alternate_name);
                }

                Ok(false)
            }
            Self::Sharded(sharded) => {
                if sharded.has_tensor(name) {
                    return Ok(true);
                }

                if let Some(alternate_name) = alternate_tensor_name(name) {
                    return Ok(sharded.has_tensor(&alternate_name));
                }

                Ok(false)
            }
        }
    }

    pub fn tensor_names(&self) -> Result<Vec<String>> {
        match self {
            Self::Single(file) => file.tensor_names(),
            Self::Sharded(sharded) => Ok(sharded.tensor_names()),
        }
    }
}

fn alternate_tensor_name(name: &str) -> Option<String> {
    if let Some(suffix) = name.strip_prefix("model.layers.") {
        return Some(format!("model.language_model.layers.{suffix}"));
    }

    if let Some(suffix) = name.strip_prefix("model.embed_tokens.") {
        return Some(format!("model.language_model.embed_tokens.{suffix}"));
    }

    if let Some(suffix) = name.strip_prefix("model.embed_tokens_per_layer.") {
        return Some(format!(
            "model.language_model.embed_tokens_per_layer.{suffix}"
        ));
    }

    if let Some(suffix) = name.strip_prefix("model.per_layer_model_projection.") {
        return Some(format!(
            "model.language_model.per_layer_model_projection.{suffix}"
        ));
    }

    if let Some(suffix) = name.strip_prefix("model.per_layer_projection_norm.") {
        return Some(format!(
            "model.language_model.per_layer_projection_norm.{suffix}"
        ));
    }

    if let Some(suffix) = name.strip_prefix("model.norm.") {
        return Some(format!("model.language_model.norm.{suffix}"));
    }

    None
}

pub struct ShardedSafeTensors {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
}

impl ShardedSafeTensors {
    pub fn open(model_dir: impl AsRef<Path>, index_path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(index_path)?;
        let index: SafeTensorIndex = serde_json::from_str(&text)?;

        Ok(Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            weight_map: index.weight_map,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor `{name}` not found in safetensors index"))?;
        let file = SafeTensorFile::open(self.model_dir.join(filename))?;
        file.tensor_f32(name)
    }

    pub fn tensor_f32_or_gemma_qat(&self, name: &str, expected_shape: &[usize]) -> Result<Tensor> {
        match self.tensor_f32(name) {
            Ok(tensor) => Ok(tensor),
            Err(_) => {
                let resolved_name = resolve_existing_tensor_name(&self.weight_map, name)?;
                let filename = self.weight_map.get(&resolved_name).ok_or_else(|| {
                    anyhow::anyhow!("tensor `{resolved_name}` not found in safetensors index")
                })?;
                let file = SafeTensorFile::open(self.model_dir.join(filename))?;
                file.tensor_f32_or_gemma_qat(&resolved_name, expected_shape)
            }
        }
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.weight_map.keys().cloned().collect();
        names.sort();
        names
    }
}

fn resolve_existing_tensor_name(
    weight_map: &HashMap<String, String>,
    name: &str,
) -> Result<String> {
    if weight_map.contains_key(name) {
        return Ok(name.to_string());
    }

    if let Some(alternate_name) = alternate_tensor_name(name) {
        if weight_map.contains_key(&alternate_name) {
            return Ok(alternate_name);
        }
    }

    if let Some(quantized_name) = gemma_quantized_tensor_name(name) {
        if weight_map.contains_key(&quantized_name) {
            return Ok(quantized_name);
        }
    }

    if let Some(alternate_name) = alternate_tensor_name(name) {
        if let Some(quantized_name) = gemma_quantized_tensor_name(&alternate_name) {
            if weight_map.contains_key(&quantized_name) {
                return Ok(quantized_name);
            }
        }
    }

    anyhow::bail!("tensor `{name}` not found in safetensors index")
}

pub fn tensor_from_safetensors(tensors: &SafeTensors<'_>, name: &str) -> Result<Tensor> {
    let view = tensors.tensor(name)?;
    let shape = view.shape().to_vec();
    let raw = view.data();

    match view.dtype() {
        Dtype::F32 => Tensor::from_f32(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            shape,
        ),

        Dtype::F16 => Tensor::from_f16_bits(
            raw.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
            shape,
        ),

        Dtype::BF16 => Tensor::from_bf16_bits(
            raw.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
            shape,
        ),

        other => anyhow::bail!("unsupported dtype for tensor `{name}`: {:?}", other),
    }
}

fn tensor_from_gemma_qat(
    tensors: &SafeTensors<'_>,
    dense_name: &str,
    expected_shape: &[usize],
) -> Result<Tensor> {
    let Some(quantized_name) = gemma_quantized_tensor_name(dense_name) else {
        anyhow::bail!("tensor `{dense_name}` is not a Gemma QAT tensor name");
    };

    let scale_name = gemma_scale_tensor_name(dense_name);

    let quantized = tensors.tensor(&quantized_name)?;
    let scale = tensor_from_safetensors(tensors, &scale_name)?;

    if expected_shape.len() != 2 {
        anyhow::bail!(
            "Gemma QAT tensor `{dense_name}` expects a rank-2 dense shape, got {:?}",
            expected_shape
        );
    }

    let rows = expected_shape[0];
    let cols = expected_shape[1];
    if scale.shape != [rows, 1] && scale.shape != [rows] && !scale.shape.is_empty() {
        anyhow::bail!(
            "Gemma QAT scale `{scale_name}` shape {:?} is incompatible with dense shape {:?}",
            scale.shape,
            expected_shape
        );
    }

    let scales = if scale.shape.is_empty() {
        vec![scale.value(0)]
    } else {
        scale.to_f32_vec()
    };

    match quantized.dtype() {
        Dtype::I8 => {
            if quantized.shape() != expected_shape {
                anyhow::bail!(
                    "Gemma QAT I8 tensor `{quantized_name}` shape {:?} does not match expected {:?}",
                    quantized.shape(),
                    expected_shape
                );
            }

            let data = quantized
                .data()
                .iter()
                .map(|byte| *byte as i8)
                .collect::<Vec<_>>();

            Tensor::from_gemma_qat_i8(data, scales, expected_shape.to_vec())
        }
        Dtype::U8 => {
            if quantized.shape().len() != 2 || quantized.shape()[0] != rows {
                anyhow::bail!(
                    "Gemma QAT U8 tensor `{quantized_name}` shape {:?} is incompatible with dense shape {:?}",
                    quantized.shape(),
                    expected_shape
                );
            }

            let packed_cols = quantized.shape()[1];
            let values_per_byte = cols
                .checked_div(packed_cols)
                .filter(|value| [1, 2, 4].contains(value))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "cannot infer Gemma QAT bit width for `{quantized_name}`: packed shape {:?}, dense shape {:?}",
                        quantized.shape(),
                        expected_shape
                    )
                })?;

            Tensor::from_gemma_qat_u8(
                quantized.data().to_vec(),
                scales,
                expected_shape.to_vec(),
                packed_cols,
                values_per_byte,
            )
        }
        other => {
            anyhow::bail!(
                "unsupported Gemma QAT dtype for `{quantized_name}`: {:?}",
                other
            );
        }
    }
}

fn gemma_quantized_tensor_name(dense_name: &str) -> Option<String> {
    if dense_name.ends_with("embed_tokens.weight") {
        return Some(dense_name.replace("embed_tokens.weight", "embed_tokens.embedding_quantized"));
    }

    if dense_name.ends_with("embed_tokens_per_layer.weight") {
        return Some(dense_name.replace(
            "embed_tokens_per_layer.weight",
            "embed_tokens_per_layer.embedding_quantized",
        ));
    }

    if dense_name.ends_with(".weight") {
        return Some(dense_name.to_string());
    }

    None
}

fn gemma_scale_tensor_name(dense_name: &str) -> String {
    if dense_name.ends_with("embed_tokens.weight") {
        return dense_name.replace("embed_tokens.weight", "embed_tokens.embedding_scale");
    }

    if dense_name.ends_with("embed_tokens_per_layer.weight") {
        return dense_name.replace(
            "embed_tokens_per_layer.weight",
            "embed_tokens_per_layer.embedding_scale",
        );
    }

    format!("{}.weight_scale", dense_name.trim_end_matches(".weight"))
}
