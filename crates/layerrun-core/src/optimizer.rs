// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use crate::safetensor_loader::SafeTensorSource;
use anyhow::Result;
use safetensors::{
    SafeTensors,
    tensor::{Dtype, View},
};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

struct BorrowedTensorView<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

struct OwnedF32TensorView {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for OwnedF32TensorView {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> View for BorrowedTensorView<'a> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

pub fn optimize_single_safetensors(
    input_model_dir: impl AsRef<Path>,
    output_model_dir: impl AsRef<Path>,
    weights_filename: &str,
    num_layers: usize,
) -> Result<()> {
    let input_model_dir = input_model_dir.as_ref();
    let output_model_dir = output_model_dir.as_ref();

    fs::create_dir_all(output_model_dir)?;

    fs::copy(
        input_model_dir.join("config.json"),
        output_model_dir.join("config.json"),
    )?;

    fs::copy(
        input_model_dir.join("tokenizer.json"),
        output_model_dir.join("tokenizer.json"),
    )?;

    let source = SafeTensorSource::open_model_weights(input_model_dir, weights_filename)?;

    write_group_from_source(
        &source,
        output_model_dir.join("embeddings.safetensors"),
        |name| {
            name.starts_with("model.embed_tokens.")
                || name.starts_with("model.language_model.embed_tokens.")
                || name.starts_with("model.embed_tokens_per_layer.")
                || name.starts_with("model.language_model.embed_tokens_per_layer.")
                || name.starts_with("model.per_layer_model_projection.")
                || name.starts_with("model.language_model.per_layer_model_projection.")
                || name.starts_with("model.per_layer_projection_norm.")
                || name.starts_with("model.language_model.per_layer_projection_norm.")
        },
    )?;

    for layer_id in 0..num_layers {
        let prefix = format!("model.layers.{layer_id}.");
        let language_model_prefix = format!("model.language_model.layers.{layer_id}.");

        write_group_from_source(
            &source,
            output_model_dir.join(format!("layer_{layer_id:03}.safetensors")),
            |name| name.starts_with(&prefix) || name.starts_with(&language_model_prefix),
        )?;
    }

    write_group_from_source(
        &source,
        output_model_dir.join("final.safetensors"),
        |name| {
            name.starts_with("model.norm.")
                || name.starts_with("model.language_model.norm.")
                || name.starts_with("lm_head.")
        },
    )?;

    fs::write(
        output_model_dir.join("layerrun.json"),
        serde_json::json!({
            "format": "layerrun-layered",
            "version": 1,
            "num_layers": num_layers,
            "source_weights": weights_filename
        })
        .to_string(),
    )?;

    Ok(())
}

fn write_group_from_source<F>(
    source: &SafeTensorSource,
    output_path: PathBuf,
    predicate: F,
) -> Result<()>
where
    F: Fn(&str) -> bool,
{
    if let SafeTensorSource::Single(file) = source {
        return file.with_tensors(|tensors| write_group(&tensors, output_path, predicate));
    }

    let mut selected: HashMap<String, OwnedF32TensorView> = HashMap::new();

    for name in source.tensor_names()? {
        if predicate(&name) {
            let tensor = source.tensor_f32(&name)?;
            let tensor_data = tensor.to_f32_vec();
            let mut data = Vec::with_capacity(tensor_data.len() * 4);

            for value in tensor_data {
                data.extend_from_slice(&value.to_le_bytes());
            }

            selected.insert(
                name,
                OwnedF32TensorView {
                    shape: tensor.shape,
                    data,
                },
            );
        }
    }

    if selected.is_empty() {
        anyhow::bail!("no tensors selected for {:?}", output_path);
    }

    safetensors::serialize_to_file(selected, None, output_path.as_path())?;

    Ok(())
}

fn write_group<F>(tensors: &SafeTensors<'_>, output_path: PathBuf, predicate: F) -> Result<()>
where
    F: Fn(&str) -> bool,
{
    let mut selected: HashMap<String, BorrowedTensorView<'_>> = HashMap::new();

    for name in tensors.names() {
        if predicate(name) {
            let view = tensors.tensor(name)?;

            selected.insert(
                name.to_string(),
                BorrowedTensorView {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    data: view.data(),
                },
            );
        }
    }

    if selected.is_empty() {
        anyhow::bail!("no tensors selected for {:?}", output_path);
    }

    safetensors::serialize_to_file(selected, None, output_path.as_path())?;

    Ok(())
}
