// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

use anyhow::{Context, Result};
use reqwest::{StatusCode, blocking::Client};
use serde::Deserialize;
use std::{
    collections::{BTreeSet, HashMap},
    env, fs, io,
    path::{Path, PathBuf},
};

const DEFAULT_REVISION: &str = "main";

#[derive(Debug, Clone)]
pub struct HuggingFaceSource {
    repo_id: String,
    revision: String,
    token: Option<String>,
    cache_dir: PathBuf,
}

#[derive(Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

impl HuggingFaceSource {
    pub fn new(repo_id: impl Into<String>) -> Result<Self> {
        Ok(Self {
            repo_id: repo_id.into(),
            revision: DEFAULT_REVISION.to_string(),
            token: env::var("HF_TOKEN").ok().filter(|value| !value.is_empty()),
            cache_dir: default_cache_dir()?,
        })
    }

    pub fn with_revision(mut self, revision: Option<String>) -> Self {
        if let Some(revision) = revision.filter(|value| !value.is_empty()) {
            self.revision = revision;
        }

        self
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        if let Some(token) = token.filter(|value| !value.is_empty()) {
            self.token = Some(token);
        }

        self
    }

    pub fn with_cache_dir(mut self, cache_dir: Option<PathBuf>) -> Self {
        if let Some(cache_dir) = cache_dir {
            self.cache_dir = cache_dir;
        }

        self
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn snapshot_dir(&self) -> PathBuf {
        self.cache_dir
            .join(sanitize_cache_component(&self.repo_id))
            .join(sanitize_cache_component(&self.revision))
    }

    pub fn prepare_model_dir(&self, weights_filename: &str) -> Result<PathBuf> {
        self.download_file("config.json")?;
        self.download_file("tokenizer.json")?;

        let weights_path = self.snapshot_dir().join(weights_filename);
        let index_filename = format!("{weights_filename}.index.json");
        let index_path = self.snapshot_dir().join(&index_filename);

        if weights_path.exists() {
            return Ok(self.snapshot_dir());
        }

        if index_path.exists() {
            self.download_shards_from_index(&index_filename)?;
            return Ok(self.snapshot_dir());
        }

        if self.download_file_optional(weights_filename)?.is_none() {
            self.download_file(&index_filename)?;
            self.download_shards_from_index(&index_filename)?;
        }

        Ok(self.snapshot_dir())
    }

    pub fn download_file(&self, filename: &str) -> Result<PathBuf> {
        self.download_file_optional(filename)?.ok_or_else(|| {
            anyhow::anyhow!(
                "Hugging Face file `{filename}` was not found in repo `{}` at revision `{}`",
                self.repo_id,
                self.revision
            )
        })
    }

    fn download_file_optional(&self, filename: &str) -> Result<Option<PathBuf>> {
        if filename.is_empty() {
            anyhow::bail!("Hugging Face filename cannot be empty");
        }

        let output_path = self.snapshot_dir().join(filename);
        if output_path.exists() {
            return Ok(Some(output_path));
        }

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            self.repo_id, self.revision, filename
        );

        let client = Client::builder()
            .user_agent(format!(
                "layerrun/{}",
                option_env!("CARGO_PKG_VERSION").unwrap_or("dev")
            ))
            .build()?;

        let mut request = client.get(&url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let mut response = request.send().with_context(|| {
            format!(
                "requesting Hugging Face file `{filename}` from `{}`",
                self.repo_id
            )
        })?;

        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            anyhow::bail!(
                "Hugging Face denied access to `{}`. Set HF_TOKEN or pass --hf-token for gated/private models.",
                self.repo_id
            );
        }

        if !status.is_success() {
            anyhow::bail!(
                "Hugging Face returned HTTP {status} for `{filename}` in repo `{}`",
                self.repo_id
            );
        }

        let partial_path = output_path.with_extension(format!(
            "{}partial",
            output_path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| format!("{extension}."))
                .unwrap_or_default()
        ));

        let mut output = fs::File::create(&partial_path)?;
        io::copy(&mut response, &mut output)
            .with_context(|| format!("downloading Hugging Face file `{filename}`"))?;
        fs::rename(&partial_path, &output_path)?;

        Ok(Some(output_path))
    }

    fn download_shards_from_index(&self, index_filename: &str) -> Result<()> {
        let index_path = self.snapshot_dir().join(index_filename);
        let text = fs::read_to_string(&index_path)?;
        let index: SafeTensorIndex = serde_json::from_str(&text)?;
        let shard_filenames: BTreeSet<String> = index.weight_map.into_values().collect();

        if shard_filenames.is_empty() {
            anyhow::bail!(
                "Hugging Face safetensors index `{index_filename}` did not list any shards"
            );
        }

        eprintln!(
            "downloading {} Hugging Face safetensors shard(s)...",
            shard_filenames.len()
        );

        for filename in shard_filenames {
            self.download_file(&filename)?;
        }

        Ok(())
    }
}

fn default_cache_dir() -> Result<PathBuf> {
    if let Some(cache_home) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache_home)
            .join("layerrun")
            .join("huggingface"));
    }

    let home = env::var_os("HOME").context("HOME is not set; pass --hf-cache-dir")?;
    Ok(Path::new(&home)
        .join(".cache")
        .join("layerrun")
        .join("huggingface"))
}

fn sanitize_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::sanitize_cache_component;

    #[test]
    fn cache_component_replaces_path_separators() {
        assert_eq!(
            sanitize_cache_component("meta-llama/Llama-3.2-1B-Instruct"),
            "meta-llama_Llama-3.2-1B-Instruct"
        );
    }
}
