use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct LayerTokenizer {
    tokenizer: Tokenizer,
}

impl LayerTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }
}
