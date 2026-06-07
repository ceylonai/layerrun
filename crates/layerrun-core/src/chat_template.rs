use crate::config::{ModelConfig, ModelFamily};
use anyhow::{Context, Result};
use serde_json::Value;
use std::{fs, path::Path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateKind {
    Gemma4,
    Plain,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    kind: ChatTemplateKind,
    source: ChatTemplateSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChatTemplateSource {
    Metadata,
    ModelFamily,
    Fallback,
}

impl ChatTemplate {
    pub fn from_model_dir(model_dir: impl AsRef<Path>, config: &ModelConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if let Some(kind) = template_kind_from_metadata(model_dir)? {
            return Ok(Self {
                kind,
                source: ChatTemplateSource::Metadata,
            });
        }

        if config.family() == ModelFamily::Gemma4 {
            return Ok(Self {
                kind: ChatTemplateKind::Gemma4,
                source: ChatTemplateSource::ModelFamily,
            });
        }

        Ok(Self {
            kind: ChatTemplateKind::Plain,
            source: ChatTemplateSource::Fallback,
        })
    }

    pub fn kind(&self) -> &ChatTemplateKind {
        &self.kind
    }

    pub fn source_name(&self) -> &'static str {
        match self.source {
            ChatTemplateSource::Metadata => "metadata",
            ChatTemplateSource::ModelFamily => "model-family",
            ChatTemplateSource::Fallback => "fallback",
        }
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        match self.kind {
            ChatTemplateKind::Gemma4 => render_gemma4(messages, add_generation_prompt),
            ChatTemplateKind::Plain => render_plain(messages, add_generation_prompt),
        }
    }
}

fn template_kind_from_metadata(model_dir: &Path) -> Result<Option<ChatTemplateKind>> {
    for file_name in ["tokenizer_config.json", "tokenizer.json", "config.json"] {
        let path = model_dir.join(file_name);
        if !path.exists() {
            continue;
        }

        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        if let Some(template) = value.get("chat_template").and_then(Value::as_str) {
            if looks_like_gemma4_template(template) {
                return Ok(Some(ChatTemplateKind::Gemma4));
            }
        }
    }

    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        let template = fs::read_to_string(&jinja_path)
            .with_context(|| format!("failed to read {}", jinja_path.display()))?;
        if looks_like_gemma4_template(&template) {
            return Ok(Some(ChatTemplateKind::Gemma4));
        }
    }

    Ok(None)
}

fn looks_like_gemma4_template(template: &str) -> bool {
    template.contains("<|turn>") && template.contains("model") && template.contains("user")
}

fn render_gemma4(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut prompt = String::from("<bos>");
    let mut index = 0;

    if let Some(first) = messages.first() {
        if first.role == "system" || first.role == "developer" {
            prompt.push_str("<|turn>system\n");
            prompt.push_str(first.content.trim());
            prompt.push_str("<turn|>\n");
            index = 1;
        }
    }

    for message in &messages[index..] {
        let role = match message.role.as_str() {
            "assistant" => "model",
            role => role,
        };
        prompt.push_str("<|turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str("<turn|>\n");
    }

    if add_generation_prompt {
        prompt.push_str("<|turn>model\n");
    }

    prompt
}

fn render_plain(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }

    if add_generation_prompt {
        prompt.push_str("assistant: ");
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::ModelConfig, tokenizer_wrap::LayerTokenizer};
    use std::path::Path;

    #[test]
    fn renders_gemma4_chat_prompt() {
        let template = ChatTemplate {
            kind: ChatTemplateKind::Gemma4,
            source: ChatTemplateSource::ModelFamily,
        };
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Say hi.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi.".to_string(),
            },
            ChatMessage {
                role: "developer".to_string(),
                content: "Keep JSON valid.".to_string(),
            },
        ];

        assert_eq!(
            template.render(&messages, true),
            "<bos><|turn>system\nBe concise.<turn|>\n<|turn>user\nSay hi.<turn|>\n<|turn>model\nHi.<turn|>\n<|turn>developer\nKeep JSON valid.<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn renders_plain_chat_prompt() {
        let template = ChatTemplate {
            kind: ChatTemplateKind::Plain,
            source: ChatTemplateSource::Fallback,
        };
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Say hi.".to_string(),
            },
        ];

        assert_eq!(
            template.render(&messages, true),
            "system: Be concise.\nuser: Say hi.\nassistant: "
        );
    }

    #[test]
    fn gemma4_prompt_token_ids_match_local_tokenizer() -> Result<()> {
        let model_dir = Path::new("models/gemma-4-E4B-it-qat-mobile-transformers");
        if !model_dir.join("tokenizer.json").exists() {
            return Ok(());
        }

        let cfg = ModelConfig::from_model_dir(model_dir)?;
        let template = ChatTemplate::from_model_dir(model_dir, &cfg)?;
        assert_eq!(template.kind(), &ChatTemplateKind::Gemma4);

        let prompt = template.render(
            &[ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            true,
        );
        assert_eq!(prompt, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n");

        let tokenizer = LayerTokenizer::from_file(model_dir.join("tokenizer.json"))?;
        assert_eq!(
            tokenizer.encode(&prompt)?,
            vec![2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107]
        );

        Ok(())
    }
}
