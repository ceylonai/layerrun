use crate::tensor::Tensor;
use anyhow::Result;
use rayon::prelude::*;

pub fn matvec(x: &[f32], w: &Tensor) -> Result<Vec<f32>> {
    if w.shape.len() != 2 {
        anyhow::bail!("matvec weight must be 2D, got {:?}", w.shape);
    }

    let rows = w.shape[0];
    let cols = w.shape[1];

    // Case 1:
    // Standard ML layout:
    // weight = [out_features, in_features]
    // y = W x
    if x.len() == cols {
        let y = (0..rows)
            .into_par_iter()
            .map(|r| w.dot_row(r, x))
            .collect::<Result<Vec<_>>>()?;

        return Ok(y);
    }

    // Case 2:
    // Transposed layout:
    // weight = [in_features, out_features]
    // y = x W
    if x.len() == rows {
        let y = (0..cols)
            .into_par_iter()
            .map(|c| {
                let mut sum = 0.0f32;

                for r in 0..rows {
                    sum += x[r] * w.value(r * cols + c);
                }

                sum
            })
            .collect::<Vec<_>>();

        return Ok(y);
    }

    anyhow::bail!(
        "matvec mismatch: x len {}, weight shape {:?}. Expected x len {} or {}",
        x.len(),
        w.shape,
        cols,
        rows
    );
}

pub fn linear(x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
    let mut y = matvec(x, weight)?;

    if let Some(b) = bias {
        if b.len() != y.len() {
            anyhow::bail!(
                "linear bias length mismatch: bias {}, output {}",
                b.len(),
                y.len()
            );
        }

        for (index, yi) in y.iter_mut().enumerate() {
            *yi += b.value(index);
        }
    }

    Ok(y)
}

pub fn rms_norm(x: &[f32], weight: &Tensor, eps: f32) -> Result<Vec<f32>> {
    if weight.len() != x.len() {
        anyhow::bail!(
            "rms_norm mismatch: x {}, weight shape {:?}",
            x.len(),
            weight.shape
        );
    }

    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();

    Ok(x.iter()
        .enumerate()
        .map(|(index, xi)| xi * scale * weight.value(index))
        .collect())
}

pub fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    x.iter().map(|xi| xi * scale).collect()
}

pub fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| silu_scalar(*v)).collect()
}

pub fn gelu_pytorch_tanh_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978846;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

pub fn gelu_pytorch_tanh_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| gelu_pytorch_tanh_scalar(*v)).collect()
}

pub fn mul_vec(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        anyhow::bail!("mul_vec length mismatch: {} != {}", a.len(), b.len());
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

pub fn add_vec(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        anyhow::bail!("add_vec length mismatch: {} != {}", a.len(), b.len());
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

pub fn softcap_vec(x: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }

    for value in x {
        *value = (*value / cap).tanh() * cap;
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return vec![];
    }

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum == 0.0 {
        return vec![0.0; x.len()];
    }

    exps.iter().map(|v| v / sum).collect()
}

pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn embed_token(embed: &Tensor, token_id: usize) -> Result<Vec<f32>> {
    if embed.shape.len() != 2 {
        anyhow::bail!("embedding tensor must be 2D, got {:?}", embed.shape);
    }

    let vocab = embed.shape[0];
    if token_id >= vocab {
        anyhow::bail!("token id {} out of range vocab {}", token_id, vocab);
    }

    embed.row_f32(token_id)
}

pub fn apply_rope_one_head(x: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    for i in (0..head_dim).step_by(2) {
        if i + 1 >= head_dim {
            break;
        }

        let inv_freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let angle = position as f32 * inv_freq;

        let cos = angle.cos();
        let sin = angle.sin();

        let x0 = x[i];
        let x1 = x[i + 1];

        x[i] = x0 * cos - x1 * sin;
        x[i + 1] = x0 * sin + x1 * cos;
    }
}

pub fn apply_llama_rope_one_head(x: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;

    for i in 0..half {
        let inv_freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        let angle = position as f32 * inv_freq;

        let cos = angle.cos();
        let sin = angle.sin();

        let x0 = x[i];
        let x1 = x[i + half];

        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

pub fn apply_rope_all_heads(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;

        if end <= x.len() {
            apply_rope_one_head(&mut x[start..end], position, head_dim, theta);
        }
    }
}

pub fn apply_llama_rope_all_heads(
    x: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;

        if end <= x.len() {
            apply_llama_rope_one_head(&mut x[start..end], position, head_dim, theta);
        }
    }
}
pub fn matvec_out_in(x: &[f32], w: &Tensor) -> Result<Vec<f32>> {
    if w.shape.len() != 2 {
        anyhow::bail!("matvec_out_in weight must be 2D, got {:?}", w.shape);
    }

    let out_features = w.shape[0];
    let in_features = w.shape[1];

    if x.len() != in_features {
        anyhow::bail!(
            "matvec_out_in mismatch: x len {}, weight shape {:?}",
            x.len(),
            w.shape
        );
    }

    let y = (0..out_features)
        .into_par_iter()
        .map(|r| w.dot_row(r, x))
        .collect::<Result<Vec<_>>>()?;

    Ok(y)
}

pub fn matvec_in_out(x: &[f32], w: &Tensor) -> Result<Vec<f32>> {
    if w.shape.len() != 2 {
        anyhow::bail!("matvec_in_out weight must be 2D, got {:?}", w.shape);
    }

    let in_features = w.shape[0];
    let out_features = w.shape[1];

    if x.len() != in_features {
        anyhow::bail!(
            "matvec_in_out mismatch: x len {}, weight shape {:?}",
            x.len(),
            w.shape
        );
    }

    let y = (0..out_features)
        .into_par_iter()
        .map(|c| {
            let mut sum = 0.0f32;

            for r in 0..in_features {
                sum += x[r] * w.value(r * out_features + c);
            }

            sum
        })
        .collect::<Vec<_>>();

    Ok(y)
}

pub fn linear_expect_out(
    x: &[f32],
    weight: &Tensor,
    bias: Option<&Tensor>,
    expected_out: usize,
) -> Result<Vec<f32>> {
    let rows = weight.shape[0];
    let cols = weight.shape[1];

    let mut y = if x.len() == cols && rows == expected_out {
        matvec_out_in(x, weight)?
    } else if x.len() == rows && cols == expected_out {
        matvec_in_out(x, weight)?
    } else {
        anyhow::bail!(
            "linear_expect_out failed: x len {}, weight shape {:?}, expected output {}",
            x.len(),
            weight.shape,
            expected_out
        );
    };

    if let Some(b) = bias {
        if b.len() != y.len() {
            anyhow::bail!(
                "linear_expect_out bias mismatch: bias {}, output {}",
                b.len(),
                y.len()
            );
        }

        for (index, yi) in y.iter_mut().enumerate() {
            *yi += b.value(index);
        }
    }

    Ok(y)
}
pub fn linear_out_in(x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
    let mut y = matvec_out_in(x, weight)?;

    if let Some(b) = bias {
        if b.len() != y.len() {
            anyhow::bail!(
                "linear_out_in bias mismatch: bias {}, output {}",
                b.len(),
                y.len()
            );
        }

        for (index, yi) in y.iter_mut().enumerate() {
            *yi += b.value(index);
        }
    }

    Ok(y)
}
