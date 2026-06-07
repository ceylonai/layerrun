use crate::{ops, tensor::Tensor};
use anyhow::Result;
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Mlx,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Mlx => "mlx",
        }
    }
}

impl Default for BackendKind {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "mlx" => Ok(Self::Mlx),
            other => anyhow::bail!("unknown backend `{other}`; expected `cpu` or `mlx`"),
        }
    }
}

pub trait BackendOps {
    fn linear_out_in(&self, x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>>;

    fn rms_norm(&self, x: &[f32], weight: &Tensor, eps: f32) -> Result<Vec<f32>>;

    fn rms_norm_no_weight(&self, x: &[f32], eps: f32) -> Vec<f32>;

    fn add_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    fn mul_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;

    fn silu_vec(&self, x: &[f32]) -> Vec<f32>;

    fn gelu_pytorch_tanh_vec(&self, x: &[f32]) -> Vec<f32>;

    fn softmax(&self, x: &[f32]) -> Vec<f32>;
}

#[derive(Debug, Default)]
pub struct CpuBackend;

impl BackendOps for CpuBackend {
    fn linear_out_in(&self, x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
        ops::linear_out_in(x, weight, bias)
    }

    fn rms_norm(&self, x: &[f32], weight: &Tensor, eps: f32) -> Result<Vec<f32>> {
        ops::rms_norm(x, weight, eps)
    }

    fn rms_norm_no_weight(&self, x: &[f32], eps: f32) -> Vec<f32> {
        ops::rms_norm_no_weight(x, eps)
    }

    fn add_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        ops::add_vec(a, b)
    }

    fn mul_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        ops::mul_vec(a, b)
    }

    fn silu_vec(&self, x: &[f32]) -> Vec<f32> {
        ops::silu_vec(x)
    }

    fn gelu_pytorch_tanh_vec(&self, x: &[f32]) -> Vec<f32> {
        ops::gelu_pytorch_tanh_vec(x)
    }

    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        ops::softmax(x)
    }
}

#[cfg(feature = "mlx")]
#[derive(Debug, Default)]
pub struct MlxBackend {
    cpu: CpuBackend,
}

#[cfg(feature = "mlx")]
impl MlxBackend {
    pub fn new() -> Self {
        let _ = std::any::type_name::<mlx_rs::Array>();
        Self { cpu: CpuBackend }
    }
}

#[cfg(feature = "mlx")]
impl BackendOps for MlxBackend {
    fn linear_out_in(&self, x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
        if let Some(mut y) = mlx_linear_out_in_f32(x, weight)? {
            if let Some(bias) = bias {
                if bias.len() != y.len() {
                    anyhow::bail!(
                        "linear_out_in bias mismatch: bias {}, output {}",
                        bias.len(),
                        y.len()
                    );
                }

                for (index, yi) in y.iter_mut().enumerate() {
                    *yi += bias.value(index);
                }
            }

            return Ok(y);
        }

        self.cpu.linear_out_in(x, weight, bias)
    }

    fn rms_norm(&self, x: &[f32], weight: &Tensor, eps: f32) -> Result<Vec<f32>> {
        self.cpu.rms_norm(x, weight, eps)
    }

    fn rms_norm_no_weight(&self, x: &[f32], eps: f32) -> Vec<f32> {
        self.cpu.rms_norm_no_weight(x, eps)
    }

    fn add_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.cpu.add_vec(a, b)
    }

    fn mul_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.cpu.mul_vec(a, b)
    }

    fn silu_vec(&self, x: &[f32]) -> Vec<f32> {
        self.cpu.silu_vec(x)
    }

    fn gelu_pytorch_tanh_vec(&self, x: &[f32]) -> Vec<f32> {
        self.cpu.gelu_pytorch_tanh_vec(x)
    }

    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        self.cpu.softmax(x)
    }
}

#[cfg(feature = "mlx")]
fn mlx_linear_out_in_f32(x: &[f32], weight: &Tensor) -> Result<Option<Vec<f32>>> {
    let crate::tensor::TensorData::F32(weight_data) = &weight.data else {
        return Ok(None);
    };

    if weight.shape.len() != 2 {
        anyhow::bail!("linear_out_in weight must be 2D, got {:?}", weight.shape);
    }

    let out_features = weight.shape[0];
    let in_features = weight.shape[1];
    if x.len() != in_features {
        anyhow::bail!(
            "linear_out_in mismatch: x len {}, weight shape {:?}",
            x.len(),
            weight.shape
        );
    }

    let in_features = i32::try_from(in_features)?;
    let out_features = i32::try_from(out_features)?;

    let x = mlx_rs::Array::from_slice(x, &[1, in_features]);
    let w = mlx_rs::Array::from_slice(weight_data, &[out_features, in_features]);
    let y = x.matmul(w.t())?;
    let y = y.as_slice::<f32>().to_vec();

    Ok(Some(y))
}

#[cfg(not(feature = "mlx"))]
#[derive(Debug, Default)]
pub struct MlxBackend;

#[cfg(not(feature = "mlx"))]
impl MlxBackend {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(feature = "mlx"))]
impl BackendOps for MlxBackend {
    fn linear_out_in(
        &self,
        _x: &[f32],
        _weight: &Tensor,
        _bias: Option<&Tensor>,
    ) -> Result<Vec<f32>> {
        mlx_feature_error()
    }

    fn rms_norm(&self, _x: &[f32], _weight: &Tensor, _eps: f32) -> Result<Vec<f32>> {
        mlx_feature_error()
    }

    fn rms_norm_no_weight(&self, _x: &[f32], _eps: f32) -> Vec<f32> {
        unreachable!("MLX backend cannot be constructed without the `mlx` feature")
    }

    fn add_vec(&self, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        mlx_feature_error()
    }

    fn mul_vec(&self, _a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
        mlx_feature_error()
    }

    fn silu_vec(&self, _x: &[f32]) -> Vec<f32> {
        unreachable!("MLX backend cannot be constructed without the `mlx` feature")
    }

    fn gelu_pytorch_tanh_vec(&self, _x: &[f32]) -> Vec<f32> {
        unreachable!("MLX backend cannot be constructed without the `mlx` feature")
    }

    fn softmax(&self, _x: &[f32]) -> Vec<f32> {
        unreachable!("MLX backend cannot be constructed without the `mlx` feature")
    }
}

#[cfg(not(feature = "mlx"))]
fn mlx_feature_error<T>() -> Result<T> {
    anyhow::bail!("MLX backend is unavailable because layerrun-core was built without `mlx`")
}

#[derive(Debug)]
pub enum Backend {
    Cpu(CpuBackend),
    Mlx(MlxBackend),
}

impl Backend {
    pub fn new(kind: BackendKind) -> Result<Self> {
        match kind {
            BackendKind::Cpu => Ok(Self::Cpu(CpuBackend)),
            BackendKind::Mlx => {
                #[cfg(not(feature = "mlx"))]
                {
                    anyhow::bail!(
                        "MLX backend requested, but layerrun-core was built without the `mlx` feature"
                    );
                }

                #[cfg(feature = "mlx")]
                {
                    Ok(Self::Mlx(MlxBackend::new()))
                }
            }
        }
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Cpu(_) => BackendKind::Cpu,
            Self::Mlx(_) => BackendKind::Mlx,
        }
    }
}

impl BackendOps for Backend {
    fn linear_out_in(&self, x: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(backend) => backend.linear_out_in(x, weight, bias),
            Self::Mlx(backend) => backend.linear_out_in(x, weight, bias),
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &Tensor, eps: f32) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(backend) => backend.rms_norm(x, weight, eps),
            Self::Mlx(backend) => backend.rms_norm(x, weight, eps),
        }
    }

    fn rms_norm_no_weight(&self, x: &[f32], eps: f32) -> Vec<f32> {
        match self {
            Self::Cpu(backend) => backend.rms_norm_no_weight(x, eps),
            Self::Mlx(backend) => backend.rms_norm_no_weight(x, eps),
        }
    }

    fn add_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(backend) => backend.add_vec(a, b),
            Self::Mlx(backend) => backend.add_vec(a, b),
        }
    }

    fn mul_vec(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Cpu(backend) => backend.mul_vec(a, b),
            Self::Mlx(backend) => backend.mul_vec(a, b),
        }
    }

    fn silu_vec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::Cpu(backend) => backend.silu_vec(x),
            Self::Mlx(backend) => backend.silu_vec(x),
        }
    }

    fn gelu_pytorch_tanh_vec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::Cpu(backend) => backend.gelu_pytorch_tanh_vec(x),
            Self::Mlx(backend) => backend.gelu_pytorch_tanh_vec(x),
        }
    }

    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::Cpu(backend) => backend.softmax(x),
            Self::Mlx(backend) => backend.softmax(x),
        }
    }
}
