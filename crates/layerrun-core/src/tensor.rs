use anyhow::Result;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    GemmaQatU8 {
        data: Vec<u8>,
        scales: Vec<f32>,
        packed_cols: usize,
        values_per_byte: usize,
    },
    GemmaQatI8 {
        data: Vec<i8>,
        scales: Vec<f32>,
    },
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::from_f32(data, shape)
    }

    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();

        if data.len() != expected {
            anyhow::bail!(
                "tensor shape mismatch: data len {}, shape {:?} expects {}",
                data.len(),
                shape,
                expected
            );
        }

        Ok(Self {
            data: TensorData::F32(data),
            shape,
        })
    }

    pub fn from_f16_bits(data: Vec<u16>, shape: Vec<usize>) -> Result<Self> {
        Self::from_half_bits(TensorData::F16(data), shape)
    }

    pub fn from_bf16_bits(data: Vec<u16>, shape: Vec<usize>) -> Result<Self> {
        Self::from_half_bits(TensorData::BF16(data), shape)
    }

    fn from_half_bits(data: TensorData, shape: Vec<usize>) -> Result<Self> {
        let expected: usize = shape.iter().product();
        let len = match &data {
            TensorData::F16(data) | TensorData::BF16(data) => data.len(),
            _ => unreachable!(),
        };

        if len != expected {
            anyhow::bail!(
                "tensor shape mismatch: data len {}, shape {:?} expects {}",
                len,
                shape,
                expected
            );
        }

        Ok(Self { data, shape })
    }

    pub fn from_gemma_qat_u8(
        data: Vec<u8>,
        scales: Vec<f32>,
        dense_shape: Vec<usize>,
        packed_cols: usize,
        values_per_byte: usize,
    ) -> Result<Self> {
        if dense_shape.len() != 2 {
            anyhow::bail!("Gemma QAT U8 tensor must be rank-2, got {dense_shape:?}");
        }

        let rows = dense_shape[0];
        if data.len() != rows * packed_cols {
            anyhow::bail!(
                "Gemma QAT U8 byte length mismatch: got {}, expected {}",
                data.len(),
                rows * packed_cols
            );
        }

        if scales.len() != 1 && scales.len() != rows {
            anyhow::bail!(
                "Gemma QAT U8 scale length mismatch: got {}, expected 1 or {}",
                scales.len(),
                rows
            );
        }

        Ok(Self {
            data: TensorData::GemmaQatU8 {
                data,
                scales,
                packed_cols,
                values_per_byte,
            },
            shape: dense_shape,
        })
    }

    pub fn from_gemma_qat_i8(
        data: Vec<i8>,
        scales: Vec<f32>,
        dense_shape: Vec<usize>,
    ) -> Result<Self> {
        let expected: usize = dense_shape.iter().product();
        if data.len() != expected {
            anyhow::bail!(
                "Gemma QAT I8 byte length mismatch: got {}, expected {}",
                data.len(),
                expected
            );
        }

        let rows = dense_shape.first().copied().unwrap_or(0);
        if scales.len() != 1 && scales.len() != rows {
            anyhow::bail!(
                "Gemma QAT I8 scale length mismatch: got {}, expected 1 or {}",
                scales.len(),
                rows
            );
        }

        Ok(Self {
            data: TensorData::GemmaQatI8 { data, scales },
            shape: dense_shape,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: TensorData::F32(vec![0.0; size]),
            shape: shape.to_vec(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.data {
            TensorData::F32(data) => Some(data),
            _ => None,
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.len()).map(|index| self.value(index)).collect()
    }

    pub fn value(&self, index: usize) -> f32 {
        match &self.data {
            TensorData::F32(data) => data[index],
            TensorData::F16(data) => f16::from_bits(data[index]).to_f32(),
            TensorData::BF16(data) => bf16::from_bits(data[index]).to_f32(),
            TensorData::GemmaQatI8 { data, scales } => {
                let cols = self.cols();
                let row = index / cols;
                data[index] as f32 * row_scale(scales, row)
            }
            TensorData::GemmaQatU8 {
                data,
                scales,
                packed_cols,
                values_per_byte,
            } => {
                let cols = self.cols();
                let row = index / cols;
                let col = index % cols;
                let packed_index = row * packed_cols + col / values_per_byte;
                let slot = col % values_per_byte;
                let bits = 8 / values_per_byte;
                let mask = (1u8 << bits) - 1;
                let zero_point = (1u8 << (bits - 1)) as i32;
                let code = ((data[packed_index] >> (slot * bits)) & mask) as i32;
                (code - zero_point) as f32 * row_scale(scales, row)
            }
        }
    }

    pub fn row_f32(&self, row: usize) -> Result<Vec<f32>> {
        if self.rank() != 2 {
            anyhow::bail!("row_f32() expects rank-2 tensor, got {:?}", self.shape);
        }

        if row >= self.rows() {
            anyhow::bail!("row index out of bounds: row {}, rows {}", row, self.rows());
        }

        let cols = self.cols();
        let start = row * cols;
        Ok((start..start + cols)
            .map(|index| self.value(index))
            .collect())
    }

    pub fn dot_row(&self, row: usize, x: &[f32]) -> Result<f32> {
        if self.rank() != 2 {
            anyhow::bail!("dot_row() expects rank-2 tensor, got {:?}", self.shape);
        }

        let cols = self.cols();
        if x.len() != cols {
            anyhow::bail!(
                "dot_row mismatch: x len {}, tensor shape {:?}",
                x.len(),
                self.shape
            );
        }

        if row >= self.rows() {
            anyhow::bail!("row index out of bounds: row {}, rows {}", row, self.rows());
        }

        Ok(match &self.data {
            TensorData::F32(data) => data[row * cols..(row + 1) * cols]
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum(),
            TensorData::F16(_) | TensorData::BF16(_) => {
                let start = row * cols;
                (0..cols).map(|col| self.value(start + col) * x[col]).sum()
            }
            TensorData::GemmaQatI8 { data, scales } => {
                let scale = row_scale(scales, row);
                let start = row * cols;
                (0..cols)
                    .map(|col| data[start + col] as f32 * scale * x[col])
                    .sum()
            }
            TensorData::GemmaQatU8 {
                data,
                scales,
                packed_cols,
                values_per_byte,
            } => {
                let scale = row_scale(scales, row);
                let bits = 8 / values_per_byte;
                let mask = (1u8 << bits) - 1;
                let zero_point = (1u8 << (bits - 1)) as i32;
                let row_start = row * packed_cols;
                let mut sum = 0.0f32;

                for packed_index in 0..*packed_cols {
                    let byte = data[row_start + packed_index];

                    for slot in 0..*values_per_byte {
                        let col = packed_index * values_per_byte + slot;
                        if col >= cols {
                            break;
                        }

                        let code = ((byte >> (slot * bits)) & mask) as i32;
                        sum += (code - zero_point) as f32 * scale * x[col];
                    }
                }

                sum
            }
        })
    }
}

fn row_scale(scales: &[f32], row: usize) -> f32 {
    if scales.len() == 1 {
        scales[0]
    } else {
        scales[row]
    }
}
