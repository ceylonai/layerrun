#[derive(Debug, Clone, Default)]
pub struct KvCache {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn position(&self) -> usize {
        self.keys.len()
    }

    pub fn push(&mut self, k: Vec<f32>, v: Vec<f32>) {
        self.keys.push(k);
        self.values.push(v);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}
