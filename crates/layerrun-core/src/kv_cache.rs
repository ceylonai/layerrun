// Copyright (C) 2026 SYIGEN (PRIVATE) LIMITED
//
// This file is part of LayerRun.
//
// LayerRun is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// any later version.

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
