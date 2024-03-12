use ndarray::{Array3, Dim};

#[derive(Debug)]
pub struct Options {
    pub eot_token: usize,
    pub sot_prev: usize,
    pub n_ctx: usize,
}

impl Options {
    pub fn new() -> Options {
        Options {
            eot_token: 50257,
            sot_prev: 50361,
            n_ctx: 448,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub k1: Array3<f32>,
    pub k2: Array3<f32>,
    pub k3: Array3<f32>,
    pub k4: Array3<f32>,
    pub k5: Array3<f32>,
    pub k6: Array3<f32>,
    pub v1: Array3<f32>,
    pub v2: Array3<f32>,
    pub v3: Array3<f32>,
    pub v4: Array3<f32>,
    pub v5: Array3<f32>,
    pub v6: Array3<f32>,
}

impl KVCache {
    pub fn default(n_ctx: usize) -> KVCache {
        let shape = Dim([1, 0, n_ctx]);
        let value: Array3<f32> = Array3::zeros(shape);

        KVCache {
            k1: value.clone(),
            k2: value.clone(),
            k3: value.clone(),
            k4: value.clone(),
            k5: value.clone(),
            k6: value.clone(),
            v1: value.clone(),
            v2: value.clone(),
            v3: value.clone(),
            v4: value.clone(),
            v5: value.clone(),
            v6: value.clone(),
        }
    }
}
