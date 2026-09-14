use crate::init::uniform;
use ndarray::{Array1, Array2};

pub struct RwkvCmix {
    pub channels: usize,
    pub layer_id: usize,
    pub n_layer: usize,
    pub x_k: Array1<f32>,
    pub key: Array2<f32>,
    pub value: Array2<f32>,
}

impl RwkvCmix {
    pub fn new(channels: usize, layer_id: usize, n_layer: usize) -> Self {
        assert!(channels > 0 && n_layer > 0 && layer_id < n_layer);
        let r = 1.0 - layer_id as f32 / n_layer as f32;
        let x_k = Array1::from_iter((0..channels).map(|i| {
            let d = i as f32 / channels as f32;
            1.0 - d.powf(r.powi(4))
        }));
        let hidden = channels * 4;
        let scale = 0.5 / (channels as f32).sqrt();
        let key = uniform(channels, hidden, -scale, scale, 0x434d_4958_4b4559 ^ layer_id as u64);
        let value = Array2::zeros((hidden, channels));
        Self { channels, layer_id, n_layer, x_k, key, value }
    }

    pub fn from_weights(x_k: Array1<f32>, key: Array2<f32>, value: Array2<f32>, layer_id: usize, n_layer: usize) -> Self {
        let channels = x_k.len();
        assert_eq!(key.dim(), (channels, value.nrows()));
        assert_eq!(value.ncols(), channels);
        assert!(n_layer > 0 && layer_id < n_layer);
        Self { channels, layer_id, n_layer, x_k, key, value }
    }

    fn relu_squared(x: f32) -> f32 {
        let x = x.max(0.0);
        x * x
    }

    pub fn forward(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>) -> (Array2<f32>, Array1<f32>) {
        assert_eq!(x.ncols(), self.channels);
        assert!(x.nrows() > 0);
        let mut mixed = x.clone();
        for t in 0..x.nrows() {
            for c in 0..self.channels {
                let previous = if t == 0 { prev.map_or(0.0, |p| p[c]) } else { x[[t - 1, c]] };
                mixed[[t, c]] = x[[t, c]] + (previous - x[[t, c]]) * self.x_k[c];
            }
        }
        let mut hidden = mixed.dot(&self.key);
        hidden.mapv_inplace(Self::relu_squared);
        let output = hidden.dot(&self.value);
        let last = x.row(x.nrows() - 1).to_owned();
        (output, last)
    }

    pub fn forward_selected(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels);
        let mut mixed = x.clone();
        for t in 0..x.nrows() {
            for c in 0..self.channels {
                let previous = if t == 0 { prev.map_or(0.0, |p| p[c]) } else { x[[t - 1, c]] };
                mixed[[t, c]] = x[[t, c]] + (previous - x[[t, c]]) * self.x_k[c];
            }
        }
        let mut hidden = mixed.dot(&self.key);
        hidden.mapv_inplace(Self::relu_squared);
        hidden.dot(&self.value)
    }

    pub fn forward_rows(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>, rows: &[usize]) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels);
        let mut mixed = Array2::zeros((rows.len(), self.channels));
        for (out_row, &t) in rows.iter().enumerate() {
            assert!(t < x.nrows());
            for c in 0..self.channels {
                let previous = if t == 0 { prev.map_or(0.0, |p| p[c]) } else { x[[t - 1, c]] };
                mixed[[out_row, c]] = x[[t, c]] + (previous - x[[t, c]]) * self.x_k[c];
            }
        }
        let mut hidden = mixed.dot(&self.key);
        hidden.mapv_inplace(Self::relu_squared);
        hidden.dot(&self.value)
    }

    pub fn parameter_count(&self) -> usize {
        self.x_k.len() + self.key.len() + self.value.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RwkvCmix;
    use ndarray::{Array1, Array2};

    #[test]
    fn forward_shapes_and_state() {
        let layer = RwkvCmix::new(8, 0, 4);
        let (y, last) = layer.forward(&Array2::<f32>::ones((3, 8)), Some(&Array1::<f32>::zeros(8)));
        assert_eq!(y.shape(), &[3, 8]);
        assert_eq!(last.len(), 8);
    }

    #[test]
    fn previous_state_affects_first_token() {
        let layer = RwkvCmix::new(4, 0, 2);
        let x = Array2::<f32>::zeros((1, 4));
        let (a, _) = layer.forward(&x, Some(&Array1::<f32>::ones(4)));
        let (b, _) = layer.forward(&x, None);
        assert_ne!(a, b);
    }

    #[test]
    fn selected_rows_match_full_forward() {
        let layer = RwkvCmix::new(4, 0, 2);
        let x = Array2::from_shape_fn((5, 4), |(r, c)| (r * 4 + c) as f32 * 0.1);
        let prev = Array1::from_elem(4, 0.25);
        let (full, _) = layer.forward(&x, Some(&prev));
        let rows = [0, 2, 4];
        let selected = layer.forward_rows(&x, Some(&prev), &rows);
        for (i, &row) in rows.iter().enumerate() {
            for c in 0..4 {
                assert!((selected[[i, c]] - full[[row, c]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn parameter_count_includes_x_k() {
        let layer = RwkvCmix::new(8, 0, 4);
        assert_eq!(layer.parameter_count(), layer.x_k.len() + layer.key.len() + layer.value.len());
    }
}
