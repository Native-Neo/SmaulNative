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
        assert!(channels > 0);
        assert!(n_layer > 0);
        assert!(layer_id < n_layer);

        let r = 1.0 - layer_id as f32 / n_layer as f32;
        let mut x_k = Array1::<f32>::zeros(channels);
        for i in 0..channels {
            let ddd = i as f32 / channels as f32;
            x_k[i] = 1.0 - ddd.powf(r.powi(4));
        }

        let hidden = channels * 4;
        let mut key = Array2::<f32>::zeros((channels, hidden));
        let value = Array2::<f32>::zeros((hidden, channels));

        let scale = 0.5 / (channels as f32).sqrt();
        for i in 0..channels {
            for j in 0..hidden {
                let phase = (i * hidden + j + 1) as f32;
                key[[i, j]] = ((phase * 12.9898).sin() * 43758.547).fract() * 2.0 * scale - scale;
            }
        }

        Self {
            channels,
            layer_id,
            n_layer,
            x_k,
            key,
            value,
        }
    }

    pub fn from_weights(
        x_k: Array1<f32>,
        key: Array2<f32>,
        value: Array2<f32>,
        layer_id: usize,
        n_layer: usize,
    ) -> Self {
        let channels = x_k.len();
        assert_eq!(key.nrows(), channels);
        assert_eq!(value.ncols(), channels);
        assert_eq!(key.ncols(), value.nrows());
        assert!(n_layer > 0 && layer_id < n_layer);

        Self {
            channels,
            layer_id,
            n_layer,
            x_k,
            key,
            value,
        }
    }

    fn relu_squared(x: f32) -> f32 {
        let r = x.max(0.0);
        r * r
    }

    pub fn forward(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>) -> (Array2<f32>, Array1<f32>) {
        assert_eq!(x.ncols(), self.channels);

        let steps = x.nrows();
        let mut mixed = x.clone();
        for t in 0..steps {
            for c in 0..self.channels {
                let previous = if t == 0 {
                    prev.map(|p| p[c]).unwrap_or(0.0)
                } else {
                    x[[t - 1, c]]
                };
                mixed[[t, c]] = x[[t, c]] + (previous - x[[t, c]]) * self.x_k[c];
            }
        }

        let hidden = mixed.dot(&self.key);
        let mut activated = hidden;
        activated.mapv_inplace(Self::relu_squared);
        let output = activated.dot(&self.value);
        let last = x.row(steps.saturating_sub(1)).to_owned();
        (output, last)
    }

    pub fn forward_selected(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>) -> Array2<f32> {
        self.forward(x, prev).0
    }

    pub fn parameter_count(&self) -> usize {
        self.key.len() + self.value.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RwkvCmix;
    use ndarray::{Array1, Array2};

    #[test]
    fn forward_shapes_and_state() {
        let layer = RwkvCmix::new(8, 0, 4);
        let x = Array2::<f32>::ones((3, 8));
        let prev = Array1::<f32>::zeros(8);
        let (y, last) = layer.forward(&x, Some(&prev));

        assert_eq!(y.shape(), &[3, 8]);
        assert_eq!(last.len(), 8);
        assert_eq!(layer.parameter_count(), 8 * 32 + 32 * 8);
    }

    #[test]
    fn previous_state_affects_first_token() {
        let layer = RwkvCmix::new(4, 0, 2);
        let x = Array2::<f32>::zeros((1, 4));
        let prev = Array1::<f32>::ones(4);
        let (with_prev, _) = layer.forward(&x, Some(&prev));
        let (without_prev, _) = layer.forward(&x, None);
        assert_ne!(with_prev, without_prev);
    }
}
