use ndarray::Array2;

pub fn clip_gradients(gradients: &mut [&mut Array2<f32>], max_norm: f32) -> f32 {
    assert!(max_norm.is_finite() && max_norm > 0.0);
    let sum = gradients.iter().flat_map(|g| g.iter()).map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
    let norm = sum.sqrt() as f32;
    if norm > max_norm {
        let scale = max_norm / norm;
        for gradient in gradients.iter_mut() { gradient.mapv_inplace(|v| v * scale); }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn clips_combined_norm() {
        let mut a = array![[3.0]];
        let mut b = array![[4.0]];
        let norm = clip_gradients(&mut [&mut a, &mut b], 2.5);
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((a[[0, 0]].powi(2) + b[[0, 0]].powi(2)).sqrt() <= 2.5 + 1e-6);
    }
}
