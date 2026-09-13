use ndarray::Array2;

pub fn clip_by_norm(gradient: &mut Array2<f32>, max_norm: f32) -> f32 {
    assert!(max_norm >= 0.0);
    let norm = gradient.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        gradient.mapv_inplace(|value| value * scale);
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn clipping_limits_gradient_norm() {
        let mut gradient = array![[3.0, 4.0]];
        let original = clip_by_norm(&mut gradient, 1.0);
        assert!((original - 5.0).abs() < 1e-6);
        let norm = gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
