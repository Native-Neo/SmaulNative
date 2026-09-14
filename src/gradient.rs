use ndarray::Array2;

pub fn global_norm(grads: &[&Array2<f32>]) -> f32 {
    grads.iter().flat_map(|g| g.iter()).map(|v| v * v).sum::<f32>().sqrt()
}

pub fn clip_by_global_norm(grads: &mut [&mut Array2<f32>], max_norm: f32) -> f32 {
    assert!(max_norm > 0.0);
    let norm = global_norm(&grads.iter().map(|g| &**g).collect::<Vec<_>>());
    if norm > max_norm {
        let scale = max_norm / norm;
        for grad in grads.iter_mut() { **grad *= scale; }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_limits_norm() {
        let mut a = Array2::from_elem((2, 2), 3.0);
        let mut grads: Vec<&mut Array2<f32>> = vec![&mut a];
        let before = clip_by_global_norm(&mut grads, 1.0);
        assert!(before > 1.0);
        assert!((global_norm(&[&a]) - 1.0).abs() < 1e-5);
    }
}
