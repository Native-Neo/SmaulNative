use ndarray::Array2;

pub fn global_l2_norm(gradients: &[&Array2<f32>]) -> f32 {
    let sum = gradients.iter().flat_map(|g| g.iter()).map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
    sum.sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn combines_multiple_gradients() {
        let a = array![[3.0]];
        let b = array![[4.0]];
        assert!((global_l2_norm(&[&a, &b]) - 5.0).abs() < 1e-6);
    }
}
