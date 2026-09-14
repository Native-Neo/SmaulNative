use ndarray::Array2;

pub fn max_abs_difference(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

pub fn all_close(a: &Array2<f32>, b: &Array2<f32>, atol: f32, rtol: f32) -> bool {
    assert!(atol >= 0.0 && rtol >= 0.0);
    a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| {
        (*x - *y).abs() <= atol + rtol * y.abs()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn comparison_reports_small_numeric_error() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 2.000001], [3.0, 4.0]];
        assert!(max_abs_difference(&a, &b) < 2e-6);
        assert!(all_close(&a, &b, 2e-6, 1e-5));
    }
}
