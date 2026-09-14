use ndarray::{Array1, Array2};

pub fn max_abs_matrix_difference(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

pub fn max_abs_vector_difference(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

pub fn assert_close_matrix(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) {
    let error = max_abs_matrix_difference(a, b);
    assert!(error <= tolerance, "maximum absolute error {error} exceeds {tolerance}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_small_numerical_error() {
        let a = Array2::from_elem((2, 2), 1.0);
        let b = Array2::from_elem((2, 2), 1.000001);
        assert_close_matrix(&a, &b, 1e-4);
    }
}
