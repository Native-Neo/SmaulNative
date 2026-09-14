use ndarray::{Array1, Array2};

pub fn sum_rows(input: &Array2<f32>) -> Array1<f32> {
    input.sum_axis(ndarray::Axis(0))
}

pub fn mean_rows(input: &Array2<f32>) -> Array1<f32> {
    assert!(input.nrows() > 0);
    sum_rows(input) / input.nrows() as f32
}

pub fn mean_all(input: &Array2<f32>) -> f32 {
    assert!(!input.is_empty());
    input.sum() / input.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn reductions_match_expected_values() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(sum_rows(&x), array![4.0, 6.0]);
        assert_eq!(mean_rows(&x), array![2.0, 3.0]);
        assert_eq!(mean_all(&x), 2.5);
    }
}
