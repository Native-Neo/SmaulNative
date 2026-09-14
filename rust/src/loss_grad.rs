use ndarray::{Array1, Array2};

use crate::softmax::softmax_rows;

pub fn cross_entropy_gradient(logits: &Array2<f32>, targets: &Array1<usize>) -> Array2<f32> {
    assert_eq!(logits.nrows(), targets.len());
    assert!(logits.ncols() > 0);
    let mut gradient = softmax_rows(logits);
    let scale = 1.0 / logits.nrows() as f32;
    for row in 0..gradient.nrows() {
        let target = targets[row];
        assert!(target < gradient.ncols());
        gradient[[row, target]] -= 1.0;
    }
    gradient *= scale;
    gradient
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Axis};

    #[test]
    fn cross_entropy_gradient_has_zero_row_sum() {
        let logits = array![[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]];
        let targets = array![2usize, 0usize];
        let gradient = cross_entropy_gradient(&logits, &targets);
        for row in gradient.axis_iter(Axis(0)) {
            assert!(row.sum().abs() < 1e-6);
        }
    }

    #[test]
    fn target_gradient_is_negative() {
        let logits = array![[1.0, 2.0, 3.0]];
        let gradient = cross_entropy_gradient(&logits, &array![2usize]);
        assert!(gradient[[0, 2]] < 0.0);
    }
}
