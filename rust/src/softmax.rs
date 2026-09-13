use ndarray::{Array1, Array2, Axis};

pub fn softmax_rows(logits: &Array2<f32>) -> Array2<f32> {
    let mut output = logits.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|x| (x - max).exp());
        let sum = row.sum();
        assert!(sum.is_finite() && sum > 0.0);
        row.mapv_inplace(|x| x / sum);
    }
    output
}

pub fn log_softmax_rows(logits: &Array2<f32>) -> Array2<f32> {
    let mut output = logits.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|x| (*x - max).exp()).sum();
        let log_sum = max + sum_exp.ln();
        row.mapv_inplace(|x| x - log_sum);
    }
    output
}

pub fn softmax_gradient(output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad_output.dim());
    let mut grad = Array2::zeros(output.raw_dim());
    for i in 0..output.nrows() {
        let dot = output.row(i).dot(&grad_output.row(i));
        for j in 0..output.ncols() {
            grad[[i, j]] = output[[i, j]] * (grad_output[[i, j]] - dot);
        }
    }
    grad
}

#[allow(dead_code)]
fn _keep_array1_available(_: Option<Array1<f32>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn softmax_rows_sum_to_one() {
        let output = softmax_rows(&array![[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        for row in output.axis_iter(Axis(0)) {
            assert!((row.sum() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn log_softmax_exponentiates_to_softmax() {
        let logits = array![[1.0, 2.0, 3.0]];
        let expected = softmax_rows(&logits);
        let actual = log_softmax_rows(&logits).mapv(f32::exp);
        assert!((expected - actual).iter().all(|x| x.abs() < 1e-6));
    }
}
