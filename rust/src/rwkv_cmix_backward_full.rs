use ndarray::{Array1, Array2};

pub struct CmixBackward {
    pub grad_x: Array2<f32>,
    pub grad_prev: Array1<f32>,
    pub grad_key: Array2<f32>,
    pub grad_value: Array2<f32>,
}

pub fn backward(
    x: &Array2<f32>,
    prev: &Array1<f32>,
    x_k: &Array1<f32>,
    key_weight: &Array2<f32>,
    key_output: &Array2<f32>,
    value_input: &Array2<f32>,
    grad_output: &Array2<f32>,
) -> CmixBackward {
    assert_eq!(x.ncols(), x_k.len());
    assert_eq!(x.nrows(), grad_output.nrows());
    let t = x.nrows();
    let c = x.ncols();
    let mut grad_x = Array2::zeros(x.dim());
    let mut grad_prev = Array1::zeros(c);
    let mut grad_key = Array2::zeros(key_weight.dim());
    let mut grad_value = Array2::zeros((key_weight.ncols(), value_input.ncols()));

    for i in 0..t {
        let previous = if i == 0 { prev } else { &x.row(i - 1).to_owned() };
        let mixed = x.row(i).to_owned() + &((&previous - &x.row(i).to_owned()) * x_k);
        let key_row = key_output.row(i);
        let mut grad_key_row = Array1::zeros(key_row.len());
        for j in 0..key_row.len() {
            let z = key_row[j];
            grad_key_row[j] = if z > 0.0 { 2.0 * z } else { 0.0 };
        }
        let upstream = grad_output.row(i);
        for j in 0..key_row.len() {
            let g = grad_key_row[j];
            for k in 0..c { grad_key[[j, k]] += upstream[j] * g * mixed[k]; }
        }
        for k in 0..c {
            let mut g = 0.0;
            for j in 0..key_row.len() {
                g += upstream[j] * grad_key_row[j] * key_weight[[j, k]];
            }
            grad_x[[i, k]] += g * (1.0 - x_k[k]);
            if i == 0 { grad_prev[k] += g * x_k[k]; } else { grad_x[[i - 1, k]] += g * x_k[k]; }
        }
        for j in 0..value_input.ncols() {
            for k in 0..value_input.nrows() { grad_value[[k, j]] += upstream[j] * value_input[[k, j]]; }
        }
    }
    CmixBackward { grad_x, grad_prev, grad_key, grad_value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_sequence_shape() {
        let x = Array2::zeros((3, 4));
        let prev = Array1::zeros(4);
        let x_k = Array1::ones(4);
        let key = Array2::ones((8, 4));
        let key_out = Array2::zeros((3, 8));
        let value_input = Array2::zeros((8, 4));
        let grad = Array2::ones((3, 4));
        let g = backward(&x, &prev, &x_k, &key, &key_out, &value_input, &grad);
        assert_eq!(g.grad_x.dim(), (3, 4));
        assert_eq!(g.grad_prev.len(), 4);
    }
}
