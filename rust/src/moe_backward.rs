use crate::moe::{MoeCmix, MoeRouter};
use ndarray::{Array1, Array2};

pub struct MoeBackward {
    pub grad_input: Array2<f32>,
    pub grad_prev: Array1<f32>,
    pub grad_router: Array2<f32>,
    pub grad_x_k: Vec<Array1<f32>>,
    pub grad_key: Vec<Array2<f32>>,
    pub grad_value: Vec<Array2<f32>>,
}

pub fn backward(moe: &MoeCmix, input: &Array2<f32>, prev: Option<&Array1<f32>>, grad_output: &Array2<f32>) -> MoeBackward {
    assert_eq!(input.ncols(), moe.router.input_size);
    assert_eq!(grad_output.dim(), input.dim());
    let gates = moe.router.route(input);
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_router = Array2::zeros(moe.router.weight.raw_dim());
    let mut grad_prev = Array1::zeros(input.ncols());
    let mut grad_x_k = moe.experts.iter().map(|e| Array1::zeros(e.x_k.len())).collect::<Vec<_>>();
    let mut grad_key = moe.experts.iter().map(|e| Array2::zeros(e.key.raw_dim())).collect::<Vec<_>>();
    let mut grad_value = moe.experts.iter().map(|e| Array2::zeros(e.value.raw_dim())).collect::<Vec<_>>();

    for expert_id in 0..moe.experts.len() {
        let rows: Vec<usize> = (0..input.nrows()).filter(|&row| gates[[row, expert_id]] != 0.0).collect();
        if rows.is_empty() { continue; }
        let mut expert_grad = Array2::zeros(input.raw_dim());
        for &row in &rows { for col in 0..input.ncols() { expert_grad[[row, col]] = grad_output[[row, col]] * gates[[row, expert_id]]; } }
        let result = crate::rwkv_cmix_backward::backward(input, prev, &expert_grad, &moe.experts[expert_id].x_k, &moe.experts[expert_id].key, &moe.experts[expert_id].value);
        grad_input += &result.grad_input;
        grad_prev += &result.grad_prev;
        grad_x_k[expert_id] += &result.grad_x_k;
        grad_key[expert_id] += &result.grad_key;
        grad_value[expert_id] += &result.grad_value;
    }

    for row in 0..input.nrows() {
        let selected: Vec<usize> = (0..moe.experts.len()).filter(|&i| gates[[row, i]] > 0.0).collect();
        if selected.is_empty() { continue; }
        let mut scores = Vec::with_capacity(selected.len());
        for &expert_id in &selected {
            let out = moe.experts[expert_id].forward_rows(input, prev, &[row]);
            scores.push(grad_output.row(row).dot(&out.row(0)));
        }
        let expected = selected.iter().enumerate().map(|(i, &expert_id)| gates[[row, expert_id]] * scores[i]).sum::<f32>();
        for (i, &expert_id) in selected.iter().enumerate() {
            let dlogit = gates[[row, expert_id]] * (scores[i] - expected);
            for col in 0..input.ncols() {
                grad_router[[expert_id, col]] += dlogit * input[[row, col]];
                grad_input[[row, col]] += dlogit * moe.router.weight[[expert_id, col]];
            }
        }
    }
    MoeBackward { grad_input, grad_prev, grad_router, grad_x_k, grad_key, grad_value }
}

pub fn router_parameter_count(router: &MoeRouter) -> usize { router.weight.len() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::MoeCmix;
    use ndarray::Array2;
    #[test]
    fn backward_shapes() {
        let moe = MoeCmix::new(8, 0, 4, 3, 2, 7);
        let x = Array2::<f32>::ones((4, 8));
        let result = backward(&moe, &x, None, &Array2::ones((4, 8)));
        assert_eq!(result.grad_input.dim(), x.dim());
        assert_eq!(result.grad_router.dim(), (3, 8));
        assert_eq!(result.grad_key.len(), 3);
    }
}
