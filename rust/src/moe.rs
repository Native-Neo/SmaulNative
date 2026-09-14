use crate::init::uniform;
use crate::rwkv_cmix::RwkvCmix;
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct MoeRouter {
    pub input_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub weight: Array2<f32>,
}

impl MoeRouter {
    pub fn new(input_size: usize, num_experts: usize, top_k: usize, seed: u64) -> Self {
        assert!(input_size > 0 && num_experts > 0);
        assert!(top_k > 0 && top_k <= num_experts);
        let scale = 1.0 / (input_size as f32).sqrt();
        Self {
            input_size,
            num_experts,
            top_k,
            weight: uniform(num_experts, input_size, -scale, scale, seed),
        }
    }

    pub fn from_weight(weight: Array2<f32>, top_k: usize) -> Self {
        assert!(weight.nrows() > 0 && weight.ncols() > 0);
        assert!(top_k > 0 && top_k <= weight.nrows());
        Self { input_size: weight.ncols(), num_experts: weight.nrows(), top_k, weight }
    }

    pub fn route(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.input_size);
        let logits = x.dot(&self.weight.t());
        let mut gates = Array2::zeros(logits.raw_dim());
        for row in 0..logits.nrows() {
            let mut order: Vec<usize> = (0..self.num_experts).collect();
            order.sort_unstable_by(|&a, &b| logits[[row, b]].total_cmp(&logits[[row, a]]));
            let selected = &order[..self.top_k];
            let max = selected.iter().map(|&i| logits[[row, i]]).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = selected.iter().map(|&i| (logits[[row, i]] - max).exp()).sum();
            for &i in selected {
                gates[[row, i]] = (logits[[row, i]] - max).exp() / sum;
            }
        }
        gates
    }

    pub fn parameter_count(&self) -> usize { self.weight.len() }
}

pub struct MoeCmix {
    pub experts: Vec<RwkvCmix>,
    pub router: MoeRouter,
}

impl MoeCmix {
    pub fn new(channels: usize, layer_id: usize, n_layer: usize, num_experts: usize, top_k: usize, seed: u64) -> Self {
        let experts = (0..num_experts)
            .map(|i| RwkvCmix::new(channels, layer_id, n_layer))
            .collect();
        let router = MoeRouter::new(channels, num_experts, top_k, seed);
        Self { experts, router }
    }

    pub fn from_experts(experts: Vec<RwkvCmix>, router: MoeRouter) -> Result<Self, String> {
        if experts.is_empty() { return Err("MoE requires at least one expert".into()); }
        if experts.len() != router.num_experts { return Err("expert count does not match router".into()); }
        if experts.iter().any(|expert| expert.channels != router.input_size) {
            return Err("expert channel count does not match router input size".into());
        }
        Ok(Self { experts, router })
    }

    pub fn forward(&self, x: &Array2<f32>, prev: Option<&Array1<f32>>) -> (Array2<f32>, Array1<f32>) {
        assert_eq!(x.ncols(), self.router.input_size);
        let gates = self.router.route(x);
        let mut output = Array2::zeros(x.raw_dim());
        let mut last = Array1::zeros(x.ncols());
        for (expert_id, expert) in self.experts.iter().enumerate() {
            let expert_output = expert.forward_selected(x, prev);
            for row in 0..x.nrows() {
                let gate = gates[[row, expert_id]];
                if gate != 0.0 {
                    for col in 0..x.ncols() {
                        output[[row, col]] += gate * expert_output[[row, col]];
                    }
                }
            }
            if expert_id == 0 {
                last = x.row(x.nrows() - 1).to_owned();
            }
        }
        (output, last)
    }

    pub fn parameter_count(&self) -> usize {
        self.router.parameter_count() + self.experts.iter().map(RwkvCmix::parameter_count).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn router_selects_exact_top_k() {
        let router = MoeRouter::from_weight(Array2::from_shape_vec((3, 2), vec![10.0, 0.0, 0.0, 10.0, 1.0, 1.0]).unwrap(), 2);
        let gates = router.route(&Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap());
        assert_eq!(gates.iter().filter(|&&v| v > 0.0).count(), 2);
        assert!((gates.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cmix_moe_preserves_shape() {
        let moe = MoeCmix::new(8, 0, 4, 3, 2, 7);
        let x = Array2::<f32>::ones((4, 8));
        let (y, last) = moe.forward(&x, None);
        assert_eq!(y.dim(), (4, 8));
        assert_eq!(last.len(), 8);
        assert_eq!(moe.parameter_count(), moe.router.parameter_count() + moe.experts.iter().map(RwkvCmix::parameter_count).sum::<usize>());
    }

    #[test]
    fn rejects_invalid_expert_router_pair() {
        let experts = vec![RwkvCmix::new(8, 0, 4)];
        let router = MoeRouter::new(8, 2, 1, 1);
        assert!(MoeCmix::from_experts(experts, router).is_err());
    }
}
