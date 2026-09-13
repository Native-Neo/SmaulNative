use ndarray::{Array1, Array2};

pub fn update_matrix(param: &mut Array2<f32>, grad: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>, lr: f32, beta1: f32, beta2: f32, weight_decay: f32, step: usize, eps: f32) {
    assert_eq!(param.dim(), grad.dim());
    assert_eq!(m.dim(), param.dim());
    assert_eq!(v.dim(), param.dim());
    let b1t = beta1.powi(step as i32);
    let b2t = beta2.powi(step as i32);
    let corr1 = 1.0 - b1t;
    let corr2 = 1.0 - b2t;
    for ((p, g), (mi, vi)) in param.iter_mut().zip(grad.iter()).zip(m.iter_mut().zip(v.iter_mut())) {
        *mi = beta1 * *mi + (1.0 - beta1) * *g;
        *vi = beta2 * *vi + (1.0 - beta2) * *g * *g;
        let mh = *mi / corr1.max(f32::MIN_POSITIVE);
        let vh = *vi / corr2.max(f32::MIN_POSITIVE);
        *p -= lr * (mh / (vh.sqrt() + eps) + weight_decay * *p);
    }
}

pub fn update_vector(param: &mut Array1<f32>, grad: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>, lr: f32, beta1: f32, beta2: f32, weight_decay: f32, step: usize, eps: f32) {
    assert_eq!(param.len(), grad.len());
    assert_eq!(m.len(), param.len());
    assert_eq!(v.len(), param.len());
    let b1t = beta1.powi(step as i32);
    let b2t = beta2.powi(step as i32);
    for (((p, g), mi), vi) in param.iter_mut().zip(grad.iter()).zip(m.iter_mut()).zip(v.iter_mut()) {
        *mi = beta1 * *mi + (1.0 - beta1) * *g;
        *vi = beta2 * *vi + (1.0 - beta2) * *g * *g;
        let mh = *mi / (1.0 - b1t).max(f32::MIN_POSITIVE);
        let vh = *vi / (1.0 - b2t).max(f32::MIN_POSITIVE);
        *p -= lr * (mh / (vh.sqrt() + eps) + weight_decay * *p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn matrix_update_changes_parameter() {
        let mut p = Array2::from_elem((1, 2), 1.0);
        let g = Array2::from_elem((1, 2), 1.0);
        let mut m = Array2::zeros((1, 2));
        let mut v = Array2::zeros((1, 2));
        update_matrix(&mut p, &g, &mut m, &mut v, 0.1, 0.9, 0.999, 0.0, 1, 1e-8);
        assert!(p[[0, 0]] < 1.0);
    }

    #[test]
    fn vector_update_is_finite() {
        let mut p = array![1.0, -1.0];
        let g = array![0.5, -0.25];
        let mut m = array![0.0, 0.0];
        let mut v = array![0.0, 0.0];
        update_vector(&mut p, &g, &mut m, &mut v, 0.01, 0.9, 0.999, 0.01, 1, 1e-8);
        assert!(p.iter().all(|x| x.is_finite()));
    }
}
