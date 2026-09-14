use ndarray::{Array1, Array2};

pub fn scale_matrix(grad: &mut Array2<f32>, scale: f32) { grad.mapv_inplace(|v| v * scale); }
pub fn scale_vector(grad: &mut Array1<f32>, scale: f32) { grad.mapv_inplace(|v| v * scale); }

pub fn unscale_and_check_matrix(grad: &mut Array2<f32>, scale: f32) -> bool {
    assert!(scale > 0.0);
    let inv = 1.0 / scale;
    let mut finite = true;
    for v in grad.iter_mut() { *v *= inv; finite &= v.is_finite(); }
    finite
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaling_round_trips() {
        let mut g = Array2::from_elem((2, 2), 3.0);
        scale_matrix(&mut g, 8.0);
        assert!(unscale_and_check_matrix(&mut g, 8.0));
        assert!(g.iter().all(|v| *v == 3.0));
    }
}
