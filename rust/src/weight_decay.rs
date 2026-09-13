pub fn decoupled_weight_decay(parameter: &mut [f32], lr: f32, weight_decay: f32) {
    let factor = 1.0 - lr * weight_decay;
    for value in parameter {
        *value *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_scales_parameters() {
        let mut values = vec![2.0, -4.0];
        decoupled_weight_decay(&mut values, 0.1, 0.01);
        assert!((values[0] - 1.998).abs() < 1e-6);
        assert!((values[1] + 3.996).abs() < 1e-6);
    }
}
