pub fn optimizer_step(parameter: &mut [f32], update: &[f32], lr: f32) {
    assert_eq!(parameter.len(), update.len());
    for (value, delta) in parameter.iter_mut().zip(update) {
        *value -= lr * *delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_step_applies_update() {
        let mut parameter = vec![1.0, -2.0];
        optimizer_step(&mut parameter, &[2.0, -3.0], 0.1);
        assert_eq!(parameter, vec![0.8, -1.7]);
    }
}
