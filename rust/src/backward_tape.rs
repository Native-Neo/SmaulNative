use ndarray::Array2;

#[derive(Default)]
pub struct BackwardTape {
    activations: Vec<Array2<f32>>,
}

impl BackwardTape {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, activation: Array2<f32>) {
        self.activations.push(activation);
    }

    pub fn len(&self) -> usize { self.activations.len() }
    pub fn is_empty(&self) -> bool { self.activations.is_empty() }

    pub fn pop(&mut self) -> Option<Array2<f32>> {
        self.activations.pop()
    }

    pub fn last(&self) -> Option<&Array2<f32>> {
        self.activations.last()
    }

    pub fn clear(&mut self) {
        self.activations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tape_reverses_activation_order() {
        let mut tape = BackwardTape::new();
        tape.push(array![[1.0]]);
        tape.push(array![[2.0]]);
        assert_eq!(tape.last().unwrap()[[0, 0]], 2.0);
        assert_eq!(tape.pop().unwrap()[[0, 0]], 2.0);
        assert_eq!(tape.pop().unwrap()[[0, 0]], 1.0);
        assert!(tape.is_empty());
    }
}
