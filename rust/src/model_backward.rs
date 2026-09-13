use ndarray::Array2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackwardBlockKind {
    Rwkv(usize),
    Moba(usize),
}

pub struct ModelBackwardTape {
    pub block_order: Vec<BackwardBlockKind>,
    pub inputs: Vec<Array2<f32>>,
    pub outputs: Vec<Array2<f32>>,
    pub normalized: Option<Array2<f32>>,
    pub logits: Option<Array2<f32>>,
}

impl ModelBackwardTape {
    pub fn new(block_order: Vec<BackwardBlockKind>) -> Self {
        Self { block_order, inputs: Vec::new(), outputs: Vec::new(), normalized: None, logits: None }
    }

    pub fn record_block(&mut self, input: Array2<f32>, output: Array2<f32>) {
        assert_eq!(input.nrows(), output.nrows());
        self.inputs.push(input);
        self.outputs.push(output);
        assert_eq!(self.inputs.len(), self.outputs.len());
    }

    pub fn record_head(&mut self, normalized: Array2<f32>, logits: Array2<f32>) {
        assert_eq!(normalized.nrows(), logits.nrows());
        self.normalized = Some(normalized);
        self.logits = Some(logits);
    }

    pub fn reverse_blocks(&self) -> impl DoubleEndedIterator<Item = (BackwardBlockKind, &Array2<f32>, &Array2<f32>)> {
        assert_eq!(self.block_order.len(), self.inputs.len());
        assert_eq!(self.inputs.len(), self.outputs.len());
        self.block_order.iter().copied().zip(self.inputs.iter()).zip(self.outputs.iter()).map(|((kind, input), output)| (kind, input, output)).rev()
    }

    pub fn len(&self) -> usize { self.inputs.len() }
    pub fn is_empty(&self) -> bool { self.inputs.is_empty() }
    pub fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
        self.normalized = None;
        self.logits = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn reverse_order_matches_model_execution_order() {
        let order = vec![BackwardBlockKind::Rwkv(0), BackwardBlockKind::Moba(0), BackwardBlockKind::Rwkv(1)];
        let mut tape = ModelBackwardTape::new(order);
        tape.record_block(Array2::zeros((1, 2)), Array2::ones((1, 2)));
        tape.record_block(Array2::ones((1, 2)), Array2::from_elem((1, 2), 2.0));
        tape.record_block(Array2::from_elem((1, 2), 2.0), Array2::from_elem((1, 2), 3.0));
        let kinds: Vec<_> = tape.reverse_blocks().map(|x| x.0).collect();
        assert_eq!(kinds, vec![BackwardBlockKind::Rwkv(1), BackwardBlockKind::Moba(0), BackwardBlockKind::Rwkv(0)]);
    }
}
