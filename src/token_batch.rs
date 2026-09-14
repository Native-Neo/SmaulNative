#[derive(Clone, Debug)]
pub struct TokenBatch {
    tokens: Vec<usize>,
}

impl TokenBatch {
    pub fn new(tokens: Vec<usize>) -> Self { Self { tokens } }
    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }
    pub fn as_slice(&self) -> &[usize] { &self.tokens }
}
