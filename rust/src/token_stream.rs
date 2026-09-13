#[derive(Clone, Debug, Default)]
pub struct TokenStream {
    tokens: Vec<usize>,
}

impl TokenStream {
    pub fn new() -> Self { Self::default() }
    pub fn from_tokens(tokens: Vec<usize>) -> Self { Self { tokens } }
    pub fn push(&mut self, token: usize) { self.tokens.push(token); }
    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }
    pub fn as_slice(&self) -> &[usize] { &self.tokens }
    pub fn clear(&mut self) { self.tokens.clear(); }
}
