pub struct TokenBatchIter<'a> {
    tokens: &'a [usize],
    batch_size: usize,
    position: usize,
}

impl<'a> TokenBatchIter<'a> {
    pub fn new(tokens: &'a [usize], batch_size: usize) -> Self {
        assert!(batch_size > 0);
        Self { tokens, batch_size, position: 0 }
    }
}

impl<'a> Iterator for TokenBatchIter<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.tokens.len() { return None; }
        let end = self.position.saturating_add(self.batch_size).min(self.tokens.len());
        let batch = &self.tokens[self.position..end];
        self.position = end;
        Some(batch)
    }
}
