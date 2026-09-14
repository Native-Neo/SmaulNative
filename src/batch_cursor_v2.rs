#[derive(Clone, Debug)]
pub struct BatchCursor {
    position: usize,
    total: usize,
}

impl BatchCursor {
    pub fn new(total: usize) -> Self { Self { position: 0, total } }
    pub fn position(&self) -> usize { self.position }
    pub fn remaining(&self) -> usize { self.total.saturating_sub(self.position) }
    pub fn finished(&self) -> bool { self.position >= self.total }
    pub fn advance(&mut self, count: usize) { self.position = self.position.saturating_add(count).min(self.total); }
    pub fn reset(&mut self) { self.position = 0; }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clamps_to_total() {
        let mut cursor = BatchCursor::new(10);
        cursor.advance(7);
        assert_eq!(cursor.remaining(), 3);
        cursor.advance(7);
        assert!(cursor.finished());
    }
}
