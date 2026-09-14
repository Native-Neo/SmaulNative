#[derive(Clone, Copy, Debug)]
pub struct ProgressInterval {
    every: usize,
}

impl ProgressInterval {
    pub fn new(every: usize) -> Self { assert!(every > 0); Self { every } }
    pub fn due(&self, step: usize) -> bool { step > 0 && step % self.every == 0 }
}
