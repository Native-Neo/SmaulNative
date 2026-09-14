use std::time::{Duration, Instant};

pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self { Self { start: Instant::now() } }
    pub fn elapsed(&self) -> Duration { self.start.elapsed() }
    pub fn seconds(&self) -> f64 { self.elapsed().as_secs_f64() }
}
