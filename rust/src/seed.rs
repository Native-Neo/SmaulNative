#[derive(Clone, Copy, Debug)]
pub struct Seed {
    state: u64,
}

impl Seed {
    pub fn new(seed: u64) -> Self { Self { state: seed } }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_reproducible() {
        let mut a = Seed::new(42);
        let mut b = Seed::new(42);
        for _ in 0..8 { assert_eq!(a.next_u64(), b.next_u64()); }
    }
}
