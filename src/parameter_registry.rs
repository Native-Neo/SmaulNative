use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct ParameterRegistry {
    offsets: BTreeMap<String, (usize, usize)>,
    total: usize,
}

impl ParameterRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, name: impl Into<String>, len: usize) -> usize {
        let name = name.into();
        assert!(!self.offsets.contains_key(&name), "duplicate parameter: {name}");
        let start = self.total;
        self.total = self.total.checked_add(len).expect("parameter count overflow");
        self.offsets.insert(name, (start, len));
        start
    }

    pub fn total_len(&self) -> usize { self.total }

    pub fn get(&self, name: &str) -> Option<(usize, usize)> { self.offsets.get(name).copied() }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.offsets.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_contiguous_offsets() {
        let mut r = ParameterRegistry::new();
        assert_eq!(r.register("a", 3), 0);
        assert_eq!(r.register("b", 5), 3);
        assert_eq!(r.get("a"), Some((0, 3)));
        assert_eq!(r.get("b"), Some((3, 5)));
        assert_eq!(r.total_len(), 8);
    }
}
