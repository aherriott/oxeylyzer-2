// Helper types

// A Key value, like "A"
pub type CacheKey = usize;
// A key position
pub type CachePos = usize;

// An IndexVec is a Vec which itself stores indexes, and thus supports reverse lookup
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeysCache {
    vec: Vec<CacheKey>,
    reverse_vec: Vec<Option<CachePos>>, // Option because some Key may not be assigned
}

impl KeysCache {
    pub fn new(layout: &Layout) -> Self {
        let mut keys = KeysCache {
            vec: Vec::new(),
            reverse_vec: Vec::new(),
        };
        for i in 0..layout.keys.len() {
            keys.vec.push(i);
            keys.reverse_vec.push(Some(i));
        }
        keys
    }

    pub fn get(&self, i: CachePos) -> CacheKey {
        self.vec[i]
    }

    pub fn set(&mut self, i: CachePos, value: CacheKey) {
        // TODO: See if this logic is right when I'm less hungover
        if self.reverse_vec[value].is_some() {
            if self.reverse_vec[value] != Some(i) {
                panic!(
                    "IndexVec: value {} is already set at index {}",
                    value,
                    self.reverse_vec[value].unwrap()
                );
            }
        }
        self.vec[i] = value;
        self.reverse_vec[value] = Some(i);
    }

    pub fn iter(&self) -> std::slice::Iter<CachePos> {
        self.vec.iter()
    }

    pub fn reverse_get(&self, i: CacheKey) -> Option<CachePos> {
        self.reverse_vec[idx]
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
            reverse_vec: vec![None; capacity],
        }
    }
}
