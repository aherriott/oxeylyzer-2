// Helper types

// A Key value, like "A"
type CacheKey = usize;
// A position for a Key
type CachePos = usize;

// An IndexVec is a Vec which itself stores indexes, and thus supports reverse lookup
pub struct KeysCache {
    vec: Vec<Option<CacheKey>>,
    reverse_vec: Vec<Option<CachePos>>,
}

impl KeysCache {
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            reverse_vec: Vec::new(),
        }
    }

    pub fn get(&self, i: CachePos) -> CacheKey {
        self.vec[i]
    }

    pub fn set(&mut self, i: CachePos, value: CacheKey) {
        if self.reverse_vec[value].is_some() {
            if self.reverse_vec[value] != Some(i) {
                panic!(
                    "IndexVec: value {} is already set at index {}",
                    value,
                    self.reverse_vec[value].unwrap()
                );
            }
            self.reverse_vec[value] = None;
        }
        self.vec[i] = value;
        self.reverse_vec[value] = Some(i);
    }

    pub fn iter(&self) -> std::slice::Iter<CachePos> {
        self.vec.iter()
    }

    pub fn reverse_at(&self, idx: usize) -> Option<CachePos> {
        self.reverse_vec[idx]
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
            reverse_vec: vec![None; capacity],
        }
    }
}
