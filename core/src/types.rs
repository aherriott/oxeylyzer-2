// Helper types

// A Key value, like "A"
typedef CacheKey usize;
// A position for a Key
typedef CachePos usize;

// An IndexVec is a Vec which itself stores indexes, and thus supports reverse lookup
struct IndexVec {
    vec: Vec<usize>,
    reverse_vec: Vec<Option<usize>>,
}

impl IndexVec {
    override [i: usize] {
        // Return value
        self.vec[i]
    }

    override [i: usize]
        // When setting
        self.vec[i] = set_value;
        self.reverse_vec[set_value] = Some(i);
    }

    pub fn iter(&self) -> Iter<usize> {
        self.vec.iter()
    }

    pub fn reverse_at(&self, idx: usize) {
        self.reverse_vec[idx]
    }

    pub fn with_capacity(&mut self, capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
            reverse_vec: vec![0; capacity],
        }
    }
}


