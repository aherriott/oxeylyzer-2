use indexmap::IndexMap;

use crate::{MAGIC_CHARS, REPLACEMENT_CHAR, SHIFT_CHAR, SPACE_CHAR};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharMapping(IndexMap<char, CacheKey>);

impl Default for CharMapping {
    fn default() -> Self {
        let mut map = Self(Default::default());
        map.push(REPLACEMENT_CHAR);
        map.push(SHIFT_CHAR);
        map.push(SPACE_CHAR);
        for char in MAGIC_CHARS.chars() {
            map.push(char);
        }

        map
    }
}

impl CharMapping {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, c: char) {
        if !self.0.contains_key(&c) {
            self.0.insert(c, self.len() as CacheKey);
        }
    }

    pub fn remove(&mut self, c: char) -> Option<CacheKey> {
        self.0.swap_remove(&c)
    }

    pub fn pop(&mut self) -> Option<(char, CacheKey)> {
        self.0.pop()
    }

    pub fn get_u(&self, c: char) -> CacheKey {
        match self.0.get(&c) {
            Some(c) => *c,
            None => 0,
        }
    }

    pub fn get_c(&self, u: CacheKey) -> char {
        match self.0.get_index(u as usize) {
            Some((c, _)) => *c,
            None => REPLACEMENT_CHAR,
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn map_cs<'a>(&'a self, s: &'a str) -> impl Iterator<Item = CacheKey> + 'a {
        s.chars().map(|c| self.get_u(c))
    }

    pub fn map_us<'a>(&'a self, u: &'a [CacheKey]) -> impl Iterator<Item = char> + 'a {
        u.iter().map(|u| self.get_c(*u))
    }
}

impl From<&str> for CharMapping {
    fn from(value: &str) -> Self {
        Self::from_iter(value.chars())
    }
}

impl From<String> for CharMapping {
    fn from(value: String) -> Self {
        Self::from_iter(value.chars())
    }
}

impl<const N: usize> From<[char; N]> for CharMapping {
    fn from(arr: [char; N]) -> Self {
        arr.into_iter().collect()
    }
}

impl From<&[char]> for CharMapping {
    fn from(slice: &[char]) -> Self {
        slice.iter().collect()
    }
}

impl FromIterator<char> for CharMapping {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        let mut res = Self::new();

        for c in iter {
            res.push(c)
        }

        res
    }
}

impl<'a> FromIterator<&'a char> for CharMapping {
    fn from_iter<T: IntoIterator<Item = &'a char>>(iter: T) -> Self {
        iter.into_iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_from() {
        let mapping_s = "abcdefhgijklmnopqrstuvwxyz ";
        let mapping = mapping_s.chars().collect::<CharMapping>();

        // CharMapping::new() starts with 3 defaults + 5 MAGIC_CHARS
        assert_eq!(mapping.len(), mapping_s.len() + 8);

        let s = "this is epic-";
        let u = mapping.map_cs(s).collect::<Vec<_>>();
        let c = mapping.map_us(&u).collect::<String>();

        assert_eq!(c, "this is epic�")
    }
}
