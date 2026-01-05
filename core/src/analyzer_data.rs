use std::sync::Arc;

use crate::{char_mapping::CharMapping, data::Data, types::CacheKey};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnalyzerData {
    pub name: String,
    pub chars: Vec<i64>,
    pub bigrams: Vec<Vec<i64>>,
    pub skipgrams: Vec<Vec<i64>>,
    pub trigrams: Vec<Vec<Vec<i64>>>,
    pub char_total: f64,
    pub bigram_total: f64,
    pub skipgram_total: f64,
    pub trigram_total: f64,
    pub mapping: Arc<CharMapping>,
}

impl AnalyzerData {
    pub fn new(data: Data) -> Self {
        let mut chars = vec![0; data.chars.len() + 8];
        let mut mapping = CharMapping::new();

        let char_total = data.char_total as f64 / 100.0;
        let bigram_total = data.bigram_total as f64 / 100.0;
        let skipgram_total = data.skipgram_total as f64 / 100.0;
        let trigram_total = data.trigram_total as f64 / 100.0;

        for (c, f) in data.chars {
            mapping.push(c);

            let i = mapping.get_u(c) as usize;
            chars[i] = (f * char_total) as i64;
        }

        debug_assert!(chars.len() >= mapping.len());

        chars.truncate(mapping.len());

        let len = chars.len();

        let mut bigrams = vec![vec![0; len]; len];

        for ([c1, c2], f) in data.bigrams {
            let u1 = mapping.get_u(c1) as usize;
            let u2 = mapping.get_u(c2) as usize;

            debug_assert_eq!(bigrams[u1][u2], 0);
            bigrams[u1][u2] = (f * bigram_total) as i64;
        }

        let mut skipgrams = vec![vec![0; len]; len];

        for ([c1, c2], f) in data.skipgrams {
            let u1 = mapping.get_u(c1) as usize;
            let u2 = mapping.get_u(c2) as usize;

            debug_assert_eq!(skipgrams[u1][u2], 0);
            skipgrams[u1][u2] = (f * skipgram_total) as i64;
        }

        let mut trigrams = vec![vec![vec![0; len]; len]; len];

        for ([c1, c2, c3], f) in data.trigrams {
            let u1 = mapping.get_u(c1) as usize;
            let u2 = mapping.get_u(c2) as usize;
            let u3 = mapping.get_u(c3) as usize;

            debug_assert_eq!(trigrams[u1][u2][u3], 0);
            trigrams[u1][u2][u3] = (f * trigram_total) as i64;
        }

        let mapping = Arc::new(mapping);

        Self {
            name: data.name,
            chars: chars,
            bigrams: bigrams,
            skipgrams: skipgrams,
            trigrams: trigrams,
            char_total,
            bigram_total,
            skipgram_total,
            trigram_total,
            mapping,
        }
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_char(&self, c: char) -> i64 {
        let i = self.mapping.get_u(c) as usize;
        self.chars[i]
    }

    pub fn get_bigram(&self, [c1, c2]: [char; 2]) -> i64 {
        let u1 = self.mapping.get_u(c1) as usize;
        let u2 = self.mapping.get_u(c2) as usize;
        self.bigrams[u1][u2]
    }

    pub fn get_skipgram(&self, [c1, c2]: [char; 2]) -> i64 {
        let u1 = self.mapping.get_u(c1) as usize;
        let u2 = self.mapping.get_u(c2) as usize;
        self.skipgrams[u1][u2]
    }

    pub fn get_trigram(&self, [c1, c2, c3]: [char; 3]) -> i64 {
        let u1 = self.mapping.get_u(c1) as usize;
        let u2 = self.mapping.get_u(c2) as usize;
        let u3 = self.mapping.get_u(c3) as usize;
        self.trigrams[u1][u2][u3]
    }

    #[inline]
    pub fn get_char_u(&self, c: CacheKey) -> i64 {
        self.chars[c as usize]
    }

    #[inline]
    pub fn get_bigram_u(&self, [c1, c2]: [CacheKey; 2]) -> i64 {
        let u1 = c1 as usize;
        let u2 = c2 as usize;
        self.bigrams[u1][u2]
    }

    #[inline]
    pub fn get_skipgram_u(&self, [c1, c2]: [CacheKey; 2]) -> i64 {
        let u1 = c1 as usize;
        let u2 = c2 as usize;
        self.skipgrams[u1][u2]
    }

    #[inline]
    pub fn get_trigram_u(&self, [c1, c2, c3]: [CacheKey; 3]) -> i64 {
        let u1 = c1 as usize;
        let u2 = c2 as usize;
        let u3 = c3 as usize;
        self.trigrams[u1][u2][u3]
    }
}
