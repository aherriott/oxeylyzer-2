use libdof::prelude::{Finger, PhysicalKey};

use crate::{
    analyze::Neighbor,
    dist::DistCache,
    layout::{Layout, MagicStealBigram, PosPair},
    magic::MagicCache,
    same_finger::SFCache,
    stretches::StretchCache,
    types::{CacheKey, CachePos},
    weights::Weights,
    REPLACEMENT_CHAR,
};

pub const EMPTY_KEY: CacheKey = CacheKey::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBigram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaSkipgram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaTrigram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub c: CacheKey,
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub p_c: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeltaGram {
    Bigram(DeltaBigram),
    Skipgram(DeltaSkipgram),
    Trigram(DeltaTrigram),
}

// CachedLayout contains the minimum mutable data used to define a layout and store scoring.
// Designed to copy quickly and without allocation. Wrapped by Analyzer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CachedLayout {
    /// Key at each position (EMPTY_KEY if unassigned)
    keys: Vec<CacheKey>,
    /// Position of each key (None if not placed)
    key_positions: Vec<Option<CachePos>>,
    possible_neighbors: Vec<Neighbor>,
    affected_grams: Vec<DeltaGram>,
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
    magic: MagicCache,
    fingers: Vec<Finger>,
}

impl CachedLayout {
    // Allocates all the required memory
    pub fn new(
        keyboard: &[PhysicalKey],
        fingers: &[Finger],
        layout: &Layout,
        num_keys: usize,
    ) -> Self {
        let len = keyboard.len();

        let keys = vec![EMPTY_KEY; len];
        let key_positions = vec![None; num_keys];

        let possible_neighbors = Vec::with_capacity(
            len * len + // Keyswaps
            (len - layout.magic.len()) * layout.magic.len(), // Steal Bigrams
        );
        let affected_grams = Vec::with_capacity(len * len);

        let dist = DistCache::new(keyboard, fingers);
        let sfb = SFCache::new(fingers, keyboard);
        let stretch = StretchCache::new(keyboard, fingers);
        let magic = MagicCache::default();

        let mut cache = CachedLayout {
            keys,
            key_positions,
            possible_neighbors,
            affected_grams,
            dist,
            sfb,
            stretch,
            magic,
            fingers: fingers.to_vec(),
        };

        layout.keys.iter().enumerate().map(|i, u| {
            cache.add_key(i, u);
        });

        layout.magic.iter().for_each(|key, rules| {
            rules.iter().for_each(|leader, output| {
                cache.steal_bigram(key, leader, output);
            });
        });

        // Initialize keys from layout
        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = key_char as CacheKey; // TODO: proper char mapping
            cache.add_key(pos, key);
        }

        cache
    }

    /// Get the key at a position
    #[inline]
    pub fn get_key(&self, pos: CachePos) -> CacheKey {
        self.keys[pos]
    }

    /// Get the position of a key
    #[inline]
    pub fn get_pos(&self, key: CacheKey) -> Option<CachePos> {
        self.key_positions.get(key).copied().flatten()
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        self.sfb.score(weights) + self.stretch.score(weights)
    }

    /// Apply a neighbor transformation
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                let key_a = self.keys[a];
                let key_b = self.keys[b];
                self.remove_key(a);
                self.remove_key(b);
                self.add_key(a, key_b);
                self.add_key(b, key_a);
            }
            Neighbor::MagicStealBigram(MagicStealBigram(key, leader, output)) => {
                self.steal_bigram(key, leader, output);
            }
        }
    }

    /// Add a key at pos. Position should currently be empty.
    pub fn add_key(&mut self, pos: CachePos, key: CacheKey) {
        debug_assert!(self.keys[pos] == EMPTY_KEY, "Position {pos} is not empty");

        self.keys[pos] = key;
        if key < self.key_positions.len() {
            self.key_positions[key] = Some(pos);
        }

        // Clear and populate affected_grams with all bigrams/skipgrams involving this position
        self.affected_grams.clear();
        self.compute_affected_grams_for_pos(pos, key, true);

        // Update caches based on affected grams
        for gram in &self.affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    self.sfb.update_bigram(&self.dist, bg);
                    self.stretch.update_bigram(bg);
                }
                DeltaGram::Skipgram(sg) => {
                    self.sfb.update_skipgram(&self.dist, sg);
                    self.stretch.update_skipgram(sg);
                }
                DeltaGram::Trigram(_) => {}
            }
        }
    }

    /// Remove a key at pos. Position should currently contain a key.
    pub fn remove_key(&mut self, pos: CachePos) {
        let key = self.keys[pos];
        debug_assert!(key != EMPTY_KEY, "Position {pos} is already empty");

        // Clear and populate affected_grams (frequencies go to 0)
        self.affected_grams.clear();
        self.compute_affected_grams_for_pos(pos, key, false);

        // Update caches based on affected grams
        for gram in &self.affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    self.sfb.update_bigram(&self.dist, bg);
                    self.stretch.update_bigram(bg);
                }
                DeltaGram::Skipgram(sg) => {
                    self.sfb.update_skipgram(&self.dist, sg);
                    self.stretch.update_skipgram(sg);
                }
                DeltaGram::Trigram(_) => {}
            }
        }

        self.keys[pos] = EMPTY_KEY;
        if key < self.key_positions.len() {
            self.key_positions[key] = None;
        }
    }

    /// Compute all affected bigrams and skipgrams when a key is added/removed at a position.
    /// If `adding` is true, old_freq=0 and new_freq=actual. If false, old_freq=actual and new_freq=0.
    fn compute_affected_grams_for_pos(&mut self, pos: CachePos, key: CacheKey, adding: bool) {
        for (other_pos, &other_key) in self.keys.iter().enumerate() {
            if other_pos == pos || other_key == EMPTY_KEY {
                continue;
            }

            // Bigram: pos -> other_pos
            let bg_freq = self.magic.get_bg_freq(key, other_key);
            if bg_freq != 0 {
                let (old, new) = if adding { (0, bg_freq) } else { (bg_freq, 0) };
                self.affected_grams.push(DeltaGram::Bigram(DeltaBigram {
                    a: key,
                    b: other_key,
                    p_a: pos,
                    p_b: other_pos,
                    old_freq: old,
                    new_freq: new,
                }));
            }

            // Bigram: other_pos -> pos
            let bg_freq_rev = self.magic.get_bg_freq(other_key, key);
            if bg_freq_rev != 0 {
                let (old, new) = if adding { (0, bg_freq_rev) } else { (bg_freq_rev, 0) };
                self.affected_grams.push(DeltaGram::Bigram(DeltaBigram {
                    a: other_key,
                    b: key,
                    p_a: other_pos,
                    p_b: pos,
                    old_freq: old,
                    new_freq: new,
                }));
            }

            // Skipgram: pos -> other_pos
            let sg_freq = self.magic.get_sg_freq(key, other_key);
            if sg_freq != 0 {
                let (old, new) = if adding { (0, sg_freq) } else { (sg_freq, 0) };
                self.affected_grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                    a: key,
                    b: other_key,
                    p_a: pos,
                    p_b: other_pos,
                    old_freq: old,
                    new_freq: new,
                }));
            }

            // Skipgram: other_pos -> pos
            let sg_freq_rev = self.magic.get_sg_freq(other_key, key);
            if sg_freq_rev != 0 {
                let (old, new) = if adding { (0, sg_freq_rev) } else { (sg_freq_rev, 0) };
                self.affected_grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                    a: other_key,
                    b: key,
                    p_a: other_pos,
                    p_b: pos,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        }
    }

    /// Steal a bigram for magic key functionality
    pub fn steal_bigram(&mut self, _key: CacheKey, _leader: CacheKey, _output: CacheKey) {
        // TODO: implement magic steal bigram
    }

    pub fn possible_neighbors(&self) -> &Vec<Neighbor> {
        &self.possible_neighbors
    }

    pub fn affected_grams(&self) -> &Vec<DeltaGram> {
        &self.affected_grams
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramPair {
    pub pair: PosPair,
    pub dist: i64,
}
