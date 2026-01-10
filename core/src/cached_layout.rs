use libdof::prelude::{Finger, PhysicalKey, Shape};
use libdof::magic::MagicKey;
use fxhash::FxHashMap as HashMap;

use crate::{
    analyze::Neighbor,
    char_mapping::CharMapping,
    dist::DistCache,
    layout::{Layout, MagicStealBigram, PosPair},
    magic::MagicCache,
    same_finger::SFCache,
    stats::Stats,
    stretches::StretchCache,
    types::{CacheKey, CachePos},
    weights::Weights,
};

pub const EMPTY_KEY: CacheKey = CacheKey::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBigram {
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaSkipgram {
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaTrigram {
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
#[derive(Debug, Clone, PartialEq)]
pub struct CachedLayout {
    // Layout identity
    name: String,
    char_mapping: CharMapping,
    keyboard: Box<[PhysicalKey]>,
    shape: Shape,

    /// Key at each position (EMPTY_KEY if unassigned)
    keys: Vec<CacheKey>,
    /// Position of each key (None if not placed)
    key_positions: Vec<Option<CachePos>>,
    /// Number of positions (for KeySwap neighbor calculation)
    num_positions: usize,
    /// Magic rules: for each (magic_key, leader), stores (current_output, all_possible_outputs)
    /// Neighbors are generated for each possible output != current_output
    magic_rules: Vec<MagicRule>,
    affected_grams: Vec<DeltaGram>,
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
    magic: MagicCache,
    fingers: Vec<Finger>,
}

/// A magic rule: (magic_key, leader) can steal to any of the possible_outputs.
/// current_output tracks which one is currently active.
#[derive(Debug, Clone, PartialEq)]
pub struct MagicRule {
    pub magic_key: CacheKey,
    pub leader: CacheKey,
    pub current_output: CacheKey,
    pub possible_outputs: Vec<CacheKey>,
}

impl Default for CachedLayout {
    fn default() -> Self {
        Self {
            name: String::new(),
            char_mapping: CharMapping::default(),
            keyboard: Box::new([]),
            shape: Shape::default(),
            keys: Vec::new(),
            key_positions: Vec::new(),
            num_positions: 0,
            magic_rules: Vec::new(),
            affected_grams: Vec::new(),
            dist: DistCache::default(),
            sfb: SFCache::default(),
            stretch: StretchCache::default(),
            magic: MagicCache::default(),
            fingers: Vec::new(),
        }
    }
}

impl CachedLayout {
    /// Create a new CachedLayout from a Layout and CharMapping
    pub fn new(
        layout: &Layout,
        char_mapping: CharMapping,
        num_keys: usize,
    ) -> Self {
        let len = layout.keyboard.len();
        let keyboard = &layout.keyboard;
        let fingers = &layout.fingers;

        let keys = vec![EMPTY_KEY; len];
        let key_positions = vec![None; num_keys];

        let affected_grams = Vec::with_capacity(len * len);

        let dist = DistCache::new(keyboard, fingers);
        let sfb = SFCache::new(fingers, keyboard);
        let stretch = StretchCache::new(keyboard, fingers);
        let magic = MagicCache::default();

        // Build magic rules from layout.magic
        let mut magic_rules = Vec::new();
        for (&magic_char, magic_key_def) in layout.magic.iter() {
            let magic_key = char_mapping.get_u(magic_char);
            // rules() returns BTreeMap<String, String> where key=leading, value=output
            // Group by leader to build MagicRule entries
            let mut leader_outputs: HashMap<CacheKey, Vec<CacheKey>> = HashMap::default();
            for (leading_str, output_str) in magic_key_def.rules().iter() {
                let leader = char_mapping.get_u(leading_str.chars().next().unwrap_or(' '));
                let output = char_mapping.get_u(output_str.chars().next().unwrap_or(' '));
                leader_outputs.entry(leader).or_default().push(output);
            }
            for (leader, outputs) in leader_outputs {
                if !outputs.is_empty() {
                    magic_rules.push(MagicRule {
                        magic_key,
                        leader,
                        current_output: outputs[0], // First output is initial
                        possible_outputs: outputs,
                    });
                }
            }
        }

        let mut cache = CachedLayout {
            name: layout.name.clone(),
            char_mapping,
            keyboard: layout.keyboard.clone(),
            shape: layout.shape.clone(),
            keys,
            key_positions,
            num_positions: len,
            magic_rules,
            affected_grams,
            dist,
            sfb,
            stretch,
            magic,
            fingers: fingers.to_vec(),
        };

        // Initialize keys from layout
        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = cache.char_mapping.get_u(key_char);
            cache.add_key(pos, key);
        }

        // Apply initial magic steals
        for rule in &cache.magic_rules.clone() {
            cache.magic.steal_bigram(
                rule.leader,
                rule.current_output,
                rule.magic_key,
                &cache.key_positions,
                cache.keys.len(),
                &mut cache.affected_grams,
            );
        }
        cache.affected_grams.clear();

        cache
    }

    /// Convert back to a Layout
    pub fn to_layout(&self) -> Layout {
        // Convert keys back to chars
        let keys: Box<[char]> = self.keys.iter()
            .map(|&k| if k == EMPTY_KEY { ' ' } else { self.char_mapping.get_c(k) })
            .collect();

        // Reconstruct magic HashMap from magic_rules
        let mut magic: HashMap<char, MagicKey> = HashMap::default();
        for rule in &self.magic_rules {
            let magic_char = self.char_mapping.get_c(rule.magic_key);
            let leader_char = self.char_mapping.get_c(rule.leader);
            let output_char = self.char_mapping.get_c(rule.current_output);

            let magic_key = magic.entry(magic_char).or_insert_with(|| {
                MagicKey::new(&magic_char.to_string())
            });
            magic_key.add_rule(&leader_char.to_string(), &output_char.to_string());
        }

        Layout {
            name: self.name.clone(),
            keys,
            fingers: self.fingers.clone().into_boxed_slice(),
            keyboard: self.keyboard.clone(),
            shape: self.shape.clone(),
            magic,
        }
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

    /// Populate stats from the caches. Requires totals for normalization.
    pub fn stats(&self, stats: &mut Stats, char_total: f64, bigram_total: f64, skipgram_total: f64, chars: &[i64]) {
        // Finger use: sum character frequencies per finger
        for (pos, &key) in self.keys.iter().enumerate() {
            if key != EMPTY_KEY && (key as usize) < chars.len() {
                let finger = self.fingers[pos] as usize;
                stats.finger_use[finger] += chars[key as usize] as f64 / char_total;
            }
        }

        // SFB/SFS stats
        self.sfb.stats(stats, bigram_total, skipgram_total);

        // Stretch stats
        self.stretch.stats(stats, bigram_total);
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
            Neighbor::MagicStealBigram(MagicStealBigram(magic_key, leader, new_output, _old_output)) => {
                self.steal_bigram(magic_key, leader, new_output);
            }
        }
    }

    /// Copy only the delta resulting from a neighbor transformation from another cache.
    /// Assumes `other` has had `neighbor` applied relative to `self`.
    /// This is more efficient than cloning the entire cache.
    pub fn copy_from(&mut self, other: &CachedLayout, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                // Copy the swapped keys
                let key_a = other.keys[a];
                let key_b = other.keys[b];
                self.keys[a] = key_a;
                self.keys[b] = key_b;

                // Copy key_positions for the affected keys
                if key_a < self.key_positions.len() {
                    self.key_positions[key_a] = other.key_positions[key_a];
                }
                if key_b < self.key_positions.len() {
                    self.key_positions[key_b] = other.key_positions[key_b];
                }
            }
            Neighbor::MagicStealBigram(MagicStealBigram(magic_key, leader, _new_output, _old_output)) => {
                // Copy the magic rule's current_output
                for (self_rule, other_rule) in self.magic_rules.iter_mut().zip(other.magic_rules.iter()) {
                    if self_rule.magic_key == magic_key && self_rule.leader == leader {
                        self_rule.current_output = other_rule.current_output;
                        break;
                    }
                }

                // Copy only affected magic frequency entries
                let keys = &other.keys;
                self.magic.copy_from(&other.magic, other.affected_grams(), |pos| keys[pos]);
            }
        }

        // Copy scoring caches - these use fixed-size arrays, no allocation
        self.sfb.copy_from(&other.sfb);
        self.stretch.copy_from(&other.stretch);
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
                    p_a: other_pos,
                    p_b: pos,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        }
    }

    /// Steal a bigram for magic key functionality.
    /// When typing leader->output, magic key intercepts and produces output.
    pub fn steal_bigram(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey) {
        self.affected_grams.clear();
        self.magic.steal_bigram(
            leader,
            new_output,
            magic_key,
            &self.key_positions,
            self.keys.len(),
            &mut self.affected_grams,
        );

        // Update magic rule tracking
        self.update_magic_rule(magic_key, leader, new_output);

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
                DeltaGram::Trigram(_) => {
                    // Trigrams don't affect SFB/stretch scores directly
                }
            }
        }
    }

    /// Total number of neighbors (KeySwaps + MagicStealBigrams)
    /// For magic, each rule contributes (possible_outputs.len() - 1) neighbors (all except current)
    #[inline]
    pub fn neighbor_count(&self) -> usize {
        let magic_count: usize = self.magic_rules.iter()
            .map(|r| r.possible_outputs.len().saturating_sub(1))
            .sum();
        self.key_swap_count() + magic_count
    }

    /// Number of KeySwap neighbors: n*(n-1)/2 for n positions
    #[inline]
    fn key_swap_count(&self) -> usize {
        let n = self.num_positions;
        n * (n - 1) / 2
    }

    /// Get neighbor by index. KeySwaps come first, then MagicStealBigrams.
    #[inline]
    pub fn get_neighbor(&self, idx: usize) -> Neighbor {
        let swap_count = self.key_swap_count();
        if idx < swap_count {
            // Decode triangular index to (a, b) where a < b
            // idx = a*n - a*(a+1)/2 + (b - a - 1)
            // Solve for a: a = floor((2n-1 - sqrt((2n-1)^2 - 8*idx)) / 2)
            let n = self.num_positions;
            let a = ((2 * n - 1) as f64 - ((2 * n - 1).pow(2) as f64 - 8.0 * idx as f64).sqrt()) / 2.0;
            let a = a.floor() as usize;
            let b = idx - (a * n - a * (a + 1) / 2) + a + 1;
            Neighbor::KeySwap(PosPair(a, b))
        } else {
            // Find which magic rule and which output this index corresponds to
            let mut magic_idx = idx - swap_count;
            for rule in &self.magic_rules {
                let alternatives = rule.possible_outputs.iter()
                    .filter(|&&o| o != rule.current_output)
                    .count();
                if magic_idx < alternatives {
                    // This is the rule - find the nth alternative output
                    let new_output = rule.possible_outputs.iter()
                        .filter(|&&o| o != rule.current_output)
                        .nth(magic_idx)
                        .copied()
                        .unwrap();
                    return Neighbor::MagicStealBigram(MagicStealBigram(
                        rule.magic_key,
                        rule.leader,
                        new_output,
                        rule.current_output,
                    ));
                }
                magic_idx -= alternatives;
            }
            panic!("Invalid neighbor index");
        }
    }

    /// Update magic rule's current output after a steal
    fn update_magic_rule(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey) {
        for rule in &mut self.magic_rules {
            if rule.magic_key == magic_key && rule.leader == leader {
                rule.current_output = new_output;
                return;
            }
        }
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
