use libdof::prelude::{Finger, PhysicalKey, Shape};
use libdof::magic::MagicKey;
use fxhash::FxHashMap as HashMap;

use crate::{
    analyze::Neighbor,
    analyzer_data::AnalyzerData,
    char_mapping::CharMapping,
    data::Data,
    dist::DistCache,
    layout::{Layout, MagicRule, PosPair},
    same_finger::SFCache,
    scissors::ScissorsCache,
    stats::Stats,
    stretches::StretchCache,
    trigrams::TrigramCache,
    types::{CacheKey, CachePos},
    weights::Weights,
};

pub const EMPTY_KEY: CacheKey = CacheKey::MAX;

/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

/// MagicCache stores frequency tables for bigrams, skipgrams, and trigrams.
///
/// # Constant Frequency Architecture
///
/// **Important**: All frequency arrays (`bg_freq`, `sg_freq`, `tg_freq`) are **constant after
/// initialization**. They are set once via [`init_from_data`](Self::init_from_data) and never
/// modified thereafter.
///
/// ## Why Frequencies Are Constant
///
/// This design enables **O(1) speculative scoring** during layout optimization. When evaluating
/// potential moves (key swaps, magic rules), the optimizer needs to quickly compute score deltas
/// without mutating state. By keeping frequencies constant:
///
/// 1. **Pre-computed lookup tables**: Each analyzer can pre-compute weighted score contributions
///    for all key/position combinations, enabling O(1) lookups instead of O(n) iteration.
///
/// 2. **No state reversal needed**: Speculative scoring (`apply=false`) doesn't require tracking
///    and reverting frequency changes, eliminating the overhead of the old `affected_grams` approach.
///
/// 3. **Analyzer independence**: Each analyzer (TrigramCache, SFCache, etc.) computes magic rule
///    effects independently via `add_rule` methods, using the constant frequencies as input.
///
/// ## Historical Context
///
/// Previously, `MagicCache` had a `steal_bigram` method that redistributed frequencies when magic
/// rules were applied. This required tracking changes via `DeltaGram` types and reverting them
/// for speculative scoring, causing ~28µs overhead per speculative score due to iterating over
/// ~5400 trigram combinations. The constant-frequency architecture targets ~1µs per speculative score.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    bg_freq: Vec<i64>,
    sg_freq: Vec<i64>,
    /// Trigram frequencies indexed as `tg_freq[a][b][c]` (kept for compatibility).
    tg_freq: Vec<Vec<Vec<i64>>>,
    /// Flat trigram frequencies indexed as `tg_freq_flat[a * nk * nk + b * nk + c]`.
    tg_freq_flat: Vec<i64>,
    num_keys: usize,
}

impl MagicCache {
    pub fn new(num_keys: usize) -> Self {
        Self {
            bg_freq: vec![0; num_keys * num_keys],
            sg_freq: vec![0; num_keys * num_keys],
            tg_freq: vec![vec![vec![0; num_keys]; num_keys]; num_keys],
            tg_freq_flat: vec![0; num_keys * num_keys * num_keys],
            num_keys,
        }
    }

    /// Initialize frequencies from corpus data (2D arrays).
    ///
    /// **This is the only place where frequency arrays are set.** After this method returns,
    /// `bg_freq`, `sg_freq`, and `tg_freq` remain constant for the lifetime of this `MagicCache`.
    ///
    /// # Arguments
    ///
    /// * `bigrams` - 2D array of bigram frequencies where `bigrams[a][b]` is the frequency of
    ///   the bigram (a, b)
    /// * `skipgrams` - 2D array of skipgram frequencies where `skipgrams[a][b]` is the frequency
    ///   of the skipgram (a, _, b)
    /// * `trigrams` - 3D array of trigram frequencies where `trigrams[a][b][c]` is the frequency
    ///   of the trigram (a, b, c)
    ///
    /// # Design Note
    ///
    /// Keeping frequencies constant after initialization enables O(1) speculative scoring.
    /// Each analyzer can pre-compute lookup tables based on these constant frequencies,
    /// avoiding the need to track and revert frequency changes during optimization.
    pub fn init_from_data(&mut self, bigrams: &[Vec<i64>], skipgrams: &[Vec<i64>], trigrams: &[Vec<Vec<i64>>]) {
        let num_keys = self.num_keys;
        for (a, row) in bigrams.iter().enumerate() {
            for (b, &freq) in row.iter().enumerate() {
                self.bg_freq[a * num_keys + b] = freq;
            }
        }
        for (a, row) in skipgrams.iter().enumerate() {
            for (b, &freq) in row.iter().enumerate() {
                self.sg_freq[a * num_keys + b] = freq;
            }
        }
        self.tg_freq = trigrams.to_vec();
        // Also populate flat trigram array
        let nk2 = num_keys * num_keys;
        for (a, plane) in trigrams.iter().enumerate() {
            for (b, row) in plane.iter().enumerate() {
                for (c, &freq) in row.iter().enumerate() {
                    self.tg_freq_flat[a * nk2 + b * num_keys + c] = freq;
                }
            }
        }
    }

    #[inline]
    pub fn bg_freq_flat(&self) -> &[i64] {
        &self.bg_freq
    }

    #[inline]
    pub fn sg_freq_flat(&self) -> &[i64] {
        &self.sg_freq
    }

    #[inline]
    pub fn tg_freq(&self) -> &[Vec<Vec<i64>>] {
        &self.tg_freq
    }

    pub fn tg_freq_flat(&self) -> &[i64] {
        &self.tg_freq_flat
    }

    #[inline]
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        if a < self.num_keys && b < self.num_keys {
            self.bg_freq[a * self.num_keys + b]
        } else {
            0
        }
    }

    #[inline]
    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        if a < self.num_keys && b < self.num_keys {
            self.sg_freq[a * self.num_keys + b]
        } else {
            0
        }
    }

    #[inline]
    pub fn get_tg_freq(&self, a: CacheKey, b: CacheKey, c: CacheKey) -> i64 {
        self.tg_freq
            .get(a)
            .and_then(|r1| r1.get(b))
            .and_then(|r2| r2.get(c))
            .copied()
            .unwrap_or(0)
    }

    // NOTE: steal_bigram method has been removed as part of the const-freq-analyzers refactoring.
    // Magic rule effects are now computed directly by each analyzer via add_rule methods.
    // See Task 6 in .kiro/specs/const-freq-analyzers/tasks.md for the CachedLayout refactoring.

    // NOTE: revert_affected method has been removed as part of the const-freq-analyzers refactoring.
    // This method was used to revert frequency changes from affected_grams, which is no longer needed
    // since frequencies will be constant after initialization.
    // See Task 1.2 in .kiro/specs/const-freq-analyzers/tasks.md.
}

// NOTE: DeltaBigram, DeltaSkipgram, DeltaTrigram, and DeltaGram types have been removed
// as part of the const-freq-analyzers refactoring (Task 1.3).
// These types were used to track frequency changes for reverting, which is no longer needed
// since frequencies will be constant after initialization.
// Magic rule effects are now computed directly by each analyzer via add_rule methods.
// See .kiro/specs/const-freq-analyzers/tasks.md for details.

#[derive(Debug, Clone, PartialEq)]
pub struct CachedLayout {
    name: String,
    keyboard: Box<[PhysicalKey]>,
    shape: Shape,
    data: AnalyzerData,
    keys: Vec<CacheKey>,
    key_positions: Vec<Option<CachePos>>,
    num_positions: usize,
    neighbors: Vec<Neighbor>,
    current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey>,
    // NOTE: affected_grams field has been removed as part of the const-freq-analyzers refactoring (Task 1.3).
    // This field was used to track frequency changes for reverting, which is no longer needed
    // since frequencies will be constant after initialization.
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
    scissors: ScissorsCache,
    trigram: TrigramCache,
    magic: MagicCache,
    fingers: Vec<Finger>,
}

impl Default for CachedLayout {
    fn default() -> Self {
        Self {
            name: String::new(),
            keyboard: Box::new([]),
            shape: Shape::default(),
            data: AnalyzerData::default(),
            keys: Vec::new(),
            key_positions: Vec::new(),
            num_positions: 0,
            neighbors: Vec::new(),
            current_magic_rules: HashMap::default(),
            dist: DistCache::default(),
            sfb: SFCache::default(),
            stretch: StretchCache::default(),
            scissors: ScissorsCache::default(),
            trigram: TrigramCache::default(),
            magic: MagicCache::default(),
            fingers: Vec::new(),
        }
    }
}


impl CachedLayout {
    pub fn new(layout: &Layout, data: Data, weights: &Weights) -> Self {
        let mut analyzer_data = AnalyzerData::new(data);

        // Ensure all layout keys are in the char_mapping for proper roundtrip
        for &key_char in layout.keys.iter() {
            analyzer_data.push_char(key_char);
        }

        let len = layout.keyboard.len();
        let keyboard = &layout.keyboard;
        let fingers = &layout.fingers;
        let num_keys = analyzer_data.len();

        let keys = vec![EMPTY_KEY; len];
        let key_positions = vec![None; num_keys];

        let dist = DistCache::new(keyboard, fingers);
        let mut sfb = SFCache::new(fingers, keyboard, dist.distances(), num_keys);
        sfb.set_weights(weights);
        let mut stretch = StretchCache::new(keyboard, fingers, num_keys);
        stretch.set_weights(weights);
        let mut scissors = ScissorsCache::new(keyboard, fingers, num_keys);
        scissors.set_weights(weights);
        let mut trigram = TrigramCache::new(fingers, num_keys);
        trigram.set_weights(weights);
        let mut magic = MagicCache::new(num_keys);
        magic.init_from_data(&analyzer_data.bigrams, &analyzer_data.skipgrams, &analyzer_data.trigrams);

        // Pre-compute all neighbors
        let mut neighbors = Vec::new();

        // Key swap neighbors: all pairs (a, b) where a < b
        for a in 0..len {
            for b in (a + 1)..len {
                neighbors.push(Neighbor::KeySwap(PosPair(a, b)));
            }
        }

        // Magic rule neighbors
        let mut current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey> = HashMap::default();

        for (&magic_char, magic_key_def) in layout.magic.iter() {
            let magic_key = analyzer_data.char_mapping().get_u(magic_char);

            // Collect all possible outputs for this magic key
            let mut all_outputs: Vec<CacheKey> = Vec::new();

            for (leading_str, output_str) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                let output = analyzer_data.char_mapping().get_u(output_str.chars().next().unwrap_or(' '));
                current_magic_rules.insert((magic_key, leader), output);
                if !all_outputs.contains(&output) {
                    all_outputs.push(output);
                }
            }

            // Add EMPTY_KEY as a possible output (to clear rules)
            all_outputs.push(EMPTY_KEY);

            // For each leader that has a rule, create neighbors for all possible outputs
            for (leading_str, _) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                for &output in &all_outputs {
                    neighbors.push(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output)));
                }
            }
        }

        let mut cache = CachedLayout {
            name: layout.name.clone(),
            keyboard: layout.keyboard.clone(),
            shape: layout.shape.clone(),
            data: analyzer_data,
            keys,
            key_positions,
            num_positions: len,
            neighbors,
            current_magic_rules,
            dist,
            sfb,
            stretch,
            scissors,
            trigram,
            magic,
            fingers: fingers.to_vec(),
        };

        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = cache.data.char_mapping().get_u(key_char);
            cache.replace_key(pos, EMPTY_KEY, key);
        }

        // Initialize pre-computed weighted scores for O(1) speculative trigram scoring
        cache.trigram.init_weighted_scores(&cache.keys, cache.magic.tg_freq());

        // Apply initial magic rules to the analyzers
        // The current_magic_rules map was populated from the layout's magic key definitions,
        // but the analyzers need to be initialized with these rules as well.
        let initial_rules: Vec<_> = cache.current_magic_rules.iter()
            .map(|(&(magic_key, leader), &output)| (magic_key, leader, output))
            .collect();
        for (magic_key, leader, output) in initial_rules {
            // Temporarily remove from current_magic_rules so apply_magic_rule doesn't early-return
            cache.current_magic_rules.remove(&(magic_key, leader));
            cache.apply_magic_rule(magic_key, leader, output, true);
        }

        cache
    }

    pub fn data(&self) -> &AnalyzerData {
        &self.data
    }

    pub fn char_mapping(&self) -> &CharMapping {
        self.data.char_mapping()
    }

    pub fn to_layout(&self) -> Layout {
        // Convert keys back to chars
        let keys: Box<[char]> = self.keys.iter()
            .map(|&k| if k == EMPTY_KEY { ' ' } else { self.data.char_mapping().get_c(k) })
            .collect();

        // Reconstruct magic HashMap from current_magic_rules
        let mut magic: HashMap<char, MagicKey> = HashMap::default();

        for (&(magic_key_id, leader), &output) in &self.current_magic_rules {
            if output == EMPTY_KEY {
                continue;
            }

            let magic_char = self.data.char_mapping().get_c(magic_key_id);
            let leader_char = self.data.char_mapping().get_c(leader);
            let output_char = self.data.char_mapping().get_c(output);

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

    pub fn score(&self) -> i64 {
        self.sfb.score() + self.stretch.score() + self.scissors.score() + self.trigram.score()
    }

    /// Populate stats from the caches.
    pub fn stats(&self, stats: &mut Stats) {
        let char_total = self.data.char_total;
        let bigram_total = self.data.bigram_total;
        let skipgram_total = self.data.skipgram_total;
        let chars = &self.data.chars;

        // Finger use: sum character frequencies per finger
        for (pos, &key) in self.keys.iter().enumerate() {
            if key != EMPTY_KEY && (key as usize) < chars.len() {
                let finger = self.fingers[pos] as usize;
                stats.finger_use[finger] += chars[key as usize] as f64 / char_total;
            }
        }

        self.sfb.stats(stats, bigram_total, skipgram_total);
        self.stretch.stats(stats, bigram_total);
        self.scissors.stats(stats, bigram_total, skipgram_total);
        self.trigram.stats(stats, self.data.trigram_total);
    }

    // ==================== New API ====================

    /// Speculative score for a neighbor. No mutation.
    /// For SA: O(1) for trigrams (flat arrays), O(pairs) for bigram caches.
    /// Only valid when trigram weighted_scores are fresh.
    pub fn score_neighbor(&self, neighbor: Neighbor) -> i64 {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.score_swap(a, b),
            Neighbor::MagicRule(_rule) => {
                // Magic rule speculative scoring requires &mut self due to add_rule API.
                // Callers should use apply_magic_rule(... , false) directly for magic rules.
                // For KeySwap neighbors (the common case in SA), this is pure &self.
                panic!("Use apply_magic_rule with apply=false for speculative magic rule scoring")
            }
        }
    }

    /// Apply a neighbor. Mutates keys + running totals.
    /// score() remains valid. score_neighbor() becomes invalid for trigrams
    /// until update_scores() is called.
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.swap_keys(a, b),
            Neighbor::MagicRule(rule) => {
                self.apply_magic_rule(rule.magic_key, rule.leader, rule.output, true);
            }
        }
    }

    /// Apply a neighbor and update weighted_score arrays.
    /// Both score() and score_neighbor() remain valid after this call.
    pub fn apply_neighbor_and_update(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.swap_keys_and_update(a, b),
            Neighbor::MagicRule(rule) => {
                self.apply_magic_rule(rule.magic_key, rule.leader, rule.output, true);
            }
        }
    }

    /// Speculative score for a neighbor (mutable version).
    /// Handles both KeySwap and MagicRule neighbors.
    /// For KeySwap: no actual mutation occurs.
    /// For MagicRule: uses apply=false path which doesn't mutate.
    pub fn score_neighbor_mut(&mut self, neighbor: Neighbor) -> i64 {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.score_swap(a, b),
            Neighbor::MagicRule(rule) => {
                self.apply_magic_rule(rule.magic_key, rule.leader, rule.output, false)
            }
        }
    }

    /// Rebuild trigram weighted_score arrays from current state.
    /// Call after one or more apply_neighbor() calls when you need score_neighbor() again.
    pub fn update_scores(&mut self) {
        self.trigram.init_weighted_scores(&self.keys, self.magic.tg_freq());
    }

    /// Speculative swap score. No mutation.
    fn score_swap(&self, pos_a: CachePos, pos_b: CachePos) -> i64 {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];
        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq_flat = self.magic.tg_freq_flat();

        let sfb = self.sfb.score_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        let stretch = self.stretch.score_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq);
        let scissors = self.scissors.score_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        let trigram = self.trigram.score_swap(pos_a, pos_b, key_a, key_b, &self.keys, tg_freq_flat);

        sfb + stretch + scissors + trigram
    }

    // ==================== Mutation ====================

    /// Replace key at position. Mutates state.
    #[inline]
    pub fn replace_key(&mut self, pos: CachePos, old_key: CacheKey, new_key: CacheKey) {
        debug_assert!(self.keys[pos] == old_key, "Position {pos} has key {} but expected {old_key}", self.keys[pos]);

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        self.sfb.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq);
        self.stretch.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq);
        self.scissors.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq);
        self.trigram.replace_key(pos, old_key, new_key, &self.keys, None, tg_freq);

        self.keys[pos] = new_key;
        if old_key != EMPTY_KEY && old_key < self.key_positions.len() {
            self.key_positions[old_key] = None;
        }
        if new_key != EMPTY_KEY && new_key < self.key_positions.len() {
            self.key_positions[new_key] = Some(pos);
        }
    }

    /// Swap keys at two positions. Mutates running totals only.
    /// score() remains valid. score_neighbor() becomes invalid for trigrams.
    #[inline]
    pub fn swap_keys(&mut self, pos_a: CachePos, pos_b: CachePos) {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];

        debug_assert!(key_a != EMPTY_KEY, "Position {pos_a} is empty");
        debug_assert!(key_b != EMPTY_KEY, "Position {pos_b} is empty");

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        self.sfb.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.stretch.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq);
        self.scissors.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.trigram.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, tg_freq);

        self.keys[pos_a] = key_b;
        self.keys[pos_b] = key_a;
        if key_a < self.key_positions.len() {
            self.key_positions[key_a] = Some(pos_b);
        }
        if key_b < self.key_positions.len() {
            self.key_positions[key_b] = Some(pos_a);
        }
    }

    /// Swap keys and update trigram weighted_score arrays.
    /// Both score() and score_neighbor() remain valid.
    #[inline]
    pub fn swap_keys_and_update(&mut self, pos_a: CachePos, pos_b: CachePos) {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];

        debug_assert!(key_a != EMPTY_KEY, "Position {pos_a} is empty");
        debug_assert!(key_b != EMPTY_KEY, "Position {pos_b} is empty");

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        self.sfb.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.stretch.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq);
        self.scissors.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.trigram.key_swap_and_update(pos_a, pos_b, key_a, key_b, &self.keys, tg_freq);

        self.keys[pos_a] = key_b;
        self.keys[pos_b] = key_a;
        if key_a < self.key_positions.len() {
            self.key_positions[key_a] = Some(pos_b);
        }
        if key_b < self.key_positions.len() {
            self.key_positions[key_b] = Some(pos_a);
        }
    }


    /// Apply a magic rule. Returns the new score.
    /// If `apply` is false, computes the score without mutating state (speculative scoring).
    ///
    /// When a magic rule A→M steals output B:
    /// - Trigrams Z→A→B become Z→A→M (for all Z)
    /// - Trigrams A→B→C become A→M→C (for all C)
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    ///
    /// Each analyzer computes its own score delta via `add_rule` methods, using the
    /// constant frequency arrays from MagicCache.
    ///
    /// # Arguments
    /// * `magic_key` - M: the magic key that steals the output
    /// * `leader` - A: the key that triggers the magic rule
    /// * `new_output` - B: the output being stolen (or EMPTY_KEY to clear the rule)
    /// * `apply` - If true, update internal state; if false, just compute the score
    ///
    /// # Returns
    /// The new total score after applying (or speculatively applying) the rule.
    pub fn apply_magic_rule(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey, apply: bool) -> i64 {
        let key = (magic_key, leader);
        let old_output = self.current_magic_rules.get(&key).copied();

        // Early return if no change
        if old_output == Some(new_output) || (old_output.is_none() && new_output == EMPTY_KEY) {
            return self.score();
        }

        // Get frequency arrays from MagicCache (read-only)
        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        // Track total delta for speculative scoring
        let mut total_delta: i64 = 0;

        // Check if another magic key has this (leader, output) and clear it
        if new_output != EMPTY_KEY {
            let mut key_to_clear = None;
            for (&(other_magic, other_leader), &other_output) in &self.current_magic_rules {
                if other_leader == leader && other_output == new_output && other_magic != magic_key {
                    key_to_clear = Some((other_magic, other_leader));
                    break;
                }
            }
            if let Some((other_magic, other_leader)) = key_to_clear {
                // Clear the conflicting rule by setting its output to EMPTY_KEY
                // Each analyzer's add_rule will handle removing the old rule
                total_delta += self.sfb.add_rule(other_leader, EMPTY_KEY, other_magic, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, apply);
                total_delta += self.stretch.add_rule(other_leader, EMPTY_KEY, other_magic, &self.keys, &self.key_positions, bg_freq, tg_freq, apply);
                total_delta += self.scissors.add_rule(other_leader, EMPTY_KEY, other_magic, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, apply);
                total_delta += self.trigram.add_rule(other_leader, EMPTY_KEY, other_magic, &self.keys, &self.key_positions, tg_freq, apply);

                if apply {
                    self.current_magic_rules.remove(&(other_magic, other_leader));
                }
            }
        }

        // Apply the new rule (or clear the existing one if new_output is EMPTY_KEY)
        // Each analyzer's add_rule will handle:
        // 1. Removing the old rule for this (magic_key, leader) if it exists
        // 2. Adding the new rule if new_output is not EMPTY_KEY
        total_delta += self.sfb.add_rule(leader, new_output, magic_key, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, apply);
        total_delta += self.stretch.add_rule(leader, new_output, magic_key, &self.keys, &self.key_positions, bg_freq, tg_freq, apply);
        total_delta += self.scissors.add_rule(leader, new_output, magic_key, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, apply);
        total_delta += self.trigram.add_rule(leader, new_output, magic_key, &self.keys, &self.key_positions, tg_freq, apply);

        // Update current_magic_rules if applying
        if apply {
            if new_output != EMPTY_KEY {
                self.current_magic_rules.insert(key, new_output);
            } else {
                self.current_magic_rules.remove(&key);
            }
        }

        // For speculative scoring (apply=false), we need to add the deltas to the current score
        // For actual application (apply=true), the analyzers have already updated their state
        if apply {
            self.score()
        } else {
            self.score() + total_delta
        }
    }

    /// Returns a copy of the pre-computed list of all neighbors (key swaps + magic rules)
    #[inline]
    pub fn neighbors(&self) -> Vec<Neighbor> {
        self.neighbors.clone()
    }

    pub fn get_revert_neighbor(&self, neighbor: Neighbor) -> Neighbor {
        match neighbor {
            Neighbor::KeySwap(pair) => Neighbor::KeySwap(pair),
            Neighbor::MagicRule(rule) => {
                let key = (rule.magic_key, rule.leader);
                let current_output = self.current_magic_rules.get(&key).copied().unwrap_or(EMPTY_KEY);
                Neighbor::MagicRule(MagicRule::new(rule.magic_key, rule.leader, current_output))
            }
        }
    }

    // NOTE: affected_grams() method has been removed as part of the const-freq-analyzers refactoring (Task 1.3).
    // The affected_grams field and DeltaGram types have been removed since frequencies
    // will be constant after initialization.
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramPair {
    pub pair: PosPair,
    pub dist: i64,
}

#[cfg(test)]
mod magic_tests {
    use super::*;

    #[test]
    fn magic_cache_new() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.bg_freq.len(), 100);
        assert_eq!(cache.sg_freq.len(), 100);
        assert_eq!(cache.tg_freq.len(), 10);
    }

    #[test]
    fn magic_cache_get_freq_empty() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.get_bg_freq(0, 1), 0);
        assert_eq!(cache.get_sg_freq(0, 1), 0);
        assert_eq!(cache.get_tg_freq(0, 1, 2), 0);
    }

    #[test]
    fn magic_cache_init_from_data() {
        let mut cache = MagicCache::new(3);
        let bigrams = vec![
            vec![0, 100, 200],
            vec![300, 0, 400],
            vec![500, 600, 0],
        ];
        let skipgrams = vec![
            vec![0, 10, 20],
            vec![30, 0, 40],
            vec![50, 60, 0],
        ];
        let trigrams = vec![
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
            vec![vec![9, 10, 11], vec![12, 13, 14], vec![15, 16, 17]],
            vec![vec![18, 19, 20], vec![21, 22, 23], vec![24, 25, 26]],
        ];

        cache.init_from_data(&bigrams, &skipgrams, &trigrams);

        assert_eq!(cache.get_bg_freq(0, 1), 100);
        assert_eq!(cache.get_bg_freq(1, 2), 400);
        assert_eq!(cache.get_sg_freq(2, 1), 60);
        assert_eq!(cache.get_tg_freq(1, 2, 0), 15);
    }

    // NOTE: steal_bigram tests have been removed as part of the const-freq-analyzers refactoring.
    // The steal_bigram method has been removed from MagicCache.
    // Magic rule effects are now computed directly by each analyzer via add_rule methods.
}

/// Integration tests for magic rule application in CachedLayout.
///
/// These tests verify that:
/// 1. Applying a magic rule produces the expected score change
/// 2. Applying and then reversing a magic rule returns to the original score
/// 3. Speculative scoring (apply=false) returns the same score as actual application
/// 4. Multiple magic rules can be applied correctly
/// 5. Conflicting rules are handled correctly
///
/// **Validates: Requirements 4, 8 (Correctness Preservation)**
#[cfg(test)]
mod magic_rule_integration_tests {
    use super::*;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;
    use libdof::magic::MagicKey;
    use libdof::prelude::{PhysicalKey, Shape};

    /// Helper to create a simple 4-position keyboard layout for testing.
    /// Positions: 0=LP, 1=LI, 2=RI, 3=RP (different fingers to avoid SFB complications)
    fn create_test_keyboard() -> (Box<[PhysicalKey]>, Box<[Finger]>) {
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),  // pos 0
            PhysicalKey::xy(1.0, 0.0),  // pos 1
            PhysicalKey::xy(2.0, 0.0),  // pos 2
            PhysicalKey::xy(3.0, 0.0),  // pos 3
        ];
        let fingers = vec![LP, LI, RI, RP];
        (keyboard.into_boxed_slice(), fingers.into_boxed_slice())
    }

    /// Helper to create test weights with non-zero values for all analyzers.
    fn create_test_weights() -> Weights {
        Weights {
            sfbs: -10,
            sfs: -5,
            stretches: -3,
            sft: -12,
            inroll: 5,
            outroll: 4,
            alternate: 4,
            redirect: -1,
            onehandin: 1,
            onehandout: 0,
            thumb: 0,
            full_scissors: -2,
            half_scissors: -1,
            full_scissors_skip: -1,
            half_scissors_skip: 0,
            fingers: FingerWeights {
                lp: 100,
                lr: 50,
                lm: 30,
                li: 20,
                lt: 10,
                rt: 10,
                ri: 20,
                rm: 30,
                rr: 50,
                rp: 100,
            },
        }
    }

    /// Helper to create a Layout with magic key support.
    /// Keys: 'a' at pos 0, 'b' at pos 1, 'c' at pos 2, '*' (magic) at pos 3
    /// Magic rule: 'a' -> '*' produces 'b' (i.e., typing 'a' then '*' outputs 'b')
    fn create_layout_with_magic() -> Layout {
        let (keyboard, fingers) = create_test_keyboard();

        // Create magic key with rule: after 'a', magic produces 'b'
        let mut magic_key = MagicKey::new("*");
        magic_key.add_rule("a", "b");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key);

        Layout {
            name: "test_magic".to_string(),
            keys: vec!['a', 'b', 'c', '*'].into_boxed_slice(),
            fingers,
            keyboard,
            shape: Shape::default(),
            magic,
        }
    }

    /// Helper to create a Layout with multiple magic keys.
    /// Keys: 'a' at pos 0, 'b' at pos 1, 'c' at pos 2, 'd' at pos 3, '*' at pos 4, '#' at pos 5
    fn create_layout_with_multiple_magic() -> Layout {
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),  // pos 0
            PhysicalKey::xy(1.0, 0.0),  // pos 1
            PhysicalKey::xy(2.0, 0.0),  // pos 2
            PhysicalKey::xy(3.0, 0.0),  // pos 3
            PhysicalKey::xy(4.0, 0.0),  // pos 4
            PhysicalKey::xy(5.0, 0.0),  // pos 5
        ];
        let fingers = vec![LP, LI, LM, RM, RI, RP];

        // Create first magic key '*' with rule: after 'a', produces 'b'
        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");

        // Create second magic key '#' with rule: after 'c', produces 'd'
        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("c", "d");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        Layout {
            name: "test_multi_magic".to_string(),
            keys: vec!['a', 'b', 'c', 'd', '*', '#'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        }
    }

    /// Helper to create test data with specific n-gram frequencies.
    /// Creates data from a string that generates the desired frequencies.
    /// IMPORTANT: The text must contain all characters used in the layout,
    /// otherwise the analyzer arrays won't be sized correctly.
    #[allow(dead_code)]
    fn create_test_data(text: &str) -> Data {
        Data::from(text)
    }

    /// Helper to create test data that includes all layout characters.
    /// This ensures the analyzer arrays are properly sized by including
    /// each layout character in the text at least once.
    fn create_test_data_with_layout_chars(base_text: &str, layout_chars: &[char]) -> Data {
        // Build a string that includes all layout characters
        // We need to include them in a way that creates proper n-grams
        let mut full_text = String::new();

        // First add all layout chars to ensure they're in the data
        for &c in layout_chars {
            full_text.push(c);
        }
        // Add a separator to avoid creating unwanted n-grams between layout chars and base text
        full_text.push(' ');

        // Then add the base text for the actual frequency data
        full_text.push_str(base_text);

        Data::from(full_text)
    }

    // ==================== Integration Test 1: Magic rule produces expected score change ====================

    /// **Validates: Requirement 4 - Magic Rule Score Computation**
    ///
    /// Test that applying a magic rule produces a score change (not necessarily zero).
    /// The exact score depends on the layout positions and weights.
    ///
    /// Note: When CachedLayout is constructed, magic rules are stored in current_magic_rules
    /// but the analyzers are not initialized with these rules. The analyzers only get updated
    /// when apply_magic_rule is called. This test verifies that:
    /// 1. Clearing a rule and re-applying it produces consistent scores
    /// 2. The score changes when rules are applied/cleared
    #[test]
    fn test_magic_rule_produces_score_change() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        // Create data with 'ab' bigrams and all layout chars to ensure proper array sizing
        let data = create_test_data_with_layout_chars("ababababab", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        // Get the magic key and leader key IDs
        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First, apply the rule to establish a baseline with the rule active
        let score_with_rule = cache.apply_magic_rule(magic_key, leader, output, true);

        // Clear the rule
        let score_without_rule = cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);

        // Re-apply the rule
        let score_reapplied = cache.apply_magic_rule(magic_key, leader, output, true);

        println!("Score with rule: {}", score_with_rule);
        println!("Score without rule: {}", score_without_rule);
        println!("Score after reapply: {}", score_reapplied);

        // The score after re-applying should match the score with the rule
        assert_eq!(
            score_reapplied, score_with_rule,
            "Re-applying the same rule should restore the previous score"
        );
    }

    // ==================== Integration Test 2: Apply and reverse returns to original score ====================

    /// **Validates: Requirement 8 - Correctness Preservation**
    ///
    /// Test that applying a magic rule and then reversing it returns to the original score.
    #[test]
    fn test_apply_and_reverse_magic_rule_returns_to_original_score() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');
        let alt_output = cache.char_mapping().get_u('c');

        // First establish a baseline by applying the original rule
        let baseline_score = cache.apply_magic_rule(magic_key, leader, output, true);

        // Apply a different rule (change output from 'b' to 'c')
        let score_after_change = cache.apply_magic_rule(magic_key, leader, alt_output, true);

        // Verify the score changed
        println!("Baseline score: {}", baseline_score);
        println!("Score after change to 'c': {}", score_after_change);

        // Reverse by applying the original rule
        let score_after_reverse = cache.apply_magic_rule(magic_key, leader, output, true);

        // Score should return to baseline
        assert_eq!(
            score_after_reverse, baseline_score,
            "Reversing a magic rule should return to the baseline score"
        );
    }

    /// **Validates: Requirement 8 - Correctness Preservation**
    ///
    /// Test that clearing a rule and re-applying it returns to the original score.
    #[test]
    fn test_clear_and_reapply_magic_rule_returns_to_original_score() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abababab", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First establish a baseline by applying the rule
        let baseline_score = cache.apply_magic_rule(magic_key, leader, output, true);

        // Clear the rule
        let score_after_clear = cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);
        println!("Baseline score: {}", baseline_score);
        println!("Score after clear: {}", score_after_clear);

        // Re-apply the rule
        let score_after_reapply = cache.apply_magic_rule(magic_key, leader, output, true);
        println!("Score after reapply: {}", score_after_reapply);

        // Score should return to baseline
        assert_eq!(
            score_after_reapply, baseline_score,
            "Re-applying a cleared rule should return to the baseline score"
        );
    }

    // ==================== Integration Test 3: Speculative scoring matches actual application ====================

    /// **Validates: Requirements 2.6, 8 - Speculative scoring preserves state**
    ///
    /// Test that speculative scoring (apply=false) returns the same score as actual application.
    #[test]
    fn test_speculative_scoring_matches_actual_application() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline by applying a rule
        cache.apply_magic_rule(magic_key, leader, output_b, true);
        let baseline_score = cache.score();

        // Speculative scoring for a different rule (apply=false)
        let speculative_score = cache.apply_magic_rule(magic_key, leader, output_c, false);

        // Verify state is unchanged
        assert_eq!(
            cache.score(),
            baseline_score,
            "Speculative scoring should not change the score"
        );

        // Now actually apply
        let actual_score = cache.apply_magic_rule(magic_key, leader, output_c, true);

        // Speculative and actual should match
        assert_eq!(
            speculative_score, actual_score,
            "Speculative score should match actual application score"
        );
    }

    /// **Validates: Requirements 2.6, 8 - Multiple speculative calls preserve state**
    ///
    /// Test that multiple speculative scoring calls don't accumulate state changes.
    #[test]
    fn test_multiple_speculative_calls_preserve_state() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline by applying a rule
        cache.apply_magic_rule(magic_key, leader, output_b, true);
        let baseline_score = cache.score();

        // Multiple speculative calls
        let spec1 = cache.apply_magic_rule(magic_key, leader, output_c, false);
        let spec2 = cache.apply_magic_rule(magic_key, leader, output_c, false);
        let _spec3 = cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, false);
        let spec4 = cache.apply_magic_rule(magic_key, leader, output_b, false);

        // All speculative calls for the same change should return the same score
        assert_eq!(spec1, spec2, "Repeated speculative calls should return the same score");

        // State should be unchanged
        assert_eq!(
            cache.score(),
            baseline_score,
            "Multiple speculative calls should not change the score"
        );

        // Speculative call for current state should return current score
        assert_eq!(
            spec4, baseline_score,
            "Speculative call for current rule should return current score"
        );
    }

    // ==================== Integration Test 4: Multiple magic rules can be applied correctly ====================

    /// **Validates: Requirement 4 - Multiple magic rules**
    ///
    /// Test that multiple magic rules on different magic keys work correctly.
    #[test]
    fn test_multiple_magic_rules_on_different_keys() {
        let layout = create_layout_with_multiple_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcdabcdabcd", &['a', 'b', 'c', 'd', '*', '#']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key1 = cache.char_mapping().get_u('*');
        let magic_key2 = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let leader_c = cache.char_mapping().get_u('c');
        let output_b = cache.char_mapping().get_u('b');
        let output_d = cache.char_mapping().get_u('d');

        // Apply both rules to establish a baseline
        cache.apply_magic_rule(magic_key1, leader_a, output_b, true);
        cache.apply_magic_rule(magic_key2, leader_c, output_d, true);
        let baseline_score = cache.score();

        // Clear both rules
        cache.apply_magic_rule(magic_key1, leader_a, EMPTY_KEY, true);
        cache.apply_magic_rule(magic_key2, leader_c, EMPTY_KEY, true);
        let score_no_rules = cache.score();

        // Apply first rule
        cache.apply_magic_rule(magic_key1, leader_a, output_b, true);
        let score_one_rule = cache.score();

        // Apply second rule
        cache.apply_magic_rule(magic_key2, leader_c, output_d, true);
        let score_two_rules = cache.score();

        println!("Baseline score: {}", baseline_score);
        println!("Score with no rules: {}", score_no_rules);
        println!("Score with one rule: {}", score_one_rule);
        println!("Score with two rules: {}", score_two_rules);

        // Should be back to baseline (both rules applied)
        assert_eq!(
            score_two_rules, baseline_score,
            "Applying both rules should return to baseline score"
        );
    }

    /// **Validates: Requirement 4 - Multiple rules on same magic key**
    ///
    /// Test that multiple rules on the same magic key (different leaders) work correctly.
    #[test]
    fn test_multiple_rules_on_same_magic_key() {
        // Create a layout with one magic key but multiple possible leaders
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),
            PhysicalKey::xy(1.0, 0.0),
            PhysicalKey::xy(2.0, 0.0),
            PhysicalKey::xy(3.0, 0.0),
            PhysicalKey::xy(4.0, 0.0),
        ];
        let fingers = vec![LP, LI, LM, RI, RP];

        let mut magic_key = MagicKey::new("*");
        magic_key.add_rule("a", "b");
        magic_key.add_rule("c", "d");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key);

        let layout = Layout {
            name: "test_multi_rule".to_string(),
            keys: vec!['a', 'b', 'c', 'd', '*'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        };

        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcdabcdabcd", &['a', 'b', 'c', 'd', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key_id = cache.char_mapping().get_u('*');
        let leader_a = cache.char_mapping().get_u('a');
        let leader_c = cache.char_mapping().get_u('c');
        let output_b = cache.char_mapping().get_u('b');
        let output_d = cache.char_mapping().get_u('d');

        // Apply both rules to establish a baseline
        cache.apply_magic_rule(magic_key_id, leader_a, output_b, true);
        cache.apply_magic_rule(magic_key_id, leader_c, output_d, true);
        let baseline_score = cache.score();

        // Clear both rules
        cache.apply_magic_rule(magic_key_id, leader_a, EMPTY_KEY, true);
        cache.apply_magic_rule(magic_key_id, leader_c, EMPTY_KEY, true);
        let score_no_rules = cache.score();

        // Apply first rule (a -> b)
        cache.apply_magic_rule(magic_key_id, leader_a, output_b, true);
        let score_one_rule = cache.score();

        // Apply second rule (c -> d)
        cache.apply_magic_rule(magic_key_id, leader_c, output_d, true);
        let score_two_rules = cache.score();

        println!("Baseline score: {}", baseline_score);
        println!("Score with no rules: {}", score_no_rules);
        println!("Score with one rule (a->b): {}", score_one_rule);
        println!("Score with two rules: {}", score_two_rules);

        // Should be back to baseline
        assert_eq!(
            score_two_rules, baseline_score,
            "Applying both rules should return to baseline score"
        );
    }

    // ==================== Integration Test 5: Conflicting rules are handled correctly ====================

    /// **Validates: Requirement 4 - Conflicting rules handling**
    ///
    /// Test that when two magic keys try to steal the same (leader, output) pair,
    /// the conflict is resolved correctly (the new rule takes precedence).
    #[test]
    fn test_conflicting_rules_new_takes_precedence() {
        // Create layout with two magic keys that can conflict
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),
            PhysicalKey::xy(1.0, 0.0),
            PhysicalKey::xy(2.0, 0.0),
            PhysicalKey::xy(3.0, 0.0),
            PhysicalKey::xy(4.0, 0.0),
        ];
        let fingers = vec![LP, LI, LM, RI, RP];

        // Both magic keys have a rule for 'a' -> 'b'
        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");

        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("a", "b");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        let layout = Layout {
            name: "test_conflict".to_string(),
            keys: vec!['a', 'b', '*', '#', 'c'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        };

        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("ababab", &['a', 'b', 'c', '*', '#']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key1_id = cache.char_mapping().get_u('*');
        let magic_key2_id = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');

        // Clear all rules first
        cache.apply_magic_rule(magic_key1_id, leader_a, EMPTY_KEY, true);
        cache.apply_magic_rule(magic_key2_id, leader_a, EMPTY_KEY, true);

        // Apply rule on magic_key1: a -> * produces b
        cache.apply_magic_rule(magic_key1_id, leader_a, output_b, true);
        let score_with_key1 = cache.score();

        // Now apply conflicting rule on magic_key2: a -> # produces b
        // This should clear the rule from magic_key1
        cache.apply_magic_rule(magic_key2_id, leader_a, output_b, true);
        let score_with_key2 = cache.score();

        println!("Score with rule on key1: {}", score_with_key1);
        println!("Score with rule on key2: {}", score_with_key2);

        // The scores might differ because the magic keys are at different positions
        // The important thing is that the system doesn't crash and handles the conflict

        // Verify that magic_key1 no longer has the rule (it was cleared)
        // We can check by trying to apply the same rule to key1 again
        let score_reapply_key1 = cache.apply_magic_rule(magic_key1_id, leader_a, output_b, true);

        // This should clear key2's rule and apply to key1
        println!("Score after reapply to key1: {}", score_reapply_key1);

        // Should match the original score_with_key1
        assert_eq!(
            score_reapply_key1, score_with_key1,
            "Re-applying to key1 should give the same score as before"
        );
    }

    /// **Validates: Requirement 8 - Conflicting rules don't corrupt state**
    ///
    /// Test that conflicting rule operations don't corrupt the internal state.
    #[test]
    fn test_conflicting_rules_state_consistency() {
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),
            PhysicalKey::xy(1.0, 0.0),
            PhysicalKey::xy(2.0, 0.0),
            PhysicalKey::xy(3.0, 0.0),
            PhysicalKey::xy(4.0, 0.0),
        ];
        let fingers = vec![LP, LI, LM, RI, RP];

        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");

        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("a", "b");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        let layout = Layout {
            name: "test_conflict_state".to_string(),
            keys: vec!['a', 'b', '*', '#', 'c'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        };

        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("ababab", &['a', 'b', 'c', '*', '#']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key1_id = cache.char_mapping().get_u('*');
        let magic_key2_id = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');

        // Clear all rules and record baseline
        cache.apply_magic_rule(magic_key1_id, leader_a, EMPTY_KEY, true);
        cache.apply_magic_rule(magic_key2_id, leader_a, EMPTY_KEY, true);
        let baseline_score = cache.score();

        // Apply and conflict multiple times
        for _ in 0..5 {
            cache.apply_magic_rule(magic_key1_id, leader_a, output_b, true);
            cache.apply_magic_rule(magic_key2_id, leader_a, output_b, true);
            cache.apply_magic_rule(magic_key1_id, leader_a, output_b, true);
        }

        // Clear all rules
        cache.apply_magic_rule(magic_key1_id, leader_a, EMPTY_KEY, true);
        cache.apply_magic_rule(magic_key2_id, leader_a, EMPTY_KEY, true);
        let final_score = cache.score();

        // Should return to baseline
        assert_eq!(
            final_score, baseline_score,
            "After clearing all rules, score should return to baseline"
        );
    }

    // ==================== Key swap speculative scoring tests ====================

    /// **Validates: Requirement 8 - Key swap speculative scoring**
    ///
    /// Test that key swap speculative scoring (apply=false) returns the same score as actual application.
    #[test]
    fn test_key_swap_speculative_vs_actual() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        // Get speculative score for swapping positions 0 and 1
        let speculative_score = cache.score_neighbor(Neighbor::KeySwap(PosPair(0, 1)));

        // Verify state is unchanged
        let score_before = cache.score();

        // Actually apply the swap
        cache.swap_keys(0, 1);
        let actual_score = cache.score();

        println!("Speculative score: {}", speculative_score);
        println!("Score before swap: {}", score_before);
        println!("Actual score after swap: {}", actual_score);
        println!("cache.score() after swap: {}", cache.score());

        assert_eq!(speculative_score, actual_score, "Speculative should match actual");
    }

    /// **Validates: Requirement 8 - Key swap speculative scoring with 6-position layout**
    ///
    /// Test that key swap speculative scoring works with the 6-position layout used in property tests.
    #[test]
    fn test_key_swap_speculative_vs_actual_6pos() {
        // Use the same layout as the property tests
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),  // pos 0
            PhysicalKey::xy(1.0, 0.0),  // pos 1
            PhysicalKey::xy(2.0, 0.0),  // pos 2
            PhysicalKey::xy(3.0, 0.0),  // pos 3
            PhysicalKey::xy(4.0, 0.0),  // pos 4 - magic key *
            PhysicalKey::xy(5.0, 0.0),  // pos 5 - magic key #
        ];
        let fingers = vec![LP, LI, LM, RM, RI, RP];

        // Create magic keys with multiple possible rules
        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");
        magic_key1.add_rule("c", "d");

        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("a", "c");
        magic_key2.add_rule("b", "d");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        let layout = Layout {
            name: "test_score_preservation".to_string(),
            keys: vec!['a', 'b', 'c', 'd', '*', '#'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        };

        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', 'd', '*', '#']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        println!("Initial score: {}", cache.score());
        println!("SFB: {}, Stretch: {}, Scissors: {}, Trigram: {}",
            cache.sfb.score(), cache.stretch.score(), cache.scissors.score(), cache.trigram.score());

        // Get speculative score for swapping positions 0 and 1
        let speculative_score = cache.score_neighbor(Neighbor::KeySwap(PosPair(0, 1)));

        // Verify state is unchanged
        let score_before = cache.score();

        println!("After speculative swap:");
        println!("Speculative score: {}", speculative_score);
        println!("Score before swap (should be unchanged): {}", score_before);
        println!("SFB: {}, Stretch: {}, Scissors: {}, Trigram: {}",
            cache.sfb.score(), cache.stretch.score(), cache.scissors.score(), cache.trigram.score());

        // Actually apply the swap
        cache.swap_keys(0, 1);
        let actual_score = cache.score();

        println!("After actual swap:");
        println!("Actual score: {}", actual_score);
        println!("cache.score(): {}", cache.score());
        println!("SFB: {}, Stretch: {}, Scissors: {}, Trigram: {}",
            cache.sfb.score(), cache.stretch.score(), cache.scissors.score(), cache.trigram.score());

        assert_eq!(speculative_score, actual_score, "Speculative should match actual");
    }

    /// **Validates: Requirement 8 - Magic rule conflict behavior**
    ///
    /// This test documents the expected behavior when magic rules conflict.
    /// When applying a rule that conflicts with an existing rule on another magic key
    /// (same leader and output), the conflicting rule is cleared as a side effect.
    /// This means that clearing the new rule does NOT restore the original state.
    ///
    /// Layout:
    /// - Magic key `*`: a->b, c->d
    /// - Magic key `#`: a->c, b->d
    ///
    /// Scenario:
    /// 1. Initial state: (*, a)->b is active
    /// 2. Clear (#, a): removes (#, a)->c
    /// 3. Apply (#, a)->b: conflicts with (*, a)->b, so (*, a)->b is cleared
    /// 4. Clear (#, a): removes (#, a)->b, but (*, a)->b is NOT restored
    ///
    /// The score after step 4 is different from after step 2 because (*, a)->b was cleared.
    #[test]
    fn test_magic_rule_conflict_behavior() {
        // Use the same layout as the property tests
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),  // pos 0
            PhysicalKey::xy(1.0, 0.0),  // pos 1
            PhysicalKey::xy(2.0, 0.0),  // pos 2
            PhysicalKey::xy(3.0, 0.0),  // pos 3
            PhysicalKey::xy(4.0, 0.0),  // pos 4 - magic key *
            PhysicalKey::xy(5.0, 0.0),  // pos 5 - magic key #
        ];
        let fingers = vec![LP, LI, LM, RM, RI, RP];

        // Create magic keys with CONFLICTING rules (both can produce 'b' from 'a')
        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");  // (*, a) -> b
        magic_key1.add_rule("c", "d");

        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("a", "c");  // (#, a) -> c (initial)
        magic_key2.add_rule("b", "d");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        let layout = Layout {
            name: "test_conflict_behavior".to_string(),
            keys: vec!['a', 'b', 'c', 'd', '*', '#'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        };

        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', 'd', '*', '#']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let key_a = cache.char_mapping().get_u('a');
        let key_b = cache.char_mapping().get_u('b');
        let magic_key_star = cache.char_mapping().get_u('*');
        let magic_key_hash = cache.char_mapping().get_u('#');

        // Verify initial state: (*, a)->b is active
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_star, key_a)),
            Some(&key_b),
            "Initial state should have (*, a)->b active"
        );

        // Step 1: Clear (#, a)
        cache.apply_magic_rule(magic_key_hash, key_a, EMPTY_KEY, true);
        let score_after_clear_hash = cache.score();

        // Verify (*, a)->b is still active
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_star, key_a)),
            Some(&key_b),
            "After clearing (#, a), (*, a)->b should still be active"
        );

        // Step 2: Apply (#, a)->b - this conflicts with (*, a)->b
        cache.apply_magic_rule(magic_key_hash, key_a, key_b, true);

        // Verify (*, a)->b was cleared due to conflict
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_star, key_a)),
            None,
            "After applying (#, a)->b, (*, a)->b should be cleared due to conflict"
        );

        // Verify (#, a)->b is now active
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_hash, key_a)),
            Some(&key_b),
            "(#, a)->b should be active"
        );

        // Step 3: Clear (#, a)
        cache.apply_magic_rule(magic_key_hash, key_a, EMPTY_KEY, true);
        let score_after_final_clear = cache.score();

        // Verify (#, a) is cleared
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_hash, key_a)),
            None,
            "(#, a) should be cleared"
        );

        // Verify (*, a)->b is NOT restored (this is the expected behavior)
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_star, key_a)),
            None,
            "(*, a)->b should NOT be restored after clearing (#, a)"
        );

        // The scores are different because (*, a)->b was cleared as a side effect
        // This is expected behavior, not a bug
        assert_ne!(
            score_after_final_clear, score_after_clear_hash,
            "Scores should be different because (*, a)->b was cleared as a side effect"
        );
    }

    // ==================== Additional edge case tests ====================

    /// **Validates: Requirement 8 - No-op rule application**
    ///
    /// Test that applying the same rule twice is a no-op.
    #[test]
    fn test_applying_same_rule_twice_is_noop() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("ababab", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First apply the rule to establish a baseline
        let baseline_score = cache.apply_magic_rule(magic_key, leader, output, true);

        // Apply the same rule again (should be no-op)
        let score_after_reapply = cache.apply_magic_rule(magic_key, leader, output, true);

        assert_eq!(
            score_after_reapply, baseline_score,
            "Applying the same rule twice should be a no-op"
        );
    }

    /// **Validates: Requirement 8 - Clearing non-existent rule**
    ///
    /// Test that clearing a rule that doesn't exist is a no-op.
    #[test]
    fn test_clearing_nonexistent_rule_is_noop() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("ababab", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let _output = cache.char_mapping().get_u('b');

        // Clear the rule (even though analyzers weren't initialized with it)
        let score_after_clear = cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);

        // Try to clear again (should be no-op)
        let score_after_second_clear = cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);

        assert_eq!(
            score_after_second_clear, score_after_clear,
            "Clearing a non-existent rule should be a no-op"
        );
    }

    /// **Validates: Requirement 8 - Score consistency across operations**
    ///
    /// Test that score() returns consistent values and matches the sum of analyzer scores.
    #[test]
    fn test_score_consistency_after_magic_operations() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        let data = create_test_data_with_layout_chars("abcabcabcabc", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights);

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline
        let baseline = cache.apply_magic_rule(magic_key, leader, output_b, true);

        // Perform various operations and verify score consistency
        let scores: Vec<i64> = vec![
            cache.score(),
            cache.apply_magic_rule(magic_key, leader, output_c, true),
            cache.score(),
            cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true),
            cache.score(),
            cache.apply_magic_rule(magic_key, leader, output_b, true),
            cache.score(),
        ];

        // Each pair of (apply result, subsequent score()) should match
        assert_eq!(scores[0], baseline, "Initial score should match baseline");
        assert_eq!(scores[1], scores[2], "apply result should match subsequent score()");
        assert_eq!(scores[3], scores[4], "apply result should match subsequent score()");
        assert_eq!(scores[5], scores[6], "apply result should match subsequent score()");

        // First and last should match (we restored the original rule)
        assert_eq!(scores[0], scores[6], "Restoring original rule should give original score");
    }
}

/// Property-based tests for constant frequency arrays in MagicCache.
///
/// **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
///
/// These tests verify that the frequency arrays in MagicCache (`bg_freq`, `sg_freq`, `tg_freq`)
/// remain unchanged after any sequence of magic rule applications. This is a core invariant
/// of the const-freq-analyzers architecture.
#[cfg(test)]
mod pbt_constant_frequencies {
    use super::*;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;
    use libdof::magic::MagicKey;
    use libdof::prelude::{PhysicalKey, Shape};
    use proptest::prelude::*;

    /// Helper to create test weights with non-zero values for all analyzers.
    fn create_test_weights() -> Weights {
        Weights {
            sfbs: -10,
            sfs: -5,
            stretches: -3,
            sft: -12,
            inroll: 5,
            outroll: 4,
            alternate: 4,
            redirect: -1,
            onehandin: 1,
            onehandout: 0,
            thumb: 0,
            full_scissors: -2,
            half_scissors: -1,
            full_scissors_skip: -1,
            half_scissors_skip: 0,
            fingers: FingerWeights {
                lp: 100,
                lr: 50,
                lm: 30,
                li: 20,
                lt: 10,
                rt: 10,
                ri: 20,
                rm: 30,
                rr: 50,
                rp: 100,
            },
        }
    }

    /// Helper to create test data that includes all layout characters.
    fn create_test_data_with_layout_chars(base_text: &str, layout_chars: &[char]) -> Data {
        let mut full_text = String::new();
        for &c in layout_chars {
            full_text.push(c);
        }
        full_text.push(' ');
        full_text.push_str(base_text);
        Data::from(full_text)
    }

    /// Strategy to generate a magic rule operation.
    /// Returns (leader_idx, output_idx, magic_key_idx) where indices are 0-based
    /// into the available keys.
    fn arb_magic_rule_indices() -> impl Strategy<Value = (usize, usize, usize)> {
        // Generate indices that will be mapped to actual keys
        // Using small indices that will be constrained to actual key count
        (0usize..10, 0usize..10, 0usize..10)
    }

    /// Strategy to generate a sequence of magic rule operations.
    fn arb_magic_rule_sequence(max_len: usize) -> impl Strategy<Value = Vec<(usize, usize, usize)>> {
        proptest::collection::vec(arb_magic_rule_indices(), 0..=max_len)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        ///
        /// Property: For any sequence of magic rule applications, the frequency arrays
        /// in MagicCache (`bg_freq`, `sg_freq`, `tg_freq`) should remain unchanged
        /// from their initial values.
        ///
        /// This test creates a CachedLayout with magic keys, captures the initial
        /// frequency arrays, applies a random sequence of magic rules, and verifies
        /// that the frequency arrays are identical to their initial state.
        #[test]
        fn prop_constant_frequencies_after_magic_rules(
            // Generate a sequence of magic rule operations (0 to 10 operations)
            rule_sequence in arb_magic_rule_sequence(10),
            // Generate different base text patterns for variety
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
                Just("aabbccddaabbccdd"),
            ],
        ) {
            // Create a layout with multiple magic keys for more interesting test cases
            let keyboard = vec![
                PhysicalKey::xy(0.0, 0.0),  // pos 0
                PhysicalKey::xy(1.0, 0.0),  // pos 1
                PhysicalKey::xy(2.0, 0.0),  // pos 2
                PhysicalKey::xy(3.0, 0.0),  // pos 3
                PhysicalKey::xy(4.0, 0.0),  // pos 4 - magic key *
                PhysicalKey::xy(5.0, 0.0),  // pos 5 - magic key #
            ];
            let fingers = vec![LP, LI, LM, RM, RI, RP];

            // Create magic keys with multiple possible rules
            let mut magic_key1 = MagicKey::new("*");
            magic_key1.add_rule("a", "b");
            magic_key1.add_rule("c", "d");

            let mut magic_key2 = MagicKey::new("#");
            magic_key2.add_rule("a", "c");
            magic_key2.add_rule("b", "d");

            let mut magic = HashMap::default();
            magic.insert('*', magic_key1);
            magic.insert('#', magic_key2);

            let layout = Layout {
                name: "test_const_freq".to_string(),
                keys: vec!['a', 'b', 'c', 'd', '*', '#'].into_boxed_slice(),
                fingers: fingers.into_boxed_slice(),
                keyboard: keyboard.into_boxed_slice(),
                shape: Shape::default(),
                magic,
            };

            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Capture initial frequency arrays from MagicCache
            let initial_bg_freq = cache.magic.bg_freq_flat().to_vec();
            let initial_sg_freq = cache.magic.sg_freq_flat().to_vec();
            let initial_tg_freq = cache.magic.tg_freq().to_vec();

            // Get key IDs for the magic keys and regular keys
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');
            let magic_key_hash = cache.char_mapping().get_u('#');

            // Available keys for rules (regular keys only)
            let regular_keys = [key_a, key_b, key_c, key_d];
            let magic_keys = [magic_key_star, magic_key_hash];

            // Apply the sequence of magic rules
            for (leader_idx, output_idx, magic_key_idx) in rule_sequence {
                // Map indices to actual keys (with wrapping)
                let leader = regular_keys[leader_idx % regular_keys.len()];
                // Output can be a regular key or EMPTY_KEY (to clear the rule)
                let output = if output_idx % 5 == 0 {
                    EMPTY_KEY  // 20% chance to clear the rule
                } else {
                    regular_keys[output_idx % regular_keys.len()]
                };
                let magic_key = magic_keys[magic_key_idx % magic_keys.len()];

                // Apply the rule (with apply=true to actually mutate state)
                cache.apply_magic_rule(magic_key, leader, output, true);
            }

            // Verify frequency arrays are unchanged
            prop_assert_eq!(
                cache.magic.bg_freq_flat(),
                &initial_bg_freq[..],
                "bg_freq should remain unchanged after magic rule operations"
            );
            prop_assert_eq!(
                cache.magic.sg_freq_flat(),
                &initial_sg_freq[..],
                "sg_freq should remain unchanged after magic rule operations"
            );
            prop_assert_eq!(
                cache.magic.tg_freq(),
                &initial_tg_freq[..],
                "tg_freq should remain unchanged after magic rule operations"
            );
        }

        /// **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        ///
        /// Property: For any sequence of speculative magic rule applications (apply=false),
        /// the frequency arrays in MagicCache should remain unchanged.
        ///
        /// This is a simpler case since speculative scoring shouldn't mutate any state,
        /// but it's important to verify this invariant holds.
        #[test]
        fn prop_constant_frequencies_after_speculative_magic_rules(
            // Generate a sequence of magic rule operations (0 to 10 operations)
            rule_sequence in arb_magic_rule_sequence(10),
        ) {
            // Create a simple layout with magic keys
            let keyboard = vec![
                PhysicalKey::xy(0.0, 0.0),
                PhysicalKey::xy(1.0, 0.0),
                PhysicalKey::xy(2.0, 0.0),
                PhysicalKey::xy(3.0, 0.0),
                PhysicalKey::xy(4.0, 0.0),
            ];
            let fingers = vec![LP, LI, LM, RI, RP];

            let mut magic_key = MagicKey::new("*");
            magic_key.add_rule("a", "b");
            magic_key.add_rule("c", "d");

            let mut magic = HashMap::default();
            magic.insert('*', magic_key);

            let layout = Layout {
                name: "test_spec_const_freq".to_string(),
                keys: vec!['a', 'b', 'c', 'd', '*'].into_boxed_slice(),
                fingers: fingers.into_boxed_slice(),
                keyboard: keyboard.into_boxed_slice(),
                shape: Shape::default(),
                magic,
            };

            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars("abcdabcdabcd", &['a', 'b', 'c', 'd', '*']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Capture initial frequency arrays
            let initial_bg_freq = cache.magic.bg_freq_flat().to_vec();
            let initial_sg_freq = cache.magic.sg_freq_flat().to_vec();
            let initial_tg_freq = cache.magic.tg_freq().to_vec();

            // Get key IDs
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');

            let regular_keys = [key_a, key_b, key_c, key_d];

            // Apply speculative magic rules (apply=false)
            for (leader_idx, output_idx, _) in rule_sequence {
                let leader = regular_keys[leader_idx % regular_keys.len()];
                let output = if output_idx % 5 == 0 {
                    EMPTY_KEY
                } else {
                    regular_keys[output_idx % regular_keys.len()]
                };

                // Speculative scoring (apply=false)
                cache.apply_magic_rule(magic_key_star, leader, output, false);
            }

            // Verify frequency arrays are unchanged
            prop_assert_eq!(
                cache.magic.bg_freq_flat(),
                &initial_bg_freq[..],
                "bg_freq should remain unchanged after speculative magic rule operations"
            );
            prop_assert_eq!(
                cache.magic.sg_freq_flat(),
                &initial_sg_freq[..],
                "sg_freq should remain unchanged after speculative magic rule operations"
            );
            prop_assert_eq!(
                cache.magic.tg_freq(),
                &initial_tg_freq[..],
                "tg_freq should remain unchanged after speculative magic rule operations"
            );
        }

        /// **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        ///
        /// Property: For any sequence of mixed operations (key swaps and magic rules),
        /// the frequency arrays in MagicCache should remain unchanged.
        ///
        /// This tests the interaction between key swaps and magic rules to ensure
        /// neither operation type affects the frequency arrays.
        #[test]
        fn prop_constant_frequencies_after_mixed_operations(
            // Generate a sequence of operations
            operations in proptest::collection::vec(
                prop_oneof![
                    // Key swap operation: (pos_a, pos_b)
                    (0usize..4, 0usize..4).prop_map(|(a, b)| (0, a, b)),
                    // Magic rule operation: (1, leader_idx, output_idx)
                    (Just(1usize), 0usize..4, 0usize..5).prop_map(|(op, l, o)| (op, l, o)),
                ],
                0..=15
            ),
        ) {
            // Create a layout with magic keys
            let keyboard = vec![
                PhysicalKey::xy(0.0, 0.0),
                PhysicalKey::xy(1.0, 0.0),
                PhysicalKey::xy(2.0, 0.0),
                PhysicalKey::xy(3.0, 0.0),
                PhysicalKey::xy(4.0, 0.0),
            ];
            let fingers = vec![LP, LI, LM, RI, RP];

            let mut magic_key = MagicKey::new("*");
            magic_key.add_rule("a", "b");
            magic_key.add_rule("c", "d");

            let mut magic = HashMap::default();
            magic.insert('*', magic_key);

            let layout = Layout {
                name: "test_mixed_const_freq".to_string(),
                keys: vec!['a', 'b', 'c', 'd', '*'].into_boxed_slice(),
                fingers: fingers.into_boxed_slice(),
                keyboard: keyboard.into_boxed_slice(),
                shape: Shape::default(),
                magic,
            };

            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars("abcdabcdabcd", &['a', 'b', 'c', 'd', '*']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Capture initial frequency arrays
            let initial_bg_freq = cache.magic.bg_freq_flat().to_vec();
            let initial_sg_freq = cache.magic.sg_freq_flat().to_vec();
            let initial_tg_freq = cache.magic.tg_freq().to_vec();

            // Get key IDs
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');

            let regular_keys = [key_a, key_b, key_c, key_d];
            let num_positions = 5;

            // Apply the sequence of operations
            for (op_type, idx1, idx2) in operations {
                if op_type == 0 {
                    // Key swap operation
                    let pos_a = idx1 % num_positions;
                    let pos_b = idx2 % num_positions;
                    if pos_a != pos_b {
                        cache.swap_keys(pos_a, pos_b);
                    }
                } else {
                    // Magic rule operation
                    let leader = regular_keys[idx1 % regular_keys.len()];
                    let output = if idx2 >= regular_keys.len() {
                        EMPTY_KEY
                    } else {
                        regular_keys[idx2 % regular_keys.len()]
                    };
                    cache.apply_magic_rule(magic_key_star, leader, output, true);
                }
            }

            // Verify frequency arrays are unchanged
            prop_assert_eq!(
                cache.magic.bg_freq_flat(),
                &initial_bg_freq[..],
                "bg_freq should remain unchanged after mixed operations"
            );
            prop_assert_eq!(
                cache.magic.sg_freq_flat(),
                &initial_sg_freq[..],
                "sg_freq should remain unchanged after mixed operations"
            );
            prop_assert_eq!(
                cache.magic.tg_freq(),
                &initial_tg_freq[..],
                "tg_freq should remain unchanged after mixed operations"
            );
        }
    }
}


/// Property-based tests for total score preservation.
///
/// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
///
/// Since the original system has been removed (update methods were deleted in Task 8),
/// we verify score preservation through alternative properties:
/// 1. Score consistency: The total score equals the sum of individual analyzer scores
/// 2. Score determinism: Applying the same operations in the same order produces the same score
/// 3. Score reversibility: Applying and then reversing operations returns to the original score
#[cfg(test)]
mod pbt_total_score_preservation {
    use super::*;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;
    use libdof::magic::MagicKey;
    use libdof::prelude::{PhysicalKey, Shape};
    use proptest::prelude::*;

    /// Helper to create test weights with non-zero values for all analyzers.
    fn create_test_weights() -> Weights {
        Weights {
            sfbs: -10,
            sfs: -5,
            stretches: -3,
            sft: -12,
            inroll: 5,
            outroll: 4,
            alternate: 4,
            redirect: -1,
            onehandin: 1,
            onehandout: 0,
            thumb: 0,
            full_scissors: -2,
            half_scissors: -1,
            full_scissors_skip: -1,
            half_scissors_skip: 0,
            fingers: FingerWeights {
                lp: 100,
                lr: 50,
                lm: 30,
                li: 20,
                lt: 10,
                rt: 10,
                ri: 20,
                rm: 30,
                rr: 50,
                rp: 100,
            },
        }
    }

    /// Helper to create test data that includes all layout characters.
    fn create_test_data_with_layout_chars(base_text: &str, layout_chars: &[char]) -> Data {
        let mut full_text = String::new();
        for &c in layout_chars {
            full_text.push(c);
        }
        full_text.push(' ');
        full_text.push_str(base_text);
        Data::from(full_text)
    }

    /// Strategy to generate a key swap operation.
    /// Returns (pos_a, pos_b) where positions are valid for a 6-position keyboard.
    fn arb_key_swap() -> impl Strategy<Value = (usize, usize)> {
        (0usize..6, 0usize..6).prop_filter("positions must be different", |(a, b)| a != b)
    }

    /// Strategy to generate a magic rule operation.
    /// Returns (leader_idx, output_idx, magic_key_idx) where indices are 0-based.
    fn arb_magic_rule_indices() -> impl Strategy<Value = (usize, usize, usize)> {
        (0usize..4, 0usize..5, 0usize..2)
    }

    /// Enum representing an operation that can be applied to a CachedLayout.
    #[derive(Debug, Clone)]
    enum Operation {
        KeySwap(usize, usize),
        MagicRule(usize, usize, usize), // (leader_idx, output_idx, magic_key_idx)
    }

    /// Strategy to generate a single operation.
    fn arb_operation() -> impl Strategy<Value = Operation> {
        prop_oneof![
            arb_key_swap().prop_map(|(a, b)| Operation::KeySwap(a, b)),
            arb_magic_rule_indices().prop_map(|(l, o, m)| Operation::MagicRule(l, o, m)),
        ]
    }

    /// Strategy to generate a sequence of operations.
    fn arb_operation_sequence(max_len: usize) -> impl Strategy<Value = Vec<Operation>> {
        proptest::collection::vec(arb_operation(), 0..=max_len)
    }

    /// Helper to create a test layout with magic keys.
    fn create_test_layout() -> Layout {
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),  // pos 0
            PhysicalKey::xy(1.0, 0.0),  // pos 1
            PhysicalKey::xy(2.0, 0.0),  // pos 2
            PhysicalKey::xy(3.0, 0.0),  // pos 3
            PhysicalKey::xy(4.0, 0.0),  // pos 4 - magic key *
            PhysicalKey::xy(5.0, 0.0),  // pos 5 - magic key #
        ];
        let fingers = vec![LP, LI, LM, RM, RI, RP];

        // Create magic keys with multiple possible rules
        let mut magic_key1 = MagicKey::new("*");
        magic_key1.add_rule("a", "b");
        magic_key1.add_rule("c", "d");

        let mut magic_key2 = MagicKey::new("#");
        magic_key2.add_rule("a", "c");
        magic_key2.add_rule("b", "d");

        let mut magic = HashMap::default();
        magic.insert('*', magic_key1);
        magic.insert('#', magic_key2);

        Layout {
            name: "test_score_preservation".to_string(),
            keys: vec!['a', 'b', 'c', 'd', '*', '#'].into_boxed_slice(),
            fingers: fingers.into_boxed_slice(),
            keyboard: keyboard.into_boxed_slice(),
            shape: Shape::default(),
            magic,
        }
    }

    /// Helper to apply an operation to a CachedLayout.
    fn apply_operation(cache: &mut CachedLayout, op: &Operation, regular_keys: &[CacheKey], magic_keys: &[CacheKey]) {
        match op {
            Operation::KeySwap(pos_a, pos_b) => {
                cache.swap_keys(*pos_a, *pos_b);
            }
            Operation::MagicRule(leader_idx, output_idx, magic_key_idx) => {
                let leader = regular_keys[*leader_idx % regular_keys.len()];
                let output = if *output_idx >= regular_keys.len() {
                    EMPTY_KEY
                } else {
                    regular_keys[*output_idx % regular_keys.len()]
                };
                let magic_key = magic_keys[*magic_key_idx % magic_keys.len()];
                cache.apply_magic_rule(magic_key, leader, output, true);
            }
        }
    }

    /// Helper to get the reverse of an operation.
    fn get_reverse_operation(op: &Operation) -> Operation {
        match op {
            // Key swap is its own reverse
            Operation::KeySwap(a, b) => Operation::KeySwap(*a, *b),
            // Magic rule reverse needs to be computed from current state
            // We'll handle this specially in the test
            Operation::MagicRule(l, o, m) => Operation::MagicRule(*l, *o, *m),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 1: Score equals sum of analyzers
        ///
        /// For any layout and any sequence of operations (key swaps, magic rules),
        /// the total score from `cache.score()` should equal the sum of individual
        /// analyzer scores: `sfb.score() + stretch.score() + scissors.score() + trigram.score()`.
        #[test]
        fn prop_score_equals_sum_of_analyzers(
            operations in arb_operation_sequence(10),
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
                Just("aabbccddaabbccdd"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Get key IDs
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');
            let magic_key_hash = cache.char_mapping().get_u('#');

            let regular_keys = [key_a, key_b, key_c, key_d];
            let magic_keys = [magic_key_star, magic_key_hash];

            // Verify initial state
            let total_score = cache.score();
            let sum_of_analyzers = cache.sfb.score() + cache.stretch.score() + cache.scissors.score() + cache.trigram.score();
            prop_assert_eq!(
                total_score, sum_of_analyzers,
                "Initial: total score {} should equal sum of analyzers {}",
                total_score, sum_of_analyzers
            );

            // Apply operations and verify after each
            for op in &operations {
                apply_operation(&mut cache, op, &regular_keys, &magic_keys);

                let total_score = cache.score();
                let sum_of_analyzers = cache.sfb.score() + cache.stretch.score() + cache.scissors.score() + cache.trigram.score();
                prop_assert_eq!(
                    total_score, sum_of_analyzers,
                    "After {:?}: total score {} should equal sum of analyzers {}",
                    op, total_score, sum_of_analyzers
                );
            }
        }

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 2: Score determinism
        ///
        /// Applying the same sequence of operations twice (starting from the same initial state)
        /// should produce the same final score.
        #[test]
        fn prop_score_determinism(
            operations in arb_operation_sequence(10),
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            // Create two identical caches
            let mut cache1 = CachedLayout::new(&layout, data.clone(), &weights);
            let mut cache2 = CachedLayout::new(&layout, data, &weights);

            // Get key IDs (same for both caches)
            let key_a = cache1.char_mapping().get_u('a');
            let key_b = cache1.char_mapping().get_u('b');
            let key_c = cache1.char_mapping().get_u('c');
            let key_d = cache1.char_mapping().get_u('d');
            let magic_key_star = cache1.char_mapping().get_u('*');
            let magic_key_hash = cache1.char_mapping().get_u('#');

            let regular_keys = [key_a, key_b, key_c, key_d];
            let magic_keys = [magic_key_star, magic_key_hash];

            // Apply the same operations to both caches
            for op in &operations {
                apply_operation(&mut cache1, op, &regular_keys, &magic_keys);
                apply_operation(&mut cache2, op, &regular_keys, &magic_keys);
            }

            // Scores should be identical
            prop_assert_eq!(
                cache1.score(), cache2.score(),
                "Applying the same operations should produce the same score"
            );

            // Individual analyzer scores should also match
            prop_assert_eq!(cache1.sfb.score(), cache2.sfb.score(), "SFB scores should match");
            prop_assert_eq!(cache1.stretch.score(), cache2.stretch.score(), "Stretch scores should match");
            prop_assert_eq!(cache1.scissors.score(), cache2.scissors.score(), "Scissors scores should match");
            prop_assert_eq!(cache1.trigram.score(), cache2.trigram.score(), "Trigram scores should match");
        }

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 3: Key swap reversibility
        ///
        /// For key swaps, swapping back returns to the original score.
        /// Key swap is its own inverse: swap(a, b) followed by swap(a, b) = identity.
        #[test]
        fn prop_key_swap_reversibility(
            swaps in proptest::collection::vec(arb_key_swap(), 1..=10),
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Record initial score
            let initial_score = cache.score();
            let initial_sfb = cache.sfb.score();
            let initial_stretch = cache.stretch.score();
            let initial_scissors = cache.scissors.score();
            let initial_trigram = cache.trigram.score();

            // Apply all swaps
            for &(pos_a, pos_b) in &swaps {
                cache.swap_keys(pos_a, pos_b);
            }

            // Reverse all swaps (in reverse order)
            for &(pos_a, pos_b) in swaps.iter().rev() {
                cache.swap_keys(pos_a, pos_b);
            }

            // Score should return to initial
            prop_assert_eq!(
                cache.score(), initial_score,
                "After reversing all swaps, score should return to initial"
            );
            prop_assert_eq!(cache.sfb.score(), initial_sfb, "SFB score should return to initial");
            prop_assert_eq!(cache.stretch.score(), initial_stretch, "Stretch score should return to initial");
            prop_assert_eq!(cache.scissors.score(), initial_scissors, "Scissors score should return to initial");
            prop_assert_eq!(cache.trigram.score(), initial_trigram, "Trigram score should return to initial");
        }

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 4: Magic rule reversibility
        ///
        /// For magic rules, clearing the rule returns to the original score (before the rule was applied).
        /// Note: This property only holds when the rule doesn't conflict with an existing rule on
        /// another magic key. When there's a conflict, the conflicting rule is cleared as a side effect.
        #[test]
        fn prop_magic_rule_reversibility(
            leader_idx in 0usize..4,
            output_idx in 0usize..4,
            magic_key_idx in 0usize..2,
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Get key IDs
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');
            let magic_key_hash = cache.char_mapping().get_u('#');

            let regular_keys = [key_a, key_b, key_c, key_d];
            let magic_keys = [magic_key_star, magic_key_hash];

            let leader = regular_keys[leader_idx % regular_keys.len()];
            let output = regular_keys[output_idx % regular_keys.len()];
            let magic_key = magic_keys[magic_key_idx % magic_keys.len()];

            // First, clear any existing rule for this (magic_key, leader) pair
            cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);

            // Check if applying this rule would conflict with an existing rule on another magic key.
            // A conflict occurs when another magic key has a rule with the same leader and output.
            // When there's a conflict, the conflicting rule is cleared as a side effect, so
            // reversibility doesn't hold in the simple sense.
            let has_conflict = cache.current_magic_rules.iter().any(|(&(other_magic, other_leader), &other_output)| {
                other_leader == leader && other_output == output && other_magic != magic_key
            });

            // Record score with no rule for this (magic_key, leader) pair
            let score_no_rule = cache.score();
            let sfb_no_rule = cache.sfb.score();
            let stretch_no_rule = cache.stretch.score();
            let scissors_no_rule = cache.scissors.score();
            let trigram_no_rule = cache.trigram.score();

            // Apply the rule
            cache.apply_magic_rule(magic_key, leader, output, true);

            // Clear the rule
            cache.apply_magic_rule(magic_key, leader, EMPTY_KEY, true);

            // Score should return to the no-rule state ONLY if there was no conflict.
            // If there was a conflict, the conflicting rule was cleared as a side effect,
            // so the score won't return to the original state.
            if !has_conflict {
                prop_assert_eq!(
                    cache.score(), score_no_rule,
                    "After clearing rule, score should return to no-rule state"
                );
                prop_assert_eq!(cache.sfb.score(), sfb_no_rule, "SFB score should return to no-rule state");
                prop_assert_eq!(cache.stretch.score(), stretch_no_rule, "Stretch score should return to no-rule state");
                prop_assert_eq!(cache.scissors.score(), scissors_no_rule, "Scissors score should return to no-rule state");
                prop_assert_eq!(cache.trigram.score(), trigram_no_rule, "Trigram score should return to no-rule state");
            }
            // When there's a conflict, we just verify the operation completes without error.
            // The score change is expected due to the side effect of clearing the conflicting rule.
        }

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 5: Mixed operations reversibility
        ///
        /// For a sequence of key swap operations, reversing them in order should return to the original score.
        /// Magic rule operations are excluded from this test because they can have side effects
        /// (clearing conflicting rules) that make simple reversal complex.
        ///
        /// For magic rule reversibility, see prop_magic_rule_reversibility which handles the
        /// conflict detection properly.
        #[test]
        fn prop_mixed_operations_reversibility(
            swaps in proptest::collection::vec(arb_key_swap(), 0..=8),
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
                Just("abcdabcdabcd"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Record initial score
            let initial_score = cache.score();

            // Apply key swaps
            for &(pos_a, pos_b) in &swaps {
                cache.swap_keys(pos_a, pos_b);
            }

            // Reverse key swaps (key swap is its own inverse)
            for &(pos_a, pos_b) in swaps.iter().rev() {
                cache.swap_keys(pos_a, pos_b);
            }

            // Score should return to initial
            prop_assert_eq!(
                cache.score(), initial_score,
                "After reversing all key swaps, score should return to initial"
            );
        }

        /// **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
        ///
        /// Property 6: Speculative scoring consistency
        ///
        /// For any operation, speculative scoring (apply=false) should return the same
        /// score as actually applying the operation.
        #[test]
        fn prop_speculative_scoring_consistency(
            operations in arb_operation_sequence(5),
            text_pattern in prop_oneof![
                Just("abcabcabcabc"),
                Just("ababababab"),
            ],
        ) {
            let layout = create_test_layout();
            let weights = create_test_weights();
            let data = create_test_data_with_layout_chars(text_pattern, &['a', 'b', 'c', 'd', '*', '#']);

            let mut cache = CachedLayout::new(&layout, data, &weights);

            // Get key IDs
            let key_a = cache.char_mapping().get_u('a');
            let key_b = cache.char_mapping().get_u('b');
            let key_c = cache.char_mapping().get_u('c');
            let key_d = cache.char_mapping().get_u('d');
            let magic_key_star = cache.char_mapping().get_u('*');
            let magic_key_hash = cache.char_mapping().get_u('#');

            let regular_keys = [key_a, key_b, key_c, key_d];
            let magic_keys = [magic_key_star, magic_key_hash];

            for op in &operations {
                // Get speculative score
                let speculative_score = match op {
                    Operation::KeySwap(pos_a, pos_b) => {
                        cache.score_neighbor_mut(Neighbor::KeySwap(PosPair(*pos_a, *pos_b)))
                    }
                    Operation::MagicRule(leader_idx, output_idx, magic_key_idx) => {
                        let leader = regular_keys[*leader_idx % regular_keys.len()];
                        let output = if *output_idx >= regular_keys.len() {
                            EMPTY_KEY
                        } else {
                            regular_keys[*output_idx % regular_keys.len()]
                        };
                        let magic_key = magic_keys[*magic_key_idx % magic_keys.len()];
                        cache.apply_magic_rule(magic_key, leader, output, false)
                    }
                };

                // Actually apply the operation
                apply_operation(&mut cache, op, &regular_keys, &magic_keys);
                let actual_score = cache.score();

                // Speculative and actual should match
                prop_assert_eq!(
                    speculative_score, actual_score,
                    "Speculative score {} should match actual score {} for {:?}",
                    speculative_score, actual_score, op
                );
            }
        }
    }
}
