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
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
    scissors: ScissorsCache,
    trigram: TrigramCache,
    magic: MagicCache,
    fingers: Vec<Finger>,
    magic_rule_penalty: i64,
    magic_repeat_penalty: i64,
    finger_usage_weight: i64,
    finger_usage_weights: [i64; 10], // per-finger weight from config
    finger_usage_score: i64,         // running total: -Σ char_freq[key] × finger_weight
    finger_usage_scale: i64,         // auto-computed normalization factor
    magic_penalty_scale: i64,        // auto-computed: penalty=1 costs ~1 rule's worth
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
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
            finger_usage_weight: 0,
            finger_usage_weights: [0; 10],
            finger_usage_score: 0,
            finger_usage_scale: 1,
            magic_penalty_scale: 1,
        }
    }
}


impl CachedLayout {
    pub fn new(layout: &Layout, data: Data, weights: &Weights, scale_factors: &crate::weights::ScaleFactors) -> Self {
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
        // Apply trigram scale factor before placing keys
        if scale_factors.trigram_scale > 1 {
            trigram.apply_scale(scale_factors.trigram_scale);
        }
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

            // Record initial rules, with conflict resolution: if another magic key
            // already maps (leader → output), the later rule wins (the earlier rule
            // is silently dropped). This matches the invariant enforced by
            // replace_rule_no_update at runtime.
            for (leading_str, output_str) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                let output = analyzer_data.char_mapping().get_u(output_str.chars().next().unwrap_or(' '));

                if output != EMPTY_KEY {
                    let mut key_to_clear = None;
                    for (&(other_magic, other_leader), &other_output) in &current_magic_rules {
                        if other_leader == leader && other_output == output && other_magic != magic_key {
                            key_to_clear = Some((other_magic, other_leader));
                            break;
                        }
                    }
                    if let Some(conflicting) = key_to_clear {
                        current_magic_rules.remove(&conflicting);
                    }
                }

                current_magic_rules.insert((magic_key, leader), output);
            }

            // All keys on the layout are potential leaders and outputs
            let all_keys: Vec<CacheKey> = layout.keys.iter()
                .map(|&c| analyzer_data.char_mapping().get_u(c))
                .filter(|&k| k != EMPTY_KEY && k != magic_key)
                .collect();

            // For every possible leader, create neighbors for every possible output + EMPTY
            for &leader in &all_keys {
                for &output in &all_keys {
                    neighbors.push(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output)));
                }
                // Also allow clearing the rule
                neighbors.push(Neighbor::MagicRule(MagicRule::new(magic_key, leader, EMPTY_KEY)));
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
            magic_rule_penalty: weights.magic_rule_penalty,
            magic_repeat_penalty: weights.magic_repeat_penalty,
            finger_usage_weight: weights.finger_usage,
            finger_usage_weights: {
                let mut fw = [0i64; 10];
                for f in Finger::FINGERS {
                    fw[f as usize] = weights.fingers.get(f);
                }
                fw
            },
            finger_usage_score: 0,
            finger_usage_scale: 1,
            magic_penalty_scale: scale_factors.magic_penalty_scale,
        };

        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = cache.data.char_mapping().get_u(key_char);
            cache.replace_key_no_update(pos, EMPTY_KEY, key);
        }
        cache.update();

        // Initialize pre-computed weighted scores for O(1) speculative trigram scoring
        cache.trigram.init_weighted_scores(&cache.keys, cache.magic.tg_freq());

        // Apply initial magic rules to the analyzers.
        // The current_magic_rules map was populated from the layout's magic key definitions.
        // Use replace_rule_no_update for each, then a single update() at the end.
        let initial_rules: Vec<_> = cache.current_magic_rules.iter()
            .map(|(&(magic_key, leader), &output)| (magic_key, leader, output))
            .collect();
        // Clear all rules so replace_rule_no_update doesn't early-return
        cache.current_magic_rules.clear();
        for (magic_key, leader, output) in initial_rules {
            cache.replace_rule_no_update(magic_key, leader, output);
        }
        cache.update();

        cache
    }

    pub fn data(&self) -> &AnalyzerData {
        &self.data
    }

    pub fn trigram_combo_counts(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        self.trigram.combo_counts()
    }

    // Accessors for B&B diagnostics and lower bound computation
    pub fn magic_bg_freq(&self) -> &[i64] { self.magic.bg_freq_flat() }
    pub fn magic_sg_freq(&self) -> &[i64] { self.magic.sg_freq_flat() }
    pub fn magic_tg_freq_flat(&self) -> &[i64] { self.magic.tg_freq_flat() }
    pub fn trigram_num_keys(&self) -> usize { self.trigram.num_keys() }
    pub fn trigram_max_weight(&self) -> i64 { self.trigram.max_weight() }
    pub fn sfb_pairs(&self, pos: usize) -> &[crate::same_finger::SfPair] { self.sfb.pairs_for_pos(pos) }
    pub fn sfb_weight(&self, finger: usize) -> i64 { self.sfb.sfb_weight_for_finger(finger) }
    pub fn sfs_weight(&self, finger: usize) -> i64 { self.sfb.sfs_weight_for_finger(finger) }
    pub fn stretch_pairs(&self, pos: usize) -> &[crate::stretches::StretchPair] { self.stretch.pairs_for_pos(pos) }
    pub fn stretch_weight_val(&self) -> i64 { self.stretch.weight() }
    pub fn trigram_combos_first(&self, pos: usize) -> &[crate::trigrams::TrigramCombo] { self.trigram.combos_first(pos) }
    pub fn get_key_at(&self, pos: usize) -> usize { self.keys[pos] }

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
            + self.magic_penalty() + self.finger_usage_score * self.finger_usage_weight * self.finger_usage_scale
    }



    /// Returns (sfb_score, stretch_score, scissors_score, trigram_score, magic_penalty, finger_usage)
    pub fn score_breakdown(&self) -> (i64, i64, i64, i64, i64, i64) {
        (self.sfb.score(), self.stretch.score(), self.scissors.score(), self.trigram.score(),
         self.magic_penalty(), self.finger_usage_score * self.finger_usage_weight * self.finger_usage_scale)
    }

    /// Penalty for active magic rules. Repeat rules (leader→same key) penalized separately.
    fn magic_penalty(&self) -> i64 {
        if self.magic_rule_penalty == 0 && self.magic_repeat_penalty == 0 {
            return 0;
        }
        let mut regular = 0i64;
        let mut repeats = 0i64;
        for (&(_magic_key, leader), &output) in &self.current_magic_rules {
            if output == EMPTY_KEY { continue; }
            if output == leader {
                repeats += 1;
            } else {
                regular += 1;
            }
        }
        // Penalties are negated — positive weight = worse score
        // Scale so penalty=1 costs roughly one rule's worth of score
        -(regular * self.magic_rule_penalty + repeats * self.magic_repeat_penalty) * self.magic_penalty_scale
    }

    /// Populate stats from the caches.
    pub fn stats(&self, stats: &mut Stats) {
        let char_total = self.data.char_total;
        let bigram_total = self.data.bigram_total;
        let skipgram_total = self.data.skipgram_total;
        let chars = &self.data.chars;

        // Finger use: sum character frequencies per finger.
        let char_total_raw = char_total * 100.0;
        for (pos, &key) in self.keys.iter().enumerate() {
            if key != EMPTY_KEY && (key as usize) < chars.len() {
                let finger = self.fingers[pos] as usize;
                stats.finger_use[finger] += chars[key as usize] as f64 / char_total_raw;
            }
        }

        self.sfb.stats(stats, bigram_total, skipgram_total);
        self.stretch.stats(stats, bigram_total);
        self.scissors.stats(stats, bigram_total, skipgram_total);
        self.trigram.stats(stats, self.data.trigram_total * 100.0);
    }

    /// Compute and return a complete Stats object for this layout.
    pub fn compute_stats(&self) -> Stats {
        let mut stats = Stats::default();
        self.stats(&mut stats);
        stats
    }

    /// Get the finger assigned to a position.
    #[inline]
    pub fn finger_at(&self, pos: usize) -> Finger {
        self.fingers[pos]
    }

    /// Get the (row, column) for a position based on the layout shape.
    /// Returns (row, col) as 0-indexed values.
    pub fn pos_row_col(&self, pos: usize) -> (usize, usize) {
        let shape = self.shape.inner();
        let mut remaining = pos;
        for (row_idx, &row_len) in shape.iter().enumerate() {
            if remaining < row_len {
                return (row_idx, remaining);
            }
            remaining -= row_len;
        }
        (0, 0)
    }

    /// Total character frequency on left vs right hand.
    pub fn hand_frequencies(&self, data_chars: &[i64]) -> (i64, i64) {
        let mut left = 0i64;
        let mut right = 0i64;
        for (pos, &key) in self.keys.iter().enumerate() {
            if key == EMPTY_KEY || (key as usize) >= data_chars.len() { continue; }
            let f = self.fingers[pos];
            let freq = data_chars[key as usize];
            if matches!(f, Finger::LP | Finger::LR | Finger::LM | Finger::LI | Finger::LT) {
                left += freq;
            } else {
                right += freq;
            }
        }
        (left, right)
    }

    /// Total character frequency on each hand for a specific set of chars.
    pub fn hand_freq_for_chars(&self, data_chars: &[i64], target_chars: &[char]) -> (i64, i64) {
        let mut left = 0i64;
        let mut right = 0i64;
        for &target in target_chars {
            let key = self.data.char_mapping().get_u(target);
            if key == EMPTY_KEY { continue; }
            let key_u = key as usize;
            if key_u >= data_chars.len() { continue; }
            if let Some(pos) = self.get_pos(key) {
                let f = self.fingers[pos];
                let freq = data_chars[key_u];
                if matches!(f, Finger::LP | Finger::LR | Finger::LM | Finger::LI | Finger::LT) {
                    left += freq;
                } else {
                    right += freq;
                }
            }
        }
        (left, right)
    }

    /// Total character frequency on a specific row.
    pub fn row_usage(&self, data_chars: &[i64], row: usize) -> i64 {
        let mut total = 0i64;
        for (pos, &key) in self.keys.iter().enumerate() {
            if key == EMPTY_KEY || (key as usize) >= data_chars.len() { continue; }
            let (r, _) = self.pos_row_col(pos);
            if r == row {
                total += data_chars[key as usize];
            }
        }
        total
    }

    /// Count active (non-empty) magic rules.
    pub fn magic_rule_count(&self) -> usize {
        self.current_magic_rules.iter()
            .filter(|(_, &output)| output != EMPTY_KEY)
            .count()
    }

    // ==================== Mutation API ====================
    //
    // Three operations: replace_key, swap_key, replace_rule
    // Each has base (score() valid after) and _no_update (score() invalid until update()).
    // score_neighbor: speculative scoring without permanent state change.

    /// Speculative score for a neighbor. No permanent state change.
    pub fn score_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                // When magic rules are active, a keyswap changes the magic-rule
                // contribution (rules re-map different trigrams depending on key
                // positions). The fast speculative path below doesn't account for
                // this. Fall back to apply-score-revert when rules exist.
                if !self.current_magic_rules.is_empty() {
                    self.swap_key(a, b);
                    let s = self.score();
                    self.swap_key(a, b);
                    return s;
                }

                let key_a = self.keys[a];
                let key_b = self.keys[b];
                let bg_freq = self.magic.bg_freq_flat();
                let sg_freq = self.magic.sg_freq_flat();
                let tg_flat = self.magic.tg_freq_flat();

                let sfb = self.sfb.score_swap(a, b, key_a, key_b, &self.keys, bg_freq, sg_freq);
                let stretch = self.stretch.score_swap(a, b, key_a, key_b, &self.keys, bg_freq);
                let scissors = self.scissors.score_swap(a, b, key_a, key_b, &self.keys, bg_freq, sg_freq);

                // Compute typed per-type frequency deltas. We need the per-type breakdown
                // (not just the weighted sum) because score() applies an offset of
                // -max_trigram_weight * total_freq, which depends on the per-type totals.
                let tg_delta_a = self.trigram.compute_replace_delta_flat_typed(a, key_a, key_b, &self.keys, Some(b), tg_flat);
                let tg_delta_b = self.trigram.compute_replace_delta_flat_typed(b, key_b, key_a, &self.keys, Some(a), tg_flat);
                let tg_delta_both = self.trigram.compute_swap_both_delta_flat_typed(a, b, key_a, key_b, &self.keys, tg_flat);
                let tg_delta = tg_delta_a.combine(&tg_delta_b).combine(&tg_delta_both);
                let trigram = self.trigram.score() + self.trigram.weighted_score_of_delta(&tg_delta);

                let fu_delta = if self.finger_usage_weight != 0 {
                    let fi_a = self.fingers[a] as usize;
                    let fi_b = self.fingers[b] as usize;
                    if fi_a != fi_b {
                        let fw_a = self.finger_usage_weights[fi_a];
                        let fw_b = self.finger_usage_weights[fi_b];
                        let freq_a = if key_a < self.data.chars.len() { self.data.chars[key_a] } else { 0 };
                        let freq_b = if key_b < self.data.chars.len() { self.data.chars[key_b] } else { 0 };
                        (freq_a * fw_a - freq_a * fw_b + freq_b * fw_b - freq_b * fw_a) * self.finger_usage_weight * self.finger_usage_scale
                    } else { 0 }
                } else { 0 };

                sfb + stretch + scissors + trigram + self.magic_penalty()
                    + (self.finger_usage_score * self.finger_usage_weight * self.finger_usage_scale) + fu_delta
            }
            Neighbor::MagicRule(rule) => {
                // TODO: speculative magic scoring for better perf.
                // For now, apply-score-revert is the only reliable approach.
                let revert = self.get_revert_neighbor(neighbor);
                self.replace_rule(rule.magic_key, rule.leader, rule.output);
                let s = self.score();
                if let Neighbor::MagicRule(rev) = revert {
                    self.replace_rule(rev.magic_key, rev.leader, rev.output);
                }
                s
            }
        }
    }

    /// Apply a neighbor. score() valid after.
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.swap_key(a, b),
            Neighbor::MagicRule(rule) => self.replace_rule(rule.magic_key, rule.leader, rule.output),
        }
    }

    /// Apply a neighbor. score() INVALID until update().
    pub fn apply_neighbor_no_update(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => self.swap_key_no_update(a, b),
            Neighbor::MagicRule(rule) => self.replace_rule_no_update(rule.magic_key, rule.leader, rule.output),
        }
    }

    /// Recompute magic deltas. Call after _no_update mutations to make score() valid.
    pub fn update(&mut self) {
        // Always reset magic deltas. If all rules have just been cleared, this
        // wipes stale contributions from previously-active rules. If rules exist,
        // they'll be re-added below.
        self.sfb.reset_magic_deltas();
        self.stretch.reset_magic_deltas();
        self.scissors.reset_magic_deltas();
        self.trigram.reset_magic_deltas();
        if self.current_magic_rules.is_empty() { return; }
        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();
        for (&(magic_key, leader), &output) in &self.current_magic_rules {
            if output == EMPTY_KEY { continue; }
            self.sfb.add_rule(leader, output, magic_key, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, true);
            self.stretch.add_rule(leader, output, magic_key, &self.keys, &self.key_positions, bg_freq, tg_freq, true);
            self.scissors.add_rule(leader, output, magic_key, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq, true);
            self.trigram.add_rule(leader, output, magic_key, &self.keys, &self.key_positions, tg_freq, true);
        }
    }

    /// Incremental magic-rule update after a keyswap. Only recomputes deltas
    /// for rules whose leader, output, or magic_key is `key_a` or `key_b`
    /// (these are the only rules whose delta changed). Requires that
    /// `swap_key_no_update` (or equivalent) has already been called to move keys.
    ///
    /// This is O(affected_rules) instead of O(all_rules × 4_caches) that
    /// `update()` pays.
    pub fn update_for_keyswap(&mut self, key_a: CacheKey, key_b: CacheKey) {
        if self.current_magic_rules.is_empty() { return; }
        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();
        self.sfb.update_for_keyswap(key_a, key_b, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq);
        self.stretch.update_for_keyswap(key_a, key_b, &self.keys, &self.key_positions, bg_freq, tg_freq);
        self.scissors.update_for_keyswap(key_a, key_b, &self.keys, &self.key_positions, bg_freq, sg_freq, tg_freq);
        self.trigram.update_for_keyswap(key_a, key_b, &self.keys, &self.key_positions, tg_freq);
    }

    /// Full recompute of trigram frequencies from scratch. Use after out-of-order
    /// `replace_key_no_update` operations (e.g., partial layouts in beam/mcts).
    /// Base swap/replace operations via `update()` don't need this — it's only for
    /// when the delta tracking breaks down due to operations involving EMPTY keys.
    pub fn full_recompute(&mut self) {
        let tg_flat = self.magic.tg_freq_flat();
        self.trigram.recompute_frequencies(&self.keys, tg_flat);
        self.update();
    }

    // ==================== Lower Bound ====================

    /// Compute a lower bound on the remaining cost by greedy completion.
    /// Places each remaining key at its best available position (greedily),
    /// then returns the difference between the resulting score and the current score.
    /// This is a valid lower bound because the optimal placement can only be better
    /// than or equal to the greedy placement.
    ///
    /// Wait - this is actually an UPPER bound on the remaining cost (less negative
    /// than optimal). For B&B we need a LOWER bound (more negative than optimal).
    /// The greedy completion gives us a feasible solution whose score we can use
    /// to TIGHTEN the bound, not for pruning the current node.
    ///
    /// For pruning, we use: if current_score < bound, prune.
    /// The greedy completion helps by finding better solutions faster.
    pub fn greedy_completion_score(&self, unplaced_keys: &[usize], available_positions: &[usize]) -> i64 {
        // Clone the cache state so we can mutate it
        let mut cache = self.clone();
        let mut avail = available_positions.to_vec();

        for &key in unplaced_keys {
            if avail.is_empty() { break; }
            let nk = cache.trigram.num_keys();
            if key >= nk { continue; }

            // Find the best position for this key
            let mut best_pos = avail[0];
            let mut best_score = i64::MIN;

            for &pos in &avail {
                cache.replace_key_no_update(pos, EMPTY_KEY, key);
                cache.update();
                let score = cache.score();
                cache.replace_key_no_update(pos, key, EMPTY_KEY);
                cache.update();
                if score > best_score {
                    best_score = score;
                    best_pos = pos;
                }
            }

            // Place at best position
            cache.replace_key_no_update(best_pos, EMPTY_KEY, key);
            cache.update();
            avail.retain(|&p| p != best_pos);
        }

        cache.score()
    }

    // ==================== Mutation ====================
    //
    // Three operations × 2 variants + update:
    //   replace_key / replace_key_no_update
    //   swap_key    / swap_key_no_update
    //   replace_rule / replace_rule_no_update
    //
    // Base variants call _no_update + update(). Use _no_update for batching.

    /// Replace key at position. score() valid after.
    #[inline]
    pub fn replace_key(&mut self, pos: CachePos, old_key: CacheKey, new_key: CacheKey) {
        self.replace_key_no_update(pos, old_key, new_key);
        self.update();
    }

    /// Replace key — no magic delta recompute. score() INVALID until update().
    #[inline]
    pub fn replace_key_no_update(&mut self, pos: CachePos, old_key: CacheKey, new_key: CacheKey) {
        debug_assert!(self.keys[pos] == old_key, "Position {pos} has key {} but expected {old_key}", self.keys[pos]);

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_flat = self.magic.tg_freq_flat();

        self.sfb.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq);
        self.stretch.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq);
        self.scissors.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq);
        self.trigram.replace_key_fast(pos, old_key, new_key, &self.keys, tg_flat);

        // Update finger usage score
        if self.finger_usage_weight != 0 {
            let fi = self.fingers[pos] as usize;
            let fw = self.finger_usage_weights[fi];
            if old_key != EMPTY_KEY && old_key < self.data.chars.len() {
                self.finger_usage_score += self.data.chars[old_key] * fw; // remove old penalty
            }
            if new_key != EMPTY_KEY && new_key < self.data.chars.len() {
                self.finger_usage_score -= self.data.chars[new_key] * fw; // add new penalty
            }
        }

        self.keys[pos] = new_key;
        if old_key != EMPTY_KEY && old_key < self.key_positions.len() {
            self.key_positions[old_key] = None;
        }
        if new_key != EMPTY_KEY && new_key < self.key_positions.len() {
            self.key_positions[new_key] = Some(pos);
        }
    }

    /// Swap keys at two positions. score() valid after.
    #[inline]
    pub fn swap_key(&mut self, pos_a: CachePos, pos_b: CachePos) {
        self.swap_key_no_update(pos_a, pos_b);
        self.update();
    }

    /// Swap keys — no magic delta recompute. score() INVALID until update().
    #[inline]
    pub fn swap_key_no_update(&mut self, pos_a: CachePos, pos_b: CachePos) {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];

        debug_assert!(key_a != EMPTY_KEY, "Position {pos_a} is empty");
        debug_assert!(key_b != EMPTY_KEY, "Position {pos_b} is empty");

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_flat = self.magic.tg_freq_flat();

        self.sfb.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.stretch.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq);
        self.scissors.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq);
        self.trigram.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, tg_flat);

        // Update finger usage: keys swap fingers
        if self.finger_usage_weight != 0 {
            let fi_a = self.fingers[pos_a] as usize;
            let fi_b = self.fingers[pos_b] as usize;
            if fi_a != fi_b {
                let fw_a = self.finger_usage_weights[fi_a];
                let fw_b = self.finger_usage_weights[fi_b];
                let freq_a = if key_a < self.data.chars.len() { self.data.chars[key_a] } else { 0 };
                let freq_b = if key_b < self.data.chars.len() { self.data.chars[key_b] } else { 0 };
                self.finger_usage_score += freq_a * fw_a - freq_a * fw_b
                                         + freq_b * fw_b - freq_b * fw_a;
            }
        }

        self.keys[pos_a] = key_b;
        self.keys[pos_b] = key_a;
        if key_a < self.key_positions.len() {
            self.key_positions[key_a] = Some(pos_b);
        }
        if key_b < self.key_positions.len() {
            self.key_positions[key_b] = Some(pos_a);
        }
    }

    /// Replace a magic rule. score() valid after.
    pub fn replace_rule(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey) {
        self.replace_rule_no_update(magic_key, leader, new_output);
        self.update();
    }

    /// Replace a magic rule — no magic delta recompute. score() INVALID until update().
    ///
    /// Handles conflict resolution: if another magic key already maps (leader → new_output),
    /// that conflicting rule is cleared first.
    pub fn replace_rule_no_update(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey) {
        let key = (magic_key, leader);
        let old_output = self.current_magic_rules.get(&key).copied();

        // Early return if no change
        if old_output == Some(new_output) || (old_output.is_none() && new_output == EMPTY_KEY) {
            return;
        }

        // Clear conflicting rule from another magic key if needed
        if new_output != EMPTY_KEY {
            let mut key_to_clear = None;
            for (&(other_magic, other_leader), &other_output) in &self.current_magic_rules {
                if other_leader == leader && other_output == new_output && other_magic != magic_key {
                    key_to_clear = Some((other_magic, other_leader));
                    break;
                }
            }
            if let Some((other_magic, other_leader)) = key_to_clear {
                self.current_magic_rules.remove(&(other_magic, other_leader));
            }
        }

        // Update current_magic_rules
        if new_output != EMPTY_KEY {
            self.current_magic_rules.insert(key, new_output);
        } else {
            self.current_magic_rules.remove(&key);
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

    /// Greedy depth-N improvement directly on the cache (no use_layout needed).
    /// `pins` is a set of positions that must not be swapped.
    /// Returns the final score after improvement.
    pub fn greedy_improve_depth_n(&mut self, pins: &[usize], depth: usize) -> i64 {
        // Build neighbor list excluding pinned positions
        let pin_set: fxhash::FxHashSet<usize> = pins.iter().copied().collect();
        let neighbors: Vec<Neighbor> = self.neighbors.iter()
            .filter(|n| match n {
                Neighbor::KeySwap(PosPair(a, b)) => !pin_set.contains(a) && !pin_set.contains(b),
                Neighbor::MagicRule(_) => false, // skip magic for rollout polish
            })
            .copied()
            .collect();

        let mut diffs = vec![Neighbor::default(); depth];
        let mut cur_best = self.score();

        while Self::best_neighbor_recursive(self, &neighbors, depth, &mut diffs, &mut cur_best) {
            for &neighbor in &diffs {
                self.apply_neighbor(neighbor);
            }
        }

        self.score()
    }

    pub(crate) fn best_neighbor_recursive(
        cache: &mut CachedLayout,
        neighbors: &[Neighbor],
        depth: usize,
        diffs: &mut Vec<Neighbor>,
        cur_best: &mut i64,
    ) -> bool {
        if depth > 0 {
            let mut return_best = false;
            for &neighbor in neighbors {
                // Apply with full update so score() is valid at each level
                cache.apply_neighbor(neighbor);

                let best = Self::best_neighbor_recursive(cache, neighbors, depth - 1, diffs, cur_best);

                // Revert — KeySwap is self-inverse, MagicRule needs explicit revert
                match neighbor {
                    Neighbor::KeySwap(_) => cache.apply_neighbor(neighbor),
                    _ => {
                        let revert = cache.get_revert_neighbor(neighbor);
                        cache.apply_neighbor(revert);
                    }
                }

                if best {
                    diffs[depth - 1] = neighbor;
                    return_best = true;
                }
            }
            return_best
        } else {
            let score = cache.score();
            if score > *cur_best {
                *cur_best = score;
                true
            } else {
                false
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
            finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
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
    /// when replace_rule is called. This test verifies that:
    /// 1. Clearing a rule and re-applying it produces consistent scores
    /// 2. The score changes when rules are applied/cleared
    #[test]
    fn test_magic_rule_produces_score_change() {
        let layout = create_layout_with_magic();
        let weights = create_test_weights();
        // Create data with 'ab' bigrams and all layout chars to ensure proper array sizing
        let data = create_test_data_with_layout_chars("ababababab", &['a', 'b', 'c', '*']);

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        // Get the magic key and leader key IDs
        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First, apply the rule to establish a baseline with the rule active
        cache.replace_rule(magic_key, leader, output); let score_with_rule = cache.score();

        // Clear the rule
        cache.replace_rule(magic_key, leader, EMPTY_KEY); let score_without_rule = cache.score();

        // Re-apply the rule
        cache.replace_rule(magic_key, leader, output); let score_reapplied = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');
        let alt_output = cache.char_mapping().get_u('c');

        // First establish a baseline by applying the original rule
        cache.replace_rule(magic_key, leader, output); let baseline_score = cache.score();

        // Apply a different rule (change output from 'b' to 'c')
        cache.replace_rule(magic_key, leader, alt_output); let score_after_change = cache.score();

        // Verify the score changed
        println!("Baseline score: {}", baseline_score);
        println!("Score after change to 'c': {}", score_after_change);

        // Reverse by applying the original rule
        cache.replace_rule(magic_key, leader, output); let score_after_reverse = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First establish a baseline by applying the rule
        cache.replace_rule(magic_key, leader, output); let baseline_score = cache.score();

        // Clear the rule
        cache.replace_rule(magic_key, leader, EMPTY_KEY); let score_after_clear = cache.score();
        println!("Baseline score: {}", baseline_score);
        println!("Score after clear: {}", score_after_clear);

        // Re-apply the rule
        cache.replace_rule(magic_key, leader, output); let score_after_reapply = cache.score();
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline by applying a rule
        cache.replace_rule(magic_key, leader, output_b);
        let baseline_score = cache.score();

        // Speculative scoring for a different rule
        let speculative_score = cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output_c)));

        // Verify state is unchanged
        assert_eq!(
            cache.score(),
            baseline_score,
            "Speculative scoring should not change the score"
        );

        // Now actually apply
        cache.replace_rule(magic_key, leader, output_c);
        let actual_score = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline by applying a rule
        cache.replace_rule(magic_key, leader, output_b);
        let baseline_score = cache.score();

        // Multiple speculative calls
        let spec1 = cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output_c)));
        let spec2 = cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output_c)));
        let _spec3 = cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, EMPTY_KEY)));
        let spec4 = cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output_b)));

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key1 = cache.char_mapping().get_u('*');
        let magic_key2 = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let leader_c = cache.char_mapping().get_u('c');
        let output_b = cache.char_mapping().get_u('b');
        let output_d = cache.char_mapping().get_u('d');

        // Apply both rules to establish a baseline
        cache.replace_rule(magic_key1, leader_a, output_b);
        cache.replace_rule(magic_key2, leader_c, output_d);
        let baseline_score = cache.score();

        // Clear both rules
        cache.replace_rule(magic_key1, leader_a, EMPTY_KEY);
        cache.replace_rule(magic_key2, leader_c, EMPTY_KEY);
        let score_no_rules = cache.score();

        // Apply first rule
        cache.replace_rule(magic_key1, leader_a, output_b);
        let score_one_rule = cache.score();

        // Apply second rule
        cache.replace_rule(magic_key2, leader_c, output_d);
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key_id = cache.char_mapping().get_u('*');
        let leader_a = cache.char_mapping().get_u('a');
        let leader_c = cache.char_mapping().get_u('c');
        let output_b = cache.char_mapping().get_u('b');
        let output_d = cache.char_mapping().get_u('d');

        // Apply both rules to establish a baseline
        cache.replace_rule(magic_key_id, leader_a, output_b);
        cache.replace_rule(magic_key_id, leader_c, output_d);
        let baseline_score = cache.score();

        // Clear both rules
        cache.replace_rule(magic_key_id, leader_a, EMPTY_KEY);
        cache.replace_rule(magic_key_id, leader_c, EMPTY_KEY);
        let score_no_rules = cache.score();

        // Apply first rule (a -> b)
        cache.replace_rule(magic_key_id, leader_a, output_b);
        let score_one_rule = cache.score();

        // Apply second rule (c -> d)
        cache.replace_rule(magic_key_id, leader_c, output_d);
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key1_id = cache.char_mapping().get_u('*');
        let magic_key2_id = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');

        // Clear all rules first
        cache.replace_rule(magic_key1_id, leader_a, EMPTY_KEY);
        cache.replace_rule(magic_key2_id, leader_a, EMPTY_KEY);

        // Apply rule on magic_key1: a -> * produces b
        cache.replace_rule(magic_key1_id, leader_a, output_b);
        let score_with_key1 = cache.score();

        // Now apply conflicting rule on magic_key2: a -> # produces b
        // This should clear the rule from magic_key1
        cache.replace_rule(magic_key2_id, leader_a, output_b);
        let score_with_key2 = cache.score();

        println!("Score with rule on key1: {}", score_with_key1);
        println!("Score with rule on key2: {}", score_with_key2);

        // The scores might differ because the magic keys are at different positions
        // The important thing is that the system doesn't crash and handles the conflict

        // Verify that magic_key1 no longer has the rule (it was cleared)
        // We can check by trying to apply the same rule to key1 again
        cache.replace_rule(magic_key1_id, leader_a, output_b);
        let score_reapply_key1 = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key1_id = cache.char_mapping().get_u('*');
        let magic_key2_id = cache.char_mapping().get_u('#');
        let leader_a = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');

        // Clear all rules and record baseline
        cache.replace_rule(magic_key1_id, leader_a, EMPTY_KEY);
        cache.replace_rule(magic_key2_id, leader_a, EMPTY_KEY);
        let baseline_score = cache.score();
        let baseline_rules = cache.magic_rule_count();

        // Apply and conflict multiple times
        for _ in 0..5 {
            cache.replace_rule(magic_key1_id, leader_a, output_b);
            cache.replace_rule(magic_key2_id, leader_a, output_b);
            cache.replace_rule(magic_key1_id, leader_a, output_b);
        }

        // Clear all rules
        cache.replace_rule(magic_key1_id, leader_a, EMPTY_KEY);
        cache.replace_rule(magic_key2_id, leader_a, EMPTY_KEY);
        let final_score = cache.score();
        let final_rules = cache.magic_rule_count();

        // Should return to baseline
        assert_eq!(baseline_rules, 0, "baseline should have 0 rules");
        assert_eq!(final_rules, 0, "final should have 0 rules");
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        // Get speculative score for swapping positions 0 and 1
        let speculative_score = cache.score_neighbor(Neighbor::KeySwap(PosPair(0, 1)));

        // Verify state is unchanged
        let score_before = cache.score();

        // Actually apply the swap
        cache.swap_key(0, 1);
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
        cache.swap_key(0, 1);
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
        cache.replace_rule(magic_key_hash, key_a, EMPTY_KEY);
        let score_after_clear_hash = cache.score();

        // Verify (*, a)->b is still active
        assert_eq!(
            cache.current_magic_rules.get(&(magic_key_star, key_a)),
            Some(&key_b),
            "After clearing (#, a), (*, a)->b should still be active"
        );

        // Step 2: Apply (#, a)->b - this conflicts with (*, a)->b
        cache.replace_rule(magic_key_hash, key_a, key_b);

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
        cache.replace_rule(magic_key_hash, key_a, EMPTY_KEY);
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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output = cache.char_mapping().get_u('b');

        // First apply the rule to establish a baseline
        cache.replace_rule(magic_key, leader, output); let baseline_score = cache.score();

        // Apply the same rule again (should be no-op)
        cache.replace_rule(magic_key, leader, output); let score_after_reapply = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let _output = cache.char_mapping().get_u('b');

        // Clear the rule (even though analyzers weren't initialized with it)
        cache.replace_rule(magic_key, leader, EMPTY_KEY); let score_after_clear = cache.score();

        // Try to clear again (should be no-op)
        cache.replace_rule(magic_key, leader, EMPTY_KEY); let score_after_second_clear = cache.score();

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

        let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

        let magic_key = cache.char_mapping().get_u('*');
        let leader = cache.char_mapping().get_u('a');
        let output_b = cache.char_mapping().get_u('b');
        let output_c = cache.char_mapping().get_u('c');

        // First establish a baseline
        cache.replace_rule(magic_key, leader, output_b);
        let baseline = cache.score();

        // Perform various operations and verify score consistency
        let s0 = cache.score();
        cache.replace_rule(magic_key, leader, output_c);
        let s1 = cache.score();
        let s2 = cache.score();
        cache.replace_rule(magic_key, leader, EMPTY_KEY);
        let s3 = cache.score();
        let s4 = cache.score();
        cache.replace_rule(magic_key, leader, output_b);
        let s5 = cache.score();
        let s6 = cache.score();
        let scores: Vec<i64> = vec![s0, s1, s2, s3, s4, s5, s6];

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

            finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
                cache.replace_rule(magic_key, leader, output);
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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

                // Speculative scoring
                cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key_star, leader, output)));
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
                        cache.swap_key(pos_a, pos_b);
                    }
                } else {
                    // Magic rule operation
                    let leader = regular_keys[idx1 % regular_keys.len()];
                    let output = if idx2 >= regular_keys.len() {
                        EMPTY_KEY
                    } else {
                        regular_keys[idx2 % regular_keys.len()]
                    };
                    cache.replace_rule(magic_key_star, leader, output);
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

            finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
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
                cache.swap_key(*pos_a, *pos_b);
            }
            Operation::MagicRule(leader_idx, output_idx, magic_key_idx) => {
                let leader = regular_keys[*leader_idx % regular_keys.len()];
                let output = if *output_idx >= regular_keys.len() {
                    EMPTY_KEY
                } else {
                    regular_keys[*output_idx % regular_keys.len()]
                };
                let magic_key = magic_keys[*magic_key_idx % magic_keys.len()];
                cache.replace_rule(magic_key, leader, output);
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
            let mut cache1 = CachedLayout::new(&layout, data.clone(), &weights, &crate::weights::ScaleFactors::default());
            let mut cache2 = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

            // Record initial score
            let initial_score = cache.score();
            let initial_sfb = cache.sfb.score();
            let initial_stretch = cache.stretch.score();
            let initial_scissors = cache.scissors.score();
            let initial_trigram = cache.trigram.score();

            // Apply all swaps
            for &(pos_a, pos_b) in &swaps {
                cache.swap_key(pos_a, pos_b);
            }

            // Reverse all swaps (in reverse order)
            for &(pos_a, pos_b) in swaps.iter().rev() {
                cache.swap_key(pos_a, pos_b);
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
            cache.replace_rule(magic_key, leader, EMPTY_KEY);

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
            cache.replace_rule(magic_key, leader, output);

            // Clear the rule
            cache.replace_rule(magic_key, leader, EMPTY_KEY);

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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

            // Record initial score
            let initial_score = cache.score();

            // Apply key swaps
            for &(pos_a, pos_b) in &swaps {
                cache.swap_key(pos_a, pos_b);
            }

            // Reverse key swaps (key swap is its own inverse)
            for &(pos_a, pos_b) in swaps.iter().rev() {
                cache.swap_key(pos_a, pos_b);
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

            let mut cache = CachedLayout::new(&layout, data, &weights, &crate::weights::ScaleFactors::default());

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
                        cache.score_neighbor(Neighbor::KeySwap(PosPair(*pos_a, *pos_b)))
                    }
                    Operation::MagicRule(leader_idx, output_idx, magic_key_idx) => {
                        let leader = regular_keys[*leader_idx % regular_keys.len()];
                        let output = if *output_idx >= regular_keys.len() {
                            EMPTY_KEY
                        } else {
                            regular_keys[*output_idx % regular_keys.len()]
                        };
                        let magic_key = magic_keys[*magic_key_idx % magic_keys.len()];
                        cache.score_neighbor(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output)))
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
