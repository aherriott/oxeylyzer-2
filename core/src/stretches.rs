/*
 **************************************
 *            Stretches
 **************************************
 */

use fxhash::FxHashMap as HashMap;

use crate::dist::{dx_dy, x_overlap};
use crate::stats::Stats;
use crate::types::{CacheKey, CachePos};
use crate::weights::Weights;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

/// Pre-computed stretch pair with distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StretchPair {
    pub other_pos: usize,
    pub dist: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    /// For each position, list of other positions that form a stretch pair with pre-computed distances
    stretch_pairs_per_key: Vec<Vec<StretchPair>>,
    /// Number of keys for frequency array indexing
    num_keys: usize,
    /// Running total (freq * dist, not yet weighted)
    total: i64,
    /// Pre-computed stretch weight
    stretch_weight: i64,

    /// Tracks active magic rules: (magic_key, leader) -> (output, delta)
    active_rules: HashMap<(CacheKey, CacheKey), (CacheKey, i64)>,
    /// Cumulative score delta from all active magic rules.
    magic_rule_score_delta: i64,
    /// Pre-computed rule deltas for O(1) `add_rule` speculative scoring.
    rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,
}

impl StretchCache {
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger], num_keys: usize) -> Self {
        let len = keyboard.len();

        let stretch_dists: Vec<Vec<i64>> = (0..len)
            .map(|i| {
                (0..len)
                    .map(|j| {
                        if i != j {
                            Self::compute_stretch(&keyboard[i], &keyboard[j], fingers[i], fingers[j])
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();

        let stretch_pairs_per_key: Vec<Vec<StretchPair>> = (0..len)
            .map(|i| {
                (0..len)
                    .filter_map(|j| {
                        let dist = stretch_dists[i][j];
                        if i != j && dist > 0 {
                            Some(StretchPair { other_pos: j, dist })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            stretch_pairs_per_key,
            num_keys,
            total: 0,
            stretch_weight: 0,
            active_rules: HashMap::default(),
            magic_rule_score_delta: 0,
            rule_delta: HashMap::default(),
        }
    }

    pub fn set_weights(&mut self, weights: &Weights) {
        // Stretches are a penalty — negate so positive weight = worse score
        self.stretch_weight = -weights.stretches;
    }

    fn compute_stretch(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> i64 {
        if f1 == f2 || f1.hand() != f2.hand() {
            return 0;
        }
        let (dx, dy) = dx_dy(k1, k2, f1, f2);
        let diff = (f1 as usize).abs_diff(f2 as usize) as f64;
        let finger_dist_allowance = diff * 1.35;
        let dist = dx.hypot(dy);
        let xo = x_overlap(dx, dy, f1, f2);
        let stretch = dist + xo - finger_dist_allowance;
        if stretch > 0.001 {
            (stretch * 100.0) as i64
        } else {
            0
        }
    }

    #[inline]
    pub fn get_stretch(&self, p1: CachePos, p2: CachePos) -> i64 {
        self.stretch_pairs_per_key[p1]
            .iter()
            .find(|sp| sp.other_pos == p2)
            .map(|sp| sp.dist)
            .unwrap_or(0)
    }

    #[inline]
    pub fn score(&self) -> i64 {
        self.total * self.stretch_weight + self.magic_rule_score_delta
    }

    #[inline]
    pub fn pairs_for_pos(&self, pos: usize) -> &[StretchPair] {
        &self.stretch_pairs_per_key[pos]
    }

    #[inline]
    pub fn weight(&self) -> i64 {
        self.stretch_weight
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64) {
        stats.stretches = self.total as f64 / (bigram_total * 100.0);
    }

    // ==================== Speculative Scoring ====================

    /// Speculative score for a key swap. No mutation.
    /// O(stretch_pairs) per position — typically 0-5 pairs, effectively O(1).
    #[inline]
    pub fn score_swap(
        &self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
    ) -> i64 {
        let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq);
        let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq);
        ((self.total + delta_a + delta_b) * self.stretch_weight) + self.magic_rule_score_delta
    }

    /// Speculative score for replacing a key. No mutation.
    #[inline]
    pub fn score_replace(
        &self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        bg_freq: &[i64],
    ) -> i64 {
        let delta = self.compute_replace_delta(pos, old_key, new_key, keys, None, bg_freq);
        ((self.total + delta) * self.stretch_weight) + self.magic_rule_score_delta
    }

    // ==================== Mutation ====================

    /// Replace key at position. Mutates running totals. Returns the new score.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
    ) -> i64 {
        let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, bg_freq);
        self.total += delta;
        self.score()
    }

    /// Swap keys at two positions. Mutates running totals. Returns the new score.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
    ) -> i64 {
        let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq);
        let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq);
        self.total += delta_a + delta_b;
        self.score()
    }

    // ==================== Internal ====================

    #[inline]
    fn compute_replace_delta(
        &self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;
        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut delta: i64 = 0;

        for sp in &self.stretch_pairs_per_key[pos] {
            let other_pos = sp.other_pos;
            if skip_pos == Some(other_pos) {
                continue;
            }
            let other_key = keys[other_pos];
            if other_key >= num_keys {
                continue;
            }

            let stretch_dist = sp.dist;
            let other_row = other_key * num_keys;

            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };

            let bg_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            delta += bg_delta * stretch_dist;
        }

        delta
    }

    #[inline]
    fn stretch_bigram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        let dist = self.get_stretch(p_a, p_b);
        if dist > 0 { dist * self.stretch_weight } else { 0 }
    }

    // ==================== Lower Bound ====================

    /// Compute a lower bound on the remaining stretch penalty from unplaced keys.
    ///
    /// For each unplaced key, finds the available position that minimizes the stretch
    /// penalty with all already-placed keys, and sums these minimums. This is a valid
    /// lower bound because it independently minimizes each key's placement.
    pub fn lower_bound_remaining(
        &self,
        unplaced_keys: &[usize],
        available_positions: &[usize],
        keys: &[usize],
        bg_freq: &[i64],
    ) -> i64 {
        let nk = self.num_keys;
        let mut total_min: i64 = 0;

        for &key in unplaced_keys {
            if key >= nk { continue; }
            let mut best_penalty = i64::MAX;
            for &pos in available_positions {
                let mut penalty: i64 = 0;
                for sp in &self.stretch_pairs_per_key[pos] {
                    let other_key = keys[sp.other_pos];
                    if other_key >= nk { continue; }
                    let bg = bg_freq[key * nk + other_key] + bg_freq[other_key * nk + key];
                    penalty += bg * sp.dist;
                }
                if penalty < best_penalty {
                    best_penalty = penalty;
                }
            }
            if best_penalty != i64::MAX {
                total_min += best_penalty;
            }
        }

        total_min * self.stretch_weight
    }

    // ==================== Magic Rules ====================

    fn compute_rule_delta(
        &self,
        leader: CacheKey,
        output: CacheKey,
        magic_key: CacheKey,
        _keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        if leader >= num_keys || output >= num_keys || magic_key >= num_keys {
            return 0;
        }

        let leader_pos = match key_positions.get(leader).copied().flatten() {
            Some(pos) => pos,
            None => return 0,
        };
        let output_pos = key_positions.get(output).copied().flatten();
        let magic_pos = key_positions.get(magic_key).copied().flatten();

        let mut delta: i64 = 0;

        // Part 1: Full steal - Bigram A→B becomes A→M
        let full_steal_freq = bg_freq[leader * num_keys + output];
        if full_steal_freq != 0 {
            let old_w = output_pos.map_or(0, |p| self.stretch_bigram_weight(leader_pos, p));
            let new_w = magic_pos.map_or(0, |p| self.stretch_bigram_weight(leader_pos, p));
            delta += full_steal_freq * (new_w - old_w);
        }

        // Part 2: Partial steal - Bigrams B→C become M→C
        for c_key in 0..num_keys {
            let c_pos = match key_positions.get(c_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };
            let stolen_freq = tg_freq[leader][output][c_key];
            if stolen_freq == 0 { continue; }

            let old_w = output_pos.map_or(0, |p| self.stretch_bigram_weight(p, c_pos));
            let new_w = magic_pos.map_or(0, |p| self.stretch_bigram_weight(p, c_pos));
            delta += stolen_freq * (new_w - old_w);
        }

        delta
    }

    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        self.rule_delta.clear();
        let num_keys = self.num_keys;
        for leader in 0..num_keys {
            if key_positions.get(leader).copied().flatten().is_none() { continue; }
            for output in 0..num_keys {
                for magic_key in 0..num_keys {
                    let delta = self.compute_rule_delta(
                        leader, output, magic_key, keys, key_positions, bg_freq, tg_freq,
                    );
                    if delta != 0 {
                        self.rule_delta.insert((leader, output, magic_key), delta as i32);
                    }
                }
            }
        }
    }

    pub fn add_rule(
        &mut self,
        leader: CacheKey,
        output: CacheKey,
        magic_key: CacheKey,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        let rule_key = (magic_key, leader);

        let old_delta = if let Some(&(old_output, _stored_delta)) = self.active_rules.get(&rule_key) {
            if old_output == output { return 0; }
            self.compute_rule_delta(leader, old_output, magic_key, keys, key_positions, bg_freq, tg_freq)
        } else {
            0
        };

        let new_delta = if output != crate::cached_layout::EMPTY_KEY {
            self.compute_rule_delta(leader, output, magic_key, keys, key_positions, bg_freq, tg_freq)
        } else {
            0
        };

        let net_delta = new_delta - old_delta;

        if apply {
            self.active_rules.remove(&rule_key);
            if output != crate::cached_layout::EMPTY_KEY {
                self.active_rules.insert(rule_key, (output, new_delta));
            }
            self.magic_rule_score_delta += net_delta;
        }

        net_delta
    }

    pub fn reset_magic_deltas(&mut self) {
        self.active_rules.clear();
        self.magic_rule_score_delta = 0;
    }
}
