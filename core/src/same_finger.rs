/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use fxhash::FxHashMap as HashMap;

use crate::stats::Stats;
use crate::types::{CacheKey, CachePos};
use crate::weights::Weights;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

/// Stores same-finger pairs for a given position with pre-computed distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfPair {
    pub other_pos: usize,
    pub finger: usize,
    pub dist: i64,
}

/// Delta representing changes to SFCache state
#[derive(Default)]
struct SFDelta {
    total_score: i64,
    sfb_score_per_finger: [i64; 10],
    sfs_score_per_finger: [i64; 10],
    sfb_freq_per_finger: [i64; 10],
    sfs_freq_per_finger: [i64; 10],
}

impl SFDelta {
    fn combine(a: &SFDelta, b: &SFDelta) -> SFDelta {
        let mut result = SFDelta::default();
        result.total_score = a.total_score + b.total_score;
        for i in 0..10 {
            result.sfb_score_per_finger[i] = a.sfb_score_per_finger[i] + b.sfb_score_per_finger[i];
            result.sfs_score_per_finger[i] = a.sfs_score_per_finger[i] + b.sfs_score_per_finger[i];
            result.sfb_freq_per_finger[i] = a.sfb_freq_per_finger[i] + b.sfb_freq_per_finger[i];
            result.sfs_freq_per_finger[i] = a.sfs_freq_per_finger[i] + b.sfs_freq_per_finger[i];
        }
        result
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFCache {
    /// Running total weighted score (updated incrementally)
    total_score: i64,
    sfb_score_per_finger: Box<[i64; 10]>,
    sfs_score_per_finger: Box<[i64; 10]>,
    sfb_freq_per_finger: Box<[i64; 10]>,
    sfs_freq_per_finger: Box<[i64; 10]>,
    /// For each position, list of other positions on the same finger
    sf_pairs_per_key: Vec<Vec<SfPair>>,
    num_keys: usize,
    sfb_finger_weights: Box<[i64; 10]>,
    sfs_finger_weights: Box<[i64; 10]>,

    /// Magic rule tracking
    active_rules: HashMap<(CacheKey, CacheKey), (CacheKey, i64)>,
    magic_rule_score_delta: i64,
    rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,
}

impl SFCache {
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey], distances: &[Vec<i64>], num_keys: usize) -> Self {
        assert_eq!(fingers.len(), keyboard.len(), "finger len is not the same as keyboard len");

        let mut sf_pairs_per_key = Vec::with_capacity(fingers.len());
        for (i, finger1) in fingers.iter().enumerate() {
            let mut pairs = Vec::new();
            for (j, finger2) in fingers.iter().enumerate() {
                if finger1 == finger2 && i != j {
                    pairs.push(SfPair {
                        other_pos: j,
                        finger: *finger1 as usize,
                        dist: distances[i][j],
                    });
                }
            }
            sf_pairs_per_key.push(pairs);
        }

        Self {
            total_score: 0,
            sfb_score_per_finger: Box::new([0i64; 10]),
            sfs_score_per_finger: Box::new([0i64; 10]),
            sfb_freq_per_finger: Box::new([0i64; 10]),
            sfs_freq_per_finger: Box::new([0i64; 10]),
            sf_pairs_per_key,
            num_keys,
            sfb_finger_weights: Box::new([0i64; 10]),
            sfs_finger_weights: Box::new([0i64; 10]),
            active_rules: HashMap::default(),
            magic_rule_score_delta: 0,
            rule_delta: HashMap::default(),
        }
    }

    pub fn set_weights(&mut self, weights: &Weights) {
        for f in Finger::FINGERS {
            let fi = f as usize;
            let finger_weight = weights.fingers.get(f);
            // SFBs and SFS are penalties — negate so positive weight = worse score
            self.sfb_finger_weights[fi] = -(finger_weight * weights.sfbs);
            self.sfs_finger_weights[fi] = -(finger_weight * weights.sfs);
        }
        self.recompute_total_score();
    }

    fn recompute_total_score(&mut self) {
        self.total_score = 0;
        for fi in 0..10 {
            self.total_score += self.sfb_score_per_finger[fi] * self.sfb_finger_weights[fi]
                + self.sfs_score_per_finger[fi] * self.sfs_finger_weights[fi];
        }
    }

    #[inline]
    pub fn score(&self) -> i64 {
        self.total_score + self.magic_rule_score_delta
    }

    #[inline]
    pub fn pairs_for_pos(&self, pos: usize) -> &[SfPair] {
        &self.sf_pairs_per_key[pos]
    }

    #[inline]
    pub fn sfb_weight_for_finger(&self, finger: usize) -> i64 {
        self.sfb_finger_weights[finger]
    }

    #[inline]
    pub fn sfs_weight_for_finger(&self, finger: usize) -> i64 {
        self.sfs_finger_weights[finger]
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64, skipgram_total: f64) {
        let total_sfb: i64 = self.sfb_freq_per_finger.iter().sum();
        let total_sfs: i64 = self.sfs_freq_per_finger.iter().sum();
        let bigram_total_raw = bigram_total * 100.0;
        let skipgram_total_raw = skipgram_total * 100.0;

        stats.sfbs = total_sfb as f64 / bigram_total_raw;
        stats.sfs = total_sfs as f64 / skipgram_total_raw;

        for (i, &freq) in self.sfb_freq_per_finger.iter().enumerate() {
            stats.finger_sfbs[i] = freq as f64 / bigram_total_raw;
        }

        let total_bg_sg = bigram_total_raw + skipgram_total_raw;
        for (i, (&sfb_score, &sfs_score)) in self.sfb_score_per_finger.iter()
            .zip(self.sfs_score_per_finger.iter()).enumerate()
        {
            stats.weighted_finger_distance[i] = (sfb_score + sfs_score) as f64 / (total_bg_sg * 100.0);
        }
    }

    #[inline]
    fn is_same_finger(&self, p_a: usize, p_b: usize) -> Option<(usize, i64)> {
        self.sf_pairs_per_key[p_a]
            .iter()
            .find(|sf| sf.other_pos == p_b)
            .map(|sf| (sf.finger, sf.dist))
    }

    // ==================== Speculative Scoring ====================

    /// Speculative score for a key swap. No mutation.
    /// O(sf_pairs) per position — typically 0-3 pairs.
    #[inline]
    pub fn score_swap(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
        let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
        self.total_score + score_a + score_b + self.magic_rule_score_delta
    }

    /// Speculative score for replacing a key. No mutation.
    #[inline]
    pub fn score_replace(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, None, bg_freq, sg_freq);
        self.total_score + delta + self.magic_rule_score_delta
    }

    // ==================== Mutation ====================

    /// Replace key at position. Mutates running totals + per-finger stats.
    /// Returns the new score.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let delta = self.compute_replace_delta_full(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
        self.apply_delta(&delta);
        self.total_score + self.magic_rule_score_delta
    }

    /// Swap keys at two positions. Mutates running totals + per-finger stats.
    /// Returns the new score.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let delta_a = self.compute_replace_delta_full(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
        let delta_b = self.compute_replace_delta_full(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
        let combined = SFDelta::combine(&delta_a, &delta_b);
        self.apply_delta(&combined);
        self.total_score + self.magic_rule_score_delta
    }

    // ==================== Internal ====================

    fn apply_delta(&mut self, delta: &SFDelta) {
        self.total_score += delta.total_score;
        for i in 0..10 {
            self.sfb_score_per_finger[i] += delta.sfb_score_per_finger[i];
            self.sfs_score_per_finger[i] += delta.sfs_score_per_finger[i];
            self.sfb_freq_per_finger[i] += delta.sfb_freq_per_finger[i];
            self.sfs_freq_per_finger[i] += delta.sfs_freq_per_finger[i];
        }
    }

    #[inline]
    fn compute_replace_delta_score_only(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;
        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut total_delta: i64 = 0;

        for sf in &self.sf_pairs_per_key[pos] {
            let other_pos = sf.other_pos;
            if skip_pos == Some(other_pos) { continue; }
            let other_key = keys[other_pos];
            if other_key >= num_keys { continue; }

            let finger = sf.finger;
            let dist = sf.dist;
            let other_row = other_key * num_keys;

            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };
            let old_sg = if old_valid { sg_freq[old_row + other_key] } else { 0 };
            let new_sg = if new_valid { sg_freq[new_row + other_key] } else { 0 };
            let old_sg_rev = if old_valid { sg_freq[other_row + old_key] } else { 0 };
            let new_sg_rev = if new_valid { sg_freq[other_row + new_key] } else { 0 };

            let bg_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            let sg_delta = (new_sg - old_sg) + (new_sg_rev - old_sg_rev);

            total_delta += bg_delta * dist * self.sfb_finger_weights[finger]
                + sg_delta * dist * self.sfs_finger_weights[finger];
        }

        total_delta
    }

    fn compute_replace_delta_full(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> SFDelta {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;
        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut delta = SFDelta::default();

        for sf in &self.sf_pairs_per_key[pos] {
            let other_pos = sf.other_pos;
            if skip_pos == Some(other_pos) { continue; }
            let other_key = keys[other_pos];
            if other_key >= num_keys { continue; }

            let finger = sf.finger;
            let dist = sf.dist;
            let other_row = other_key * num_keys;

            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };
            let old_sg = if old_valid { sg_freq[old_row + other_key] } else { 0 };
            let new_sg = if new_valid { sg_freq[new_row + other_key] } else { 0 };
            let old_sg_rev = if old_valid { sg_freq[other_row + old_key] } else { 0 };
            let new_sg_rev = if new_valid { sg_freq[other_row + new_key] } else { 0 };

            let bg_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            let sg_delta = (new_sg - old_sg) + (new_sg_rev - old_sg_rev);

            delta.sfb_score_per_finger[finger] += bg_delta * dist;
            delta.sfb_freq_per_finger[finger] += bg_delta;
            delta.sfs_score_per_finger[finger] += sg_delta * dist;
            delta.sfs_freq_per_finger[finger] += sg_delta;

            delta.total_score += bg_delta * dist * self.sfb_finger_weights[finger]
                + sg_delta * dist * self.sfs_finger_weights[finger];
        }

        delta
    }

    #[inline]
    fn sf_bigram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            dist * self.sfb_finger_weights[finger]
        } else { 0 }
    }

    #[inline]
    fn sf_skipgram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            dist * self.sfs_finger_weights[finger]
        } else { 0 }
    }

    // ==================== Lower Bound ====================

    /// Compute a lower bound on the remaining SFB/SFS penalty from unplaced keys.
    ///
    /// For each unplaced key, finds the available position that minimizes the
    /// same-finger penalty with all already-placed keys, and sums these minimums.
    pub fn lower_bound_remaining(
        &self,
        unplaced_keys: &[usize],
        available_positions: &[usize],
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let nk = self.num_keys;
        let mut total_min: i64 = 0;

        for &key in unplaced_keys {
            if key >= nk { continue; }
            let mut best_penalty = i64::MAX;
            for &pos in available_positions {
                let mut penalty: i64 = 0;
                for sf in &self.sf_pairs_per_key[pos] {
                    let other_key = keys[sf.other_pos];
                    if other_key >= nk { continue; }
                    let bg = bg_freq[key * nk + other_key] + bg_freq[other_key * nk + key];
                    let sg = sg_freq[key * nk + other_key] + sg_freq[other_key * nk + key];
                    penalty += bg * sf.dist * self.sfb_finger_weights[sf.finger]
                        + sg * sf.dist * self.sfs_finger_weights[sf.finger];
                }
                if penalty < best_penalty {
                    best_penalty = penalty;
                }
            }
            if best_penalty != i64::MAX {
                total_min += best_penalty;
            }
        }

        total_min
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
        _sg_freq: &[i64],
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
            let old_w = output_pos.map_or(0, |p| self.sf_bigram_weight(leader_pos, p));
            let new_w = magic_pos.map_or(0, |p| self.sf_bigram_weight(leader_pos, p));
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

            let old_w = output_pos.map_or(0, |p| self.sf_bigram_weight(p, c_pos));
            let new_w = magic_pos.map_or(0, |p| self.sf_bigram_weight(p, c_pos));
            delta += stolen_freq * (new_w - old_w);
        }

        // Part 3: Skipgram partial steal - Skipgrams Z→B become Z→M
        for z_key in 0..num_keys {
            let z_pos = match key_positions.get(z_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };
            let stolen_freq = tg_freq[z_key][leader][output];
            if stolen_freq == 0 { continue; }

            let old_w = output_pos.map_or(0, |p| self.sf_skipgram_weight(z_pos, p));
            let new_w = magic_pos.map_or(0, |p| self.sf_skipgram_weight(z_pos, p));
            delta += stolen_freq * (new_w - old_w);
        }

        delta
    }

    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        self.rule_delta.clear();
        let num_keys = self.num_keys;
        for leader in 0..num_keys {
            if key_positions.get(leader).copied().flatten().is_none() { continue; }
            for output in 0..num_keys {
                for magic_key in 0..num_keys {
                    let delta = self.compute_rule_delta(
                        leader, output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq,
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
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        let rule_key = (magic_key, leader);

        // Prefer the stored delta (kept current by update_for_keyswap after swaps).
        // Fall back to recomputing only if stored delta is absent.
        let old_delta = if let Some(&(old_output, stored_delta)) = self.active_rules.get(&rule_key) {
            if old_output == output { return 0; }
            stored_delta
        } else {
            0
        };

        let new_delta = if output != crate::cached_layout::EMPTY_KEY {
            self.compute_rule_delta(leader, output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq)
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

    /// Reset all magic rule tracking. Called before recomputing from scratch.
    pub fn reset_magic_deltas(&mut self) {
        self.active_rules.clear();
        self.magic_rule_score_delta = 0;
    }

    /// Incremental magic-rule update after a keyswap. See TrigramCache version
    /// for rationale. Assumes `key_positions` is already updated for the swap.
    pub fn update_for_keyswap(
        &mut self,
        key_a: CacheKey,
        key_b: CacheKey,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        if self.active_rules.is_empty() {
            return;
        }
        let mut net_change: i64 = 0;
        let affected: Vec<((CacheKey, CacheKey), CacheKey, i64)> = self
            .active_rules
            .iter()
            .filter(|(&(magic_key, leader), &(output, _stored))| {
                leader == key_a || leader == key_b
                    || output == key_a || output == key_b
                    || magic_key == key_a || magic_key == key_b
            })
            .map(|(&k, &(output, stored))| (k, output, stored))
            .collect();

        for ((magic_key, leader), output, old_stored) in affected {
            let new_delta = self.compute_rule_delta(
                leader, output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq,
            );
            net_change += new_delta - old_stored;
            self.active_rules.insert((magic_key, leader), (output, new_delta));
        }
        self.magic_rule_score_delta += net_change;
    }
}
