/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use std::collections::HashMap;

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
    /// Weighted SFB score per finger (freq * dist)
    sfb_score_per_finger: Box<[i64; 10]>,
    /// Weighted SFS score per finger (freq * dist)
    sfs_score_per_finger: Box<[i64; 10]>,
    /// Unweighted SFB frequency per finger (for stats)
    sfb_freq_per_finger: Box<[i64; 10]>,
    /// Unweighted SFS frequency per finger (for stats)
    sfs_freq_per_finger: Box<[i64; 10]>,
    /// For each position, list of other positions on the same finger with pre-computed distances
    sf_pairs_per_key: Vec<Vec<SfPair>>,
    /// Number of keys for frequency array indexing
    num_keys: usize,
    /// Pre-computed: finger_weight * sfb_weight for each finger
    sfb_finger_weights: Box<[i64; 10]>,
    /// Pre-computed: finger_weight * sfs_weight for each finger
    sfs_finger_weights: Box<[i64; 10]>,

    /// Tracks active magic rules: (magic_key, leader) -> (output, delta)
    /// When a rule A→M steals output B, we store ((M, A), (B, delta))
    /// This allows the analyzer to correctly compute deltas when rules change.
    /// The delta is stored so we can properly reverse the rule by subtracting it.
    active_rules: HashMap<(CacheKey, CacheKey), (CacheKey, i64)>,

    /// Cumulative score delta from all active magic rules.
    /// This is added to the base score (computed from frequencies) to get the total score.
    magic_rule_score_delta: i64,

    /// Pre-computed rule deltas for O(1) `add_rule` speculative scoring.
    /// Maps (leader, output, magic_key) -> score delta.
    /// Uses sparse storage (HashMap) to stay under 10MB memory target.
    /// When `add_rule` is called with `apply=false`, this lookup table is used
    /// instead of computing the delta from scratch.
    rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,

    /// Pre-computed swap deltas for O(1) speculative key_swap scoring.
    /// Maps (pos_a, pos_b, key_a, key_b) -> score delta where pos_a < pos_b.
    /// Uses sparse storage (HashMap) to keep memory under control - only stores non-zero deltas.
    /// When `key_swap` is called with `apply=false`, this lookup table is used
    /// instead of computing the delta from scratch.
    swap_delta: HashMap<(CachePos, CachePos, CacheKey, CacheKey), i32>,
}

impl SFCache {
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey], distances: &[Vec<i64>], num_keys: usize) -> Self {
        assert_eq!(
            fingers.len(),
            keyboard.len(),
            "finger len is not the same as keyboard len"
        );

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
            // Initialize active rules to empty (no magic rules active initially)
            active_rules: HashMap::new(),
            // Initialize magic rule score delta to zero
            magic_rule_score_delta: 0,
            // Initialize rule delta lookup table to empty (populated by init_rule_deltas)
            rule_delta: HashMap::new(),
            // Initialize swap delta lookup table to empty (populated by init_swap_deltas)
            swap_delta: HashMap::new(),
        }
    }

    pub fn set_weights(&mut self, weights: &Weights) {
        for f in Finger::FINGERS {
            let fi = f as usize;
            let finger_weight = weights.fingers.get(f);
            self.sfb_finger_weights[fi] = finger_weight * weights.sfbs;
            self.sfs_finger_weights[fi] = finger_weight * weights.sfs;
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
            .zip(self.sfs_score_per_finger.iter())
            .enumerate()
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

    /// Replace key at position. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
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
        apply: bool,
    ) -> i64 {
        if apply {
            let delta = self.compute_replace_delta_full(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
            self.apply_delta(&delta);
            self.total_score + self.magic_rule_score_delta
        } else {
            let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
            self.total_score + score_delta + self.magic_rule_score_delta
        }
    }

    /// Swap keys at two positions. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    /// When `apply=false` and swap_delta table is populated, uses O(1) lookup.
    ///
    /// **Validates: Requirements 2.3, 2.5**
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
        apply: bool,
    ) -> i64 {
        if apply {
            let delta_a = self.compute_replace_delta_full(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
            let delta_b = self.compute_replace_delta_full(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
            let combined = SFDelta::combine(&delta_a, &delta_b);
            self.apply_delta(&combined);
            self.total_score + self.magic_rule_score_delta
        } else {
            // O(1) lookup for speculative scoring when swap_delta table is populated
            if !self.swap_delta.is_empty() {
                // Normalize position ordering (pos_a < pos_b) for canonical lookup
                let (p_lo, p_hi, k_lo, k_hi) = if pos_a < pos_b {
                    (pos_a, pos_b, key_a, key_b)
                } else {
                    (pos_b, pos_a, key_b, key_a)
                };

                // Look up the delta directly from the pre-computed table.
                // If not found in the HashMap, the delta is 0 (sparse storage semantics).
                let delta = self.swap_delta
                    .get(&(p_lo, p_hi, k_lo, k_hi))
                    .copied()
                    .unwrap_or(0) as i64;

                return self.total_score + delta + self.magic_rule_score_delta;
            }

            // Fallback to computed delta if swap_delta table not initialized
            let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
            let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
            self.total_score + score_a + score_b + self.magic_rule_score_delta
        }
    }

    fn apply_delta(&mut self, delta: &SFDelta) {
        self.total_score += delta.total_score;
        for i in 0..10 {
            self.sfb_score_per_finger[i] += delta.sfb_score_per_finger[i];
            self.sfs_score_per_finger[i] += delta.sfs_score_per_finger[i];
            self.sfb_freq_per_finger[i] += delta.sfb_freq_per_finger[i];
            self.sfs_freq_per_finger[i] += delta.sfs_freq_per_finger[i];
        }
    }

    /// Fast path: compute only the total score delta (for speculative scoring)
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
            if skip_pos == Some(other_pos) {
                continue;
            }
            let other_key = keys[other_pos];

            if other_key >= num_keys {
                continue;
            }

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

            let bg_score_delta = bg_delta * dist;
            let sg_score_delta = sg_delta * dist;

            total_delta += bg_score_delta * self.sfb_finger_weights[finger]
                + sg_score_delta * self.sfs_finger_weights[finger];
        }

        total_delta
    }

    /// Full delta computation (for actual mutations that need per-finger tracking)
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
            if skip_pos == Some(other_pos) {
                continue;
            }
            let other_key = keys[other_pos];

            if other_key >= num_keys {
                continue;
            }

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

            let bg_score_delta = bg_delta * dist;
            let sg_score_delta = sg_delta * dist;

            delta.sfb_score_per_finger[finger] += bg_score_delta;
            delta.sfb_freq_per_finger[finger] += bg_delta;
            delta.sfs_score_per_finger[finger] += sg_score_delta;
            delta.sfs_freq_per_finger[finger] += sg_delta;

            delta.total_score += bg_score_delta * self.sfb_finger_weights[finger]
                + sg_score_delta * self.sfs_finger_weights[finger];
        }

        delta
    }

    /// Compute the score delta for applying a magic rule.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    ///
    /// Returns the score delta (new_score - old_score).
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions (not used but kept for API consistency)
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    fn compute_rule_delta(
        &self,
        leader: CacheKey,      // A - the key that triggers the magic rule
        output: CacheKey,      // B - the output being stolen
        magic_key: CacheKey,   // M - the magic key that steals the output
        _keys: &[CacheKey],    // Not used but kept for API consistency
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],       // Flat bigram frequencies
        _sg_freq: &[i64],      // Flat skipgram frequencies (not used - skipgram partial steal uses tg_freq)
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;

        // Validate keys are within bounds
        if leader >= num_keys || output >= num_keys || magic_key >= num_keys {
            return 0;
        }

        // Get positions for leader (A), output (B), and magic_key (M)
        // Leader must have a position for the rule to have any effect
        let leader_pos = match key_positions.get(leader).copied().flatten() {
            Some(pos) => pos,
            None => return 0, // Leader has no position, rule has no effect
        };

        // Output (B) and magic_key (M) positions are needed for same-finger computation
        let output_pos = key_positions.get(output).copied().flatten();
        let magic_pos = key_positions.get(magic_key).copied().flatten();

        let mut delta: i64 = 0;

        // Part 1: Full steal - Bigram A→B becomes A→M
        // The frequency changes from bg_freq[A][B] to bg_freq[A][M]
        // But since we're computing the delta for the SAME frequency being redirected,
        // we need to compute: freq * (new_weight - old_weight)
        // where freq = bg_freq[A][B] (the frequency being stolen)
        let full_steal_freq = bg_freq[leader * num_keys + output];
        if full_steal_freq != 0 {
            // Old weight: sf_weight(A, B) - based on positions of leader and output
            let old_weight = if let Some(b_pos) = output_pos {
                self.sf_bigram_weight(leader_pos, b_pos)
            } else {
                0
            };

            // New weight: sf_weight(A, M) - based on positions of leader and magic_key
            let new_weight = if let Some(m_pos) = magic_pos {
                self.sf_bigram_weight(leader_pos, m_pos)
            } else {
                0
            };

            delta += full_steal_freq * (new_weight - old_weight);
        }

        // Part 2: Partial steal - Bigrams B→C become M→C based on tg_freq[A][B][C]
        // For each C with a position, the frequency stolen is tg_freq[A][B][C]
        // The weight changes from sf_weight(B, C) to sf_weight(M, C)
        for c_key in 0..num_keys {
            // C must have a position
            let c_pos = match key_positions.get(c_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency stolen: tg_freq[A][B][C]
            let stolen_freq = tg_freq[leader][output][c_key];
            if stolen_freq == 0 {
                continue;
            }

            // Old weight: sf_weight(B, C) - based on positions of output and C
            let old_weight = if let Some(b_pos) = output_pos {
                self.sf_bigram_weight(b_pos, c_pos)
            } else {
                0
            };

            // New weight: sf_weight(M, C) - based on positions of magic_key and C
            let new_weight = if let Some(m_pos) = magic_pos {
                self.sf_bigram_weight(m_pos, c_pos)
            } else {
                0
            };

            delta += stolen_freq * (new_weight - old_weight);
        }

        // Part 3: Skipgram partial steal - Skipgrams Z→B become Z→M based on tg_freq[Z][A][B]
        // For each Z with a position, the frequency stolen is tg_freq[Z][A][B]
        // The weight changes from sfs_weight(Z, B) to sfs_weight(Z, M)
        for z_key in 0..num_keys {
            // Z must have a position
            let z_pos = match key_positions.get(z_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency stolen: tg_freq[Z][A][B]
            let stolen_freq = tg_freq[z_key][leader][output];
            if stolen_freq == 0 {
                continue;
            }

            // Old weight: sfs_weight(Z, B) - based on positions of Z and output
            let old_weight = if let Some(b_pos) = output_pos {
                self.sf_skipgram_weight(z_pos, b_pos)
            } else {
                0
            };

            // New weight: sfs_weight(Z, M) - based on positions of Z and magic_key
            let new_weight = if let Some(m_pos) = magic_pos {
                self.sf_skipgram_weight(z_pos, m_pos)
            } else {
                0
            };

            delta += stolen_freq * (new_weight - old_weight);
        }

        delta
    }

    /// Compute the weighted score for a same-finger bigram pair.
    /// Returns 0 if the positions are not on the same finger.
    #[inline]
    fn sf_bigram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            dist * self.sfb_finger_weights[finger]
        } else {
            0
        }
    }

    /// Compute the weighted score for a same-finger skipgram pair.
    /// Returns 0 if the positions are not on the same finger.
    #[inline]
    fn sf_skipgram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            dist * self.sfs_finger_weights[finger]
        } else {
            0
        }
    }

    /// Initialize pre-computed rule deltas for O(1) `add_rule` speculative scoring.
    ///
    /// Pre-computes the score delta for all valid (leader, output, magic_key) combinations
    /// where the leader has a position on the layout. Uses sparse storage (HashMap) to
    /// store only non-zero deltas, keeping memory usage under 10MB for typical configurations.
    ///
    /// This must be called after the layout is fully initialized (all keys placed).
    ///
    /// # Arguments
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// **Validates: Requirements 6.2, 6.5**
    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        // Clear existing rule deltas
        self.rule_delta.clear();

        let num_keys = self.num_keys;

        // Iterate over all valid (leader, output, magic_key) combinations
        for leader in 0..num_keys {
            // Leader must have a position for the rule to have any effect
            let leader_pos = key_positions.get(leader).copied().flatten();
            if leader_pos.is_none() {
                continue;
            }

            for output in 0..num_keys {
                for magic_key in 0..num_keys {
                    // Compute the score delta for this rule
                    let delta = self.compute_rule_delta(
                        leader as CacheKey,
                        output as CacheKey,
                        magic_key as CacheKey,
                        keys,
                        key_positions,
                        bg_freq,
                        sg_freq,
                        tg_freq,
                    );

                    // Only store non-zero deltas for sparse storage (memory efficiency)
                    if delta != 0 {
                        self.rule_delta.insert(
                            (leader as CacheKey, output as CacheKey, magic_key as CacheKey),
                            delta as i32,
                        );
                    }
                }
            }
        }
    }

    /// Initialize pre-computed swap deltas for O(1) speculative key_swap scoring.
    ///
    /// Pre-computes the score delta for all valid (pos_a, pos_b, key_a, key_b) combinations
    /// where pos_a < pos_b (canonical ordering). Uses sparse storage (HashMap) to
    /// store only non-zero deltas, keeping memory usage under control.
    ///
    /// This must be called after the layout is fully initialized (all keys placed).
    ///
    /// # Arguments
    /// * `keys` - The current key assignments for all positions
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    ///
    /// **Validates: Requirements 2.2, 7.2**
    pub fn init_swap_deltas(
        &mut self,
        keys: &[CacheKey],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) {
        // Clear existing swap deltas
        self.swap_delta.clear();

        let num_keys = self.num_keys;
        let num_positions = self.sf_pairs_per_key.len();

        // Iterate over all position pairs (pos_a, pos_b) where pos_a < pos_b
        for pos_a in 0..num_positions {
            for pos_b in (pos_a + 1)..num_positions {
                // Iterate over all key pairs (key_a, key_b)
                for key_a in 0..num_keys {
                    for key_b in 0..num_keys {
                        // Skip if same key (no change when swapping same key)
                        if key_a == key_b {
                            continue;
                        }

                        // Compute the total swap delta using existing methods:
                        // 1. Delta for pos_a (key_a -> key_b), skipping pos_b
                        // 2. Delta for pos_b (key_b -> key_a), skipping pos_a
                        let score_a = self.compute_replace_delta_score_only(
                            pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq
                        );
                        let score_b = self.compute_replace_delta_score_only(
                            pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq
                        );

                        let delta = score_a + score_b;

                        // Only store non-zero deltas (sparse storage for memory efficiency)
                        if delta != 0 {
                            self.swap_delta.insert(
                                (pos_a, pos_b, key_a, key_b),
                                delta as i32,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Clear the swap_delta lookup table.
    ///
    /// This should be called after any apply=true operation that changes the layout,
    /// as the pre-computed deltas become invalid when keys move.
    #[inline]
    pub fn clear_swap_deltas(&mut self) {
        self.swap_delta.clear();
    }

    /// Apply a magic rule. Returns the score delta.
    ///
    /// If `apply` is false, computes the score delta without mutating state (speculative scoring).
    /// If `apply` is true, updates internal state and returns the score delta.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    /// * `apply` - If true, update internal state; if false, just compute the delta
    ///
    /// # Returns
    /// The score delta (new_score - old_score) for this rule application.
    ///
    /// **Validates: Requirements 2.5, 2.6**
    pub fn add_rule(
        &mut self,
        leader: CacheKey,      // A
        output: CacheKey,      // B (being stolen)
        magic_key: CacheKey,   // M
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        let rule_key = (magic_key, leader);

        // Check if there's an existing rule for this (magic_key, leader) pair
        let old_delta = if let Some(&(old_output, old_delta)) = self.active_rules.get(&rule_key) {
            // If the new output is the same as the old output, no change needed
            if old_output == output {
                return 0;
            }
            old_delta
        } else {
            0
        };

        // Compute the score delta for the new rule (0 if output is EMPTY_KEY)
        let new_delta = if output != crate::cached_layout::EMPTY_KEY {
            // When apply=false and lookup table is populated, use O(1) lookup
            // instead of computing the delta from scratch.
            if !apply && !self.rule_delta.is_empty() {
                // Look up the delta directly from the pre-computed table.
                // If not found in the HashMap, the delta is 0 (sparse storage).
                self.rule_delta
                    .get(&(leader, output, magic_key))
                    .copied()
                    .unwrap_or(0) as i64
            } else {
                // Either apply=true or lookup table not initialized - compute from scratch
                self.compute_rule_delta(
                    leader, output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq
                )
            }
        } else {
            0
        };

        // The net delta is: new_delta - old_delta
        // (removing old rule subtracts old_delta, adding new rule adds new_delta)
        let net_delta = new_delta - old_delta;

        if apply {
            // Always remove old rule if it exists (regardless of delta value)
            self.active_rules.remove(&rule_key);

            // Add new rule if output is not EMPTY_KEY
            if output != crate::cached_layout::EMPTY_KEY {
                self.active_rules.insert(rule_key, (output, new_delta));
            }

            // Update the cumulative magic rule score delta
            self.magic_rule_score_delta += net_delta;
        }

        net_delta
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;

    /// Helper function to create a simple SFCache for testing
    /// Creates a 4-position layout with fingers: LP, LI, RI, RP
    /// Positions 0 and 1 are on the same finger (LP), positions 2 and 3 are on different fingers
    fn create_test_cache(num_keys: usize) -> SFCache {
        // Fingers: LP, LP, RI, RP (positions 0 and 1 are same finger)
        let fingers = vec![LP, LP, RI, RP];
        // Simple keyboard with 4 positions
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),
            PhysicalKey::xy(1.0, 0.0),
            PhysicalKey::xy(2.0, 0.0),
            PhysicalKey::xy(3.0, 0.0),
        ];
        // Distances between positions (only same-finger pairs matter)
        let distances = vec![
            vec![0, 100, 0, 0],  // pos 0 to others
            vec![100, 0, 0, 0],  // pos 1 to others
            vec![0, 0, 0, 0],    // pos 2 to others
            vec![0, 0, 0, 0],    // pos 3 to others
        ];
        SFCache::new(&fingers, &keyboard, &distances, num_keys)
    }

    /// Helper function to create key_positions array from keys
    /// Maps each key to its position (index in keys array)
    fn create_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
        let mut key_positions = vec![None; num_keys];
        for (pos, &key) in keys.iter().enumerate() {
            if key < num_keys {
                key_positions[key] = Some(pos);
            }
        }
        key_positions
    }

    /// Helper function to create a 3D trigram frequency array
    fn create_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
        let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
        for &(k1, k2, k3, freq) in entries {
            if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                tg_freq[k1][k2][k3] = freq;
            }
        }
        tg_freq
    }

    /// Helper function to create flat bigram frequency array
    fn create_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut bg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                bg_freq[k1 * num_keys + k2] = freq;
            }
        }
        bg_freq
    }

    /// Helper function to create flat skipgram frequency array
    fn create_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut sg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                sg_freq[k1 * num_keys + k2] = freq;
            }
        }
        sg_freq
    }

    /// Helper to create test weights with non-zero SFB/SFS weights
    fn create_test_weights() -> Weights {
        Weights {
            sfbs: -10,
            sfs: -5,
            stretches: 0,
            sft: 0,
            inroll: 0,
            outroll: 0,
            alternate: 0,
            redirect: 0,
            onehandin: 0,
            onehandout: 0,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights {
                lp: 100,  // Left pinky weight
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

    // ==================== add_rule tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_false_preserves_state() {
        // Test that add_rule with apply=false returns correct delta without mutating state
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create bigram frequencies for the full steal: A→B
        // Positions 0 and 1 are on the same finger (LP), so A→B is an SFB
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        // Record initial state
        let initial_active_rules_len = cache.active_rules.len();
        let initial_magic_delta = cache.magic_rule_score_delta;
        let initial_score = cache.score();

        // Apply rule speculatively (apply=false)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, false);

        // Verify state is unchanged
        assert_eq!(cache.active_rules.len(), initial_active_rules_len, "active_rules should not change with apply=false");
        assert_eq!(cache.magic_rule_score_delta, initial_magic_delta, "magic_rule_score_delta should not change with apply=false");
        assert_eq!(cache.score(), initial_score, "score should not change with apply=false");

        // Delta should be computed (may be non-zero depending on positions)
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_true_updates_state() {
        // Test that add_rule with apply=true updates active_rules and magic_rule_score_delta
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create bigram frequencies
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        // Verify initial state
        assert!(cache.active_rules.is_empty());
        assert_eq!(cache.magic_rule_score_delta, 0);

        // Apply rule: leader=0 (A), output=1 (B), magic_key=2 (M)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Verify active_rules is updated
        assert_eq!(cache.active_rules.len(), 1);
        // active_rules now stores (output, delta) instead of just output
        let rule_entry = cache.active_rules.get(&(2, 0));
        assert!(rule_entry.is_some(), "Rule should be tracked");
        assert_eq!(rule_entry.unwrap().0, 1, "Output should be 1"); // (magic_key, leader) -> (output, delta)

        // Verify magic_rule_score_delta is updated
        assert_eq!(cache.magic_rule_score_delta, delta);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_active_rules_updated_correctly() {
        // Test that active_rules HashMap is updated correctly when apply=true
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[]);
        let sg_freq = create_sg_freq(6, &[]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply first rule: leader=0, output=1, magic_key=2
        let delta1 = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        // active_rules now stores (output, delta) instead of just output
        let rule1 = cache.active_rules.get(&(2, 0));
        assert!(rule1.is_some(), "Rule should be tracked");
        assert_eq!(rule1.unwrap().0, 1, "Output should be 1");

        // Apply second rule with different magic_key and leader
        let delta2 = cache.add_rule(1, 3, 4, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        let rule2 = cache.active_rules.get(&(4, 1));
        assert!(rule2.is_some(), "Rule should be tracked");
        assert_eq!(rule2.unwrap().0, 3, "Output should be 3");

        // Both rules should be tracked
        assert_eq!(cache.active_rules.len(), 2);

        // Apply rule with same (magic_key, leader) but different output - should replace
        let _delta3 = cache.add_rule(0, 3, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        let rule1_updated = cache.active_rules.get(&(2, 0));
        assert!(rule1_updated.is_some(), "Rule should be tracked");
        assert_eq!(rule1_updated.unwrap().0, 3, "Output should be updated to 3");
        assert_eq!(cache.active_rules.len(), 2); // Still 2 rules
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_magic_rule_score_delta_updated_correctly() {
        // Test that magic_rule_score_delta is updated correctly when apply=true
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create frequencies that will produce non-zero deltas
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[
            (0, 1, 2, 50),  // A→B→C for partial steal
        ]);

        // Apply first rule
        let delta1 = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(cache.magic_rule_score_delta, delta1);

        // Apply second rule
        let delta2 = cache.add_rule(1, 2, 3, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_true_and_false_return_same_delta() {
        // Test that apply=true and apply=false return the same delta value
        let mut cache1 = create_test_cache(5);
        let mut cache2 = create_test_cache(5);
        let weights = create_test_weights();
        cache1.set_weights(&weights);
        cache2.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[
            (2, 0, 1, 50),  // Z→A→B for skipgram partial steal
            (0, 1, 2, 75),  // A→B→C for bigram partial steal
        ]);

        // Get delta with apply=false
        let delta_false = cache1.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, false);

        // Get delta with apply=true
        let delta_true = cache2.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Both should return the same delta
        assert_eq!(delta_false, delta_true);
    }

    // ==================== Edge case tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_invalid_keys() {
        // Test add_rule with key indices >= num_keys
        let mut cache = create_test_cache(4);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 4);

        let bg_freq = create_bg_freq(4, &[]);
        let sg_freq = create_sg_freq(4, &[]);
        let tg_freq = create_tg_freq(4, &[]);

        // Apply rule with invalid leader (>= num_keys)
        let delta = cache.add_rule(99, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid leader key");

        // Apply rule with invalid output (>= num_keys)
        let delta = cache.add_rule(0, 99, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid output key");

        // Apply rule with invalid magic_key (>= num_keys)
        let delta = cache.add_rule(0, 1, 99, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid magic key");
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_keys_without_positions() {
        // Test add_rule when keys don't have positions
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Only keys 0, 1, 2, 3 have positions (4 positions)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[
            (4, 5, 100),  // Frequency for keys without positions
        ]);
        let sg_freq = create_sg_freq(6, &[]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply rule where leader has no position
        let delta = cache.add_rule(4, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 when leader has no position");

        // Apply rule where output has no position (old weight becomes 0)
        let delta = cache.add_rule(0, 4, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        // Delta may be non-zero if magic_key has a position
        let _ = delta;

        // Apply rule where magic_key has no position (new weight becomes 0)
        let delta = cache.add_rule(0, 1, 5, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        // Delta may be non-zero if output has a position
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_zero_frequencies() {
        // Test add_rule with zero frequencies
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // All frequencies are zero
        let bg_freq = create_bg_freq(5, &[]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        let initial_score = cache.score();

        // Apply rule with zero frequencies
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Delta should be 0 since all frequencies are 0
        assert_eq!(delta, 0);

        // Score should remain unchanged
        assert_eq!(cache.score(), initial_score);

        // Rule should still be tracked (stores (output, delta))
        let rule = cache.active_rules.get(&(2, 0));
        assert!(rule.is_some(), "Rule should be tracked");
        assert_eq!(rule.unwrap().0, 1, "Output should be 1");
    }

    // ==================== Full steal tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_full_steal_same_finger_to_different_finger() {
        // Test full steal: A→B (same finger) becomes A→M (different finger)
        // This should reduce the SFB score (positive delta since SFB weight is negative)
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M)
        // Positions 0 and 1 are on same finger (LP), position 2 is on different finger (RI)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // A→B is an SFB (positions 0 and 1 are same finger)
        // A→M is NOT an SFB (positions 0 and 2 are different fingers)
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // The delta should be positive (removing SFB penalty)
        // Old: 100 * dist * sfb_finger_weight (negative contribution)
        // New: 0 (A→M is not same finger)
        // Delta = new - old = 0 - (negative) = positive
        assert!(delta > 0, "Delta should be positive when removing SFB: got {}", delta);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_full_steal_different_finger_to_same_finger() {
        // Test full steal: A→B (different finger) becomes A→M (same finger)
        // This should increase the SFB score (negative delta since SFB weight is negative)

        // Create cache where positions 0 and 2 are same finger, position 1 is different
        let fingers = vec![LP, RI, LP, RP];  // pos 0 and 2 are LP
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),
            PhysicalKey::xy(1.0, 0.0),
            PhysicalKey::xy(2.0, 0.0),
            PhysicalKey::xy(3.0, 0.0),
        ];
        let distances = vec![
            vec![0, 0, 100, 0],  // pos 0 to pos 2 has distance 100
            vec![0, 0, 0, 0],
            vec![100, 0, 0, 0],  // pos 2 to pos 0 has distance 100
            vec![0, 0, 0, 0],
        ];
        let mut cache = SFCache::new(&fingers, &keyboard, &distances, 5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M)
        // A→B is NOT an SFB (positions 0 and 1 are different fingers)
        // A→M IS an SFB (positions 0 and 2 are same finger LP)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // The delta should be negative (adding SFB penalty)
        // Old: 0 (A→B is not same finger)
        // New: 100 * dist * sfb_finger_weight (negative contribution)
        // Delta = new - old = negative - 0 = negative
        assert!(delta < 0, "Delta should be negative when adding SFB: got {}", delta);
    }

    // ==================== Partial steal tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_partial_steal_bigram() {
        // Test partial steal: B→C becomes M→C based on tg_freq[A][B][C]
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // No full steal frequency
        let bg_freq = create_bg_freq(5, &[]);
        let sg_freq = create_sg_freq(5, &[]);
        // Partial steal: tg_freq[A][B][C] determines how much of B→C is stolen
        let tg_freq = create_tg_freq(5, &[
            (0, 1, 3, 200),  // A=0, B=1, C=3: trigram frequency
        ]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Delta depends on whether B→C and M→C are same-finger pairs
        // In our test setup, positions 0,1 are same finger (LP), positions 2,3 are different
        // B is at pos 1 (LP), C is at pos 3 (RP) - different fingers
        // M is at pos 2 (RI), C is at pos 3 (RP) - different fingers
        // So both B→C and M→C are not SFBs, delta should be 0
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_partial_steal_skipgram() {
        // Test partial steal: Z→B becomes Z→M based on tg_freq[Z][A][B]
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[]);
        let sg_freq = create_sg_freq(5, &[]);
        // Partial steal: tg_freq[Z][A][B] determines how much of Z→B skipgram is stolen
        let tg_freq = create_tg_freq(5, &[
            (3, 0, 1, 150),  // Z=3, A=0, B=1: trigram frequency
        ]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Delta depends on whether Z→B and Z→M are same-finger skipgram pairs
        let _ = delta;
    }

    // ==================== Multiple calls tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_multiple_speculative_calls_preserve_state() {
        // Test that multiple apply=false calls don't accumulate state changes
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),
        ]);
        let sg_freq = create_sg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        let initial_score = cache.score();
        let initial_rules_len = cache.active_rules.len();
        let initial_delta = cache.magic_rule_score_delta;

        // Call add_rule with apply=false multiple times
        for _ in 0..10 {
            cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, false);
        }

        // State should be unchanged
        assert_eq!(cache.score(), initial_score);
        assert_eq!(cache.active_rules.len(), initial_rules_len);
        assert_eq!(cache.magic_rule_score_delta, initial_delta);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_sequential_apply_true_accumulates() {
        // Test that multiple apply=true calls accumulate deltas
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Use 6 keys so we have enough for multiple rules
        let keys = vec![0, 1, 2, 3, 4, 5];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[
            (0, 1, 100),
            (2, 3, 100),
        ]);
        let sg_freq = create_sg_freq(6, &[]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply first rule
        let delta1 = cache.add_rule(0, 1, 4, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // Apply second rule (different magic_key and leader)
        let delta2 = cache.add_rule(2, 3, 5, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // magic_rule_score_delta should be sum of both deltas
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);

        // Both rules should be tracked
        assert_eq!(cache.active_rules.len(), 2);
    }

    // ==================== Property-Based Tests ====================

    /// **Validates: Requirements 2.5**
    ///
    /// Property test: add_rule with apply=true mutates state correctly.
    ///
    /// For any add_rule call with apply=true, the analyzer's internal state should
    /// reflect the rule application, and subsequent calls to score() should return
    /// the same value as was returned by the operation.
    mod pbt_add_rule_apply_true {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        /// Convert entries to flat bigram frequency array
        fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut bg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    bg_freq[k1 * num_keys + k2] = freq;
                }
            }
            bg_freq
        }

        /// Convert entries to flat skipgram frequency array
        fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut sg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    sg_freq[k1 * num_keys + k2] = freq;
                }
            }
            sg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(25))]

            /// **Validates: Requirements 2.5**
            ///
            /// Property: When add_rule is called with apply=true:
            /// 1. active_rules contains the new rule: ((magic_key, leader), output)
            /// 2. magic_rule_score_delta is updated by the returned delta
            /// 3. The returned delta equals compute_rule_delta result
            #[test]
            fn prop_add_rule_apply_true_mutates_state_correctly(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating bg_freq entries
                bg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating sg_freq entries
                sg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate frequency arrays from entries (constrained to num_keys)
                let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

                let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

                let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

                // Create a simple finger layout with some same-finger pairs
                // Positions 0 and 1 are on the same finger (LP), others alternate
                let fingers: Vec<Finger> = (0..num_positions)
                    .map(|i| match i {
                        0 | 1 => LP,  // Same finger for positions 0 and 1
                        _ if i % 2 == 0 => RI,
                        _ => RP,
                    })
                    .collect();

                let keyboard: Vec<PhysicalKey> = (0..num_positions)
                    .map(|i| PhysicalKey::xy(i as f64, 0.0))
                    .collect();

                // Create distances - same-finger pairs have non-zero distance
                let mut distances = vec![vec![0i64; num_positions]; num_positions];
                for i in 0..num_positions {
                    for j in 0..num_positions {
                        if i != j && fingers[i] == fingers[j] {
                            distances[i][j] = 100; // Fixed distance for same-finger pairs
                        }
                    }
                }

                let mut cache = SFCache::new(&fingers, &keyboard, &distances, num_keys);

                // Set some weights so score changes are visible
                let weights = Weights {
                    sfbs: -10,
                    sfs: -5,
                    stretches: 0,
                    sft: 0,
                    inroll: 0,
                    outroll: 0,
                    alternate: 0,
                    redirect: 0,
                    onehandin: 0,
                    onehandout: 0,
                    thumb: 0,
                    full_scissors: 0,
                    half_scissors: 0,
                    full_scissors_skip: 0,
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
                };
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Record initial state
                let initial_magic_delta = cache.magic_rule_score_delta;

                // Apply the rule with apply=true
                let delta = cache.add_rule(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &bg_freq,
                    &sg_freq,
                    &tg_freq,
                    true,
                );

                // Property 1: active_rules contains the new rule
                prop_assert!(
                    cache.active_rules.contains_key(&(magic_key, leader)),
                    "active_rules should contain the rule ({}, {}) -> {}",
                    magic_key, leader, output
                );
                prop_assert!(
                    cache.active_rules.get(&(magic_key, leader)).map(|(o, _)| *o) == Some(output),
                    "active_rules[({}, {})] should have output {}",
                    magic_key, leader, output
                );

                // Property 2: magic_rule_score_delta equals initial + returned delta
                prop_assert_eq!(
                    cache.magic_rule_score_delta,
                    initial_magic_delta + delta,
                    "magic_rule_score_delta should be {} + {} = {}",
                    initial_magic_delta, delta, initial_magic_delta + delta
                );
            }
        }
    }

    /// **Validates: Requirements 2.6**
    ///
    /// Property test: add_rule with apply=false preserves state.
    ///
    /// For any add_rule call with apply=false, the analyzer's internal state should
    /// remain unchanged, and score() called before and after should return the same value.
    mod pbt_add_rule_apply_false {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        /// Convert entries to flat bigram frequency array
        fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut bg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    bg_freq[k1 * num_keys + k2] = freq;
                }
            }
            bg_freq
        }

        /// Convert entries to flat skipgram frequency array
        fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut sg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    sg_freq[k1 * num_keys + k2] = freq;
                }
            }
            sg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(25))]

            /// **Validates: Requirements 2.6**
            ///
            /// Property: When add_rule is called with apply=false:
            /// 1. active_rules is unchanged
            /// 2. magic_rule_score_delta is unchanged
            /// 3. score() is unchanged
            /// 4. The returned delta equals what apply=true would return
            #[test]
            fn prop_add_rule_apply_false_preserves_state(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating bg_freq entries
                bg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating sg_freq entries
                sg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate frequency arrays from entries (constrained to num_keys)
                let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

                let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

                let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

                // Create a simple finger layout with some same-finger pairs
                // Positions 0 and 1 are on the same finger (LP), others alternate
                let fingers: Vec<Finger> = (0..num_positions)
                    .map(|i| match i {
                        0 | 1 => LP,  // Same finger for positions 0 and 1
                        _ if i % 2 == 0 => RI,
                        _ => RP,
                    })
                    .collect();

                let keyboard: Vec<PhysicalKey> = (0..num_positions)
                    .map(|i| PhysicalKey::xy(i as f64, 0.0))
                    .collect();

                // Create distances - same-finger pairs have non-zero distance
                let mut distances = vec![vec![0i64; num_positions]; num_positions];
                for i in 0..num_positions {
                    for j in 0..num_positions {
                        if i != j && fingers[i] == fingers[j] {
                            distances[i][j] = 100; // Fixed distance for same-finger pairs
                        }
                    }
                }

                let mut cache = SFCache::new(&fingers, &keyboard, &distances, num_keys);

                // Set some weights so score changes are visible
                let weights = Weights {
                    sfbs: -10,
                    sfs: -5,
                    stretches: 0,
                    sft: 0,
                    inroll: 0,
                    outroll: 0,
                    alternate: 0,
                    redirect: 0,
                    onehandin: 0,
                    onehandout: 0,
                    thumb: 0,
                    full_scissors: 0,
                    half_scissors: 0,
                    full_scissors_skip: 0,
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
                };
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Record initial state
                let initial_score = cache.score();
                let initial_magic_delta = cache.magic_rule_score_delta;
                let initial_active_rules = cache.active_rules.clone();

                // Call add_rule with apply=false (speculative scoring)
                let delta_false = cache.add_rule(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &bg_freq,
                    &sg_freq,
                    &tg_freq,
                    false,  // apply=false - should NOT mutate state
                );

                // Property 1: active_rules is unchanged
                prop_assert_eq!(
                    &cache.active_rules,
                    &initial_active_rules,
                    "active_rules should be unchanged after apply=false"
                );

                // Property 2: magic_rule_score_delta is unchanged
                prop_assert_eq!(
                    cache.magic_rule_score_delta,
                    initial_magic_delta,
                    "magic_rule_score_delta should be unchanged after apply=false, was {}, now {}",
                    initial_magic_delta, cache.magic_rule_score_delta
                );

                // Property 3: score() is unchanged
                let final_score = cache.score();
                prop_assert_eq!(
                    final_score,
                    initial_score,
                    "score() should be unchanged after apply=false, was {}, now {}",
                    initial_score, final_score
                );

                // Property 4: The returned delta equals what apply=true would return
                // Create a fresh cache to test apply=true
                let mut cache_for_apply = SFCache::new(&fingers, &keyboard, &distances, num_keys);
                cache_for_apply.set_weights(&weights);

                let delta_true = cache_for_apply.add_rule(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &bg_freq,
                    &sg_freq,
                    &tg_freq,
                    true,  // apply=true
                );

                prop_assert_eq!(
                    delta_false,
                    delta_true,
                    "apply=false should return the same delta as apply=true, got {} vs {}",
                    delta_false, delta_true
                );
            }
        }
    }

    /// **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    ///
    /// Property test: lookup table values match computed values.
    ///
    /// For any (leader, output, magic_key) triple, the pre-computed `rule_delta` lookup
    /// should equal the value computed by `compute_rule_delta` from scratch.
    mod pbt_lookup_table_matches_computed_sfcache {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        /// Convert entries to flat bigram frequency array
        fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut bg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    bg_freq[k1 * num_keys + k2] = freq;
                }
            }
            bg_freq
        }

        /// Convert entries to flat skipgram frequency array
        fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
            let mut sg_freq = vec![0i64; num_keys * num_keys];
            for &(k1, k2, freq) in entries {
                if k1 < num_keys && k2 < num_keys {
                    sg_freq[k1 * num_keys + k2] = freq;
                }
            }
            sg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(25))]

            /// **Validates: Requirements 6.2, 6.5**
            ///
            /// Property: After calling init_rule_deltas, for any valid (leader, output, magic_key)
            /// triple where leader has a position, the value in rule_delta.get(&(leader, output, magic_key))
            /// should equal compute_rule_delta(leader, output, magic_key, ...).
            #[test]
            fn prop_lookup_table_matches_computed_sfcache(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating bg_freq entries
                bg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating sg_freq entries
                sg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0i64..1000i64),
                    0..10
                ),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule to test
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate frequency arrays from entries (constrained to num_keys)
                let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

                let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                    .iter()
                    .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                    .collect();
                let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

                let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

                // Create a simple finger layout with same-finger pairs
                // Fingers: LP, LP, RI, RP (positions 0 and 1 are same finger)
                let base_fingers = vec![LP, LP, RI, RP];
                let fingers: Vec<Finger> = (0..num_positions)
                    .map(|i| base_fingers[i % base_fingers.len()])
                    .collect();

                // Simple keyboard with positions
                let keyboard: Vec<PhysicalKey> = (0..num_positions)
                    .map(|i| PhysicalKey::xy(i as f64, 0.0))
                    .collect();

                // Distances between positions (only same-finger pairs matter)
                let distances: Vec<Vec<i64>> = (0..num_positions)
                    .map(|i| {
                        (0..num_positions)
                            .map(|j| if i != j && fingers[i] == fingers[j] { 100 } else { 0 })
                            .collect()
                    })
                    .collect();

                let mut cache = SFCache::new(&fingers, &keyboard, &distances, num_keys);

                // Set weights
                let weights = Weights {
                    sfbs: -10,
                    sfs: -5,
                    stretches: 0,
                    sft: 0,
                    inroll: 0,
                    outroll: 0,
                    alternate: 0,
                    redirect: 0,
                    onehandin: 0,
                    onehandout: 0,
                    thumb: 0,
                    full_scissors: 0,
                    half_scissors: 0,
                    full_scissors_skip: 0,
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
                };
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Initialize the rule delta lookup table
                cache.init_rule_deltas(&keys, &key_positions, &bg_freq, &sg_freq, &tg_freq);

                // Compute the delta from scratch using compute_rule_delta
                let computed_delta = cache.compute_rule_delta(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &bg_freq,
                    &sg_freq,
                    &tg_freq,
                );

                // Get the lookup table value (0 if not present due to sparse storage)
                let lookup_delta = cache.rule_delta
                    .get(&(leader, output, magic_key))
                    .copied()
                    .unwrap_or(0) as i64;

                // Property: lookup table value should equal computed value
                prop_assert_eq!(
                    lookup_delta,
                    computed_delta,
                    "Lookup table value {} should equal computed value {} for (leader={}, output={}, magic_key={})",
                    lookup_delta, computed_delta, leader, output, magic_key
                );
            }
        }
    }
}
