/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use crate::stats::Stats;
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
        self.total_score
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

    pub fn update_bigram(&mut self, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfb_score_per_finger[finger] += score_delta;
            self.sfb_freq_per_finger[finger] += freq_delta;
            self.total_score += score_delta * self.sfb_finger_weights[finger];
        }
    }

    pub fn update_skipgram(&mut self, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfs_score_per_finger[finger] += score_delta;
            self.sfs_freq_per_finger[finger] += freq_delta;
            self.total_score += score_delta * self.sfs_finger_weights[finger];
        }
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
            self.total_score
        } else {
            let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
            self.total_score + score_delta
        }
    }

    /// Swap keys at two positions. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
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
            self.total_score
        } else {
            let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
            let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
            self.total_score + score_a + score_b
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
}
