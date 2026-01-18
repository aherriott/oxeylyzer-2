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
    pub dist: i64,  // Pre-computed distance
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

        // Build same-finger pair lookup with pre-computed distances
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

    /// Set weights and pre-compute finger weight products
    pub fn set_weights(&mut self, weights: &Weights) {
        for f in Finger::FINGERS {
            let fi = f as usize;
            let finger_weight = weights.fingers.get(f);
            self.sfb_finger_weights[fi] = finger_weight * weights.sfbs;
            self.sfs_finger_weights[fi] = finger_weight * weights.sfs;
        }
        // Recompute total score with new weights
        self.recompute_total_score();
    }

    /// Recompute total score from per-finger scores
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
        // Total SFB/SFS frequencies (sum across all fingers)
        let total_sfb: i64 = self.sfb_freq_per_finger.iter().sum();
        let total_sfs: i64 = self.sfs_freq_per_finger.iter().sum();

        // Note: bigram_total and skipgram_total are already divided by 100 in AnalyzerData,
        // but the stored frequencies are raw counts (percentage * total / 100).
        // To get the correct percentage, we need to multiply the totals by 100.
        let bigram_total_raw = bigram_total * 100.0;
        let skipgram_total_raw = skipgram_total * 100.0;

        stats.sfbs = total_sfb as f64 / bigram_total_raw;
        stats.sfs = total_sfs as f64 / skipgram_total_raw;

        // Per-finger SFB frequencies
        for (i, &freq) in self.sfb_freq_per_finger.iter().enumerate() {
            stats.finger_sfbs[i] = freq as f64 / bigram_total_raw;
        }

        // Per-finger weighted distance (score / 100 to convert back from centiunits)
        let total_bg_sg = bigram_total_raw + skipgram_total_raw;
        for (i, (&sfb_score, &sfs_score)) in self.sfb_score_per_finger.iter()
            .zip(self.sfs_score_per_finger.iter())
            .enumerate()
        {
            stats.weighted_finger_distance[i] = (sfb_score + sfs_score) as f64 / (total_bg_sg * 100.0);
        }
    }

    /// Check if two positions are on the same finger, return (finger, dist) if so
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
            // Update running total
            self.total_score += score_delta * self.sfb_finger_weights[finger];
        }
    }

    pub fn update_skipgram(&mut self, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some((finger, dist)) = self.is_same_finger(p_a, p_b) {
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfs_score_per_finger[finger] += score_delta;
            self.sfs_freq_per_finger[finger] += freq_delta;
            // Update running total
            self.total_score += score_delta * self.sfs_finger_weights[finger];
        }
    }

    /// Copy scoring data from another SFCache. No allocations.
    #[inline]
    pub fn copy_from(&mut self, other: &SFCache) {
        self.total_score = other.total_score;
        *self.sfb_score_per_finger = *other.sfb_score_per_finger;
        *self.sfs_score_per_finger = *other.sfs_score_per_finger;
        *self.sfb_freq_per_finger = *other.sfb_freq_per_finger;
        *self.sfs_freq_per_finger = *other.sfs_freq_per_finger;
    }

    /// Replace key at position: update scores for changing from old_key to new_key.
    /// Use EMPTY_KEY for old_key when adding, or new_key when removing.
    /// `skip_pos` allows skipping a position (used by key_swap to avoid double-counting).
    /// `bg_freq` and `sg_freq` are flat arrays indexed by `a * num_keys + b`.
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
    ) {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        for sf in &self.sf_pairs_per_key[pos] {
            let other_pos = sf.other_pos;
            if skip_pos == Some(other_pos) {
                continue;
            }
            let finger = sf.finger;
            let dist = sf.dist;
            let other_key = keys[other_pos];

            // Skip if other_key is EMPTY_KEY
            if other_key >= num_keys {
                continue;
            }

            // Bigram: pos -> other_pos
            let old_bg = if old_valid { bg_freq[old_key * num_keys + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_key * num_keys + other_key] } else { 0 };
            let bg_delta = new_bg - old_bg;
            let bg_score_delta = bg_delta * dist;
            self.sfb_score_per_finger[finger] += bg_score_delta;
            self.sfb_freq_per_finger[finger] += bg_delta;
            self.total_score += bg_score_delta * self.sfb_finger_weights[finger];

            // Bigram: other_pos -> pos
            let old_bg_rev = if old_valid { bg_freq[other_key * num_keys + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_key * num_keys + new_key] } else { 0 };
            let bg_delta_rev = new_bg_rev - old_bg_rev;
            let bg_score_delta_rev = bg_delta_rev * dist;
            self.sfb_score_per_finger[finger] += bg_score_delta_rev;
            self.sfb_freq_per_finger[finger] += bg_delta_rev;
            self.total_score += bg_score_delta_rev * self.sfb_finger_weights[finger];

            // Skipgram: pos -> other_pos
            let old_sg = if old_valid { sg_freq[old_key * num_keys + other_key] } else { 0 };
            let new_sg = if new_valid { sg_freq[new_key * num_keys + other_key] } else { 0 };
            let sg_delta = new_sg - old_sg;
            let sg_score_delta = sg_delta * dist;
            self.sfs_score_per_finger[finger] += sg_score_delta;
            self.sfs_freq_per_finger[finger] += sg_delta;
            self.total_score += sg_score_delta * self.sfs_finger_weights[finger];

            // Skipgram: other_pos -> pos
            let old_sg_rev = if old_valid { sg_freq[other_key * num_keys + old_key] } else { 0 };
            let new_sg_rev = if new_valid { sg_freq[other_key * num_keys + new_key] } else { 0 };
            let sg_delta_rev = new_sg_rev - old_sg_rev;
            let sg_score_delta_rev = sg_delta_rev * dist;
            self.sfs_score_per_finger[finger] += sg_score_delta_rev;
            self.sfs_freq_per_finger[finger] += sg_delta_rev;
            self.total_score += sg_score_delta_rev * self.sfs_finger_weights[finger];
        }
    }

    /// Optimized key swap: update scores for swapping keys at pos_a and pos_b.
    /// Handles the direct pair between pos_a and pos_b specially to avoid double-counting.
    /// `bg_freq` and `sg_freq` are flat arrays indexed by `a * num_keys + b`.
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
    ) {
        let num_keys = self.num_keys;

        // Handle the direct pair between pos_a and pos_b (if same finger and both keys valid)
        if key_a < num_keys && key_b < num_keys {
            if let Some((finger, dist)) = self.is_same_finger(pos_a, pos_b) {
                // Bigram a->b: was (key_a, key_b), now (key_b, key_a)
                let bg_delta_ab = bg_freq[key_b * num_keys + key_a] - bg_freq[key_a * num_keys + key_b];
                let bg_score_ab = bg_delta_ab * dist;
                self.sfb_score_per_finger[finger] += bg_score_ab;
                self.sfb_freq_per_finger[finger] += bg_delta_ab;
                self.total_score += bg_score_ab * self.sfb_finger_weights[finger];

                // Bigram b->a: was (key_b, key_a), now (key_a, key_b)
                let bg_delta_ba = bg_freq[key_a * num_keys + key_b] - bg_freq[key_b * num_keys + key_a];
                let bg_score_ba = bg_delta_ba * dist;
                self.sfb_score_per_finger[finger] += bg_score_ba;
                self.sfb_freq_per_finger[finger] += bg_delta_ba;
                self.total_score += bg_score_ba * self.sfb_finger_weights[finger];

                // Skipgram a->b
                let sg_delta_ab = sg_freq[key_b * num_keys + key_a] - sg_freq[key_a * num_keys + key_b];
                let sg_score_ab = sg_delta_ab * dist;
                self.sfs_score_per_finger[finger] += sg_score_ab;
                self.sfs_freq_per_finger[finger] += sg_delta_ab;
                self.total_score += sg_score_ab * self.sfs_finger_weights[finger];

                // Skipgram b->a
                let sg_delta_ba = sg_freq[key_a * num_keys + key_b] - sg_freq[key_b * num_keys + key_a];
                let sg_score_ba = sg_delta_ba * dist;
                self.sfs_score_per_finger[finger] += sg_score_ba;
                self.sfs_freq_per_finger[finger] += sg_delta_ba;
                self.total_score += sg_score_ba * self.sfs_finger_weights[finger];
            }
        }

        // Replace key at pos_a (key_a -> key_b), skipping pos_b
        self.replace_key(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
        // Replace key at pos_b (key_b -> key_a), skipping pos_a
        self.replace_key(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
    }
}
