/*
 **************************************
 *            Stretches
 **************************************
 */

use crate::dist::{dx_dy, x_overlap};
use crate::stats::Stats;
use crate::types::CachePos;
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
}

impl StretchCache {
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger], num_keys: usize) -> Self {
        let len = keyboard.len();

        // Compute stretch distances
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

        // Build stretch pair lookup with pre-computed distances
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
        }
    }

    /// Set weights
    pub fn set_weights(&mut self, weights: &Weights) {
        self.stretch_weight = weights.stretches;
    }

    /// Compute stretch distance for a key pair.
    /// Returns 0 if not a stretch (same finger or different hand).
    fn compute_stretch(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> i64 {
        // Stretch only applies to different fingers on the same hand
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
        self.total * self.stretch_weight
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64) {
        // Convert from centiunits back to units, normalized by bigram total
        stats.stretches = self.total as f64 / (bigram_total * 100.0);
    }

    pub fn update_bigram(&mut self, p_a: CachePos, p_b: CachePos, old_freq: i64, new_freq: i64) {
        if let Some(sp) = self.stretch_pairs_per_key[p_a].iter().find(|sp| sp.other_pos == p_b) {
            let delta = (new_freq - old_freq) * sp.dist;
            self.total += delta;
        }
    }

    /// Replace key at position. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        apply: bool,
    ) -> i64 {
        let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, bg_freq);
        if apply {
            self.total += delta;
        }
        (self.total + if apply { 0 } else { delta }) * self.stretch_weight
    }

    /// Compute the delta for replacing a key without mutating state.
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

    /// Swap keys at two positions. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        apply: bool,
    ) -> i64 {
        // Compute delta for pos_a (key_a -> key_b), skipping pos_b
        let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq);
        // Compute delta for pos_b (key_b -> key_a), skipping pos_a
        let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq);
        let total_delta = delta_a + delta_b;

        if apply {
            self.total += total_delta;
        }
        (self.total + if apply { 0 } else { total_delta }) * self.stretch_weight
    }
}
