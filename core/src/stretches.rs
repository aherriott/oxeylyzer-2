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
        // Look up in stretch_pairs_per_key
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
        // Look up stretch distance from pre-computed pairs
        if let Some(sp) = self.stretch_pairs_per_key[p_a].iter().find(|sp| sp.other_pos == p_b) {
            let delta = (new_freq - old_freq) * sp.dist;
            self.total += delta;
        }
    }

    pub fn update_skipgram(&mut self, _p_a: CachePos, _p_b: CachePos, _old_freq: i64, _new_freq: i64) {
        // Stretches don't track skipgrams
    }

    /// Copy scoring data from another StretchCache. No allocations.
    #[inline]
    pub fn copy_from(&mut self, other: &StretchCache) {
        self.total = other.total;
    }

    /// Replace key at position: update scores for changing from old_key to new_key.
    /// Use EMPTY_KEY for old_key when adding, or new_key when removing.
    /// `skip_pos` allows skipping a position (used by key_swap to avoid double-counting).
    /// `bg_freq` is a flat array indexed by `a * num_keys + b`.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
    ) {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        for sp in &self.stretch_pairs_per_key[pos] {
            let other_pos = sp.other_pos;
            if skip_pos == Some(other_pos) {
                continue;
            }
            let stretch_dist = sp.dist;
            let other_key = keys[other_pos];

            // Skip if other_key is EMPTY_KEY
            if other_key >= num_keys {
                continue;
            }

            // Bigram: pos -> other_pos
            let old_bg = if old_valid { bg_freq[old_key * num_keys + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_key * num_keys + other_key] } else { 0 };
            let bg_delta = new_bg - old_bg;
            self.total += bg_delta * stretch_dist;

            // Bigram: other_pos -> pos
            let old_bg_rev = if old_valid { bg_freq[other_key * num_keys + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_key * num_keys + new_key] } else { 0 };
            let bg_delta_rev = new_bg_rev - old_bg_rev;
            self.total += bg_delta_rev * stretch_dist;
        }
    }

    /// Optimized key swap: update scores for swapping keys at pos_a and pos_b.
    /// Handles the direct pair between pos_a and pos_b specially to avoid double-counting.
    /// `bg_freq` is a flat array indexed by `a * num_keys + b`.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
    ) {
        let num_keys = self.num_keys;

        // Handle the direct pair between pos_a and pos_b (only if both keys are valid)
        if key_a < num_keys && key_b < num_keys {
            if let Some(sp) = self.stretch_pairs_per_key[pos_a].iter().find(|sp| sp.other_pos == pos_b) {
                let stretch_ab = sp.dist;

                // Bigram a->b: was (key_a, key_b), now (key_b, key_a)
                let bg_delta_ab = bg_freq[key_b * num_keys + key_a] - bg_freq[key_a * num_keys + key_b];
                self.total += bg_delta_ab * stretch_ab;

                // Bigram b->a: was (key_b, key_a), now (key_a, key_b)
                let bg_delta_ba = bg_freq[key_a * num_keys + key_b] - bg_freq[key_b * num_keys + key_a];
                self.total += bg_delta_ba * stretch_ab;
            }
        }

        // Replace key at pos_a (key_a -> key_b), skipping pos_b
        self.replace_key(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq);
        // Replace key at pos_b (key_b -> key_a), skipping pos_a
        self.replace_key(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq);
    }
}
