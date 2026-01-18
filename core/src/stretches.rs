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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    /// Precomputed stretch distances for each position pair
    stretch_dists: Vec<Vec<i64>>,
    /// For each position, list of other positions that form a stretch pair
    stretch_pairs_per_key: Vec<Vec<usize>>,
    total: i64,
}

impl StretchCache {
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger]) -> Self {
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

        // Build stretch pair lookup - for each position, which other positions form a stretch
        let stretch_pairs_per_key: Vec<Vec<usize>> = (0..len)
            .map(|i| {
                (0..len)
                    .filter(|&j| i != j && stretch_dists[i][j] > 0)
                    .collect()
            })
            .collect();

        Self {
            stretch_dists,
            stretch_pairs_per_key,
            total: 0,
        }
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
        self.stretch_dists[p1][p2]
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        self.total * weights.stretches
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64) {
        // Convert from centiunits back to units, normalized by bigram total
        stats.stretches = self.total as f64 / (bigram_total * 100.0);
    }

    pub fn update_bigram(&mut self, p_a: CachePos, p_b: CachePos, old_freq: i64, new_freq: i64) {
        let stretch_dist = self.get_stretch(p_a, p_b);
        if stretch_dist > 0 {
            let delta = (new_freq - old_freq) * stretch_dist;
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
    #[inline]
    pub fn replace_key<F>(
        &mut self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        get_bg_freq: F,
    ) where
        F: Fn(usize, usize) -> i64,
    {
        for &other_pos in &self.stretch_pairs_per_key[pos] {
            if skip_pos == Some(other_pos) {
                continue;
            }
            let stretch_dist = self.stretch_dists[pos][other_pos];
            let other_key = keys[other_pos];

            // Bigram: pos -> other_pos
            let bg_delta = get_bg_freq(new_key, other_key) - get_bg_freq(old_key, other_key);
            self.total += bg_delta * stretch_dist;

            // Bigram: other_pos -> pos
            let bg_delta_rev = get_bg_freq(other_key, new_key) - get_bg_freq(other_key, old_key);
            self.total += bg_delta_rev * stretch_dist;
        }
    }

    /// Optimized key swap: update scores for swapping keys at pos_a and pos_b.
    /// Handles the direct pair between pos_a and pos_b specially to avoid double-counting.
    #[inline]
    pub fn key_swap<F>(
        &mut self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        get_bg_freq: F,
    ) where
        F: Fn(usize, usize) -> i64,
    {
        // Handle the direct pair between pos_a and pos_b
        let stretch_ab = self.get_stretch(pos_a, pos_b);
        if stretch_ab > 0 {
            // Bigram a->b: was (key_a, key_b), now (key_b, key_a)
            let bg_delta_ab = get_bg_freq(key_b, key_a) - get_bg_freq(key_a, key_b);
            self.total += bg_delta_ab * stretch_ab;

            // Bigram b->a: was (key_b, key_a), now (key_a, key_b)
            let bg_delta_ba = get_bg_freq(key_a, key_b) - get_bg_freq(key_b, key_a);
            self.total += bg_delta_ba * stretch_ab;
        }

        // Replace key at pos_a (key_a -> key_b), skipping pos_b
        self.replace_key(pos_a, key_a, key_b, keys, Some(pos_b), &get_bg_freq);
        // Replace key at pos_b (key_b -> key_a), skipping pos_a
        self.replace_key(pos_b, key_b, key_a, keys, Some(pos_a), &get_bg_freq);
    }
}
