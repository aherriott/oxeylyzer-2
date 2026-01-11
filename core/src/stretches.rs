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
    total: i64,
}

impl StretchCache {
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger]) -> Self {
        let len = keyboard.len();
        let stretch_dists = (0..len)
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

        Self {
            stretch_dists,
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
}
