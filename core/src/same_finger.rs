/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use crate::dist::DistCache;
use crate::stats::Stats;
use crate::weights::Weights;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

/// Stores same-finger pairs for a given position
#[derive(Debug, Clone, PartialEq)]
pub struct SfPair {
    pub other_pos: usize,
    pub finger: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFCache {
    /// Weighted SFB score per finger (freq * dist)
    sfb_score_per_finger: Box<[i64; 10]>,
    /// Weighted SFS score per finger (freq * dist)
    sfs_score_per_finger: Box<[i64; 10]>,
    /// Unweighted SFB frequency per finger (for stats)
    sfb_freq_per_finger: Box<[i64; 10]>,
    /// Unweighted SFS frequency per finger (for stats)
    sfs_freq_per_finger: Box<[i64; 10]>,
    /// For each position, list of other positions on the same finger
    sf_pairs_per_key: Vec<Vec<SfPair>>,
}

impl SFCache {
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey]) -> Self {
        assert_eq!(
            fingers.len(),
            keyboard.len(),
            "finger len is not the same as keyboard len"
        );

        // Build same-finger pair lookup
        let mut sf_pairs_per_key = Vec::with_capacity(fingers.len());
        for (i, finger1) in fingers.iter().enumerate() {
            let mut pairs = Vec::new();
            for (j, finger2) in fingers.iter().enumerate() {
                if finger1 == finger2 && i != j {
                    pairs.push(SfPair {
                        other_pos: j,
                        finger: *finger1 as usize,
                    });
                }
            }
            sf_pairs_per_key.push(pairs);
        }

        Self {
            sfb_score_per_finger: Box::new([0i64; 10]),
            sfs_score_per_finger: Box::new([0i64; 10]),
            sfb_freq_per_finger: Box::new([0i64; 10]),
            sfs_freq_per_finger: Box::new([0i64; 10]),
            sf_pairs_per_key,
        }
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        Finger::FINGERS
            .iter()
            .map(|f| -> i64 {
                let fi = *f as usize;
                self.sfb_score_per_finger[fi] * weights.fingers.get(*f) * weights.sfbs
                    + self.sfs_score_per_finger[fi] * weights.fingers.get(*f) * weights.sfs
            })
            .sum()
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

    /// Check if two positions are on the same finger
    #[inline]
    fn is_same_finger(&self, p_a: usize, p_b: usize) -> Option<usize> {
        self.sf_pairs_per_key[p_a]
            .iter()
            .find(|sf| sf.other_pos == p_b)
            .map(|sf| sf.finger)
    }

    pub fn update_bigram(&mut self, dist_cache: &DistCache, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(finger) = self.is_same_finger(p_a, p_b) {
            let dist = dist_cache.get(p_a, p_b);
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfb_score_per_finger[finger] += score_delta;
            self.sfb_freq_per_finger[finger] += freq_delta;
        }
    }

    pub fn update_skipgram(&mut self, dist_cache: &DistCache, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(finger) = self.is_same_finger(p_a, p_b) {
            let dist = dist_cache.get(p_a, p_b);
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfs_score_per_finger[finger] += score_delta;
            self.sfs_freq_per_finger[finger] += freq_delta;
        }
    }

    /// Copy scoring data from another SFCache. No allocations.
    #[inline]
    pub fn copy_from(&mut self, other: &SFCache) {
        *self.sfb_score_per_finger = *other.sfb_score_per_finger;
        *self.sfs_score_per_finger = *other.sfs_score_per_finger;
        *self.sfb_freq_per_finger = *other.sfb_freq_per_finger;
        *self.sfs_freq_per_finger = *other.sfs_freq_per_finger;
    }
}
