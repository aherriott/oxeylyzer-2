/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use crate::cached_layout::{DeltaBigram, DeltaSkipgram};
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
    sfb_per_finger: Box<[i64; 10]>,
    total_sfbs: i64,
    sfs_per_finger: Box<[i64; 10]>,
    total_sfs: i64,
    use_per_finger: Box<[i64; 10]>,
    total_bg: i64,
    total_sg: i64,
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
            sfb_per_finger: Box::new([0i64; 10]),
            total_sfbs: 0,
            sfs_per_finger: Box::new([0i64; 10]),
            total_sfs: 0,
            use_per_finger: Box::new([0i64; 10]),
            total_bg: 0,
            total_sg: 0,
            sf_pairs_per_key,
        }
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        // TODO: normalize
        Finger::FINGERS
            .iter()
            .map(|f| -> i64 {
                let fi = *f as usize;
                self.sfb_per_finger[fi] * weights.fingers.get(*f) * weights.sfbs
                    + self.sfs_per_finger[fi] * weights.fingers.get(*f) * weights.sfs
            })
            .sum()
    }

    pub fn stats(&self, _stats: &mut Stats) {
        // TODO
    }

    /// Check if two positions are on the same finger
    #[inline]
    fn is_same_finger(&self, p_a: usize, p_b: usize) -> Option<usize> {
        self.sf_pairs_per_key[p_a]
            .iter()
            .find(|sf| sf.other_pos == p_b)
            .map(|sf| sf.finger)
    }

    pub fn update_bigram(&mut self, dist_cache: &DistCache, bg: &DeltaBigram) {
        if let Some(finger) = self.is_same_finger(bg.p_a, bg.p_b) {
            let dist = dist_cache.get(bg.p_a, bg.p_b);
            let delta = (bg.new_freq - bg.old_freq) * dist;
            self.sfb_per_finger[finger] += delta;
        }
    }

    pub fn update_skipgram(&mut self, dist_cache: &DistCache, sg: &DeltaSkipgram) {
        if let Some(finger) = self.is_same_finger(sg.p_a, sg.p_b) {
            let dist = dist_cache.get(sg.p_a, sg.p_b);
            let delta = (sg.new_freq - sg.old_freq) * dist;
            self.sfs_per_finger[finger] += delta;
        }
    }

    /// Copy scoring data from another SFCache. No allocations.
    #[inline]
    pub fn copy_from(&mut self, other: &SFCache) {
        *self.sfb_per_finger = *other.sfb_per_finger;
        *self.sfs_per_finger = *other.sfs_per_finger;
    }
}
