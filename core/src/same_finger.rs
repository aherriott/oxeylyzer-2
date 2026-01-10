/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use crate::char_mapping::EMPTY_KEY;
use crate::magic::{DeltaGram, MagicCache};
use crate::stats::Stats;
use crate::types::CacheKey;
use crate::weights::{FingerWeights, Weights};
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;
use std::fmt::{self, Debug};

#[derive(Clone)]
pub struct SfBigramPair {
    pub other_pos: usize,
    pub dist: i64,
}

impl Debug for SfBigramPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SfBigramPair {{ other_pos: {}, dist: {} }}",
            self.other_pos, self.dist
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFCache {
    sfb_per_finger: Box<[i64; 10]>, // Cumulative SFB dist per finger
    total_sfbs: i64,                // SFB count
    sfs_per_finger: Box<[i64; 10]>, // Cumulative SFS dist per finger
    total_sfs: i64,                 // SFS count
    use_per_finger: Box<[i64; 10]>, // Finger use
    total_bg: i64,                  // Total non-zero-freq BGs (for normalizing)
    total_sg: i64,                  // Total non-zero-freq SGs (for normalizing)
    sfbg_dist_per_key: Vec<Vec<SfBigramPair>>, // One-time calculated key pos -> list of bg dist on same finger
}

impl SFCache {
    // Zero initialize
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey], keys: &[CacheKey]) -> Self {
        assert!(
            fingers.len() <= CacheKey::MAX as usize,
            "Too many keys to index with CacheKey, max is {}",
            CacheKey::MAX
        );
        assert_eq!(
            fingers.len(),
            keyboard.len(),
            "finger len is not the same as keyboard len: "
        );
        let mut sfb_per_finger = Box::new([0i64; 10]);
        let mut sfs_per_finger = Box::new([0i64; 10]);
        let mut use_per_finger = Box::new([0i64; 10]);
        // compute distances
        let mut sfbg_dist_per_key = Vec::with_capacity(fingers.len());
        for (i, (finger1, _phys1)) in fingers.iter().zip(keyboard).enumerate() {
            let mut pairs = Vec::new();
            for (j, (finger2, _phys2)) in fingers.iter().zip(keyboard).enumerate() {
                if finger1 == finger2 && i != j {
                    // distance function placeholder
                    let dist = 0i64; // TODO: compute actual distance
                    pairs.push(SfBigramPair { other_pos: j, dist });
                }
            }
            sfbg_dist_per_key.push(pairs);
        }
        Self {
            sfb_per_finger,
            total_sfbs: 0,
            sfs_per_finger,
            total_sfs: 0,
            use_per_finger,
            total_bg: 0,
            total_sg: 0,
            sfbg_dist_per_key,
        }
    }

    pub fn initialize(&mut self, _fingers: &[Finger], _keyboard: &[PhysicalKey]) {
        // stub
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        // TODO: normalize
        Fingers::FINGERS
            .iter()
            .map(|f| -> i64 {
                self.sfb_per_finger[f] * weights.finger[f] * weights.sfb
                    + self.sfs_per_finger[f] * weights.finger[f] * weights.sfs
            })
            .sum()
    }

    pub fn stats(&self, _stats: &mut Stats) {
        // TODO
    }

    pub fn add_key(
        &mut self,
        keys: &Box<[CacheKey]>,
        magic: &MagicCache,
        pos: usize,
        key: CacheKey,
    ) {
        let sfb = self.sfbg_dist_per_key[pos]
            .iter()
            .map(|sfbg: SfBigramPair| -> i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_bg_freq(u1, u2) + magic.get_bg_freq(u2, u1)) * sfbg.dist
            })
            .sum();
        self.sfb_per_finger[fingers[pos]] += sfb;
        // TODO: sfb count, normalizing

        let sfs = self.sfbg_dist_per_key[pos]
            .iter()
            .map(|sfbg: SfBigramPair| -> i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_sg_freq(u1, u2) + magic.get_sg_freq(u2, u1)) * sfbg.dist
            })
            .sum();
        self.sfs_per_finger[fingers[pos]] += sfs;

        self.keys.reverse_get(key) = pos;
    }

    pub fn remove_key(
        &mut self,
        keys: &Box<[CacheKey]>,
        magic: &MagicCache,
        pos: usize,
        key: CacheKey,
    ) {
        let sfb = self.sfbg_dist_per_key[pos]
            .iter()
            .map(|sfbg: SfBigramPair| -> i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_bg_freq(u1, u2) + magic.get_bg_freq(u2, u1)) * sfbg.dist
            })
            .sum();
        self.sfb_per_finger[fingers[pos]] -= sfb;
        // TODO: sfb count, normalizing

        let sfs = self.sfbg_dist_per_key[pos]
            .iter()
            .map(|sfbg: SfBigramPair| -> i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_sg_freq(u1, u2) + magic.get_sg_freq(u2, u1)) * sfbg.dist
            })
            .sum();
        self.sfs_per_finger[fingers[pos]] -= sfs;

        self.keys.reverse_get(key) = EMPTY_KEY;
    }

    pub fn steal_bigram(&mut self, magic: &MagicCache, affected_grams: &[DeltaGram]) {
        for gram in &cache.magic.affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    self.sfbg_dist_per_key[self.keys.reverse_get(bg.a)]
                        .iter_mut()
                        .filter(|(pos, dist)| (self.keys.reverse_get(bg.b) == pos)) // Is SF
                        .map(|(pos, dist)| {
                            self.sfb_per_finger(fingers[pos]) += (bg.new - bg.old) * dist;
                        })
                }
                DeltaGram::Skipgram(bg) => {
                    self.sfbg_dist_per_key[self.keys.reverse_get(bg.a)]
                        .iter_mut()
                        .filter(|(pos, dist)| (self.keys.reverse_get(bg.b) == pos)) // Is SF
                        .map(|(pos, dist)| {
                            self.sfs_per_finger(fingers[pos]) += (bg.new - bg.old) * dist;
                        })
                }
                _ => { /* Trigrams are not part of SFBs */ }
            }
        }
    }
}
