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
    pub fn initialize(&mut self, fingers: &[Finger], keyboard: &[PhysicalKey]) {
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
        self.sfb_per_finger.iter_mut().map(|x| *x = 0);
        self.total_sfbs = 0;
        self.sfs_per_finger.iter_mut().map(|x| *x = 0);
        self.total_sfs = 0;
        self.use_per_finger.iter_mut().map(|x| *x = 0);
        self.total_bg = 0;
        self.total_sg = 0;
        self.key_to_pos.iter_mut().map(|x| *x = EMPTY_KEY); // Invalid pos (CacheKey::Max + 1)

        fingers.iter().zip(keyboard).map(|(finger, physical_1)| {
            self.sfbg_dist_per_key.push(Vec::new());
            fingers
                .iter()
                .zip(keyboard)
                .zip(0)
                .filter(|((f, physical_2), pos)| (f == &finger))
                .map(|((f, physical_2), pos)| {
                    self.sfbg_dist_per_key
                        .last_mut()
                        .unwrap()
                        .push(SfBigramPair {
                            other_pos: pos,
                            dist: (Self::dist(physical_1, physical_2, Finger::LP, Finger::LP)
                                * 100.0) as i64,
                        });
                });
        });
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

    pub fn stats(&self, stats: &mut Stats) {
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

    pub fn add_rule(&mut self, magic: &MagicCache, affected_grams: &[DeltaGram]) {
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
