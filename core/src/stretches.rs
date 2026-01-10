/*
 **************************************
 *            Stretches
 **************************************
 */

use crate::{
    analyze::Neighbor,
    analyzer_data::AnalyzerData,
    cached_layout::{BigramPair, CachedLayout},
    layout::PosPair,
    magic::{DeltaGram, MagicCache},
    types::CacheKey,
    weights::Weights,
};
use itertools::Itertools;
use libdof::{dofinitions::Finger, keyboard::PhysicalKey};
use std::collections::HashMap;

const KEY_EDGE_OFFSET: f64 = 0.5;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    pub all_pairs: Vec<BigramPair>,
    pub per_keypair: Vec<Vec<BigramPair>>,
    pub total: i64,
}

impl StretchCache {
    pub fn new(
        keys: &[CacheKey],
        fingers: &[Finger],
        keyboard: &[PhysicalKey],
        magic: &MagicCache,
    ) -> Self {
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

        let all_pairs = keyboard
            .iter()
            .zip(fingers)
            .zip(keys)
            .enumerate()
            .tuple_combinations::<(_, _)>()
            .filter(|((_, ((_, &f1), _)), (_, ((_, &f2), _)))| f1 != f2 && (f1.hand() == f2.hand()))
            .filter_map(|((i1, ((k1, &f1), _c1)), (i2, ((k2, &f2), _c2)))| {
                let diff = (f1 as CacheKey).abs_diff(f2 as CacheKey) as f64;
                let fd = diff * 1.35;
                // let minimum_diff = diff * 0.9;
                let (dx, dy) = Self::dx_dy(k1, k2, f1, f2);
                let negative_lsb = 0.0; //(minimum_diff - dx.abs() - 1.0).max(0.0) * 2.0;
                let dist = dx.hypot(dy);

                let xo = x_overlap(dx, dy, f1, f2);

                let stretch = dist + xo + negative_lsb - fd;

                // if stretch > 0.001 {
                //     println!("{_c1}{_c2}: {}", (stretch * 100.0) as i64);
                // }

                (stretch > 0.001).then_some(BigramPair {
                    pair: PosPair(i1 as CacheKey, i2 as CacheKey),
                    dist: (stretch * 100.0) as i64,
                })
            })
            .collect();

        let per_keypair = (0..(fingers.len() as CacheKey))
            .cartesian_product(0..(fingers.len() as CacheKey))
            .map(|(i1, i2)| {
                let is = [i1, i2];

                let pairs = all_pairs
                    .iter()
                    .filter(move |b| is.contains(&b.pair.0) || is.contains(&b.pair.1))
                    .copied()
                    .collect::<Box<[_]>>();

                (PosPair(i1, i2), pairs)
            })
            .collect::<HashMap<_, _>>();

        let total = all_pairs
            .iter()
            .map(
                |BigramPair {
                     dist,
                     pair: PosPair(a, b),
                 }| {
                    let u1 = *a as usize;
                    let u2 = *b as usize;
                    let bg = magic.get_bg_freq(u1 + u2 * keys.len()).unwrap()
                        + magic.get_bg_freq(u2 + u1 * keys.len()).unwrap();
                    let sg = magic.get_sg_freq(u1 + u2 * keys.len()).unwrap()
                        + magic.get_sg_freq(u2 + u1 * keys.len()).unwrap();

                    // TODO: should this be sfb / sfs? If you weight sfbs more, the skipgrams instead get weighted more here.
                    // Should it be the other way around? Would a weighted average make more sense?
                    let sfb_over_sfs = (weights.sfbs as f64) / (weights.sfs as f64);
                    (bg + (sg as f64 * sfb_over_sfs) as i64) * weights.stretches * dist
                },
            )
            .sum();

        Self {
            all_pairs,
            per_keypair,
            total,
        }
    }

    pub fn update_bigram(&mut self, bg: &DeltaBigram) {
        // stub
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        // TODO
        self.total * weights.stretches
    }

    // pub fn stretch_neighbor_update(
    //     &self,
    //     neighbor: &Neighbor,
    //     apply: bool,
    // ) -> i64 {
    //     match neighbor {
    //         Neighbor::KeySwap(pair) => {
    //             if pair.0 == pair.1 {
    //                 // nothing to do
    //                 return self.total;
    //             }

    //             // TODO: Does this even work? Doesn't seems to take into account new cache.keys after all_pairs is initialized
    //             let revert = cache.apply_neighbor(*neighbor);
    //             let stretch_new = self.keypair_stretch(cache, pair);

    //             cache.apply_neighbor(revert);
    //             let stretch_old = self.keypair_stretch(cache, pair);

    //             if apply {
    //                 // TODO: is this correct? Feels like it should be new - old
    //                 stretch.total += stretch_old - stretch_new;
    //                 stretch.total
    //             } else {
    //                 stretch.total + stretch_old - stretch_new
    //             }
    //         }
    //         Neighbor::MagicRule(_) => 0,
    //     }
    // }

    fn keypair_stretch(&self, pair: &PosPair) -> i64 {
        self
            .per_keypair
            .get(&pair)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(
                        |BigramPair {
                             pair: PosPair(a, b),
                             dist,
                         }| { self.stretch_get_bigram(cache, a, b) * dist },
                    )
                    .sum()
            })
            .unwrap_or_default()
    }

    pub fn stretch_get_bigram(&self, cache: &CachedLayout, a: &CacheKey, b: &CacheKey) -> i64 {
        let mapping_len = self.data.mapping.len();
        let u1 = *a as usize;
        let u2 = *b as usize;

        let bg = cache.magic.get_bg_freq(u1 * mapping_len + u2).unwrap()
            + cache.magic.get_bg_freq(u2 * mapping_len + u1).unwrap();
        let sg = cache.magic.get_sg_freq(u1 * mapping_len + u2).unwrap()
            + cache.magic.get_sg_freq(u2 * mapping_len + u1).unwrap();

        // TODO: should this be sfb / sfs? If you weight sfbs more, the skipgrams instead get weighted more here.
        // Should it be the other way around? Would a weighted average make more sense?
        let sfb_over_sfs = (self.weights.sfbs as f64) / (self.weights.sfs as f64);
        (bg + (sg as f64 * sfb_over_sfs) as i64) * self.weights.stretches
    }
}
