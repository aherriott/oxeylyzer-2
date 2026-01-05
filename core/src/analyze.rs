use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use itertools::Itertools;
use libdof::{dofinitions::Finger, magic::MagicKey, prelude::PhysicalKey};
use nanorand::{Rng, WyRand};
use std::sync::Arc;

use crate::cached_layout;
use crate::{
    analyzer_data::AnalyzerData,
    cached_layout::*,
    data::Data,
    layout::*,
    trigrams::TRIGRAMS,
    weights::{FingerWeights, Weights},
};

// The difference between two neighboring layouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Neighbor {
    KeySwap(PosPair),
    MagicStealBigram(MagicStealBigram),
}

impl Neighbor {
    pub fn default() -> Self {
        Neighbor::KeySwap(PosPair(0, 0))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analyzer {
    data: AnalyzerData,
    weights: Weights,
    analyze_bigrams: bool,
    analyze_stretches: bool,
    analyze_trigrams: bool,
    current_cache: Option<CachedLayout>,
    working_cache: Option<CachedLayout>,
}

impl Analyzer {
    pub fn new(data: Data, weights: Weights) -> Self {
        let data = AnalyzerData::new(data);

        Self {
            data,
            weights,
            analyze_bigrams,
            analyze_stretches,
            analyze_trigrams,
            None,
            None,
        }
    }

    pub fn use_layout(&mut self, layout: &Layout, pins: &[usize]) {
        self.current_cache = Some(CachedLayout::new(self.data, layout));
        // Clone the current cache to allocate the memory we need. Everything from here is alloc-free
        self.working_cache = self.current_cache.clone();
    }

    pub fn score(&self) -> i64 {
        self.current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .score()
    }

    /*
     **************************************
     *         Neighbor actions
     **************************************
     */

    // possible_neighbors only needs to be called once per layout + pins combo
    pub fn possible_neighbors(&self) -> &Vec<Neighbor> {
        assert!(self.current_cache.is_some(), "Analyzer has no Layout set");
        self.current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .possible_neighbors()
    }

    pub fn random_neighbor(&self, cache: &CachedLayout, rng: &mut WyRand) -> Neighbor {
        assert!(self.current_cache.is_some(), "Analyzer has no Layout set");
        let pos_neighbors = self
            .current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .possible_neighbors();
        pos_neighbors[rng.generate_range(0..pos_neighbors.len())]
    }

    /**
     * Returns the best neighbor
     */
    pub fn best_neighbor(&self) -> Option<(Neighbor, i64)> {
        let mut best_score = self
            .current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .score();
        let mut best = None;

        for neighbor in self
            .current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .possible_neighbors()
        {
            let score = self.test_neighbor(*neighbor);
            if score > best_score {
                best_score = score;
                best = Some((neighbor.clone(), score));
            }
        }
        best
    }

    // Calculates the score of a neighbor without updating the cache
    pub fn test_neighbor(&self, neighbor: Neighbor) -> i64 {
        // Copy the current cache to the working cache
        let working = self
            .working_cache
            .as_ref()
            .expect("Analyzer has no Layout set");
        let current = self
            .current_cache
            .as_ref()
            .expect("Analyzer has no Layout set");
        debug_assert_eq!(
            working, current,
            "Working cache out of sync with current cache"
        );
        working.apply_neighbor(neighbor);
        let score = working.score();
        working.copy_from(current, neighbor);
    }

    // Calculates the score of a neighbor without updating the cache
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        // Copy the current cache to the working cache
        return Self::apply_neighbor_to_cache(neighbor);
    }

    /*
     **************************************
     *              Stats
     **************************************
     */

    // pub fn sfbs(&self) -> i64 {
    //     let mapping_len = self.data.mapping.len();
    //     self.current_cache
    //         .unwrap()
    //         .sfb
    //         .weighted_sfb_indices
    //         .all
    //         .iter()
    //         .map(
    //             |BigramPair {
    //                  pair: PosPair(a, b),
    //                  ..
    //              }| {
    //                 let u1 = cache.keys[*a as usize] as usize;
    //                 let u2 = cache.keys[*b as usize] as usize;

    //                 cache.magic.bigrams.get(u1 * mapping_len + u2).unwrap()
    //                     + cache.magic.bigrams.get(u2 * mapping_len + u1).unwrap()
    //             },
    //         )
    //         .sum()
    // }

    // pub fn sfs(&self) -> i64 {
    //     let mapping_len = self.data.mapping.len();
    //     cache
    //         .sfb
    //         .weighted_sfb_indices
    //         .all
    //         .iter()
    //         .map(
    //             |BigramPair {
    //                  pair: PosPair(a, b),
    //                  ..
    //              }| {
    //                 let u1 = cache.keys[*a as usize] as usize;
    //                 let u2 = cache.keys[*b as usize] as usize;

    //                 cache.magic.skipgrams.get(u1 * mapping_len + u2).unwrap()
    //                     + cache.magic.skipgrams.get(u2 * mapping_len + u1).unwrap()
    //             },
    //         )
    //         .sum()
    // }

    // pub fn stretches(&self) -> i64 {
    //     cache
    //         .stretch
    //         .all_pairs
    //         .iter()
    //         .map(
    //             |BigramPair {
    //                  pair: PosPair(a, b),
    //                  dist,
    //              }| {
    //                 let u1 = cache.keys[*a as usize];
    //                 let u2 = cache.keys[*b as usize];

    //                 (self.stretch_get_bigram(cache, &u1, &u2)
    //                     + self.stretch_get_bigram(cache, &u2, &u1))
    //                     * dist
    //             },
    //         )
    //         .sum()
    // }

    // pub fn finger_use(&self) -> [i64; 10] {
    //     let mut res = [0; 10];

    //     for (&k, &f) in cache.keys.iter().zip(cache.fingers.iter()) {
    //         // TODO: Magic mapping
    //         res[f as usize] += self.data.get_char_u(k);
    //     }

    //     res
    // }

    // pub fn unweighted_finger_distance(&self, cache: &CachedLayout) -> [i64; 10] {
    //     Finger::FINGERS.map(|f| self.sfb_finger_unweighted_bigrams(cache, f))
    // }

    // pub fn finger_sfbs(&self) -> [i64; 10] {
    //     let mapping_len = self.data.mapping.len();
    //     self.current_cache
    //         .sfb
    //         .weighted_sfb_indices
    //         .fingers
    //         .clone()
    //         .map(|pairs| {
    //             pairs
    //                 .iter()
    //                 .map(
    //                     |BigramPair {
    //                          pair: PosPair(a, b),
    //                          ..
    //                      }| {
    //                         let u1 = self.current_cache.keys[*a as usize] as usize;
    //                         let u2 = self.current_cache.keys[*b as usize] as usize;

    //                         self.current_cache
    //                             .magic
    //                             .bigrams
    //                             .get(u1 * mapping_len + u2)
    //                             .unwrap()
    //                             + self
    //                                 .current_cache
    //                                 .magic
    //                                 .bigrams
    //                                 .get(u2 * mapping_len + u1)
    //                                 .unwrap()
    //                     },
    //                 )
    //                 .sum()
    //         })
    // }

    // pub fn trigrams(&self) -> TrigramData {
    //     use crate::trigrams::TrigramType::*;

    //     let mapping_len = self.data.mapping.len();
    //     let mut trigrams = TrigramData::default();

    //     for (&c1, &f1) in self.current_cache.keys.iter().zip(&self.current_cache.fingers) {
    //         for (&c2, &f2) in self.current_cache.keys.iter().zip(&self.current_cache.fingers) {
    //             for (&c3, &f3) in self.current_cache.keys.iter().zip(&self.current_cache.fingers) {
    //                 let u1 = c1 as usize;
    //                 let u2 = c2 as usize;
    //                 let u3 = c3 as usize;
    //                 let freq = self.current_cache
    //                     .magic
    //                     .trigrams
    //                     .get(u1 * mapping_len.pow(2) + u2 * mapping_len + u3)
    //                     .unwrap();
    //                 let ttype = TRIGRAMS[f1 as usize * 100 + f2 as usize * 10 + f3 as usize];

    //                 match ttype {
    //                     Sft => trigrams.sft += freq,
    //                     Sfb => trigrams.sfb += freq,
    //                     Inroll => trigrams.inroll += freq,
    //                     Outroll => trigrams.outroll += freq,
    //                     Alternate => trigrams.alternate += freq,
    //                     Redirect => trigrams.redirect += freq,
    //                     OnehandIn => trigrams.onehandin += freq,
    //                     OnehandOut => trigrams.onehandout += freq,
    //                     Thumb => trigrams.thumb += freq,
    //                     Invalid => trigrams.invalid += freq,
    //                 }
    //             }
    //         }
    //     }

    //     trigrams
    // }

    pub fn similarity(&self, layout1: &Layout, layout2: &Layout) -> i64 {
        // TODO: Magic
        let _key_sim = layout1
            .keys
            .iter()
            .zip(&layout2.keys)
            .filter(|&(c1, c2)| (c1 == c2))
            .map(|(c1, _)| self.data.get_char(*c1))
            .sum::<i64>();

        let per_column = Finger::FINGERS
            .into_iter()
            .map(|f| {
                let col1 = layout1
                    .keys
                    .iter()
                    .zip(&layout1.fingers)
                    .filter_map(|(c1, f1)| (f == *f1).then_some(*c1))
                    .collect::<HashSet<_>>();

                layout2
                    .keys
                    .iter()
                    .zip(&layout1.fingers)
                    .filter(|&(c2, f2)| (f == *f2 && col1.contains(c2)))
                    .map(|(c2, _)| self.data.get_char(*c2))
                    .sum::<i64>()
            })
            .sum::<i64>();

        per_column
    }
}

impl std::fmt::Display for Analyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.keys.iter().map(|&u| self.char_mapping.get_c(u));

        for l in self.shape.inner().iter() {
            let mut i = 0;
            for c in iter.by_ref() {
                write!(f, "{c} ")?;
                i += 1;

                if *l == i {
                    break;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use crate::weights::dummy_weights;

    use super::*;

    fn analyzer_layout(layout_name: &str) -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("this should exist");

        let weights = dummy_weights();

        let analyzer = Analyzer::new(data, weights);

        let layout = Layout::load(format!("../layouts/{layout_name}.dof"))
            .expect("this layout is valid and exists, soooo");

        (analyzer, layout)
    }

    #[test]
    fn update_cache_bigrams() {
        let (analyzer, layout) = analyzer_layout("rstn-oxey");

        let mut cache = analyzer.cached_layout(layout, &[]);
        let reference = cache.clone();

        let possible_swaps = cache.possible_neighbors.clone();

        for (i, &swap) in possible_swaps.iter().enumerate() {
            let initial = analyzer.score_cache(&cache);

            analyzer.apply_neighbor(&mut cache, swap);
            let revert = swap.revert(&cache);
            analyzer.apply_neighbor(&mut cache, revert);

            let returned = analyzer.score_cache(&cache);

            assert_eq!(initial, returned, "iteration {i}: ");
            assert_eq!(cache, reference, "iteration {i}: ");
        }
    }

    #[test]
    fn stretch_cache_consistency() {
        let (analyzer, layout) = analyzer_layout("qwerty-minimal");
        let mut cache = analyzer.cached_layout(layout, &[]);
        let swap = PosPair(0, 8);

        let total = cache.stretch.total;
        let stretches = analyzer.stretches(&cache);

        assert_eq!(total, stretches);

        dbg!(total);

        analyzer.apply_neighbor(&mut cache, Neighbor::KeySwap(swap));
        analyzer.apply_neighbor(&mut cache, Neighbor::KeySwap(swap));

        let total2 = cache.stretch.total;

        println!("total stretch cache: {total}");
        println!("diff after swaps:    {}", total - total2);

        // println!("{:#?}", cache.stretch_cache.all_pairs);

        // cache.stretch_cache.per_keypair.get(&swap).unwrap().iter()
        //     .for_each(|pair| {
        //         let [p1, p2] = [pair.pair.0, pair.pair.1];
        //         let [c1, c2] = [cache.char(p1).unwrap(), cache.char(p2).unwrap()];

        //         println!("{c1}{c2}: {}", pair.dist);
        //     })
    }

    #[test]
    fn test_magic_bigram_frequency_basic() {
        // Test basic magic key frequency calculation
        let (analyzer, layout) = analyzer_layout("test/magic");
        let cache = analyzer.cached_layout(layout, &[]);

        // Test that 'a' -> 'b' bigram returns 0 (stolen by magic)
        let a = cache.char_mapping.get_u('a');
        let b = cache.char_mapping.get_u('b');

        let freq = analyzer.get_bigram_frequency_magic(&cache, a, b);
        assert_eq!(freq, 0, "a->b should be stolen by magic key");

        // Test that 'a' -> 'z' bigram is unaffected
        let z = cache.char_mapping.get_u('z');

        let freq = analyzer.get_bigram_frequency_magic(&cache, a, z);
        assert_ne!(freq, 0, "a->z should be unaffected");

        // Todo: test that the mag -> 'z' bigram is zero
        // Todo: test that 'a' -> mag has the stolen frequency
        // Todo: test that 'z' -> mag is zero
    }

    #[test]
    fn test_magic_bigram_multiple_keys() {
        // Test priority ordering when multiple magic keys have same rules
        // test/magic has two magic keys. The first has rule 'a'->'b', the second has 'a'->'b' and 'b'->'c'
        let (analyzer, _) = analyzer_layout("test/magic");

        // Todo: test that 'a' -> mag returns bg(a,b)
        // Todo: test that 'a' -> mag2 returns 0
        // Todo: test that 'b' -> mag2 returns bg(b,c)
    }

    #[test]
    fn test_magic_skipgram_frequency() {
        let (analyzer, layout) = analyzer_layout("test/magic");
        let cache = analyzer.cached_layout(layout, &[]);

        // Test skipgram calculation with magic keys
        let a = cache.char_mapping.get_u('a');
        let b = cache.char_mapping.get_u('b');

        let freq = analyzer.get_skipgram_frequency_magic(&cache, a, b);

        // Should be base skipgram minus any stolen by magic rules
        let base = analyzer.data.get_skipgram_u([a, b]);
        assert!(
            freq <= base,
            "Magic keys can only reduce skipgram frequency"
        );
    }

    #[test]
    fn test_magic_trigram_frequency() {
        let (analyzer, layout) = analyzer_layout("test/magic");
        let cache = analyzer.cached_layout(layout, &[]);

        // Test trigram calculation with magic keys
        let a = cache.char_mapping.get_u('a');
        let b = cache.char_mapping.get_u('b');
        let c = cache.char_mapping.get_u('c');

        let freq = analyzer.get_trigram_frequency_magic(&cache, a, b, c);

        // Trigram should either be 0 (stolen) or base frequency
        let base = analyzer.data.get_trigram_u([a, b, c]);
        assert!(
            freq == 0 || freq == base,
            "Trigram should be either stolen (0) or not ({base}), got {freq}"
        );
    }

    #[test]
    fn test_magic_frequency_symmetry() {
        // Test that changing a rule and reverting it preserves frequencies
        let (analyzer, layout) = analyzer_layout("test/magic");
        let mut cache = analyzer.cached_layout(layout, &[]);

        // Extract values first to avoid borrow conflicts
        let (magic_key, leader, output) = if let Some((mk, rules)) = cache.magic.rules.iter().next()
        {
            if let Some((&l, &o)) = rules.iter().next() {
                (*mk, l, o)
            } else {
                return; // No rules
            }
        } else {
            return; // No magic keys
        };

        let a = cache.char_mapping.get_u('t');
        let b = cache.char_mapping.get_u('e');

        // Get initial frequency
        let initial_bg = analyzer.get_bigram_frequency_magic(&cache, a, b);
        let initial_sg = analyzer.get_skipgram_frequency_magic(&cache, a, b);

        // Change the rule
        let new_output = cache.char_mapping.get_u('x');
        cache
            .magic
            .rules
            .get_mut(&magic_key)
            .unwrap()
            .insert(leader, new_output);

        // Revert the rule
        cache
            .magic
            .rules
            .get_mut(&magic_key)
            .unwrap()
            .insert(leader, output);

        // Frequencies should be restored
        let final_bg = analyzer.get_bigram_frequency_magic(&cache, a, b);
        let final_sg = analyzer.get_skipgram_frequency_magic(&cache, a, b);

        assert_eq!(initial_bg, final_bg, "Bigram frequency should be restored");
        assert_eq!(
            initial_sg, final_sg,
            "Skipgram frequency should be restored"
        );
    }
}
