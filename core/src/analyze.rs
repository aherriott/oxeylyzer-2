use fxhash::FxHashSet as HashSet;
use libdof::dofinitions::Finger;
use nanorand::{Rng, WyRand};

use crate::{
    cached_layout::CachedLayout,
    data::Data,
    layout::*,
    analyzer_data::AnalyzerData,
    weights::Weights,
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

    pub fn revert(&self) -> Self {
        match self {
            // KeySwap is its own inverse
            Neighbor::KeySwap(pair) => Neighbor::KeySwap(*pair),
            // TODO: MagicStealBigram revert would need the old output - for now just return self
            Neighbor::MagicStealBigram(msb) => Neighbor::MagicStealBigram(*msb),
        }
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
        let analyze_bigrams = weights.has_bigram_weights();
        let analyze_stretches = weights.has_stretch_weights();
        let analyze_trigrams = weights.has_trigram_weights();

        Self {
            data,
            weights,
            analyze_bigrams,
            analyze_stretches,
            analyze_trigrams,
            current_cache: None,
            working_cache: None,
        }
    }

    pub fn use_layout(&mut self, layout: &Layout, _pins: &[usize]) {
        // TODO: use pins
        self.current_cache = Some(CachedLayout::new(
            &layout.keyboard,
            layout,
        ));
        // Clone the current cache to allocate the memory we need. Everything from here is alloc-free
        self.working_cache = self.current_cache.clone();
    }

    pub fn layout(&self) -> Layout {
        self.current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .to_layout()
    }

    pub fn score(&self) -> i64 {
        self.current_cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .score(&self.weights)
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

    pub fn random_neighbor(&self, rng: &mut WyRand) -> Neighbor {
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
            .score(&self.weights);
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
    pub fn test_neighbor(&mut self, neighbor: Neighbor) -> i64 {
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
        score
    }

    // Calculates the score of a neighbor without updating the cache
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) -> i64 {
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

        current.apply_neighbor(neighbor);
        working.copy_from(current, neighbor);
        current.score()
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

    //                 cache.magic.get_bg_freq(u1 * mapping_len + u2).unwrap()
    //                     + cache.magic.get_bg_freq(u2 * mapping_len + u1).unwrap()
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

    //                 cache.magic.get_sg_freq(u1 * mapping_len + u2).unwrap()
    //                     + cache.magic.get_sg_freq(u2 * mapping_len + u1).unwrap()
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

    // TODO
    fn _analyzer_layout(layout_name: &str) -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("this should exist");

        let weights = dummy_weights();

        let analyzer = Analyzer::new(data, weights);

        let layout = Layout::load(format!("../layouts/{layout_name}.dof"))
            .expect("this layout is valid and exists, soooo");

        (analyzer, layout)
    }
}
