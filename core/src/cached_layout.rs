use fxhash::FxHashMap as HashMap;
use itertools::Itertools;
use libdof::prelude::{Finger, PhysicalKey, Shape};
use std::sync::Arc;

use crate::{
    analyze::Neighbor,
    analyzer_data::AnalyzerData,
    char_mapping::CharMapping,
    layout::{Layout, MagicStealBigram, PosPair},
    magic::DeltaGram,
    magic::MagicCache,
    same_finger::SFCache,
    stretches::StretchCache,
    types::{CacheKey, KeysCache},
    weights::{FingerWeights, Weights},
    REPLACEMENT_CHAR,
};

// CachedLayout contains the minimum mutable data used to define a layout and store scoring. Designed to copy quickly and without allocation.
// It is wrapped by Analyzer
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CachedLayout {
    pub keys: KeysCache,
    pub possible_neighbors: Vec<Neighbor>,
    pub affected_grams: Vec<DeltaGram>,
    pub sfb: SFCache,
    pub stretch: StretchCache,
    pub magic: MagicCache,
    pub fingers: Vec<Finger>, // Internal storage for key:finger mapping
    pub char_mapping: Arc<CharMapping>,
    pub keyboard: Box<[PhysicalKey]>,
    pub shape: Shape,
    pub name: String,
}

impl CachedLayout {
    // Allocates all the required memory
    pub fn new(
        data: &AnalyzerData,
        keyboard: &[PhysicalKey],
        char_mapping: &CharMapping,
        layout: &Layout,
        weights: &Weights,
    ) -> Self {
        // Zero initialize all of the cache data
        let keys = KeysCache::new(layout);
        let mut possible_neighbors = Vec::with_capacity(
            layout.keys.len() * layout.keys.len() + // Keyswaps
            (layout.keys.len() - layout.magic.len()) * layout.magic.len(), // Steal Bigrams
        );
        // TODO: This with_capacity probably isn't right
        let affected_grams =
            Vec::with_capacity(layout.keys.len().pow(2) + layout.magic.len() * layout.keys.len());
        let magic = MagicCache::new(data, &keys, layout, &mut possible_neighbors);
        let sfb = SFCache::new(&layout.fingers, &layout.keyboard, keys.as_slice());
        let stretch = StretchCache::new(
            keys.as_slice(),
            &layout.fingers,
            &layout.keyboard,
            weights,
            &magic,
        );

        let char_mapping_arc = Arc::new(char_mapping.clone());
        let mut cache = CachedLayout {
            keys,
            possible_neighbors,
            affected_grams,
            sfb,
            stretch,
            magic,
            fingers: layout.fingers.to_vec(),
            char_mapping: char_mapping_arc.clone(),
            keyboard: keyboard.to_vec().into_boxed_slice(),
            shape: layout.shape.clone(),
            name: layout.name.clone(),
        };

        for (i, u) in layout.keys.iter().enumerate() {
            let cache_key = char_mapping_arc.get_u(*u);
            cache.add_key(i, cache_key);
        }

        // TODO: magic iteration
        // layout.magic.iter().for_each(|(key, rules)| {
        //     rules.iter().for_each(|(leader, output)| {
        //         cache.steal_bigram(*key, *leader, *output);
        //     });
        // });

        cache
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        return self.sfb.score(weights) + self.stretch.score(weights);
    }

    // Calculates the score of a neighbor and applies it to the cache
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                let key_a = self.keys.get(a);
                let key_b = self.keys.get(b);
                self.remove_key(a);
                self.remove_key(b);
                self.add_key(a, key_b);
                self.add_key(b, key_a);
            }
            Neighbor::MagicStealBigram(MagicStealBigram(key, leader, output)) => {
                self.steal_bigram(key, leader, output);
            }
        }
    }

    // Add a key at pos. Key should currently be empty
    pub fn add_key(&mut self, pos: usize, u: CacheKey) {
        debug_assert!(self.keys.get(pos) == 0); // REPLACEMENT_CHAR maps to CacheKey 0
        self.keys.set(pos, u);
        self.sfb.add_key(pos, u);
        self.stretch.add_key(pos, u);

        // update fingers (no-op, fingers are static)
    }

    // Remove a key at pos. Key should currently contain something
    pub fn remove_key(&mut self, pos: usize) {
        debug_assert!(self.keys.get(pos) != 0); // REPLACEMENT_CHAR maps to CacheKey 0
        self.keys.set(pos, 0); // set to REPLACEMENT_CHAR index
        self.sfb.remove_key(pos);
        self.stretch.remove_key(pos);

        // update fingers
    }

    // Add a rule. Rule should currently be empty
    pub fn steal_bigram(&mut self, key: CacheKey, leader: CacheKey, output: CacheKey) {
        debug_assert!(self.magic.rules[key][leader] == 0); // REPLACEMENT_CHAR maps to CacheKey 0
        self.affected_grams = self.magic.add_rule(key, leader, output);
        self.sfb.steal_bigram(&self.affected_grams);
        self.stretch.steal_bigram(&self.affected_grams);
    }

    pub fn possible_neighbors(&self) -> &Vec<Neighbor> {
        &self.possible_neighbors
    }

    // Restore the cache to a previous state after testing a neighbor.
    pub fn copy_from(&mut self, other: &Self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                self.remove_key(a);
                self.remove_key(b);
                self.add_key(a, other.keys.get(a));
                self.add_key(b, other.keys.get(b));
            }
            Neighbor::MagicStealBigram(MagicStealBigram(key, leader, output)) => {
                // Revert the rule (assuming we can remove it)
                // This is a stub; actual implementation may need to track previous state
                // For now, just copy the whole magic cache? Not efficient.
                // We'll implement later.
                todo!("copy_from for MagicStealBigram")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramPair {
    pub pair: PosPair,
    pub dist: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{self, Analyzer};
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;

    #[test]
    fn test_apply_revert_neighbor_cache_integrity() {
        let data = Data::load("../data/english.json").expect("this should exist");
        let weights = dummy_weights();
        let layout = Layout::load(format!("../layouts/test/magic.dof"))
            .expect("this layout is valid and exists, soooo");
        let cache = CachedLayout::new(
            &AnalyzerData::new(&data),
            &layout.keyboard, // TODO
            &CharMapping::default(),
            &layout,
        );
        let reference = cache.clone();

        // Test key swap
        let diff = Neighbor::KeySwap(PosPair(0, 1));
        cache.apply_neighbor(diff);
        let revert = diff.revert(&cache);
        cache.apply_neighbor(revert);
        assert!(cache == reference);

        // Test arbitrary magic rule
        let diff = Neighbor::MagicRule(MagicRule(
            *self.magic.rules.iter().next().unwrap().0,
            analyzer.char_mapping.get_u('c'),
            analyzer.char_mapping.get_u('d'),
        ));
        analyzer.apply_neighbor(&mut cache, diff);
        let revert = diff.revert(&cache);
        analyzer.apply_neighbor(&mut cache, revert);
        assert!(cache == reference);
    }

    #[test]
    fn test_key_dist() {
        let k1 = "1 0 0 0"
            .parse::<PhysicalKey>()
            .expect("couldn't create k1");

        let k2 = "2 1 0 0"
            .parse::<PhysicalKey>()
            .expect("couldn't create k2");

        let d = dist(&k1, &k2, Finger::RP, Finger::RP);

        approx::assert_abs_diff_eq!(d, 2f64.sqrt(), epsilon = 1e-9);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_square_shapes() {
        fn print_key_info(layout: &crate::layout::Layout, c: char) {
            let i = match layout.keys.iter().position(|k| k == &c) {
                Some(i) => i,
                None => {
                    println!("layout '{}' does not contain '{c}'", layout.name);
                    return;
                }
            };

            let p = &layout.keyboard[i];
            let f = &layout.fingers[i];

            println!("{c} uses {f}, key: {p:?}")
        }

        let k1 = "6.25 3 1 1"
            .parse::<PhysicalKey>()
            .expect("couldn't create k1");

        let k2 = "3.75 4 6.25 1 "
            .parse::<PhysicalKey>()
            .expect("couldn't create k2");

        let d = dist(&k1, &k2, Finger::LP, Finger::LP);

        approx::assert_abs_diff_eq!(d, 1.0, epsilon = 1e-9);

        let layout = crate::layout::Layout::load("../layouts/qwerty.dof").unwrap();

        print_key_info(&layout, 'b');
        print_key_info(&layout, '␣');
    }

    #[test]
    fn update_cache_bigrams() {
        let (analyzer, layout) = analyzer_layout("rstn-oxey");

        analyzer.use_layout(layout, &[]);
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
}
