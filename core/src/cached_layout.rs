use fxhash::FxHashMap as HashMap;
use itertools::Itertools;
use libdof::prelude::{Finger, PhysicalKey, Shape};

use crate::{
    magic::MagicCache,
    same_finger::SFCache,
    stretches::StretchCache,
    trigrams::TrigramCache,
    layout::PosPair,
    weights::{FingerWeights, Weights},
    REPLACEMENT_CHAR,
};

const KEY_EDGE_OFFSET: f64 = 0.5;

// CachedLayout contains the minimum mutable data used to define a layout and store scoring. Designed to copy quickly and without allocation.
// It is wrapped by Analyzer
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CachedLayout {
    pub keys: Vec<u8>,
    pub sfb: SFCache,
    pub stretch: StretchCache,
    pub magic: MagicCache,
    fingers: Vec<Vec<BigramPair>>, // Internal storage for key:finger mapping
}

impl CachedLayout {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(data: &AnalyzerData, layout: &Layout) -> Self {
        // Zero initialize all of the cache data
        let keys = Box::new([REPLACEMENT_CHAR; layout.keys.len()]);
        let magic = initialize_magic_cache(&layout.magic, &char_mapping, &keys);
        let stretch = StretchCache::new(&keys, &fingers, &keyboard, &self.weights);
        let sfb = self.sfb_cache_initialize(&keys, &fingers, &keyboard);

        let cache = CachedLayout { keys, stretch, sfb };

        // Build the current score for the layout by adding all the current keys and rules
        layout.keys.iter().for_each( (i: usize, u: u8) => {
            cache.add_key(i, u);
        });

        layout.magic.iter().for_each( (key: u8, leader: u8, output: u8) => {
            cache.add_rule( key, leader, output);
        })

        cache
    }

    pub fn score(&self) -> i64 {
        return self.sfb.total + self.stretch.total;
    }

    // Add a key at pos. Key should currently be empty
    pub fn add_key(&mut self, pos: usize, u: u8) {
        debug_assert!(self.keys[pos] == REPLACEMENT_CHAR);
        self.sfb.add_key(pos, u);
        self.stretch.add_key(pos, u);

        // update fingers
        finger
    }

    // Remove a key at pos. Key should currently contain something
    pub fn remove_key(&mut self, pos: usize) {
        debug_assert!(self.keys[pos] != REPLACEMENT_CHAR);
        self.sfb.remove_key(pos);
        self.stretch.remove_key(pos);

        // update fingers

    }

    // Add a rule. Rule should currently be empty
    pub fn add_rule(&mut self, key: u8, leader: u8, output: u8) {
        debug_assert!(self.magic.rules[key][leader] == REPLACEMENT_CHAR);
        affected_grams = self.magic.add_rule(key, leader, output);
        self.sf.add_rule(affected_grams);
        self.stretch.add_rule(affected_bgs);
    }

    // Remove a rule. Rule should currently contain something
    pub fn remove_rule(&mut self, key: u8, leader: u8, output: u8) {
        debug_assert!(self.magic.rules[key][leader] != REPLACEMENT_CHAR);
        affected_grams = self.magic.remove_rule(key, leader, output);
        self.sf.add_rule(affected_grams);
        self.stretch.add_rule(affected_grams);
    }

    pub fn copy(&mut self, other: &Self) {
        // TODO: move to metrics files
        // self.keys.copy_from_slice(&other.keys);
        // let _ = self
        //     .sfb
        //     .weighted_sfb_indices
        //     .fingers
        //     .iter_mut()
        //     .zip(other.sfb.weighted_sfb_indices.fingers.iter())
        //     .map(|(s, o)| s.copy_from_slice(o));
        // self.sfb
        //     .weighted_sfb_indices
        //     .all
        //     .copy_from_slice(&other.sfb.weighted_sfb_indices.all);

        // let _ = self
        //     .sfb
        //     .unweighted_sfb_indices
        //     .fingers
        //     .iter_mut()
        //     .zip(other.sfb.unweighted_sfb_indices.fingers.iter())
        //     .map(|(s, o)| s.copy_from_slice(o));
        // self.sfb
        //     .unweighted_sfb_indices
        //     .all
        //     .copy_from_slice(&other.sfb.unweighted_sfb_indices.all);

        // self.sfb.per_finger.copy_from_slice(&*other.sfb.per_finger);
        // self.sfb.total = other.sfb.total;

        // self.stretch
        //     .all_pairs
        //     .copy_from_slice(&other.stretch.all_pairs);
        // let _ = self
        //     .stretch
        //     .per_keypair
        //     .iter_mut()
        //     .map(|(pair, bg)| bg.copy_from_slice(other.stretch.per_keypair.get(pair).unwrap()));
        // self.stretch.total = other.stretch.total;
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
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load(format!("../layouts/test/magic.dof"))
            .expect("this layout is valid and exists, soooo");
        let mut cache = analyzer.cached_layout(layout, &[]);
        let reference = cache.clone();

        // Test key swap
        let diff = Neighbor::KeySwap(PosPair(0, 1));
        analyzer.apply_neighbor(&mut cache, diff);
        let revert = diff.revert(&cache);
        analyzer.apply_neighbor(&mut cache, revert);
        assert!(cache == reference);

        // Test arbitrary magic rule
        let diff = Neighbor::MagicRule(MagicRule(
            *cache.magic.rules.iter().next().unwrap().0,
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
}
