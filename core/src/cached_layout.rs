use fxhash::FxHashMap as HashMap;
use itertools::Itertools;
use libdof::prelude::{Finger, PhysicalKey, Shape};

use crate::{
    MagicCache,
    SFCache,
    StretchCache,
    TrigramCache,
    layout::PosPair,
    weights::{FingerWeights, Weights},
    REPLACEMENT_CHAR,
};

const KEY_EDGE_OFFSET: f64 = 0.5;

// CachedLayout contains the minimum mutable data used to define a layout and store scoring. Designed to copy quickly and without allocation.
// It is wrapped by Analyzer
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CachedLayout {
    keys: Vec<u8>,
    possible_neighbors: Vec<Neighbor>,
    sfb: SFCache,
    stretch: StretchCache,
    magic: MagicCache,
    fingers: Vec<Finger>, // Internal storage for key:finger mapping
}

impl CachedLayout {
    pub fn initialize(&mut self, data: &AnalyzerData, keyboard: &[PhysicalKey], char_mapping: &CharMapping, layout: &Layout) {
        // Zero initialize all of the cache data
        self.keys.clear();
        for _ in 0..layout.keys.len() {
            self.keys.push(REPLACEMENT_CHAR);
        }
        self.magic.initialize(&layout.magic, &char_mapping, &self.keys);
        self.sfb.initialize(&layout.fingers, &layout.keyboard, &self.keys);
        self.stretch.initialize(&keys, &fingers, &keyboard);

        layout.magic.iter().for_each( (key: u8, leader: u8, output: u8) => {
            self.add_rule( key, leader, output);
        });

        layout.keys.iter().enumerate().map(|(i: usize, u: u8)| {
            self.add_key(i, u);
        });
    }

    pub fn score(&self) -> i64 {
        return self.sfb.total + self.stretch.total;
    }

    // Calculates the score of a neighbor and applies it to the cache
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                self.remove_key(a);
                self.remove_key(b);
                self.add_key(a);
                self.add_key(b);
            }
            Neighbor::MagicStealBigram(MagicStealBigram(key, leader, output)) => {
                self.remove_rule(key, leader);
                self.add_rule(key, leader, output);
            }
        }
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

    pub fn possible_neighbors(&self) -> &Vec<Neighbor> {
        &self.possible_neighbors
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
        let reference = self.clone();

        // Test key swap
        let diff = Neighbor::KeySwap(PosPair(0, 1));
        analyzer.apply_neighbor(&mut cache, diff);
        let revert = diff.revert(&cache);
        analyzer.apply_neighbor(&mut cache, revert);
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
}
