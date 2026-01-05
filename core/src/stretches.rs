/*
 **************************************
 *            Stretches
 **************************************
 */

use crate::{
    cached_layout::BigramPair, layout::PosPair, magic::MagicCache, types::CacheKey,
    weights::Weights,
};
use itertools::Itertools;
use libdof::{dofinitions::Finger, keyboard::PhysicalKey};

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
        weights: &Weights,
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

                let xo = Self::x_overlap(dx, dy, f1, f2);

                let stretch = dist + xo + negative_lsb - fd;

                // if stretch > 0.001 {
                //     println!("{_c1}{_c2}: {}", (stretch * 100.0) as i64);
                // }

                (stretch > 0.001).then_some(BigramPair {
                    pair: PosPair(i1 as CacheKey, i2 as CacheKey),
                    dist: (stretch * 100.0) as i64,
                })
            })
            .collect::<Box<[_]>>();

        // println!("pair count: {}", all_pairs.len());

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
                    let bg = magic.bigrams.get(u1 + u2 * keys.len()).unwrap()
                        + magic.bigrams.get(u2 + u1 * keys.len()).unwrap();
                    let sg = magic.skipgrams.get(u1 + u2 * keys.len()).unwrap()
                        + magic.skipgrams.get(u2 + u1 * keys.len()).unwrap();

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

    pub fn add_key() {}

    pub fn remove_key() {}

    pub fn score(&self, weights: &Weights) -> i64 {
        // TODO
        self.total * weights.stretches
    }

    fn x_finger_overlap(f1: Finger, f2: Finger) -> f64 {
        match (f1, f2) {
            (LP, LR) => 0.8,
            (LR, LP) => 0.8,
            (LR, LM) => 0.4,
            (LM, LR) => 0.4,
            (LM, LI) => 0.1,
            (LI, LM) => 0.1,
            (LI, LT) => -2.5,
            (LT, LI) => -2.5,
            (RT, RI) => -2.5,
            (RI, RT) => -2.5,
            (RI, RM) => 0.1,
            (RM, RI) => 0.1,
            (RM, RR) => 0.4,
            (RR, RM) => 0.4,
            (RR, RP) => 0.8,
            (RP, RR) => 0.8,
            _ => 0.0,
        }
    }

    fn x_overlap(dx: f64, dy: f64, f1: Finger, f2: Finger) -> f64 {
        let x_offset = Self::x_finger_overlap(f1, f2);

        let dx_offset = x_offset - dx * 1.3;
        let dy_offset = 0.3333 * dy;

        (dx_offset + dy_offset).max(0.0)
    }

    fn dx_dy(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> (f64, f64) {
        let flen = |f: Finger| match f {
            Finger::LP | Finger::RP => -0.15,
            Finger::LR | Finger::RR => 0.35,
            Finger::LM | Finger::RM => 0.25,
            Finger::LI | Finger::RI => -0.30,
            Finger::LT | Finger::RT => -1.80,
        };

        let ox1 = (k1.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
        let ox2 = (k1.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);

        let oy1 = (k2.height() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
        let oy2 = (k2.height() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);

        let l1 = k1.x() + ox1;
        let r1 = k1.x() - ox1 + k1.width();
        let t1 = k1.y() + oy1 + flen(f1);
        let b1 = k1.y() - oy1 + k1.height() + flen(f1);

        let l2 = k2.x() + ox2;
        let r2 = k2.x() - ox2 + k2.width();
        let t2 = k2.y() + oy2 + flen(f2);
        let b2 = k2.y() - oy2 + k2.height() + flen(f2);

        let dx = (l1.max(l2) - r1.min(r2)).max(0.0);
        let dy = (t1.max(t2) - b1.min(b2)).max(0.0);

        // Checks whether or not a finger is below or to the side of another finger, in which case the
        // distance is considered negative. To the side meaning, where the distance between qwerty `er`
        // pressed with middle and index is considered 1, if each key were pressed with the other
        // finger, the distance is negative (because who the fuck is doing that, that's not good).

        let xo = x_finger_overlap(f1, f2);

        // match (f1.hand(), f2.hand()) {
        //     (Hand::Left, Hand::Left) => match ((f1 as CacheKey) > (f2 as CacheKey), (f1 as CacheKey) < (f2 as CacheKey)) {
        //         (true, false) if r1 < l2 => (-dx, dy),
        //         (false, true) if l1 > r2 => (-dx, dy),
        //         _ => (dx, dy),
        //     },
        //     (Hand::Right, Hand::Right) => match ((f2 as CacheKey) > (f1 as CacheKey), (f2 as CacheKey) < (f1 as CacheKey)) {
        //         (true, false) if r1 > l2 => (-dx, dy),
        //         (false, true) if l1 < r2 => (-dx, dy),
        //         _ => (dx, dy),
        //     },
        //     _ => (dx, dy)
        // }
        match (
            (f1 as CacheKey) > (f2 as CacheKey),
            (f1 as CacheKey) < (f2 as CacheKey),
        ) {
            (true, false) if r1 < l2 + xo => (-dx, dy),
            (false, true) if l1 + xo > r2 => (-dx, dy),
            _ => (dx, dy),
        }
    }

    fn dist(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> f64 {
        let (dx, dy) = dx_dy(k1, k2, f1, f2);

        dx.hypot(dy)
    }

    pub fn stretch_neighbor_update(
        &self,
        cache: &mut CachedLayout,
        neighbor: &Neighbor,
        apply: bool,
    ) -> i64 {
        if !self.analyze_stretches {
            return 0;
        }
        match neighbor {
            Neighbor::KeySwap(pair) => {
                if pair.0 == pair.1 {
                    // nothing to do
                    return cache.stretch.total;
                }

                // TODO: Does this even work? Doesn't seems to take into account new cache.keys after all_pairs is initialized
                let revert = cache.apply_neighbor(*neighbor);
                let stretch_new = self.keypair_stretch(cache, pair);

                cache.apply_neighbor(revert);
                let stretch_old = self.keypair_stretch(cache, pair);

                if apply {
                    // TODO: is this correct? Feels like it should be new - old
                    cache.stretch.total += stretch_old - stretch_new;
                    cache.stretch.total
                } else {
                    cache.stretch.total + stretch_old - stretch_new
                }
            }
            Neighbor::MagicRule(_) => 0,
        }
    }

    fn keypair_stretch(&self, cache: &CachedLayout, pair: &PosPair) -> i64 {
        cache
            .stretch
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

        let bg = cache.magic.bigrams.get(u1 * mapping_len + u2).unwrap()
            + cache.magic.bigrams.get(u2 * mapping_len + u1).unwrap();
        let sg = cache.magic.skipgrams.get(u1 * mapping_len + u2).unwrap()
            + cache.magic.skipgrams.get(u2 * mapping_len + u1).unwrap();

        // TODO: should this be sfb / sfs? If you weight sfbs more, the skipgrams instead get weighted more here.
        // Should it be the other way around? Would a weighted average make more sense?
        let sfb_over_sfs = (self.weights.sfbs as f64) / (self.weights.sfs as f64);
        (bg + (sg as f64 * sfb_over_sfs) as i64) * self.weights.stretches
    }
}
