use fxhash::FxHashMap as HashMap;
use itertools::Itertools;
use libdof::prelude::{Finger, PhysicalKey, Shape};
use std::sync::Arc;

use crate::{
    analyzer_data::AnalyzerData, char_mapping::CharMapping, layout::PosPair,
    weights::FingerWeights, REPLACEMENT_CHAR,
};

const KEY_EDGE_OFFSET: f64 = 0.5;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CachedLayout {
    pub name: String,
    pub keys: Box<[u8]>,
    pub fingers: Box<[Finger]>,
    pub keyboard: Box<[PhysicalKey]>,
    pub shape: Shape,
    pub char_mapping: Arc<CharMapping>,
    pub possible_swaps: Box<[PosPair]>,
    pub weighted_sfb_indices: SfbIndices,
    pub unweighted_sfb_indices: SfbIndices,
    pub weighted_bigrams: BigramCache,
    pub stretch_bigrams: StretchCache,
}

impl CachedLayout {
    #[inline]
    pub fn swap(&mut self, PosPair(k1, k2): PosPair) {
        self.keys.swap(k1 as usize, k2 as usize);
    }

    pub fn char(&self, pos: u8) -> Option<char> {
        let u = self.keys.get(pos as usize)?;

        match self.char_mapping.get_c(*u) {
            REPLACEMENT_CHAR => None,
            c => Some(c),
        }
    }
}

impl std::fmt::Display for CachedLayout {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramPair {
    pub pair: PosPair,
    pub dist: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SfbIndices {
    pub fingers: Box<[Box<[BigramPair]>; 10]>,
    pub all: Box<[BigramPair]>,
}

impl SfbIndices {
    pub fn get_finger(&self, finger: Finger) -> &[BigramPair] {
        &self.fingers[finger as usize]
    }

    pub fn new(
        fingers: &[Finger],
        keyboard: &[PhysicalKey],
        finger_weights: &FingerWeights,
    ) -> Self {
        assert!(
            fingers.len() <= u8::MAX as usize,
            "Too many keys to index with u8, max is {}",
            u8::MAX
        );
        assert_eq!(
            fingers.len(),
            keyboard.len(),
            "finger len is not the same as keyboard len: "
        );

        let fingers: Box<[_; 10]> = Finger::FINGERS
            .map(|finger| {
                fingers
                    .iter()
                    .zip(keyboard)
                    .zip(0u8..)
                    .filter_map(|((f, k), i)| (f == &finger).then_some((k, i)))
                    .tuple_combinations::<(_, _)>()
                    .map(|((k1, i1), (k2, i2))| BigramPair {
                        pair: PosPair(i1, i2),
                        dist: (dist(k1, k2, Finger::LP, Finger::LP) * 100.0) as i64
                            * finger_weights.get(finger),
                    })
                    .collect::<Box<_>>()
            })
            .into();

        let all = fingers
            .iter()
            .flat_map(|f| f.iter())
            .cloned()
            .collect::<Box<_>>();

        Self { fingers, all }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BigramCache {
    pub total: i64,
    pub per_finger: Box<[i64; 10]>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    pub all_pairs: Box<[BigramPair]>,
    pub per_keypair: HashMap<PosPair, Box<[BigramPair]>>,
    pub total: i64,
}

impl StretchCache {
    pub fn new(
        keys: &[char],
        fingers: &[Finger],
        keyboard: &[PhysicalKey],
        data: &AnalyzerData,
    ) -> Self {
        assert!(
            fingers.len() <= u8::MAX as usize,
            "Too many keys to index with u8, max is {}",
            u8::MAX
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
                let diff = (f1 as u8).abs_diff(f2 as u8) as f64;
                let fd = diff * 1.35;
                // let minimum_diff = diff * 0.9;
                let (dx, dy) = dx_dy(k1, k2, f1, f2);
                let negative_lsb = 0.0; //(minimum_diff - dx.abs() - 1.0).max(0.0) * 2.0;
                let dist = dx.hypot(dy);

                let xo = x_overlap(dx, dy, f1, f2);

                let stretch = dist + xo + negative_lsb - fd;

                // if stretch > 0.001 {
                //     println!("{_c1}{_c2}: {}", (stretch * 100.0) as i64);
                // }

                (stretch > 0.001).then_some(BigramPair {
                    pair: PosPair(i1 as u8, i2 as u8),
                    dist: (stretch * 100.0) as i64,
                })
            })
            .collect::<Box<[_]>>();

        // println!("pair count: {}", all_pairs.len());

        let per_keypair = (0..(fingers.len() as u8))
            .cartesian_product(0..(fingers.len() as u8))
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
                    let u1 = keys[*a as usize];
                    let u2 = keys[*b as usize];

                    (data.get_stretch_weighted_bigram([u1, u2])
                        + data.get_stretch_weighted_bigram([u2, u1]))
                        * dist
                },
            )
            .sum();

        Self {
            all_pairs,
            per_keypair,
            total,
        }
    }
}

fn x_finger_overlap(f1: Finger, f2: Finger) -> f64 {
    use Finger::*;

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
    let x_offset = x_finger_overlap(f1, f2);

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
    //     (Hand::Left, Hand::Left) => match ((f1 as u8) > (f2 as u8), (f1 as u8) < (f2 as u8)) {
    //         (true, false) if r1 < l2 => (-dx, dy),
    //         (false, true) if l1 > r2 => (-dx, dy),
    //         _ => (dx, dy),
    //     },
    //     (Hand::Right, Hand::Right) => match ((f2 as u8) > (f1 as u8), (f2 as u8) < (f1 as u8)) {
    //         (true, false) if r1 > l2 => (-dx, dy),
    //         (false, true) if l1 < r2 => (-dx, dy),
    //         _ => (dx, dy),
    //     },
    //     _ => (dx, dy)
    // }
    match ((f1 as u8) > (f2 as u8), (f1 as u8) < (f2 as u8)) {
        (true, false) if r1 < l2 + xo => (-dx, dy),
        (false, true) if l1 + xo > r2 => (-dx, dy),
        _ => (dx, dy),
    }
}

fn dist(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> f64 {
    let (dx, dy) = dx_dy(k1, k2, f1, f2);

    dx.hypot(dy)
}

#[cfg(test)]
mod tests {
    use super::*;

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
