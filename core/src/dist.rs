/*
 **************************************
 *         Distance Utilities
 **************************************
 */

use crate::types::CachePos;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

const KEY_EDGE_OFFSET: f64 = 0.5;

/// Precomputed distances between all pairs of key positions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DistCache {
    distances: Vec<Vec<i64>>,
}

impl DistCache {
    /// Create a new distance cache from keyboard layout and finger assignments.
    /// Distances are stored as i64 (multiplied by 100 for precision).
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger]) -> Self {
        assert_eq!(
            keyboard.len(),
            fingers.len(),
            "keyboard and fingers must have same length"
        );

        let len = keyboard.len();
        let distances = (0..len)
            .map(|i| {
                (0..len)
                    .map(|j| {
                        if i != j {
                            let (dx, dy) = dx_dy(&keyboard[i], &keyboard[j], fingers[i], fingers[j]);
                            (dx.hypot(dy) * 100.0) as i64
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();

        Self { distances }
    }

    /// Get the distance between two positions.
    #[inline]
    pub fn get(&self, p1: CachePos, p2: CachePos) -> i64 {
        self.distances[p1][p2]
    }
}

/// Compute dx/dy between two keys, accounting for finger lengths and key edges.
pub fn dx_dy(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> (f64, f64) {
    let flen = |f: Finger| match f {
        Finger::LP | Finger::RP => -0.15,
        Finger::LR | Finger::RR => 0.35,
        Finger::LM | Finger::RM => 0.25,
        Finger::LI | Finger::RI => -0.30,
        Finger::LT | Finger::RT => -1.80,
    };

    let ox1 = (k1.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
    let ox2 = (k2.width() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);

    let oy1 = (k1.height() * KEY_EDGE_OFFSET).min(KEY_EDGE_OFFSET);
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

    let xo = x_finger_overlap(f1, f2);

    match (
        (f1 as usize) > (f2 as usize),
        (f1 as usize) < (f2 as usize),
    ) {
        (true, false) if r1 < l2 + xo => (-dx, dy),
        (false, true) if l1 + xo > r2 => (-dx, dy),
        _ => (dx, dy),
    }
}

/// Compute x-overlap penalty based on finger positions.
pub fn x_overlap(dx: f64, dy: f64, f1: Finger, f2: Finger) -> f64 {
    let x_offset = x_finger_overlap(f1, f2);
    let dx_offset = x_offset - dx * 1.3;
    let dy_offset = 0.3333 * dy;
    (dx_offset + dy_offset).max(0.0)
}

/// Get the natural x-overlap between two fingers.
pub fn x_finger_overlap(f1: Finger, f2: Finger) -> f64 {
    match (f1, f2) {
        (Finger::LP, Finger::LR) | (Finger::LR, Finger::LP) => 0.8,
        (Finger::LR, Finger::LM) | (Finger::LM, Finger::LR) => 0.4,
        (Finger::LM, Finger::LI) | (Finger::LI, Finger::LM) => 0.1,
        (Finger::LI, Finger::LT) | (Finger::LT, Finger::LI) => -2.5,
        (Finger::RT, Finger::RI) | (Finger::RI, Finger::RT) => -2.5,
        (Finger::RI, Finger::RM) | (Finger::RM, Finger::RI) => 0.1,
        (Finger::RM, Finger::RR) | (Finger::RR, Finger::RM) => 0.4,
        (Finger::RR, Finger::RP) | (Finger::RP, Finger::RR) => 0.8,
        _ => 0.0,
    }
}
