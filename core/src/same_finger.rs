/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

pub struct SfBigramPair {
    pub other_pos: usize,
    pub dist: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFCache {
    sfb_per_finger: Box<[i64; 10]>,                 // Cumulative SFB dist per finger
    total_sfbs: i64,                                // SFB count
    sfs_per_finger: Box<[i64; 10]>,                 // Cumulative SFS dist per finger
    total_sfs: i64,                                 // SFS count
    use_per_finger: Box<[i64; 10]>,                 // Finger use
    total_bg: i64,                                  // Total non-zero-freq BGs (for normalizing)
    total_sg: i64,                                  // Total non-zero-freq SGs (for normalizing)
    key_to_pos: Box<[usize]>,                       // Key position from key code
    sfbg_dist_per_key: Box<[Box<[SfBigramPair]>]>,  // One-time calculated key pos -> list of bg dist on same finger
}

impl SFCache {
    // Zero initialize
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey]) -> Self {
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
        let sfb_per_finger = Box<i64>::new([0; 10]);
        let total_sfbs = 0;
        let sfs_per_finger = Box<i64>::new([0; 10]);
        let total_sfs = 0;
        let use_per_finger = Box<i64>::new([0; 10]);
        let total_bg = 0;
        let total_sg = 0;
        let key_to_pos = Box<usize>::new([256; 256]); // invalid index, u8 max size

        let sfbg_dist_per_key: Box<[Box<[SfBigramPair]>]> = 
            fingers
                .iter()
                .zip(keyboard)
                .map(|(finger, physical_1)| {
                    fingers.iter()
                    .zip(keyboard)
                    .zip(0u8)
                    .filter(|((f, physical_2), pos)| (f == &finger))
                    .map(|((f, physical_2), pos)| SfBigramPair {
                        other_pos: pos,
                        dist: (dist(physical_1, physical_2, Finger::LP, Finger::LP) * 100.0) as i64,
                    })
                    .collect::<Box<[SfBigramPair]>>()
                }).collect::<Box<Box<[SfBigramPair]>>>();

        let key_to_pos = Box<CharMapping::

        Self {
            sfb_per_finger,
            sfs_per_finger,
            keypair_dist_per_finger,
        }
    }

    pub fn fast_copy(&mut self, other: &SFCache) {
        // TODO: Copy vars, but not pre-calc
    }

    pub fn score(&self, weights: &Weights) {
        // TODO: normalize
        Fingers::FINGERS.iter().map(|f| i64 {
            self.sfb_per_finger[f] * weights.finger[f] * weights.sfb +
            self.sfs_per_finger[f] * weights.finger[f] * weights.sfs
        }).sum();
    }

    pub fn stats(&self, stats: &mut Stats) {
        // TODO
    }

pub fn add_key(&mut self, keys: &Box<[u8]>, magic: &MagicCache, pos: usize, key: u8) {
    let sfb = self.sfbg_dist_per_key[pos]
        .iter()
        .map(|sfbg: SfBigramPair| i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_bigram(u1, u2)
                    + magic.get_bigram(u2, u1))
                    * sfbg.dist
            }
        )
        .sum();
    self.sfb_per_finger[fingers[pos]] += sfb;
    // TODO: sfb count, normalizing

    let sfs = self.sfbg_dist_per_key[pos]
    .iter()
    .map(|sfbg: SfBigramPair| i64 {
            let u1 = keys[pos];
            let u2 = keys[sfbg.other_pos];

            (magic.get_skipgram(u1, u2)
                + magic.get_skipgram(u2, u1))
                * sfbg.dist
        }
    )
    .sum();
    self.sfs_per_finger[fingers[pos]] += sfs;

    self.key_to_pos[key] = pos;
}

pub fn remove_key(&mut self, keys: &Box<[u8]>, magic: &MagicCache, pos: usize, key: u8) {
    let sfb = self.sfbg_dist_per_key[pos]
        .iter()
        .map(|sfbg: SfBigramPair| i64 {
                let u1 = keys[pos];
                let u2 = keys[sfbg.other_pos];

                (magic.get_bigram(u1, u2)
                    + magic.get_bigram(u2, u1))
                    * sfbg.dist
            }
        )
        .sum();
    self.sfb_per_finger[fingers[pos]] -= sfb;
    // TODO: sfb count, normalizing

    let sfs = self.sfbg_dist_per_key[pos]
    .iter()
    .map(|sfbg: SfBigramPair| i64 {
            let u1 = keys[pos];
            let u2 = keys[sfbg.other_pos];

            (magic.get_skipgram(u1, u2)
                + magic.get_skipgram(u2, u1))
                * sfbg.dist
        }
    )
    .sum();
    self.sfs_per_finger[fingers[pos]] -= sfs;

    self.key_to_pos[key] = 256;
}

    pub fn add_rule(&mut self, magic: &MagicCache, affected_grams: &[DeltaGram]) {
            for gram in &cache.magic.affected_grams {
                match gram {
                    DeltaGram::Bigram(bg) => {
                        self.sfbg_dist_per_key[self.key_to_pos[bg.a]]
                        .iter_mut()
                        .filter(|(pos, dist)| (self.key_to_pos[bg.b] == pos)) // Is SF
                        .map(|(pos, dist)| {
                            self.sfb_per_finger(fingers[pos]) += (bg.new - bg.old) * dist;
                        })
                    }
                    DeltaGram::Skipgram(bg) => {
                        self.sfbg_dist_per_key[self.key_to_pos[bg.a]]
                        .iter_mut()
                        .filter(|(pos, dist)| (self.key_to_pos[bg.b] == pos)) // Is SF
                        .map(|(pos, dist)| {
                            self.sfs_per_finger(fingers[pos]) += (bg.new - bg.old) * dist;
                        })
                    }
                    _ => { /* Trigrams are not part of SFBs */ }
                }
            }
        }
    }


