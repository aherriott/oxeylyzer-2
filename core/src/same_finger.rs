/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

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
pub struct SFBCache {
    pub total: i64,
    pub weighted_sfb_indices: SfbIndices,
    pub unweighted_sfb_indices: SfbIndices,
    pub per_finger: Box<[i64; 10]>,
}

// Top level update function for SFBCache
pub fn sfb_neighbor_update(
    &self,
    cache: &mut CachedLayout,
    neighbor: &Neighbor,
    apply: bool,
) -> i64 {
    if !self.analyze_bigrams {
        return 0;
    }

    match neighbor {
        Neighbor::KeySwap(PosPair(a, b)) => {
            // For keyswaps, it's easier to apply and revert the change to the cache keys (final cache.keys apply handled elsewhere)
            cache.apply_neighbor(*neighbor);
            let f1 = cache.fingers[*a as usize];
            let f2 = cache.fingers[*b as usize];

            if f1 == f2 {
                let b1 = self.sfb_finger_weighted_bigrams(cache, f1);

                let cache1 = cache.sfb.per_finger[f1 as usize];
                cache.apply_neighbor(*neighbor); // revert
                if apply {
                    // Update sfb cache
                    cache.sfb.per_finger[f1 as usize] += b1;
                    cache.sfb.total += b1 - cache1;
                    cache.sfb.total
                } else {
                    // just return the score
                    cache.sfb.total + b1 - cache1
                }
            } else {
                let b1 = self.sfb_finger_weighted_bigrams(cache, f1);
                let b2 = self.sfb_finger_weighted_bigrams(cache, f2);

                let cache1 = cache.sfb.per_finger[f1 as usize];
                let cache2 = cache.sfb.per_finger[f2 as usize];
                cache.apply_neighbor(*neighbor); // revert
                if apply {
                    cache.sfb.per_finger[f1 as usize] += b1;
                    cache.sfb.per_finger[f2 as usize] += b2;
                    cache.sfb.total += b1 + b2 - cache1 - cache2;
                    cache.sfb.total
                } else {
                    cache.sfb.total + b1 + b2 - cache1 - cache2
                }
            }
        }
        Neighbor::MagicRule(_) => {
            // update_affected_bigrams() needs to have been called first
            let mut score = cache.sfb.total;
            for gram in &cache.magic.affected_grams {
                match gram {
                    DeltaGram::Bigram(bg) => {
                        for (i, f) in cache.sfb.weighted_sfb_indices.fingers.iter().enumerate() {
                            for pair in f {
                                if (bg.a == pair.pair.0 || bg.a == pair.pair.1)
                                    && (bg.b == pair.pair.0 || bg.b == pair.pair.1)
                                {
                                    // The affected bg is part of an sfb
                                    let old = self.weights.sfbs * bg.old * pair.dist;
                                    let new = self.weights.sfbs * bg.new * pair.dist;
                                    if apply {
                                        cache.sfb.per_finger[i] += new - old;
                                        cache.sfb.total += new - old;
                                    } else {
                                        score += new - old;
                                    }
                                }
                            }
                        }
                    }
                    DeltaGram::Skipgram(bg) => {
                        for (i, f) in cache.sfb.weighted_sfb_indices.fingers.iter().enumerate() {
                            for pair in f {
                                if (bg.a == pair.pair.0 || bg.a == pair.pair.1)
                                    && (bg.b == pair.pair.0 || bg.b == pair.pair.1)
                                {
                                    // The affected bg is part of an sfb
                                    let old = self.weights.sfs * bg.old * pair.dist;
                                    let new = self.weights.sfs * bg.new * pair.dist;
                                    if apply {
                                        cache.sfb.per_finger[i] += new - old;
                                        cache.sfb.total += new - old;
                                    } else {
                                        score += new - old;
                                    }
                                }
                            }
                        }
                    }
                    _ => { /* Trigrams are not part of SFBs */ }
                }
            }

            if apply {
                cache.sfb.total
            } else {
                score
            }
        }
    }
}

fn sfb_cache_initialize(
    &self,
    keys: &[u8],
    fingers: &[Finger],
    keyboard: &[PhysicalKey],
    magic: &MagicCache,
) -> SFBCache {
    let weighted_sfb_indices = SfbIndices::new(&fingers, &keyboard, &self.weights.fingers);
    let unweighted_sfb_indices = SfbIndices::new(&fingers, &keyboard, &FingerWeights::default());

    // Save bigram scores per-finger
    let per_finger = Box::new(
        Finger::FINGERS.map(|f| self.sfb_finger_bigrams(f, keys, &weighted_sfb_indices, magic)),
    );
    // Save total
    let total = per_finger.iter().sum();

    let sfb = SFBCache {
        weighted_sfb_indices,
        unweighted_sfb_indices,
        per_finger,
        total,
    };
    sfb
}

// Bigram score for a finger, with finger weighting applied (also SFB & SFS weights)
fn sfb_finger_weighted_bigrams(&self, cache: &CachedLayout, f: Finger) -> i64 {
    self.sfb_finger_bigrams(
        f,
        &cache.keys,
        &cache.sfb.weighted_sfb_indices,
        &cache.magic,
    )
}

// Bigram score for a finger, no finger weighting applied (but with SFB & SFS weights)
fn sfb_finger_unweighted_bigrams(&self, cache: &CachedLayout, f: Finger) -> i64 {
    self.sfb_finger_bigrams(
        f,
        &cache.keys,
        &cache.sfb.unweighted_sfb_indices,
        &cache.magic,
    )
}

fn sfb_finger_bigrams(
    &self,
    f: Finger,
    keys: &[u8],
    indices: &SfbIndices,
    magic: &MagicCache,
) -> i64 {
    indices
        .get_finger(f)
        .iter()
        .map(
            |BigramPair {
                 pair: PosPair(a, b),
                 dist,
             }| {
                let u1 = keys[*a as usize];
                let u2 = keys[*b as usize];

                (self.sfb_get_weighted_bigram(keys, magic, u1, u2)
                    + self.sfb_get_weighted_bigram(keys, magic, u2, u1))
                    * dist
            },
        )
        .sum()
}

// Simple getter that maps a keycode pair to the associated weighted bigram score
fn sfb_get_weighted_bigram(&self, keys: &[u8], magic: &MagicCache, a: u8, b: u8) -> i64 {
    let mapping_len = self.data.mapping.len();
    let u1 = a as usize;
    let u2 = b as usize;
    let bg = *magic.bigrams.get(u1 * mapping_len + u2).unwrap();
    let sg = *magic.skipgrams.get(u1 * mapping_len + u2).unwrap();

    self.weights.sfbs * bg + self.weights.sfs * sg
}
