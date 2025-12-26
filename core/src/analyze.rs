use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use itertools::Itertools;
use libdof::{dofinitions::Finger, magic::MagicKey, prelude::PhysicalKey};
use nanorand::{Rng, WyRand};

use crate::{
    analyzer_data::AnalyzerData,
    cached_layout::*,
    char_mapping::CharMapping,
    data::Data,
    layout::*,
    trigrams::TRIGRAMS,
    weights::{FingerWeights, Weights},
};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrigramData {
    pub sft: i64,
    pub sfb: i64,
    pub inroll: i64,
    pub outroll: i64,
    pub alternate: i64,
    pub redirect: i64,
    pub onehandin: i64,
    pub onehandout: i64,
    pub thumb: i64,
    pub invalid: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analyzer {
    pub data: AnalyzerData,
    pub weights: Weights,
    pub analyze_bigrams: bool,
    pub analyze_stretches: bool,
    pub analyze_trigrams: bool,
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
        }
    }

    pub fn score(&self, layout: &Layout) -> i64 {
        let cache = self.cached_layout(layout.clone(), &[]);

        self.score_cache(&cache)
    }

    pub fn score_cache(&self, cache: &CachedLayout) -> i64 {
        // more metrics will obviously also go here
        cache.sfb.total + cache.stretch.total
    }

    pub fn mapping(&self) -> &CharMapping {
        &self.data.mapping
    }

    pub fn cached_layout(&self, layout: Layout, pins: &[usize]) -> CachedLayout {
        let keys = layout
            .keys
            .iter()
            .map(|&c| self.data.mapping.get_u(c))
            .collect::<Box<_>>();

        let name = layout.name;
        let fingers = layout.fingers;
        let shape = layout.shape;
        let char_mapping = self.data.mapping.clone();
        let keyboard = layout.keyboard;

        // Initialize the Magic/Repeat key cache
        let magic = self.initialize_magic_cache(&layout.magic, &char_mapping, &keys);

        // All possible keyswaps to create a neighbor layout
        let possible_swaps = (0..(keys.len() as u8))
            .filter(|v| !pins.contains(&(*v as usize)))
            .tuple_combinations::<(_, _)>()
            .map(|pair| Neighbor::KeySwap(pair.into()));
        // All possible new magic rules to create a neighbor layout
        let possible_rules = magic.rules.iter().flat_map(|(key, _)| {
            keys.iter().flat_map(|lead| {
                keys.iter()
                    .map(|output| Neighbor::MagicRule(MagicRule(*key, *lead, *output)))
            })
        });
        let possible_neighbors = possible_swaps.chain(possible_rules).collect();

        // Initialize strech cache
        let stretch = StretchCache::new(&keys, &fingers, &keyboard, &self.weights, &magic);

        // Initialize the SFBCache
        let sfb = self.sfb_cache_initialize(&keys, &fingers, &keyboard, &magic);

        let cache = CachedLayout {
            name,
            keys,
            fingers,
            keyboard,
            shape,
            char_mapping,
            possible_neighbors,
            stretch,
            sfb,
            magic,
        };

        cache
    }

    pub fn greedy_improve(&self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        let mut cache = self.cached_layout(layout, pins);
        let mut best_score = self.score_cache(&cache);

        while let Some((neighbor, score)) = self.best_neighbor(&mut cache) {
            if score <= best_score {
                break;
            }

            best_score = score;
            self.apply_neighbor(&mut cache, neighbor);
        }

        (cache.into(), best_score)
    }

    /*
     **************************************
     *         Neighbor actions
     **************************************
     */

    pub fn random_neighbor(&self, cache: &CachedLayout, rng: &mut WyRand) -> Neighbor {
        cache.possible_neighbors[rng.generate_range(0..cache.possible_neighbors.len())]
    }

    /**
     * Returns the best neighbor
     */
    pub fn best_neighbor(&self, cache: &mut CachedLayout) -> Option<(Neighbor, i64)> {
        let mut best_score = self.score_cache(cache);
        let mut best = None;

        // TODO: can we remove this clone?
        let neighbors = cache.possible_neighbors.clone();
        for neighbor in neighbors {
            let score = self.test_neighbor(cache, neighbor);

            if score > best_score {
                best_score = score;
                best = Some((neighbor.clone(), score));
            }
        }
        best
    }

    // Calculates the score of a neighbor without updating the cache
    pub fn test_neighbor(&self, cache: &mut CachedLayout, neighbor: Neighbor) -> i64 {
        self.update_cache(cache, &neighbor, false)
    }

    // Calculates the score of a neighbor and applies it to the cache
    pub fn apply_neighbor(&self, cache: &mut CachedLayout, neighbor: Neighbor) -> i64 {
        self.update_cache(cache, &neighbor, true)
    }

    // TODO(AI): Used to be pub(crate). Safe?
    fn update_cache(&self, cache: &mut CachedLayout, neighbor: &Neighbor, apply: bool) -> i64 {
        match neighbor {
            Neighbor::KeySwap(swap) => {
                // Nothing to do, just return
                if swap.0 == swap.1 {
                    return self.score_cache(cache);
                }
            }
            Neighbor::MagicRule(rule) => {
                // Calculate the impacted bigrams/skipgrams/trigrams
                self.update_affected_bigrams(cache, &rule, apply);
            }
        }

        let mut score = 0;

        if self.analyze_bigrams {
            score += self.sfb_neighbor_update(cache, &neighbor, apply);
        }

        if self.analyze_stretches {
            self.stretch_neighbor_update(cache, &neighbor, apply);
        }

        score
    }

    /*
     **************************************
     *            SFBs & SFSs
     **************************************
     */

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
                            for (i, f) in cache.sfb.weighted_sfb_indices.fingers.iter().enumerate()
                            {
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
                            for (i, f) in cache.sfb.weighted_sfb_indices.fingers.iter().enumerate()
                            {
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
        let unweighted_sfb_indices =
            SfbIndices::new(&fingers, &keyboard, &FingerWeights::default());

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
        let u1 = a as usize;
        let u2 = b as usize;
        let bg = *magic.bigrams.get(u1 * keys.len() + u2).unwrap();
        let sg = *magic.skipgrams.get(u1 * keys.len() + u2).unwrap();

        self.weights.sfbs * bg + self.weights.sfs * sg
    }

    /*
     **************************************
     *    Magic Frequency (On-Demand)
     **************************************
     */

    /// Get effective bigram frequency accounting for magic key rules
    /// This works on the premise that magic keys "steal" frequency from their non-magic counterparts
    fn get_bigram_frequency_magic(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
        // Check if a or b are magic keys
        let a_is_magic = cache.magic.rules.contains_key(&a);
        let b_is_magic = cache.magic.rules.contains_key(&b);

        match (a_is_magic, b_is_magic) {
            (false, false) => {
                // Neither is magic: check if this bigram is stolen by a magic rule
                // Priority order: iterate magic keys in order, first match wins
                for (magic_key, rules) in &cache.magic.rules {
                    if let Some(&output) = rules.get(&a) {
                        if output == b {
                            // a -> b is fully stolen by a -> magic_key
                            return 0;
                        }
                    }
                }
                // Not stolen, return base frequency
                self.data.get_bigram_u([a, b])
            }
            (true, false) => {
                // a is magic: return sum of tg(l,a,b) where l -> a is a magic rule
                let mut freq = 0;
                if let Some(rules) = cache.magic.rules.get(&a) {
                    for (&leader, &output) in rules.iter() {
                        if output == a {
                            // Leader produces 'a' as magic output
                            freq += self.data.get_trigram_u([leader, a, b]);
                        }
                    }
                }
                freq
            }
            (false, true) => {
                // b is magic: check if there's a rule a -> b
                if let Some(rules) = cache.magic.rules.get(&b) {
                    if let Some(&output) = rules.get(&a) {
                        if output == b {
                            // There IS a rule a -> b
                            let mut freq = self.data.get_bigram_u([a, b]);

                            // Subtract frequency stolen by other magic keys with same rule
                            for (other_magic, other_rules) in &cache.magic.rules {
                                if other_magic == &b {
                                    continue;
                                } // Skip the current magic key
                                if let Some(&other_output) = other_rules.get(&a) {
                                    if other_output == b {
                                        // Higher priority magic key with same rule - it steals all frequency
                                        return 0;
                                    }
                                }
                            }

                            // Subtract tg(l,a,b) where l -> a is a magic rule
                            for (magic_key, rules) in &cache.magic.rules {
                                for (&leader, &output) in rules.iter() {
                                    if output == a {
                                        freq -= self.data.get_trigram_u([leader, a, b]);
                                    }
                                }
                            }

                            freq
                        } else {
                            // No rule a -> b
                            0
                        }
                    } else {
                        // No rule a -> b
                        0
                    }
                } else {
                    0
                }
            }
            (true, true) => {
                // Both are magic: return 0
                0
            }
        }
    }

    /// Get effective skipgram frequency accounting for magic key rules
    fn get_skipgram_frequency_magic(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
        // Check if a or b are magic keys
        let a_is_magic = cache.magic.rules.contains_key(&a);
        let b_is_magic = cache.magic.rules.contains_key(&b);

        match (a_is_magic, b_is_magic) {
            (false, false) => {
                // Neither is magic: return sg(a,b) minus sum(tg(a,l,b)) where l -> b is a magic rule
                let mut freq = self.data.get_skipgram_u([a, b]);

                for (magic_key, rules) in &cache.magic.rules {
                    if magic_key == &b {
                        // b is the magic key, find all leaders that produce b
                        for (&leader, &output) in rules.iter() {
                            if output == b {
                                // Leader produces b as magic output
                                freq -= self.data.get_trigram_u([a, leader, b]);
                            }
                        }
                    }
                }

                freq
            }
            (true, _) => {
                // a is magic: return 0 (need quadgrams for precise calculation)
                0
            }
            (false, true) => {
                // b is magic: return sum(tg(a,l,b)) where l -> b is a magic rule
                let mut freq = 0;
                if let Some(rules) = cache.magic.rules.get(&b) {
                    for (&leader, &output) in rules.iter() {
                        if output == b {
                            freq += self.data.get_trigram_u([a, leader, b]);
                        }
                    }
                }
                freq
            }
        }
    }

    /// Get effective trigram frequency accounting for magic key rules
    fn get_trigram_frequency_magic(&self, cache: &CachedLayout, a: u8, b: u8, c: u8) -> i64 {
        // Check which positions are magic keys
        let a_is_magic = cache.magic.rules.contains_key(&a);
        let b_is_magic = cache.magic.rules.contains_key(&b);
        let c_is_magic = cache.magic.rules.contains_key(&c);

        if a_is_magic {
            // a is magic: return 0
            return 0;
        }

        if b_is_magic && c_is_magic {
            // Both b and c are magic: return 0
            return 0;
        }

        if b_is_magic && !c_is_magic {
            // b is magic, c is not: if a -> b is a magic rule, return tg(a,b,c)
            if let Some(rules) = cache.magic.rules.get(&b) {
                if let Some(&output) = rules.get(&a) {
                    if output == b {
                        return self.data.get_trigram_u([a, b, c]);
                    }
                }
            }
            return 0;
        }

        if c_is_magic {
            // c is magic (b is not): if b -> c is a magic rule, return tg(a,b,c)
            if let Some(rules) = cache.magic.rules.get(&c) {
                if let Some(&output) = rules.get(&b) {
                    if output == c {
                        return self.data.get_trigram_u([a, b, c]);
                    }
                }
            }
            return 0;
        }

        // None are magic: check if a -> b or b -> c is a magic rule
        for (magic_key, rules) in &cache.magic.rules {
            if let Some(&output) = rules.get(&a) {
                if output == b {
                    // a -> b is a magic rule, trigram stolen
                    return 0;
                }
            }
            if let Some(&output) = rules.get(&b) {
                if output == c {
                    // b -> c is a magic rule, trigram stolen
                    return 0;
                }
            }
        }

        // No magic rules affect this trigram
        self.data.get_trigram_u([a, b, c])
    }

    /*
     **************************************
     *            Stretches
     **************************************
     */

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

    pub fn stretch_get_bigram(&self, cache: &CachedLayout, a: &u8, b: &u8) -> i64 {
        let u1 = *a as usize;
        let u2 = *b as usize;

        let bg = cache.magic.bigrams.get(u1 + u2 * cache.keys.len()).unwrap()
            + cache.magic.bigrams.get(u2 + u1 * cache.keys.len()).unwrap();
        let sg = cache
            .magic
            .skipgrams
            .get(u1 + u2 * cache.keys.len())
            .unwrap()
            + cache
                .magic
                .skipgrams
                .get(u2 + u1 * cache.keys.len())
                .unwrap();

        // TODO: should this be sfb / sfs? If you weight sfbs more, the skipgrams instead get weighted more here.
        // Should it be the other way around? Would a weighted average make more sense?
        let sfb_over_sfs = (self.weights.sfbs as f64) / (self.weights.sfs as f64);
        (bg + (sg as f64 * sfb_over_sfs) as i64) * self.weights.stretches
    }

    /*
     **************************************
     *     Magic and Repeat Remapping
     **************************************
     */

    fn initialize_magic_cache(
        &self,
        magic: &HashMap<char, MagicKey>,
        char_mapping: &CharMapping,
        keys: &Box<[u8]>,
    ) -> MagicCache {
        // Format the Layout magic rules into CachedLayout
        let mut rules: HashMap<u8, HashMap<u8, u8>> = HashMap::default();
        if !magic.is_empty() {
            for (key, magic_key) in magic {
                assert!(
                    magic_key.max_leading_length() == 1 && magic_key.max_output_length() == 1,
                    "Only \"simple\" magic keys are supported (one leader & one output key)"
                );
                let magic_u = char_mapping.get_u(*key);
                rules.insert(magic_u, HashMap::default());
                for k in keys {
                    match magic_key.rule(char_mapping.get_c(*k).to_string().as_str()) {
                        Some(output) => {
                            // If there's a rule already, add it to the cache
                            rules
                                .get_mut(&magic_u)
                                .unwrap()
                                .insert(*k, char_mapping.get_u(output.chars().next().unwrap()));
                        }
                        None => {
                            // If there's no rule, it's a repeat
                            rules.get_mut(&magic_u).unwrap().insert(*k, *k);
                        }
                    }
                }
            }
        }

        let mut bigrams = vec![0 as i64; keys.len().pow(2)].into_boxed_slice();
        let mut skipgrams = vec![0 as i64; keys.len().pow(2)].into_boxed_slice();
        let mut trigrams = vec![0 as i64; keys.len().pow(3)].into_boxed_slice();

        // First, copy all bg/sg/tg frequencies from data
        keys.iter().tuple_combinations().for_each(|(key1, key2)| {
            let u1 = *key1 as usize;
            let u2 = *key2 as usize;
            bigrams[u1 * keys.len() + u2] = self.data.get_bigram_u([*key1, *key2]);
            skipgrams[u1 * keys.len() + u2] = self.data.get_skipgram_u([*key1, *key2]);
            keys.iter().for_each(|key3| {
                let u3 = *key3 as usize;
                trigrams[u1 * keys.len().pow(2) + u2 * keys.len() + u3] =
                    self.data.get_trigram_u([*key1, *key2, *key3]);
            });
        });

        // Convenience closures
        let mut add_bg = |a: u8, b: u8, val: i64| {
            let u1 = a as usize;
            let u2 = b as usize;
            bigrams[u1 * keys.len() + u2] += val;
        };
        let mut add_sg = |a: u8, b: u8, val: i64| {
            let u1 = a as usize;
            let u2 = b as usize;
            skipgrams[u1 * keys.len() + u2] += val;
        };
        let mut add_tg = |a: u8, b: u8, c: u8, val: i64| {
            let u1 = a as usize;
            let u2 = b as usize;
            let u3 = c as usize;
            trigrams[u1 * keys.len().pow(2) + u2 * keys.len() + u3] += val;
        };

        // Next, apply the magic rules
        rules.iter().for_each(|(magic_key, rule)| {
            rule.iter().for_each(|(lead, output)| {
                // TODO: multiple magic key handling for when magic keys have duplicate rules
                // Leader -> Output bigram is now never pressed (vs leader -> magic key),
                add_bg(*lead, *magic_key, self.data.get_bigram_u([*lead, *output]));
                add_bg(
                    *lead,
                    *output,
                    -1 * self.data.get_bigram_u([*lead, *output]),
                );

                // For now, the magic-magic double-press bigram is not considered

                for key1 in keys {
                    // output -> key1 bigram stolen whenever the leader is pressed first
                    add_bg(
                        *magic_key,
                        *key1,
                        self.data.get_trigram_u([*lead, *output, *key1]),
                    );
                    add_bg(
                        *output,
                        *key1,
                        -1 * self.data.get_trigram_u([*lead, *output, *key1]),
                    );

                    // key1 -> output skipgram stolen whenever the leader is pressed in the middle
                    add_sg(
                        *key1,
                        *magic_key,
                        self.data.get_trigram_u([*key1, *lead, *output]),
                    );
                    add_sg(
                        *key1,
                        *output,
                        -1 * self.data.get_trigram_u([*key1, *lead, *output]),
                    );

                    // key1 -> lead -> output trigram always stolen
                    add_tg(
                        *key1,
                        *lead,
                        *magic_key,
                        self.data.get_trigram_u([*key1, *lead, *output]),
                    );
                    add_tg(
                        *key1,
                        *lead,
                        *output,
                        -1 * self.data.get_trigram_u([*key1, *lead, *output]),
                    );

                    // lead -> output -> key1 trigram always stolen
                    add_tg(
                        *lead,
                        *magic_key,
                        *key1,
                        self.data.get_trigram_u([*lead, *output, *key1]),
                    );
                    add_tg(
                        *lead,
                        *output,
                        *key1,
                        -1 * self.data.get_trigram_u([*lead, *output, *key1]),
                    );

                    // TODO: The following skipgrams/trigrams are not currently attributed to the magic key. We need quadgrams to do it correctly
                    // for key2 in keys {
                    //     // output -> key1 skipgram stolen whenever the leader is pressed first
                    //     *sg(*magic_key, *key2) += self.data.get_quadgram_u([*lead, *output, *key1, *key2]);
                    //     *sg(*output, *key2) -= self.data.get_quadgram_u([*lead, *output, *key1, *key2]);

                    //     // output -> key1 -> key2 trigram stolen whenever the leader is pressed first
                    //     *tg(*magic_key, *key1, *key2) += self.data.get_quadgram_u([*lead, *output, *key1, *key2]);
                    //     *tg(*output, *key1, *key2) -= self.data.get_quadgram_u([*lead, *output, *key1, *key2]);
                    // }
                }
                // rules.iter().for_each(|(magic_key2, rule2)| {
                //     rule2.iter().for_each(|(lead2, output2)| {
                //         // output -> output2 skipgram stolen when double magic occurs
                //         *sg(*magic_key, *magic_key2) += self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
                //         *sg(*output, *output2)  -= self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);

                //         // output -> lead2 -> output2 trigram stolen when double magic occurs
                //         *tg(*magic_key, *lead2, *magic_key2) += self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
                //         *tg(*output, *lead2, *output2)  -= self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
            });
        });

        // See the below update_affected_bigrams function for where this number comes from
        let affected_grams_len = 3 + keys.len() * 12;

        let cache = MagicCache {
            rules,
            affected_grams: vec![DeltaGram::default(); affected_grams_len].into_boxed_slice(),
            bigrams,
            skipgrams,
            trigrams,
        };
        cache
    }

    // Updates pre-allocated AffectedBigrams in the cached layout to help calculate new scores after a new MagicRule
    fn update_affected_bigrams(
        &self,
        cache: &mut CachedLayout,
        MagicRule(magic_key, lead, output): &MagicRule,
        apply: bool,
    ) {
        let mut i = 0;

        // Get the old output
        let old = *cache.magic.rules.get(magic_key).unwrap().get(lead).unwrap();

        // Convenience closures to increment bigrams/skipgrams/trigrams
        let add_bg = |magic: &mut MagicCache, len: usize, a: u8, b: u8, val: i64, i: &mut usize| {
            let u1 = a as usize;
            let u2 = b as usize;
            magic.affected_grams[*i] = DeltaGram::Bigram(DeltaBigram {
                a,
                b,
                old: magic.bigrams[u1 * len + u2],
                new: magic.bigrams[u1 * len + u2] + val,
            });
            if apply {
                magic.bigrams[u1 * len + u2] += val;
            }
            *i += 1;
        };
        let add_sg = |magic: &mut MagicCache, len: usize, a: u8, b: u8, val: i64, i: &mut usize| {
            let u1 = a as usize;
            let u2 = b as usize;
            magic.affected_grams[*i] = DeltaGram::Skipgram(DeltaBigram {
                a,
                b,
                old: magic.skipgrams[u1 * len + u2],
                new: magic.skipgrams[u1 * len + u2] + val,
            });
            if apply {
                magic.skipgrams[u1 * len + u2] += val;
            }
            *i += 1;
        };
        let add_tg =
            |magic: &mut MagicCache, len: usize, a: u8, b: u8, c: u8, val: i64, i: &mut usize| {
                let u1 = a as usize;
                let u2 = b as usize;
                let u3 = c as usize;
                magic.affected_grams[*i] = DeltaGram::Trigram(DeltaTrigram {
                    a,
                    b,
                    c,
                    old: magic.trigrams[u1 * len.pow(2) + u2 * len + u3],
                    new: magic.trigrams[u1 * len.pow(2) + u2 * len + u3] + val,
                });
                if apply {
                    magic.trigrams[u1 * len.pow(2) + u2 * len + u3] += val;
                }
                *i += 1;
            };

        // TODO: multiple magic/repeat key handling. Would need some mechanism to decide which magic/repeat key to use in case of duplicate rules
        // Leader -> Output bigram
        add_bg(
            &mut cache.magic,
            cache.keys.len(),
            *lead,
            old,
            self.data.get_bigram_u([*lead, old]),
            &mut i,
        ); // Restore old output
        add_bg(
            &mut cache.magic,
            cache.keys.len(),
            *lead,
            *magic_key,
            self.data.get_bigram_u([*lead, *output]) - self.data.get_bigram_u([*lead, old]),
            &mut i,
        ); // New magic_key bigram
        add_bg(
            &mut cache.magic,
            cache.keys.len(),
            *lead,
            *output,
            -1 * self.data.get_bigram_u([*lead, old]),
            &mut i,
        ); // New output bigram never pressed

        cache.keys.iter().for_each(|key1| {
            // output -> key1 bigram stolen whenever the leader is pressed first
            add_bg(
                &mut cache.magic,
                cache.keys.len(),
                old,
                *key1,
                self.data.get_trigram_u([*lead, old, *key1]),
                &mut i,
            );
            add_bg(
                &mut cache.magic,
                cache.keys.len(),
                *magic_key,
                *key1,
                self.data.get_trigram_u([*lead, *output, *key1])
                    - self.data.get_trigram_u([*lead, old, *key1]),
                &mut i,
            );
            add_bg(
                &mut cache.magic,
                cache.keys.len(),
                *output,
                *key1,
                -1 * self.data.get_trigram_u([*lead, *output, *key1]),
                &mut i,
            );

            // key1 -> output skipgram stolen whenever the leader is pressed in the middle
            add_sg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                old,
                self.data.get_trigram_u([*key1, *lead, old]),
                &mut i,
            );
            add_sg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                *magic_key,
                self.data.get_trigram_u([*key1, *lead, *output])
                    - self.data.get_trigram_u([*key1, *lead, old]),
                &mut i,
            );
            add_sg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                *output,
                -1 * self.data.get_trigram_u([*key1, *lead, *output]),
                &mut i,
            );

            // key1 -> lead -> output trigram always stolen
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                *lead,
                old,
                self.data.get_trigram_u([*key1, *lead, old]),
                &mut i,
            );
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                *lead,
                *magic_key,
                self.data.get_trigram_u([*key1, *lead, *output])
                    - self.data.get_trigram_u([*key1, *lead, old]),
                &mut i,
            );
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *key1,
                *lead,
                *output,
                -1 * self.data.get_trigram_u([*key1, *lead, *output]),
                &mut i,
            );

            // lead -> output -> key1 trigram always stolen
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *lead,
                old,
                *key1,
                self.data.get_trigram_u([*lead, old, *key1]),
                &mut i,
            );
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *lead,
                *magic_key,
                *key1,
                self.data.get_trigram_u([*lead, *output, *key1])
                    - self.data.get_trigram_u([*lead, old, *key1]),
                &mut i,
            );
            add_tg(
                &mut cache.magic,
                cache.keys.len(),
                *lead,
                *output,
                *key1,
                -1 * self.data.get_trigram_u([*lead, *output, *key1]),
                &mut i,
            );

            // TODO: The following skipgrams/trigrams are not currently attributed to the magic key. We need quadgrams to do it correctly
            // for key2 in keys {
            //     // output -> key1 skipgram stolen whenever the leader is pressed first
            //     add_sg(*magic_key, *key2, self.data.get_quadgram_u([*lead, *old, *key1, *key2]);
            //     add_sg(*magic_key, *key2, self.data.get_quadgram_u([*lead, *output, *key1, *key2] - self.data.get_quadgram_u([*lead, *old, *key1, *key2]));
            //     add_sg(*output, *key2, -1 * self.data.get_quadgram_u([*lead, *output, *key1, *key2]);

            //     // output -> key1 -> key2 trigram stolen whenever the leader is pressed first
            //     *tg(*magic_key, *key1, *key2) += self.data.get_quadgram_u([*lead, *output, *key1, *key2]);
            //     *tg(*output, *key1, *key2) -= self.data.get_quadgram_u([*lead, *output, *key1, *key2]);
            // }
        });
        // caceh.magic.rules.iter().for_each(|(magic_key2, rule2)| {
        //     rule2.iter().for_each(|(lead2, output2)| {
        //         // output -> output2 skipgram stolen when double magic occurs
        //         *sg(*magic_key, *magic_key2) += self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
        //         *sg(*output, *output2)  -= self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);

        //         // output -> lead2 -> output2 trigram stolen when double magic occurs
        //         *tg(*magic_key, *lead2, *magic_key2) += self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
        //         *tg(*output, *lead2, *output2)  -= self.data.get_quadgram_u([*lead, *output, *lead2, *output2]);
    }

    /*
     **************************************
     *              Stats
     **************************************
     */

    pub fn sfbs(&self, cache: &CachedLayout) -> i64 {
        cache
            .sfb
            .weighted_sfb_indices
            .all
            .iter()
            .map(
                |BigramPair {
                     pair: PosPair(a, b),
                     ..
                 }| {
                    let u1 = cache.keys[*a as usize] as usize;
                    let u2 = cache.keys[*b as usize] as usize;

                    cache.magic.bigrams.get(u1 * cache.keys.len() + u2).unwrap()
                        + cache.magic.bigrams.get(u2 * cache.keys.len() + u1).unwrap()
                },
            )
            .sum()
    }

    pub fn sfs(&self, cache: &CachedLayout) -> i64 {
        cache
            .sfb
            .weighted_sfb_indices
            .all
            .iter()
            .map(
                |BigramPair {
                     pair: PosPair(a, b),
                     ..
                 }| {
                    let u1 = cache.keys[*a as usize] as usize;
                    let u2 = cache.keys[*b as usize] as usize;

                    cache
                        .magic
                        .skipgrams
                        .get(u1 * cache.keys.len() + u2)
                        .unwrap()
                        + cache
                            .magic
                            .skipgrams
                            .get(u2 * cache.keys.len() + u1)
                            .unwrap()
                },
            )
            .sum()
    }

    pub fn stretches(&self, cache: &CachedLayout) -> i64 {
        cache
            .stretch
            .all_pairs
            .iter()
            .map(
                |BigramPair {
                     pair: PosPair(a, b),
                     dist,
                 }| {
                    let u1 = cache.keys[*a as usize];
                    let u2 = cache.keys[*b as usize];

                    (self.stretch_get_bigram(cache, &u1, &u2)
                        + self.stretch_get_bigram(cache, &u2, &u1))
                        * dist
                },
            )
            .sum()
    }

    pub fn finger_use(&self, cache: &CachedLayout) -> [i64; 10] {
        let mut res = [0; 10];

        for (&k, &f) in cache.keys.iter().zip(cache.fingers.iter()) {
            // TODO: Magic mapping
            res[f as usize] += self.data.get_char_u(k);
        }

        res
    }

    pub fn weighted_finger_distance(&self, cache: &CachedLayout) -> [i64; 10] {
        Finger::FINGERS.map(|f| self.sfb_finger_weighted_bigrams(cache, f))
    }

    pub fn unweighted_finger_distance(&self, cache: &CachedLayout) -> [i64; 10] {
        Finger::FINGERS.map(|f| self.sfb_finger_unweighted_bigrams(cache, f))
    }

    pub fn finger_sfbs(&self, cache: &CachedLayout) -> [i64; 10] {
        cache.sfb.weighted_sfb_indices.fingers.clone().map(|pairs| {
            pairs
                .iter()
                .map(
                    |BigramPair {
                         pair: PosPair(a, b),
                         ..
                     }| {
                        let u1 = cache.keys[*a as usize] as usize;
                        let u2 = cache.keys[*b as usize] as usize;

                        cache.magic.bigrams.get(u1 * cache.keys.len() + u2).unwrap()
                            + cache.magic.bigrams.get(u2 * cache.keys.len() + u1).unwrap()
                    },
                )
                .sum()
        })
    }

    pub fn trigrams(&self, cache: &CachedLayout) -> TrigramData {
        use crate::trigrams::TrigramType::*;

        let mut trigrams = TrigramData::default();

        for (&c1, &f1) in cache.keys.iter().zip(&cache.fingers) {
            for (&c2, &f2) in cache.keys.iter().zip(&cache.fingers) {
                for (&c3, &f3) in cache.keys.iter().zip(&cache.fingers) {
                    let u1 = c1 as usize;
                    let u2 = c2 as usize;
                    let u3 = c3 as usize;
                    let freq = cache
                        .magic
                        .trigrams
                        .get(u1 * cache.keys.len().pow(2) + u2 * cache.keys.len() + u3)
                        .unwrap();
                    let ttype = TRIGRAMS[f1 as usize * 100 + f2 as usize * 10 + f3 as usize];

                    match ttype {
                        Sft => trigrams.sft += freq,
                        Sfb => trigrams.sfb += freq,
                        Inroll => trigrams.inroll += freq,
                        Outroll => trigrams.outroll += freq,
                        Alternate => trigrams.alternate += freq,
                        Redirect => trigrams.redirect += freq,
                        OnehandIn => trigrams.onehandin += freq,
                        OnehandOut => trigrams.onehandout += freq,
                        Thumb => trigrams.thumb += freq,
                        Invalid => trigrams.invalid += freq,
                    }
                }
            }
        }

        trigrams
    }

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
        assert_eq!(freq, 0, "a->z should be unaffected");

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
