/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

// For now, only "simple" magic rules are supported. One leader key -> one output key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    // magic_key -> leader -> output
    pub rules: Vec<Vec<u8>>,
}

impl MagicCache {
    pub fn initialize(&mut self, layout: &Layout) {
        // TODO: allocate rules based on num magic keys and num keys
    }

    // Main entry point for neighbor updates. Updates the rules for the key under the hood
    // from, to may be non-magic keys that has already had its bg stolen by another magic key
    // At least one of from, to must be magic/repeat
    pub fn steal_bigram(&mut self, from: u8, to: u8, a: u8, b: u8) {
        
    }

    // Updates the list of possible neighbors that can be reached from the current rules
    fn possible_neighbors(&self, possible_neighbors: &mut Vec<Neighbor>) {
        self.rules.iter().enumerate())
        .map(|(magic, rule)| {
            rule.iter().enumerate()
            .map(|(leader, output)| {
                if *output == REPLACEMENT_CHAR {

                }
            })
        })
    }


/// Get effective bigram frequency accounting for magic key rules
/// This works on the premise that magic keys "steal" frequency from their non-magic counterparts
fn get_bigram_freq(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
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
fn get_skipgram_freq(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
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
fn get_trigram_freq(&self, cache: &CachedLayout, a: u8, b: u8, c: u8) -> i64 {
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

    let mapping_len = self.data.mapping.len();
    let mut bigrams = vec![0 as i64; mapping_len.pow(2)].into_boxed_slice();
    let mut skipgrams = vec![0 as i64; mapping_len.pow(2)].into_boxed_slice();
    let mut trigrams = vec![0 as i64; mapping_len.pow(3)].into_boxed_slice();

    // First, copy all bg/sg/tg frequencies from data
    keys.iter().tuple_combinations().for_each(|(key1, key2)| {
        let u1 = *key1 as usize;
        let u2 = *key2 as usize;
        bigrams[u1 * mapping_len + u2] = self.data.get_bigram_u([*key1, *key2]);
        skipgrams[u1 * mapping_len + u2] = self.data.get_skipgram_u([*key1, *key2]);
        keys.iter().for_each(|key3| {
            let u3 = *key3 as usize;
            trigrams[u1 * mapping_len.pow(2) + u2 * mapping_len + u3] =
                self.data.get_trigram_u([*key1, *key2, *key3]);
        });
    });

    // Convenience closures
    let mut add_bg = |a: u8, b: u8, val: i64| {
        let u1 = a as usize;
        let u2 = b as usize;
        bigrams[u1 * mapping_len + u2] += val;
    };
    let mut add_sg = |a: u8, b: u8, val: i64| {
        let u1 = a as usize;
        let u2 = b as usize;
        skipgrams[u1 * mapping_len + u2] += val;
    };
    let mut add_tg = |a: u8, b: u8, c: u8, val: i64| {
        let u1 = a as usize;
        let u2 = b as usize;
        let u3 = c as usize;
        trigrams[u1 * mapping_len.pow(2) + u2 * mapping_len + u3] += val;
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
            self.data.mapping.len(),
            old,
            *key1,
            self.data.get_trigram_u([*lead, old, *key1]),
            &mut i,
        );
        add_bg(
            &mut cache.magic,
            self.data.mapping.len(),
            *magic_key,
            *key1,
            self.data.get_trigram_u([*lead, *output, *key1])
                - self.data.get_trigram_u([*lead, old, *key1]),
            &mut i,
        );
        add_bg(
            &mut cache.magic,
            self.data.mapping.len(),
            *output,
            *key1,
            -1 * self.data.get_trigram_u([*lead, *output, *key1]),
            &mut i,
        );

        // key1 -> output skipgram stolen whenever the leader is pressed in the middle
        add_sg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            old,
            self.data.get_trigram_u([*key1, *lead, old]),
            &mut i,
        );
        add_sg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            *magic_key,
            self.data.get_trigram_u([*key1, *lead, *output])
                - self.data.get_trigram_u([*key1, *lead, old]),
            &mut i,
        );
        add_sg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            *output,
            -1 * self.data.get_trigram_u([*key1, *lead, *output]),
            &mut i,
        );

        // key1 -> lead -> output trigram always stolen
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            *lead,
            old,
            self.data.get_trigram_u([*key1, *lead, old]),
            &mut i,
        );
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            *lead,
            *magic_key,
            self.data.get_trigram_u([*key1, *lead, *output])
                - self.data.get_trigram_u([*key1, *lead, old]),
            &mut i,
        );
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
            *key1,
            *lead,
            *output,
            -1 * self.data.get_trigram_u([*key1, *lead, *output]),
            &mut i,
        );

        // lead -> output -> key1 trigram always stolen
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
            *lead,
            old,
            *key1,
            self.data.get_trigram_u([*lead, old, *key1]),
            &mut i,
        );
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
            *lead,
            *magic_key,
            *key1,
            self.data.get_trigram_u([*lead, *output, *key1])
                - self.data.get_trigram_u([*lead, old, *key1]),
            &mut i,
        );
        add_tg(
            &mut cache.magic,
            self.data.mapping.len(),
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
