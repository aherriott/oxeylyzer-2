/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

use crate::{
    analyze::Neighbor,
    analyzer_data::AnalyzerData,
    layout::{Layout, MagicStealBigram},
    types::CacheKey,
    types::KeysCache,
};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBigram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaSkipgram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaTrigram {
    pub a: CacheKey,
    pub b: CacheKey,
    pub c: CacheKey,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeltaGram {
    Bigram(DeltaBigram),
    Skipgram(DeltaSkipgram),
    Trigram(DeltaTrigram),
}

// For now, only "simple" magic rules are supported. One leader key -> one output key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    // Freq lists are the mapped frequencies of bg, sg, tg post-magic rule
    pub bg_freq: Vec<Vec<i64>>,
    pub sg_freq: Vec<Vec<i64>>,
    pub tg_freq: Vec<Vec<Vec<i64>>>,
    pub rules: HashMap<CacheKey, HashMap<CacheKey, CacheKey>>,
    // Flattened versions for fast indexing
    pub bigrams: Vec<i64>,
    pub skipgrams: Vec<i64>,
}

impl MagicCache {
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.bg_freq[a][b]
    }

    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.sg_freq[a][b]
    }

    pub fn get_tg_freq(&self, a: CacheKey, b: CacheKey, c: CacheKey) -> i64 {
        self.tg_freq[a][b][c]
    }

    pub fn new(
        data: &AnalyzerData,
        keys: &KeysCache,
        layout: &Layout,
        possible_neighbors: &mut Vec<Neighbor>,
    ) -> Self {
        /* This data structure leads to wholesale copying of all bg values. Since we're biasing for time > space, this is fine. */
        let bg_freq = data.bigrams.clone();
        let sg_freq = data.skipgrams.clone();
        let tg_freq = data.trigrams.clone();
        // Flatten bigrams and skipgrams for fast indexing
        let n = bg_freq.len();
        let mut bigrams = Vec::with_capacity(n * n);
        let mut skipgrams = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                bigrams.push(bg_freq[i][j]);
                skipgrams.push(sg_freq[i][j]);
            }
        }
        let mut magic = MagicCache {
            bg_freq,
            sg_freq,
            tg_freq,
            rules: HashMap::new(),
            bigrams,
            skipgrams,
        };

        // First, flesh out possible_neighbors with magic rules. steal_bigram will update as needed
        // TODO: fix keys.magic missing
        // layout.magic.iter().map(|mag, rules| {
        //     rules.iter().map(|lead, out| {
        //         for m in 0..keys.magic.len() {
        //             possible_neighbors.push(Neighbor::MagicStealBigram(MagicStealBigram {
        //                 key: keys.magic[m], // TODO need the CacheKey, not the layout version
        //                 m_old: *out,
        //                 leader: *lead,
        //                 output: *out,
        //             }));
        //         }
        //     });
        // });

        // Call steal_bigram for all magic rules to set frequencies & possible neighbors
        // layout.magic.iter().map(|mag, rules| {
        //     rules.iter().map(|lead, out| {
        //         // Ignore affected_grams. Magic needs to run first to set all the freqs anyways before other metrics can init.
        //         magic.steal_bigram(lead, out, mag, lead, possible_neighbors, keys, None);
        //     });
        // });

        magic
    }

    // Main entry point for neighbor updates. Updates the rules for the key under the hood
    // from, to may be non-magic keys that has already had its bg stolen by another magic key
    // At least one of from, to must be magic/repeat
    pub fn steal_bigram(
        &mut self,
        a: CacheKey,
        b: CacheKey,
        m: CacheKey,
        m_old: CacheKey,
        possible_neighbors: &mut Vec<Neighbor>,
        keys: &KeysCache,
        affected_grams: Option<&mut Vec<DeltaGram>>,
    ) {
        // Helper functions to set frequencies and update affected grams
        let set_bg = |a: CacheKey, b: CacheKey, new: i64| {
            let old = self.bg_freq[a][b];
            self.bg_freq[a][b] = new;
            if let Some(grams) = affected_grams {
                grams.push(DeltaGram::Bigram(DeltaBigram {
                    a,
                    b,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        let set_sg = |a: CacheKey, b: CacheKey, new: i64| {
            let old = self.sg_freq[a][b];
            self.sg_freq[a][b] = new;
            if let Some(grams) = affected_grams {
                grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                    a,
                    b,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        let set_tg = |a: CacheKey, b: CacheKey, c: CacheKey, new: i64| {
            let old = self.tg_freq[a][b][c];
            self.tg_freq[a][b][c] = new;
            if let Some(grams) = affected_grams {
                grams.push(DeltaGram::Trigram(DeltaTrigram {
                    a,
                    b,
                    c,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        /*
         * The following calculations are based on the idea that magic keys "steal" bigrams from regular keys.
         * i.e. when you add the rule A->B to magic key M, the bigram A->B is never typed, because you always
         * type A->M instead.
         *
         * Assuming key order: Z->A->B->C->D
         *
         * If B is magic:
         * 1. The bigram A->B is fully stolen by A->M
         * 2. The bigram B->C is partially stolen by M->C, based on the rate of the A->B->C trigram (which is now typed A->M->C)
         * 3. The skipgram Z->B is partially stolen by Z->M, unless Z->A is magic (TODO, does this wash out in the math?) based on the rate of the Z->A->B trigram
         * 4. The trigram Z->A->B is fully stolen by Z->A->M, unless Z->A is magic (TODO, does this wash out in the math?) (This is not fully accurate, since we can't know if Z is magic w/o quadgrams)
         * 5. The trigram  A->B->C is fully stolen by A->M->C (This is not fully accurate, since we can't know if A is magic w/o quadgrams)
         *  */

        // The exact bigram is fully stolen:
        set_bg(a, m, self.bg_freq[a][m] + self.bg_freq[a][b]);
        set_bg(a, b, 0i64);

        for c in 0..self.bg_freq.len() {
            debug_assert!(self.bg_freq[b][c] - self.tg_freq[a][b][c] >= 0);
            set_bg(m, c, self.bg_freq[m][c] + self.tg_freq[a][b][c]);
            set_bg(b, c, self.bg_freq[b][c] - self.tg_freq[a][b][c]);
        }

        for z in 0..self.sg_freq.len() {
            debug_assert!(self.sg_freq[z][b] - self.tg_freq[z][a][b] >= 0);
            set_sg(z, m, self.sg_freq[z][m] + self.tg_freq[z][a][b]);
            set_sg(z, b, self.sg_freq[z][b] - self.tg_freq[z][a][b]);
        }

        for z in 0..self.tg_freq.len() {
            set_tg(z, a, m, self.tg_freq[z][a][m] + self.tg_freq[z][a][b]);
            set_tg(z, a, b, 0i64);
        }

        for c in 0..self.tg_freq[a].len() {
            set_tg(a, m, c, self.tg_freq[a][m][c] + self.tg_freq[a][b][c]);
            set_tg(a, b, c, 0i64);
        }

        // Update possible neighbors with the revert of the steal
        // Since keys.magic doesn't exist, we need to know the magic keys from layout.
        // For now, stub.
        // possible_neighbors[(a * keys.len() + b) * layout.magic.len() + m] =
        //     Neighbor::MagicStealBigram(MagicStealBigram(key, leader, output));
        // TODO: implement
    }
    // Add a rule and return affected grams
    pub fn add_rule(
        &mut self,
        key: CacheKey,
        leader: CacheKey,
        output: CacheKey,
    ) -> Vec<DeltaGram> {
        // This is a stub; real implementation should compute delta grams
        // For now, return empty vec
        Vec::new()
    }
}

mod test {}
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
    assert_ne!(freq, 0, "a->z should be unaffected");

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
    let (magic_key, leader, output) = if let Some((mk, rules)) = cache.magic.rules.iter().next() {
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
