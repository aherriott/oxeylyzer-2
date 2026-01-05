/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

use crate::{
    analyze::Neighbor, analyzer_data::AnalyzerData, layout::Layout, types::CacheKey,
    types::KeysCache,
};

pub struct DeltaBigram {
    a: CacheKey,
    b: CacheKey,
    old_freq: i64,
    new_freq: i64,
}

pub struct DeltaSkipgram {
    a: CacheKey,
    b: CacheKey,
    old_freq: i64,
    new_freq: i64,
}

pub struct DeltaTrigram {
    a: CacheKey,
    b: CacheKey,
    c: CacheKey,
    old_freq: i64,
    new_freq: i64,
}

pub enum DeltaGram {
    Bigram(DeltaBigram),
    Skipgram(DeltaSkipgram),
    Trigram(DeltaTrigram),
}

// For now, only "simple" magic rules are supported. One leader key -> one output key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    // Freq lists are the mapped frequencies of bg, sg, tg post-magic rule
    bg_freq: Vec<Vec<i64>>,
    sg_freq: Vec<Vec<i64>>,
    tg_freq: Vec<Vec<Vec<i64>>>,
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
    ) {
        /* This data structure leads to wholesale copying of all bg values. Since we're biasing for time > space, this is fine. */
        let bg_freq = data.bigrams.clone();
        let sg_freq = data.skipgrams.clone();
        let tg_freq = data.trigrams.clone();
        let mut magic = MagicCache {
            bg_freq,
            sg_freq,
            tg_freq,
        };

        // First, flesh out possible_neighbors with magic rules. steal_bigram will update as needed
        layout.magic.iter().map(|mag, rules| {
            rules.iter().map(|lead, out| {
                for m in 0..keys.magic.len() {
                    possible_neighbors.push(Neighbor::MagicStealBigram(MagicStealBigram {
                        key: keys.magic[m], // TODO need the CacheKey, not the layout version
                        m_old: *out,
                        leader: *lead,
                        output: *out,
                    }));
                }
            });
        });

        // Call steal_bigram for all magic rules to set frequencies & possible neighbors
        layout.magic.iter().map(|mag, rules| {
            rules.iter().map(|lead, out| {
                // Ignore affected_grams. Magic needs to run first to set all the freqs anyways before other metrics can init.
                magic.steal_bigram(lead, out, mag, lead, possible_neighbors, keys, None);
            });
        });

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
        let set_bg = |a, b, new| {
            let old = self.bg_freq[a][b];
            bg_freq[a][b] = new_freq;
            if let Some(affected_grams) = grams {
                grams.push(DeltaGram::Bigram(DeltaBigram { a, b, old, new }));
            }
        };

        let set_sg = |a, b, new| {
            let old = self.sg_freq[a][b];
            sg_freq[a][b] = new_freq;
            if let Some(affected_grams) = grams {
                grams.push(DeltaGram::Skipgram(DeltaSkipgram { a, b, old, new }));
            }
        };

        let set_tg = |a, b, c, new| {
            let old = self.bg_freq[a][b];
            tg_freq[a][b][c] = new_freq;
            if let Some(affected_grams) = grams {
                grams.push(DeltaGram::Trigram(DeltaTrigram { a, b, c, old, new }));
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

        for c in keys {
            debug_assert!(self.bg_freq[b][c] - self.tg_freq[a][b][c] >= 0);
            set_bg(m, c, self.bg_freq[m][c] + self.tg_freq[a][b][c]);
            set_bg(b, c, self.bg_freq[b][c] - self.tg_freq[a][b][c]);
        }

        for z in keys {
            debug_assert!(self.sg_freq[z][b] - self.tg_freq[z][a][b] >= 0);
            set_sg(z, m, self.sg_freq[z][m] + self.tg_freq[z][a][b]);
            set_sg(z, b, self.sg_freq[z][b] - self.tg_freq[z][a][b]);
        }

        for z in keys {
            set_tg(z, a, m, self.tg_freq[z][a][m] + self.tg_freq[z][a][b]);
            set_tg(z, a, b, 0i64);
        }

        for c in keys {
            set_tg(a, m, c, self.tg_freq[a][m][c] + self.tg_freq[a][b][c]);
            set_tg(a, b, c, 0i64);
        }

        // Update possible neighbors with the revert of the steal
        for m in 0..keys.magic.len() {
            // TODO update with m_old being the original magic key
            possible_neighbors[(a * keys.len() + b) * keys.magic.len() + m] =
                Neighbor::MagicStealBigram(MagicStealBigram {
                    key: keys.magic[m],
                    m_old: m,
                    leader: a,
                    output: b,
                });
        }
    }
}
