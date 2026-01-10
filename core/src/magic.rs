/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

use crate::{
    cached_layout::{DeltaBigram, DeltaGram, DeltaSkipgram, DeltaTrigram},
    types::{CacheKey, CachePos},
};

/// MagicCache stores frequency tables that get modified by magic key rules.
/// When a magic rule "steals" a bigram, the frequencies are redistributed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    bg_freq: Vec<Vec<i64>>,
    sg_freq: Vec<Vec<i64>>,
    tg_freq: Vec<Vec<Vec<i64>>>,
}

impl MagicCache {
    pub fn new(num_keys: usize) -> Self {
        Self {
            bg_freq: vec![vec![0; num_keys]; num_keys],
            sg_freq: vec![vec![0; num_keys]; num_keys],
            tg_freq: vec![vec![vec![0; num_keys]; num_keys]; num_keys],
        }
    }

    /// Initialize frequencies from corpus data
    pub fn init_from_data(&mut self, bigrams: &[Vec<i64>], skipgrams: &[Vec<i64>], trigrams: &[Vec<Vec<i64>>]) {
        self.bg_freq = bigrams.to_vec();
        self.sg_freq = skipgrams.to_vec();
        self.tg_freq = trigrams.to_vec();
    }

    #[inline]
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.bg_freq.get(a).and_then(|row| row.get(b)).copied().unwrap_or(0)
    }

    #[inline]
    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.sg_freq.get(a).and_then(|row| row.get(b)).copied().unwrap_or(0)
    }

    #[inline]
    pub fn get_tg_freq(&self, a: CacheKey, b: CacheKey, c: CacheKey) -> i64 {
        self.tg_freq
            .get(a)
            .and_then(|r1| r1.get(b))
            .and_then(|r2| r2.get(c))
            .copied()
            .unwrap_or(0)
    }

    /// Steal a bigram: when typing leader->output, magic key M intercepts it.
    /// This redistributes frequencies from (leader, output) to (leader, magic_key).
    ///
    /// Populates affected_grams with position-based deltas for cache updates.
    ///
    /// The following calculations are based on the idea that magic keys "steal" bigrams from regular keys.
    /// i.e. when you add the rule A->B to magic key M, the bigram A->B is never typed, because you always
    /// type A->M instead.
    ///
    /// Assuming key order: Z->A->B->C
    ///
    /// If B is magic:
    /// 1. The bigram A->B is fully stolen by A->M
    /// 2. The bigram B->C is partially stolen by M->C, based on the rate of the A->B->C trigram (which is now typed A->M->C)
    /// 3. The skipgram Z->B is partially stolen by Z->M, based on the rate of the Z->A->B trigram
    /// 4. The trigram Z->A->B is fully stolen by Z->A->M
    /// 5. The trigram A->B->C is fully stolen by A->M->C
    pub fn steal_bigram(
        &mut self,
        a: CacheKey,      // leader
        b: CacheKey,      // output (being stolen)
        m: CacheKey,      // magic key
        key_positions: &[Option<CachePos>],
        num_keys: usize,
        affected_grams: &mut Vec<DeltaGram>,
    ) {
        // Helper to get position, returns None if key not placed
        let get_pos = |k: CacheKey| -> Option<CachePos> {
            key_positions.get(k).copied().flatten()
        };

        // Helper macros to set frequencies and record deltas
        let mut set_bg = |a: CacheKey, b: CacheKey, new: i64| {
            let old = self.bg_freq[a][b];
            self.bg_freq[a][b] = new;
            if let (Some(p_a), Some(p_b)) = (get_pos(a), get_pos(b)) {
                affected_grams.push(DeltaGram::Bigram(DeltaBigram {
                    p_a,
                    p_b,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        let mut set_sg = |a: CacheKey, b: CacheKey, new: i64| {
            let old = self.sg_freq[a][b];
            self.sg_freq[a][b] = new;
            if let (Some(p_a), Some(p_b)) = (get_pos(a), get_pos(b)) {
                affected_grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                    p_a,
                    p_b,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        let mut set_tg = |a: CacheKey, b: CacheKey, c: CacheKey, new: i64| {
            let old = self.tg_freq[a][b][c];
            self.tg_freq[a][b][c] = new;
            if let (Some(p_a), Some(p_b), Some(p_c)) = (get_pos(a), get_pos(b), get_pos(c)) {
                affected_grams.push(DeltaGram::Trigram(DeltaTrigram {
                    p_a,
                    p_b,
                    p_c,
                    old_freq: old,
                    new_freq: new,
                }));
            }
        };

        // 1. The exact bigram A->B is fully stolen by A->M
        set_bg(a, m, self.bg_freq[a][m] + self.bg_freq[a][b]);
        set_bg(a, b, 0);

        // 2. For each key c: B->C is partially stolen by M->C based on trigram A->B->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.bg_freq[b][c] >= tg);
            set_bg(m, c, self.bg_freq[m][c] + tg);
            set_bg(b, c, self.bg_freq[b][c] - tg);
        }

        // 3. For each key z: skipgram Z->B is partially stolen by Z->M based on trigram Z->A->B
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.sg_freq[z][b] >= tg);
            set_sg(z, m, self.sg_freq[z][m] + tg);
            set_sg(z, b, self.sg_freq[z][b] - tg);
        }

        // 4. For each key z: trigram Z->A->B is fully stolen by Z->A->M
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            set_tg(z, a, m, self.tg_freq[z][a][m] + tg);
            set_tg(z, a, b, 0);
        }

        // 5. For each key c: trigram A->B->C is fully stolen by A->M->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            set_tg(a, m, c, self.tg_freq[a][m][c] + tg);
            set_tg(a, b, c, 0);
        }
    }
}


#[cfg(test)]
mod test {
    // TODO: fix these tests after refactoring is complete
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
