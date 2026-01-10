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
        // Helper to get position
        let get_pos = |k: CacheKey| -> Option<CachePos> {
            key_positions.get(k).copied().flatten()
        };

        // Inline helper for setting bigram freq and recording delta
        macro_rules! set_bg {
            ($ka:expr, $kb:expr, $new:expr) => {{
                let old = self.bg_freq[$ka][$kb];
                self.bg_freq[$ka][$kb] = $new;
                if let (Some(p_a), Some(p_b)) = (get_pos($ka), get_pos($kb)) {
                    affected_grams.push(DeltaGram::Bigram(DeltaBigram {
                        p_a,
                        p_b,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        macro_rules! set_sg {
            ($ka:expr, $kb:expr, $new:expr) => {{
                let old = self.sg_freq[$ka][$kb];
                self.sg_freq[$ka][$kb] = $new;
                if let (Some(p_a), Some(p_b)) = (get_pos($ka), get_pos($kb)) {
                    affected_grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                        p_a,
                        p_b,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        macro_rules! set_tg {
            ($ka:expr, $kb:expr, $kc:expr, $new:expr) => {{
                let old = self.tg_freq[$ka][$kb][$kc];
                self.tg_freq[$ka][$kb][$kc] = $new;
                if let (Some(p_a), Some(p_b), Some(p_c)) = (get_pos($ka), get_pos($kb), get_pos($kc)) {
                    affected_grams.push(DeltaGram::Trigram(DeltaTrigram {
                        p_a,
                        p_b,
                        p_c,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        // 1. The exact bigram A->B is fully stolen by A->M
        let new_am = self.bg_freq[a][m] + self.bg_freq[a][b];
        set_bg!(a, m, new_am);
        set_bg!(a, b, 0);

        // 2. For each key c: B->C is partially stolen by M->C based on trigram A->B->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.bg_freq[b][c] >= tg);
            let new_mc = self.bg_freq[m][c] + tg;
            let new_bc = self.bg_freq[b][c] - tg;
            set_bg!(m, c, new_mc);
            set_bg!(b, c, new_bc);
        }

        // 3. For each key z: skipgram Z->B is partially stolen by Z->M based on trigram Z->A->B
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.sg_freq[z][b] >= tg);
            let new_zm = self.sg_freq[z][m] + tg;
            let new_zb = self.sg_freq[z][b] - tg;
            set_sg!(z, m, new_zm);
            set_sg!(z, b, new_zb);
        }

        // 4. For each key z: trigram Z->A->B is fully stolen by Z->A->M
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            let new_zam = self.tg_freq[z][a][m] + tg;
            set_tg!(z, a, m, new_zam);
            set_tg!(z, a, b, 0);
        }

        // 5. For each key c: trigram A->B->C is fully stolen by A->M->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            let new_amc = self.tg_freq[a][m][c] + tg;
            set_tg!(a, m, c, new_amc);
            set_tg!(a, b, c, 0);
        }
    }

    /// Copy only the frequency entries that were affected by a steal operation.
    /// `affected_grams` should contain the deltas from the steal that was applied to `other`.
    pub fn copy_from(&mut self, other: &MagicCache, affected_grams: &[DeltaGram], key_at_pos: impl Fn(CachePos) -> CacheKey) {
        for gram in affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    let a = key_at_pos(bg.p_a);
                    let b = key_at_pos(bg.p_b);
                    self.bg_freq[a][b] = other.bg_freq[a][b];
                }
                DeltaGram::Skipgram(sg) => {
                    let a = key_at_pos(sg.p_a);
                    let b = key_at_pos(sg.p_b);
                    self.sg_freq[a][b] = other.sg_freq[a][b];
                }
                DeltaGram::Trigram(tg) => {
                    let a = key_at_pos(tg.p_a);
                    let b = key_at_pos(tg.p_b);
                    let c = key_at_pos(tg.p_c);
                    self.tg_freq[a][b][c] = other.tg_freq[a][b][c];
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cached_layout::DeltaGram;

    #[test]
    fn magic_cache_new() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.bg_freq.len(), 10);
        assert_eq!(cache.sg_freq.len(), 10);
        assert_eq!(cache.tg_freq.len(), 10);
    }

    #[test]
    fn magic_cache_get_freq_empty() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.get_bg_freq(0, 1), 0);
        assert_eq!(cache.get_sg_freq(0, 1), 0);
        assert_eq!(cache.get_tg_freq(0, 1, 2), 0);
    }

    #[test]
    fn magic_cache_init_from_data() {
        let mut cache = MagicCache::new(3);
        let bigrams = vec![
            vec![0, 100, 200],
            vec![300, 0, 400],
            vec![500, 600, 0],
        ];
        let skipgrams = vec![
            vec![0, 10, 20],
            vec![30, 0, 40],
            vec![50, 60, 0],
        ];
        let trigrams = vec![
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
            vec![vec![9, 10, 11], vec![12, 13, 14], vec![15, 16, 17]],
            vec![vec![18, 19, 20], vec![21, 22, 23], vec![24, 25, 26]],
        ];

        cache.init_from_data(&bigrams, &skipgrams, &trigrams);

        assert_eq!(cache.get_bg_freq(0, 1), 100);
        assert_eq!(cache.get_bg_freq(1, 2), 400);
        assert_eq!(cache.get_sg_freq(2, 1), 60);
        assert_eq!(cache.get_tg_freq(1, 2, 0), 15);
    }

    #[test]
    fn magic_cache_steal_bigram_basic() {
        let mut cache = MagicCache::new(4);
        // Setup: a=0, b=1, m=2, c=3
        // Bigram a->b = 100
        cache.bg_freq[0][1] = 100;
        cache.bg_freq[0][2] = 0; // a->m initially 0

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        // Steal bigram a->b with magic key m
        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        // a->b should now be 0
        assert_eq!(cache.get_bg_freq(0, 1), 0, "a->b should be stolen");
        // a->m should now have the stolen frequency
        assert_eq!(cache.get_bg_freq(0, 2), 100, "a->m should have stolen frequency");
    }

    #[test]
    fn magic_cache_steal_records_affected_grams() {
        let mut cache = MagicCache::new(4);
        cache.bg_freq[0][1] = 100;

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        // Should have recorded at least the two bigram changes (a->b and a->m)
        let bigram_count = affected.iter().filter(|g| matches!(g, DeltaGram::Bigram(_))).count();
        assert!(bigram_count >= 2, "Should record at least 2 bigram changes, got {bigram_count}");
    }

    #[test]
    fn magic_cache_copy_from_selective() {
        let mut cache1 = MagicCache::new(4);
        let mut cache2 = MagicCache::new(4);

        // Setup cache1 with some data
        cache1.bg_freq[0][1] = 100;
        cache1.bg_freq[1][2] = 200;
        cache1.sg_freq[0][2] = 50;

        // Setup cache2 differently
        cache2.bg_freq[0][1] = 999;
        cache2.bg_freq[1][2] = 999;
        cache2.sg_freq[0][2] = 999;

        // Create affected grams that only include [0][1] bigram
        let affected = vec![
            DeltaGram::Bigram(DeltaBigram {
                p_a: 0,
                p_b: 1,
                old_freq: 999,
                new_freq: 100,
            }),
        ];

        // Copy only affected entries from cache1 to cache2
        cache2.copy_from(&cache1, &affected, |pos| pos);

        // [0][1] should be copied
        assert_eq!(cache2.get_bg_freq(0, 1), 100, "Affected bigram should be copied");
        // [1][2] should NOT be copied (not in affected)
        assert_eq!(cache2.get_bg_freq(1, 2), 999, "Unaffected bigram should not be copied");
        // skipgram should NOT be copied
        assert_eq!(cache2.get_sg_freq(0, 2), 999, "Unaffected skipgram should not be copied");
    }
}
