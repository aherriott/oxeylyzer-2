/*
 **************************************
 *            SFBs & SFSs
 **************************************
 */

use crate::dist::DistCache;
use crate::stats::Stats;
use crate::weights::Weights;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

/// Stores same-finger pairs for a given position
#[derive(Debug, Clone, PartialEq)]
pub struct SfPair {
    pub other_pos: usize,
    pub finger: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFCache {
    /// Weighted SFB score per finger (freq * dist)
    sfb_score_per_finger: Box<[i64; 10]>,
    /// Weighted SFS score per finger (freq * dist)
    sfs_score_per_finger: Box<[i64; 10]>,
    /// Unweighted SFB frequency per finger (for stats)
    sfb_freq_per_finger: Box<[i64; 10]>,
    /// Unweighted SFS frequency per finger (for stats)
    sfs_freq_per_finger: Box<[i64; 10]>,
    /// For each position, list of other positions on the same finger
    sf_pairs_per_key: Vec<Vec<SfPair>>,
}

impl SFCache {
    pub fn new(fingers: &[Finger], keyboard: &[PhysicalKey]) -> Self {
        assert_eq!(
            fingers.len(),
            keyboard.len(),
            "finger len is not the same as keyboard len"
        );

        // Build same-finger pair lookup
        let mut sf_pairs_per_key = Vec::with_capacity(fingers.len());
        for (i, finger1) in fingers.iter().enumerate() {
            let mut pairs = Vec::new();
            for (j, finger2) in fingers.iter().enumerate() {
                if finger1 == finger2 && i != j {
                    pairs.push(SfPair {
                        other_pos: j,
                        finger: *finger1 as usize,
                    });
                }
            }
            sf_pairs_per_key.push(pairs);
        }

        Self {
            sfb_score_per_finger: Box::new([0i64; 10]),
            sfs_score_per_finger: Box::new([0i64; 10]),
            sfb_freq_per_finger: Box::new([0i64; 10]),
            sfs_freq_per_finger: Box::new([0i64; 10]),
            sf_pairs_per_key,
        }
    }

    pub fn score(&self, weights: &Weights) -> i64 {
        Finger::FINGERS
            .iter()
            .map(|f| -> i64 {
                let fi = *f as usize;
                self.sfb_score_per_finger[fi] * weights.fingers.get(*f) * weights.sfbs
                    + self.sfs_score_per_finger[fi] * weights.fingers.get(*f) * weights.sfs
            })
            .sum()
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64, skipgram_total: f64) {
        // Total SFB/SFS frequencies (sum across all fingers)
        let total_sfb: i64 = self.sfb_freq_per_finger.iter().sum();
        let total_sfs: i64 = self.sfs_freq_per_finger.iter().sum();

        // Note: bigram_total and skipgram_total are already divided by 100 in AnalyzerData,
        // but the stored frequencies are raw counts (percentage * total / 100).
        // To get the correct percentage, we need to multiply the totals by 100.
        let bigram_total_raw = bigram_total * 100.0;
        let skipgram_total_raw = skipgram_total * 100.0;

        stats.sfbs = total_sfb as f64 / bigram_total_raw;
        stats.sfs = total_sfs as f64 / skipgram_total_raw;

        // Per-finger SFB frequencies
        for (i, &freq) in self.sfb_freq_per_finger.iter().enumerate() {
            stats.finger_sfbs[i] = freq as f64 / bigram_total_raw;
        }

        // Per-finger weighted distance (score / 100 to convert back from centiunits)
        let total_bg_sg = bigram_total_raw + skipgram_total_raw;
        for (i, (&sfb_score, &sfs_score)) in self.sfb_score_per_finger.iter()
            .zip(self.sfs_score_per_finger.iter())
            .enumerate()
        {
            stats.weighted_finger_distance[i] = (sfb_score + sfs_score) as f64 / (total_bg_sg * 100.0);
        }
    }

    /// Check if two positions are on the same finger
    #[inline]
    fn is_same_finger(&self, p_a: usize, p_b: usize) -> Option<usize> {
        self.sf_pairs_per_key[p_a]
            .iter()
            .find(|sf| sf.other_pos == p_b)
            .map(|sf| sf.finger)
    }

    pub fn update_bigram(&mut self, dist_cache: &DistCache, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(finger) = self.is_same_finger(p_a, p_b) {
            let dist = dist_cache.get(p_a, p_b);
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfb_score_per_finger[finger] += score_delta;
            self.sfb_freq_per_finger[finger] += freq_delta;
        }
    }

    pub fn update_skipgram(&mut self, dist_cache: &DistCache, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(finger) = self.is_same_finger(p_a, p_b) {
            let dist = dist_cache.get(p_a, p_b);
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * dist;
            self.sfs_score_per_finger[finger] += score_delta;
            self.sfs_freq_per_finger[finger] += freq_delta;
        }
    }

    /// Get same-finger pairs for a position (for optimized add_key/remove_key)
    #[inline]
    pub fn sf_pairs(&self, pos: usize) -> &[SfPair] {
        &self.sf_pairs_per_key[pos]
    }

    /// Copy scoring data from another SFCache. No allocations.
    #[inline]
    pub fn copy_from(&mut self, other: &SFCache) {
        *self.sfb_score_per_finger = *other.sfb_score_per_finger;
        *self.sfs_score_per_finger = *other.sfs_score_per_finger;
        *self.sfb_freq_per_finger = *other.sfb_freq_per_finger;
        *self.sfs_freq_per_finger = *other.sfs_freq_per_finger;
    }

    /// Optimized key swap: update scores for swapping keys at pos_a and pos_b.
    /// More efficient than remove_key(a) + remove_key(b) + add_key(a, new) + add_key(b, new)
    /// because it only iterates each relevant pair once.
    ///
    /// `keys` is the current key array (before swap).
    #[inline]
    pub fn key_swap<F, G>(
        &mut self,
        dist_cache: &DistCache,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,  // old key at pos_a, will move to pos_b
        key_b: usize,  // old key at pos_b, will move to pos_a
        keys: &[usize],
        get_bg_freq: F,
        get_sg_freq: G,
    ) where
        F: Fn(usize, usize) -> i64,
        G: Fn(usize, usize) -> i64,
    {
        // Check if pos_a and pos_b are on the same finger
        let same_finger_ab = self.is_same_finger(pos_a, pos_b);

        // If they're on the same finger, handle the direct pair between them
        if let Some(finger) = same_finger_ab {
            let dist = dist_cache.get(pos_a, pos_b);

            // Bigram a->b: was (key_a, key_b), now (key_b, key_a)
            let old_bg_ab = get_bg_freq(key_a, key_b);
            let new_bg_ab = get_bg_freq(key_b, key_a);
            let bg_delta_ab = new_bg_ab - old_bg_ab;
            self.sfb_score_per_finger[finger] += bg_delta_ab * dist;
            self.sfb_freq_per_finger[finger] += bg_delta_ab;

            // Bigram b->a: was (key_b, key_a), now (key_a, key_b)
            let old_bg_ba = get_bg_freq(key_b, key_a);
            let new_bg_ba = get_bg_freq(key_a, key_b);
            let bg_delta_ba = new_bg_ba - old_bg_ba;
            self.sfb_score_per_finger[finger] += bg_delta_ba * dist;
            self.sfb_freq_per_finger[finger] += bg_delta_ba;

            // Skipgram a->b
            let old_sg_ab = get_sg_freq(key_a, key_b);
            let new_sg_ab = get_sg_freq(key_b, key_a);
            let sg_delta_ab = new_sg_ab - old_sg_ab;
            self.sfs_score_per_finger[finger] += sg_delta_ab * dist;
            self.sfs_freq_per_finger[finger] += sg_delta_ab;

            // Skipgram b->a
            let old_sg_ba = get_sg_freq(key_b, key_a);
            let new_sg_ba = get_sg_freq(key_a, key_b);
            let sg_delta_ba = new_sg_ba - old_sg_ba;
            self.sfs_score_per_finger[finger] += sg_delta_ba * dist;
            self.sfs_freq_per_finger[finger] += sg_delta_ba;
        }

        // Process same-finger pairs for pos_a (excluding pos_b if same finger)
        for sf in &self.sf_pairs_per_key[pos_a] {
            let other_pos = sf.other_pos;
            if other_pos == pos_b {
                continue; // Already handled above
            }
            let finger = sf.finger;
            let dist = dist_cache.get(pos_a, other_pos);
            let other_key = keys[other_pos];

            // At pos_a: old key was key_a, new key is key_b
            // Bigram: pos_a -> other_pos
            let old_bg = get_bg_freq(key_a, other_key);
            let new_bg = get_bg_freq(key_b, other_key);
            self.sfb_score_per_finger[finger] += (new_bg - old_bg) * dist;
            self.sfb_freq_per_finger[finger] += new_bg - old_bg;

            // Bigram: other_pos -> pos_a
            let old_bg_rev = get_bg_freq(other_key, key_a);
            let new_bg_rev = get_bg_freq(other_key, key_b);
            self.sfb_score_per_finger[finger] += (new_bg_rev - old_bg_rev) * dist;
            self.sfb_freq_per_finger[finger] += new_bg_rev - old_bg_rev;

            // Skipgram: pos_a -> other_pos
            let old_sg = get_sg_freq(key_a, other_key);
            let new_sg = get_sg_freq(key_b, other_key);
            self.sfs_score_per_finger[finger] += (new_sg - old_sg) * dist;
            self.sfs_freq_per_finger[finger] += new_sg - old_sg;

            // Skipgram: other_pos -> pos_a
            let old_sg_rev = get_sg_freq(other_key, key_a);
            let new_sg_rev = get_sg_freq(other_key, key_b);
            self.sfs_score_per_finger[finger] += (new_sg_rev - old_sg_rev) * dist;
            self.sfs_freq_per_finger[finger] += new_sg_rev - old_sg_rev;
        }

        // Process same-finger pairs for pos_b (excluding pos_a if same finger)
        for sf in &self.sf_pairs_per_key[pos_b] {
            let other_pos = sf.other_pos;
            if other_pos == pos_a {
                continue; // Already handled above
            }
            let finger = sf.finger;
            let dist = dist_cache.get(pos_b, other_pos);
            let other_key = keys[other_pos];

            // At pos_b: old key was key_b, new key is key_a
            // Bigram: pos_b -> other_pos
            let old_bg = get_bg_freq(key_b, other_key);
            let new_bg = get_bg_freq(key_a, other_key);
            self.sfb_score_per_finger[finger] += (new_bg - old_bg) * dist;
            self.sfb_freq_per_finger[finger] += new_bg - old_bg;

            // Bigram: other_pos -> pos_b
            let old_bg_rev = get_bg_freq(other_key, key_b);
            let new_bg_rev = get_bg_freq(other_key, key_a);
            self.sfb_score_per_finger[finger] += (new_bg_rev - old_bg_rev) * dist;
            self.sfb_freq_per_finger[finger] += new_bg_rev - old_bg_rev;

            // Skipgram: pos_b -> other_pos
            let old_sg = get_sg_freq(key_b, other_key);
            let new_sg = get_sg_freq(key_a, other_key);
            self.sfs_score_per_finger[finger] += (new_sg - old_sg) * dist;
            self.sfs_freq_per_finger[finger] += new_sg - old_sg;

            // Skipgram: other_pos -> pos_b
            let old_sg_rev = get_sg_freq(other_key, key_b);
            let new_sg_rev = get_sg_freq(other_key, key_a);
            self.sfs_score_per_finger[finger] += (new_sg_rev - old_sg_rev) * dist;
            self.sfs_freq_per_finger[finger] += new_sg_rev - old_sg_rev;
        }
    }
}
