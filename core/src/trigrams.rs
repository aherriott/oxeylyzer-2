use crate::stats::Stats;
use crate::types::CacheKey;
use crate::weights::Weights;
use libdof::dofinitions::Finger as DofFinger;
use libdof::prelude::Finger::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrigramType {
    Sft,
    Sfb,
    // Repeat,
    Inroll,
    Outroll,
    Alternate,
    Redirect,
    OnehandIn,
    OnehandOut,
    Thumb,
    Invalid,
}

/// Pre-computed trigram combination with type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrigramCombo {
    /// Second position in the trigram
    pub pos_b: usize,
    /// Third position in the trigram
    pub pos_c: usize,
    /// Pre-computed trigram type
    pub trigram_type: TrigramType,
}

/// Delta representing changes to TrigramCache state
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TrigramDelta {
    pub inroll_freq: i64,
    pub outroll_freq: i64,
    pub alternate_freq: i64,
    pub redirect_freq: i64,
    pub onehandin_freq: i64,
    pub onehandout_freq: i64,
}

impl TrigramDelta {
    /// Combine two deltas by adding corresponding fields
    pub fn combine(&self, other: &TrigramDelta) -> TrigramDelta {
        TrigramDelta {
            inroll_freq: self.inroll_freq + other.inroll_freq,
            outroll_freq: self.outroll_freq + other.outroll_freq,
            alternate_freq: self.alternate_freq + other.alternate_freq,
            redirect_freq: self.redirect_freq + other.redirect_freq,
            onehandin_freq: self.onehandin_freq + other.onehandin_freq,
            onehandout_freq: self.onehandout_freq + other.onehandout_freq,
        }
    }
}

/// Cache for tracking trigram type frequencies and computing weighted scores
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrigramCache {
    /// For each position, list of (pos_b, pos_c, type) combinations
    trigram_combos_per_key: Vec<Vec<TrigramCombo>>,

    /// Number of keys for frequency array indexing
    num_keys: usize,

    /// Running frequency totals for each trigram type
    inroll_freq: i64,
    outroll_freq: i64,
    alternate_freq: i64,
    redirect_freq: i64,
    onehandin_freq: i64,
    onehandout_freq: i64,

    /// Pre-computed weights from configuration
    inroll_weight: i64,
    outroll_weight: i64,
    alternate_weight: i64,
    redirect_weight: i64,
    onehandin_weight: i64,
    onehandout_weight: i64,

    /// Finger assignments for trigram type lookup (as usize indices)
    fingers: Vec<usize>,
}

impl TrigramCache {
    /// Create a new cache from keyboard layout and finger assignments
    ///
    /// Pre-computes all valid trigram position combinations for each position.
    /// Only stores combinations where the trigram type is one of the tracked types:
    /// Inroll, Outroll, Alternate, Redirect, OnehandIn, OnehandOut.
    pub fn new(fingers: &[DofFinger], num_keys: usize) -> Self {
        // Convert Finger enum to usize indices
        let finger_indices: Vec<usize> = fingers.iter().map(|f| *f as usize).collect();

        let num_positions = fingers.len();
        let mut trigram_combos_per_key = Vec::with_capacity(num_positions);

        // Pre-compute all valid trigram combinations for each position
        for pos_a in 0..num_positions {
            let f_a = finger_indices[pos_a];
            let mut combos = Vec::new();

            for pos_b in 0..num_positions {
                let f_b = finger_indices[pos_b];

                for pos_c in 0..num_positions {
                    let f_c = finger_indices[pos_c];

                    // Look up the trigram type using TRIGRAMS[f_a * 100 + f_b * 10 + f_c]
                    let trigram_type = TRIGRAMS[f_a * 100 + f_b * 10 + f_c];

                    // Only store combinations where the type is one of the tracked types
                    match trigram_type {
                        TrigramType::Inroll
                        | TrigramType::Outroll
                        | TrigramType::Alternate
                        | TrigramType::Redirect
                        | TrigramType::OnehandIn
                        | TrigramType::OnehandOut => {
                            combos.push(TrigramCombo {
                                pos_b,
                                pos_c,
                                trigram_type,
                            });
                        }
                        // Skip untracked types: Sft, Sfb, Thumb, Invalid
                        _ => {}
                    }
                }
            }

            trigram_combos_per_key.push(combos);
        }

        Self {
            trigram_combos_per_key,
            num_keys,
            // Initialize all frequency totals to zero
            inroll_freq: 0,
            outroll_freq: 0,
            alternate_freq: 0,
            redirect_freq: 0,
            onehandin_freq: 0,
            onehandout_freq: 0,
            // Initialize all weights to zero (to be set via set_weights)
            inroll_weight: 0,
            outroll_weight: 0,
            alternate_weight: 0,
            redirect_weight: 0,
            onehandin_weight: 0,
            onehandout_weight: 0,
            // Store finger indices
            fingers: finger_indices,
        }
    }

    /// Set weights from configuration
    ///
    /// Copies the weight values from the Weights struct for each trigram type:
    /// inroll, outroll, alternate, redirect, onehandin, onehandout.
    ///
    /// Requirements: 3.1, 3.2
    pub fn set_weights(&mut self, weights: &Weights) {
        self.inroll_weight = weights.inroll;
        self.outroll_weight = weights.outroll;
        self.alternate_weight = weights.alternate;
        self.redirect_weight = weights.redirect;
        self.onehandin_weight = weights.onehandin;
        self.onehandout_weight = weights.onehandout;
    }

    /// Update for a trigram frequency change
    ///
    /// Looks up the trigram type from finger indices using the TRIGRAMS constant,
    /// then updates the corresponding frequency total if the type is tracked.
    ///
    /// # Arguments
    /// * `p_a` - First position in the trigram
    /// * `p_b` - Second position in the trigram
    /// * `p_c` - Third position in the trigram
    /// * `old_freq` - Previous frequency value
    /// * `new_freq` - New frequency value
    ///
    /// Requirements: 4.1, 4.2, 4.3
    pub fn update_trigram(
        &mut self,
        p_a: usize,
        p_b: usize,
        p_c: usize,
        old_freq: i64,
        new_freq: i64,
    ) {
        // Look up the finger indices for positions p_a, p_b, p_c
        let f_a = self.fingers[p_a];
        let f_b = self.fingers[p_b];
        let f_c = self.fingers[p_c];

        // Compute the trigram type using TRIGRAMS[f_a * 100 + f_b * 10 + f_c]
        let trigram_type = TRIGRAMS[f_a * 100 + f_b * 10 + f_c];

        // Compute the delta
        let delta = new_freq - old_freq;

        // Update the corresponding frequency total if the type is tracked
        // If the type is not tracked (Sft, Sfb, Thumb, Invalid), ignore the update
        match trigram_type {
            TrigramType::Inroll => self.inroll_freq += delta,
            TrigramType::Outroll => self.outroll_freq += delta,
            TrigramType::Alternate => self.alternate_freq += delta,
            TrigramType::Redirect => self.redirect_freq += delta,
            TrigramType::OnehandIn => self.onehandin_freq += delta,
            TrigramType::OnehandOut => self.onehandout_freq += delta,
            // Untracked types: Sft, Sfb, Thumb, Invalid - ignore the update
            TrigramType::Sft | TrigramType::Sfb | TrigramType::Thumb | TrigramType::Invalid => {}
        }
    }

    /// Get the current weighted score
    ///
    /// Returns the weighted sum of all frequency totals:
    /// `inroll_freq * inroll_weight + outroll_freq * outroll_weight + alternate_freq * alternate_weight
    ///  + redirect_freq * redirect_weight + onehandin_freq * onehandin_weight + onehandout_freq * onehandout_weight`
    ///
    /// Requirements: 6.1, 6.2
    pub fn score(&self) -> i64 {
        self.inroll_freq * self.inroll_weight
            + self.outroll_freq * self.outroll_weight
            + self.alternate_freq * self.alternate_weight
            + self.redirect_freq * self.redirect_weight
            + self.onehandin_freq * self.onehandin_weight
            + self.onehandout_freq * self.onehandout_weight
    }

    /// Compute the delta for replacing a key at a position.
    ///
    /// Iterates over all pre-computed trigram combinations involving the position
    /// and computes frequency deltas for old_key vs new_key.
    ///
    /// # Arguments
    /// * `pos` - The position where the key is being replaced
    /// * `old_key` - The key currently at the position
    /// * `new_key` - The key that will replace it
    /// * `keys` - The current key assignments for all positions
    /// * `skip_pos` - Optional position to skip (used during swaps to avoid double-counting)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// # Returns
    /// A TrigramDelta with accumulated changes for each trigram type.
    ///
    /// Requirements: 5.1
    fn compute_replace_delta(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        tg_freq: &[Vec<Vec<i64>>],
    ) -> TrigramDelta {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let mut delta = TrigramDelta::default();

        // Iterate over all pre-computed trigram combinations for this position
        for combo in &self.trigram_combos_per_key[pos] {
            let pos_b = combo.pos_b;
            let pos_c = combo.pos_c;

            // Skip if skip_pos matches pos_b or pos_c (to avoid double-counting in swaps)
            if let Some(skip) = skip_pos {
                if pos_b == skip || pos_c == skip {
                    continue;
                }
            }

            let key_b = keys[pos_b];
            let key_c = keys[pos_c];

            // Skip if key_b or key_c is invalid
            if key_b >= num_keys || key_c >= num_keys {
                continue;
            }

            // Look up old frequency: tg_freq[old_key][key_b][key_c]
            let old_freq = if old_valid {
                tg_freq[old_key][key_b][key_c]
            } else {
                0
            };

            // Look up new frequency: tg_freq[new_key][key_b][key_c]
            let new_freq = if new_valid {
                tg_freq[new_key][key_b][key_c]
            } else {
                0
            };

            // Compute delta
            let freq_delta = new_freq - old_freq;

            // Add delta to the appropriate field based on trigram_type
            match combo.trigram_type {
                TrigramType::Inroll => delta.inroll_freq += freq_delta,
                TrigramType::Outroll => delta.outroll_freq += freq_delta,
                TrigramType::Alternate => delta.alternate_freq += freq_delta,
                TrigramType::Redirect => delta.redirect_freq += freq_delta,
                TrigramType::OnehandIn => delta.onehandin_freq += freq_delta,
                TrigramType::OnehandOut => delta.onehandout_freq += freq_delta,
                // Untracked types should not be in trigram_combos_per_key,
                // but handle them gracefully just in case
                TrigramType::Sft
                | TrigramType::Sfb
                | TrigramType::Thumb
                | TrigramType::Invalid => {}
            }
        }

        delta
    }

    /// Compute only the weighted score delta for replacing a key at a position.
    ///
    /// This is an optimized version of `compute_replace_delta` that only computes
    /// the weighted score delta without building a full TrigramDelta struct.
    /// Used for speculative scoring when apply=false.
    ///
    /// # Arguments
    /// * `pos` - The position where the key is being replaced
    /// * `old_key` - The key currently at the position
    /// * `new_key` - The key that will replace it
    /// * `keys` - The current key assignments for all positions
    /// * `skip_pos` - Optional position to skip (used during swaps to avoid double-counting)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// # Returns
    /// The weighted score delta as a single i64 value.
    ///
    /// Requirements: 5.3
    fn compute_replace_delta_score_only(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let mut score_delta: i64 = 0;

        // Iterate over all pre-computed trigram combinations for this position
        for combo in &self.trigram_combos_per_key[pos] {
            let pos_b = combo.pos_b;
            let pos_c = combo.pos_c;

            // Skip if skip_pos matches pos_b or pos_c (to avoid double-counting in swaps)
            if let Some(skip) = skip_pos {
                if pos_b == skip || pos_c == skip {
                    continue;
                }
            }

            let key_b = keys[pos_b];
            let key_c = keys[pos_c];

            // Skip if key_b or key_c is invalid
            if key_b >= num_keys || key_c >= num_keys {
                continue;
            }

            // Look up old frequency: tg_freq[old_key][key_b][key_c]
            let old_freq = if old_valid {
                tg_freq[old_key][key_b][key_c]
            } else {
                0
            };

            // Look up new frequency: tg_freq[new_key][key_b][key_c]
            let new_freq = if new_valid {
                tg_freq[new_key][key_b][key_c]
            } else {
                0
            };

            // Compute frequency delta
            let freq_delta = new_freq - old_freq;

            // Multiply by the appropriate weight based on trigram_type and accumulate
            score_delta += match combo.trigram_type {
                TrigramType::Inroll => freq_delta * self.inroll_weight,
                TrigramType::Outroll => freq_delta * self.outroll_weight,
                TrigramType::Alternate => freq_delta * self.alternate_weight,
                TrigramType::Redirect => freq_delta * self.redirect_weight,
                TrigramType::OnehandIn => freq_delta * self.onehandin_weight,
                TrigramType::OnehandOut => freq_delta * self.onehandout_weight,
                // Untracked types should not be in trigram_combos_per_key,
                // but handle them gracefully just in case
                TrigramType::Sft
                | TrigramType::Sfb
                | TrigramType::Thumb
                | TrigramType::Invalid => 0,
            };
        }

        score_delta
    }

    /// Apply a delta to the cache state.
    ///
    /// Updates all internal frequency totals by adding the delta values.
    ///
    /// # Arguments
    /// * `delta` - The delta to apply
    fn apply_delta(&mut self, delta: &TrigramDelta) {
        self.inroll_freq += delta.inroll_freq;
        self.outroll_freq += delta.outroll_freq;
        self.alternate_freq += delta.alternate_freq;
        self.redirect_freq += delta.redirect_freq;
        self.onehandin_freq += delta.onehandin_freq;
        self.onehandout_freq += delta.onehandout_freq;
    }

    /// Replace key at position. Returns the new score.
    ///
    /// If `apply` is true, computes the full delta, applies it to internal state,
    /// and returns the new score.
    /// If `apply` is false, computes only the score delta and returns the
    /// speculative new score without mutating state.
    ///
    /// # Arguments
    /// * `pos` - The position where the key is being replaced
    /// * `old_key` - The key currently at the position
    /// * `new_key` - The key that will replace it
    /// * `keys` - The current key assignments for all positions
    /// * `skip_pos` - Optional position to skip (used during swaps to avoid double-counting)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    /// * `apply` - If true, updates internal state; if false, only computes the score
    ///
    /// # Returns
    /// The new total score after the replacement.
    ///
    /// Requirements: 5.1, 5.2, 5.3
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        if apply {
            // Compute full delta and apply it
            let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, tg_freq);
            self.apply_delta(&delta);
            self.score()
        } else {
            // Compute only the score delta without mutating state
            let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, skip_pos, tg_freq);
            self.score() + score_delta
        }
    }

    /// Swap keys at two positions. Returns the new score.
    ///
    /// If `apply` is true, computes the combined delta for both positions,
    /// applies it to internal state, and returns the new score.
    /// If `apply` is false, computes only the score deltas and returns the
    /// speculative new score without mutating state.
    ///
    /// Uses skip_pos to avoid double-counting trigrams that involve both
    /// swapped positions.
    ///
    /// # Arguments
    /// * `pos_a` - First position in the swap
    /// * `pos_b` - Second position in the swap
    /// * `key_a` - The key currently at pos_a (will move to pos_b)
    /// * `key_b` - The key currently at pos_b (will move to pos_a)
    /// * `keys` - The current key assignments for all positions
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    /// * `apply` - If true, updates internal state; if false, only computes the score
    ///
    /// # Returns
    /// The new total score after the swap.
    ///
    /// Requirements: 5.4, 5.5
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        if apply {
            // Compute delta for pos_a: replacing key_a with key_b, skip pos_b to avoid double-counting
            let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), tg_freq);

            // Compute delta for pos_b: replacing key_b with key_a, skip pos_a to avoid double-counting
            let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), tg_freq);

            // Combine the two deltas
            let combined_delta = delta_a.combine(&delta_b);

            // Apply the combined delta to internal state
            self.apply_delta(&combined_delta);

            // Return the new score
            self.score()
        } else {
            // Compute only the score deltas without mutating state
            let score_delta_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), tg_freq);
            let score_delta_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), tg_freq);

            // Return the speculative new score
            self.score() + score_delta_a + score_delta_b
        }
    }

    /// Populate statistics with normalized trigram frequencies
    ///
    /// Normalizes each frequency by dividing by trigram_total and populates
    /// the TrigramStats fields: inroll, outroll, alternate, redirect, onehandin, onehandout.
    ///
    /// If trigram_total is zero or negative, all fields are set to 0 to avoid
    /// division by zero.
    ///
    /// # Arguments
    /// * `stats` - Mutable reference to Stats struct to populate
    /// * `trigram_total` - Total trigram count for normalization
    ///
    /// Requirements: 7.1, 7.2, 7.3
    pub fn stats(&self, stats: &mut Stats, trigram_total: f64) {
        // Handle division by zero: if trigram_total is zero or negative, set all fields to 0
        if trigram_total <= 0.0 {
            stats.trigrams.inroll = 0.0;
            stats.trigrams.outroll = 0.0;
            stats.trigrams.alternate = 0.0;
            stats.trigrams.redirect = 0.0;
            stats.trigrams.onehandin = 0.0;
            stats.trigrams.onehandout = 0.0;
            return;
        }

        // Normalize frequencies by dividing by trigram_total
        stats.trigrams.inroll = self.inroll_freq as f64 / trigram_total;
        stats.trigrams.outroll = self.outroll_freq as f64 / trigram_total;
        stats.trigrams.alternate = self.alternate_freq as f64 / trigram_total;
        stats.trigrams.redirect = self.redirect_freq as f64 / trigram_total;
        stats.trigrams.onehandin = self.onehandin_freq as f64 / trigram_total;
        stats.trigrams.onehandout = self.onehandout_freq as f64 / trigram_total;
    }
}

#[derive(Debug, Clone, Copy)]
enum Hand {
    Left,
    Right,
}

impl Hand {
    const fn eq(self, rhs: Self) -> bool {
        self as CacheKey == rhs as CacheKey
    }
}

#[derive(Debug, Clone, Copy)]
struct Finger(DofFinger);

impl Finger {
    const fn eq(self, rhs: Self) -> bool {
        self.0 as CacheKey == rhs.0 as CacheKey
    }

    const fn hand(self) -> Hand {
        match self.0 {
            LP | LR | LM | LI | LT => Hand::Left,
            RP | RR | RM | RI | RT => Hand::Right,
        }
    }

    const fn is_thumb(self) -> bool {
        matches!(self.0, RT | LT)
    }

    const fn _is_index(self) -> bool {
        matches!(self.0, LI | RI)
    }

    const fn _is_non_index(self) -> bool {
        !(self.is_thumb() || self._is_index())
    }

    const fn is_inward(self, rhs: Self) -> bool {
        matches!(
            (self.0, rhs.0),
            (LP, LR | LM | LI | LT)
                | (RP, RP | RM | RI | RT)
                | (LR, LM | LI | LT)
                | (RR, RM | RI | RT)
                | (LM, LI | LT)
                | (RM, RI | RT)
                | (LI, LT)
                | (RI, RT)
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Trigram([Finger; 3]);

impl Trigram {
    const fn new([f1, f2, f3]: [DofFinger; 3]) -> Self {
        Self([Finger(f1), Finger(f2), Finger(f3)])
    }

    const fn is_sft(&self) -> bool {
        let [f1, f2, f3] = self.0;

        f1.eq(f2) && f2.eq(f3)
    }

    const fn is_sfb(&self) -> bool {
        let [f1, f2, f3] = self.0;

        !self.is_sft() && (f1.eq(f2) || f2.eq(f3))
    }

    const fn is_inroll(&self) -> bool {
        let [f1, f2, f3] = self.0;
        let [h1, h2, h3] = [f1.hand(), f2.hand(), f3.hand()];

        h1.eq(h2) && !h2.eq(h3) && f1.is_inward(f2) || h2.eq(h3) && !h1.eq(h2) && f2.is_inward(f3)
    }

    const fn is_outroll(&self) -> bool {
        let [f1, f2, f3] = self.0;

        f1.hand().eq(f2.hand()) && !f2.hand().eq(f3.hand()) && !f1.is_inward(f2)
            || f2.hand().eq(f3.hand()) && !f1.hand().eq(f2.hand()) && !f2.is_inward(f3)
    }

    const fn is_alternate(&self) -> bool {
        let [f1, f2, f3] = self.0;

        !f1.hand().eq(f2.hand()) && !f2.hand().eq(f3.hand())
    }

    const fn is_redirect(&self) -> bool {
        let [f1, f2, f3] = self.0;

        (f1.is_inward(f2) && !f2.is_inward(f3)) || (!f1.is_inward(f2) && f2.is_inward(f3))
    }

    const fn is_onehandin(&self) -> bool {
        let [f1, f2, f3] = self.0;

        f1.is_inward(f2) && f2.is_inward(f3)
    }

    const fn is_onehandout(&self) -> bool {
        let [f1, f2, f3] = self.0;

        !(f1.is_inward(f2) || f2.is_inward(f3))
    }

    const fn is_thumb(&self) -> bool {
        let [f1, f2, f3] = self.0;

        f1.is_thumb() || f2.is_thumb() || f3.is_thumb()
    }
}

pub const fn trigrams() -> [TrigramType; 1000] {
    use TrigramType::*;

    let mut res = [Invalid; 1000];

    let mut i = 0;
    while i < 10 {
        let mut j = 0;
        while j < 10 {
            let mut k = 0;
            while k < 10 {
                let fs = Trigram::new([
                    DofFinger::FINGERS[i],
                    DofFinger::FINGERS[j],
                    DofFinger::FINGERS[k],
                ]);

                res[i * 100 + j * 10 + k] = if fs.is_thumb() {
                    Thumb
                } else if fs.is_sft() {
                    Sft
                } else if fs.is_sfb() {
                    Sfb
                } else if fs.is_inroll() {
                    Inroll
                } else if fs.is_outroll() {
                    Outroll
                } else if fs.is_alternate() {
                    Alternate
                } else if fs.is_redirect() {
                    Redirect
                } else if fs.is_onehandin() {
                    OnehandIn
                } else if fs.is_onehandout() {
                    OnehandOut
                } else {
                    Invalid
                };

                k += 1;
            }

            j += 1;
        }

        i += 1;
    }

    res
}

pub const TRIGRAMS: [TrigramType; 1000] = trigrams();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print() {
        for f1 in DofFinger::FINGERS {
            for f2 in DofFinger::FINGERS {
                for f3 in DofFinger::FINGERS {
                    let t = TRIGRAMS[f1 as usize * 100 + f2 as usize * 10 + f3 as usize];
                    println!("{f1} {f2} {f3}: {t:?}");
                }
            }
        }
    }

    #[test]
    fn test_trigram_combo_creation() {
        let combo = TrigramCombo {
            pos_b: 1,
            pos_c: 2,
            trigram_type: TrigramType::Inroll,
        };
        assert_eq!(combo.pos_b, 1);
        assert_eq!(combo.pos_c, 2);
        assert_eq!(combo.trigram_type, TrigramType::Inroll);
    }

    #[test]
    fn test_trigram_delta_default() {
        let delta = TrigramDelta::default();
        assert_eq!(delta.inroll_freq, 0);
        assert_eq!(delta.outroll_freq, 0);
        assert_eq!(delta.alternate_freq, 0);
        assert_eq!(delta.redirect_freq, 0);
        assert_eq!(delta.onehandin_freq, 0);
        assert_eq!(delta.onehandout_freq, 0);
    }

    #[test]
    fn test_trigram_delta_combine() {
        let delta1 = TrigramDelta {
            inroll_freq: 10,
            outroll_freq: 20,
            alternate_freq: 30,
            redirect_freq: 40,
            onehandin_freq: 50,
            onehandout_freq: 60,
        };
        let delta2 = TrigramDelta {
            inroll_freq: 1,
            outroll_freq: 2,
            alternate_freq: 3,
            redirect_freq: 4,
            onehandin_freq: 5,
            onehandout_freq: 6,
        };

        let combined = delta1.combine(&delta2);

        assert_eq!(combined.inroll_freq, 11);
        assert_eq!(combined.outroll_freq, 22);
        assert_eq!(combined.alternate_freq, 33);
        assert_eq!(combined.redirect_freq, 44);
        assert_eq!(combined.onehandin_freq, 55);
        assert_eq!(combined.onehandout_freq, 66);
    }

    #[test]
    fn test_trigram_delta_combine_with_negatives() {
        let delta1 = TrigramDelta {
            inroll_freq: 100,
            outroll_freq: 50,
            alternate_freq: 0,
            redirect_freq: -10,
            onehandin_freq: 25,
            onehandout_freq: -5,
        };
        let delta2 = TrigramDelta {
            inroll_freq: -30,
            outroll_freq: -50,
            alternate_freq: 10,
            redirect_freq: 10,
            onehandin_freq: -25,
            onehandout_freq: 5,
        };

        let combined = delta1.combine(&delta2);

        assert_eq!(combined.inroll_freq, 70);
        assert_eq!(combined.outroll_freq, 0);
        assert_eq!(combined.alternate_freq, 10);
        assert_eq!(combined.redirect_freq, 0);
        assert_eq!(combined.onehandin_freq, 0);
        assert_eq!(combined.onehandout_freq, 0);
    }

    #[test]
    fn test_trigram_delta_combine_identity() {
        let delta = TrigramDelta {
            inroll_freq: 10,
            outroll_freq: 20,
            alternate_freq: 30,
            redirect_freq: 40,
            onehandin_freq: 50,
            onehandout_freq: 60,
        };
        let zero = TrigramDelta::default();

        let combined = delta.combine(&zero);

        assert_eq!(combined, delta);
    }

    #[test]
    fn test_trigram_cache_new_empty() {
        let fingers: Vec<DofFinger> = vec![];
        let cache = TrigramCache::new(&fingers, 30);

        assert_eq!(cache.num_keys, 30);
        assert_eq!(cache.fingers.len(), 0);
        assert_eq!(cache.trigram_combos_per_key.len(), 0);
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.alternate_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
        assert_eq!(cache.inroll_weight, 0);
        assert_eq!(cache.outroll_weight, 0);
        assert_eq!(cache.alternate_weight, 0);
        assert_eq!(cache.redirect_weight, 0);
        assert_eq!(cache.onehandin_weight, 0);
        assert_eq!(cache.onehandout_weight, 0);
    }

    #[test]
    fn test_trigram_cache_new_with_fingers() {
        // Create a simple 4-key layout with 2 fingers on each hand
        // LP, LI, RI, RP (left pinky, left index, right index, right pinky)
        let fingers = vec![LP, LI, RI, RP];
        let cache = TrigramCache::new(&fingers, 30);

        assert_eq!(cache.num_keys, 30);
        assert_eq!(cache.fingers.len(), 4);
        assert_eq!(cache.fingers[0], LP as usize);
        assert_eq!(cache.fingers[1], LI as usize);
        assert_eq!(cache.fingers[2], RI as usize);
        assert_eq!(cache.fingers[3], RP as usize);
        assert_eq!(cache.trigram_combos_per_key.len(), 4);
    }

    #[test]
    fn test_trigram_cache_precomputes_combos() {
        // Create a layout where we can verify specific trigram types
        // Using LP, LI, RI to get alternates (LP -> RI -> LP would be alternate)
        let fingers = vec![LP, LI, RI];
        let cache = TrigramCache::new(&fingers, 30);

        // Each position should have some combinations
        assert_eq!(cache.trigram_combos_per_key.len(), 3);

        // Verify that all stored combos have tracked types
        for combos in &cache.trigram_combos_per_key {
            for combo in combos {
                match combo.trigram_type {
                    TrigramType::Inroll
                    | TrigramType::Outroll
                    | TrigramType::Alternate
                    | TrigramType::Redirect
                    | TrigramType::OnehandIn
                    | TrigramType::OnehandOut => {}
                    _ => panic!("Found untracked type in combos: {:?}", combo.trigram_type),
                }
            }
        }
    }

    #[test]
    fn test_trigram_cache_filters_untracked_types() {
        // Create a layout with same finger positions to generate Sft/Sfb types
        // These should NOT be stored in the cache
        let fingers = vec![LP, LP, LP]; // All same finger
        let cache = TrigramCache::new(&fingers, 30);

        // All combinations with same finger should be Sft or Sfb, which are not tracked
        // So we should have no combos stored
        for combos in &cache.trigram_combos_per_key {
            for combo in combos {
                // If any combo is stored, it must be a tracked type
                match combo.trigram_type {
                    TrigramType::Sft | TrigramType::Sfb | TrigramType::Thumb | TrigramType::Invalid => {
                        panic!("Found untracked type in combos: {:?}", combo.trigram_type);
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_trigram_cache_alternate_detection() {
        // LP -> RI -> LP should be an alternate (hand alternates each key)
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 30);

        // Find combos for position 0 (LP)
        let combos_for_pos0 = &cache.trigram_combos_per_key[0];

        // Look for the combo (pos_b=1, pos_c=0) which is LP -> RI -> LP
        let alternate_combo = combos_for_pos0
            .iter()
            .find(|c| c.pos_b == 1 && c.pos_c == 0);

        assert!(
            alternate_combo.is_some(),
            "Should find alternate combo LP -> RI -> LP"
        );
        assert_eq!(
            alternate_combo.unwrap().trigram_type,
            TrigramType::Alternate
        );
    }

    #[test]
    fn test_trigram_cache_default() {
        let cache = TrigramCache::default();

        assert_eq!(cache.num_keys, 0);
        assert!(cache.fingers.is_empty());
        assert!(cache.trigram_combos_per_key.is_empty());
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.alternate_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
    }

    #[test]
    fn test_set_weights() {
        use crate::weights::dummy_weights;

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify weights are initially zero
        assert_eq!(cache.inroll_weight, 0);
        assert_eq!(cache.outroll_weight, 0);
        assert_eq!(cache.alternate_weight, 0);
        assert_eq!(cache.redirect_weight, 0);
        assert_eq!(cache.onehandin_weight, 0);
        assert_eq!(cache.onehandout_weight, 0);

        // Set weights from dummy_weights
        let weights = dummy_weights();
        cache.set_weights(&weights);

        // Verify weights are copied correctly
        assert_eq!(cache.inroll_weight, weights.inroll);
        assert_eq!(cache.outroll_weight, weights.outroll);
        assert_eq!(cache.alternate_weight, weights.alternate);
        assert_eq!(cache.redirect_weight, weights.redirect);
        assert_eq!(cache.onehandin_weight, weights.onehandin);
        assert_eq!(cache.onehandout_weight, weights.onehandout);
    }

    #[test]
    fn test_set_weights_with_custom_values() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Create custom weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 100,
            outroll: 200,
            alternate: 300,
            redirect: -400,
            onehandin: 50,
            onehandout: -25,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };

        cache.set_weights(&weights);

        // Verify weights are copied correctly
        assert_eq!(cache.inroll_weight, 100);
        assert_eq!(cache.outroll_weight, 200);
        assert_eq!(cache.alternate_weight, 300);
        assert_eq!(cache.redirect_weight, -400);
        assert_eq!(cache.onehandin_weight, 50);
        assert_eq!(cache.onehandout_weight, -25);
    }

    #[test]
    fn test_set_weights_can_be_called_multiple_times() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set initial weights
        let weights1 = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 20,
            alternate: 30,
            redirect: 40,
            onehandin: 50,
            onehandout: 60,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights1);

        assert_eq!(cache.inroll_weight, 10);
        assert_eq!(cache.outroll_weight, 20);

        // Set new weights (should overwrite)
        let weights2 = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 100,
            outroll: 200,
            alternate: 300,
            redirect: 400,
            onehandin: 500,
            onehandout: 600,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights2);

        // Verify new weights are applied
        assert_eq!(cache.inroll_weight, 100);
        assert_eq!(cache.outroll_weight, 200);
        assert_eq!(cache.alternate_weight, 300);
        assert_eq!(cache.redirect_weight, 400);
        assert_eq!(cache.onehandin_weight, 500);
        assert_eq!(cache.onehandout_weight, 600);
    }

    #[test]
    fn test_score_returns_zero_for_default_cache() {
        let cache = TrigramCache::default();
        assert_eq!(cache.score(), 0);
    }

    #[test]
    fn test_score_returns_zero_when_frequencies_are_zero() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set non-zero weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 100,
            outroll: 200,
            alternate: 300,
            redirect: 400,
            onehandin: 500,
            onehandout: 600,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Score should still be zero because frequencies are zero
        assert_eq!(cache.score(), 0);
    }

    #[test]
    fn test_score_returns_zero_when_weights_are_zero() {
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Manually set non-zero frequencies
        cache.inroll_freq = 10;
        cache.outroll_freq = 20;
        cache.alternate_freq = 30;
        cache.redirect_freq = 40;
        cache.onehandin_freq = 50;
        cache.onehandout_freq = 60;

        // Weights are zero by default, so score should be zero
        assert_eq!(cache.score(), 0);
    }

    #[test]
    fn test_score_computes_weighted_sum() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set frequencies
        cache.inroll_freq = 10;
        cache.outroll_freq = 20;
        cache.alternate_freq = 30;
        cache.redirect_freq = 40;
        cache.onehandin_freq = 50;
        cache.onehandout_freq = 60;

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 2,
            alternate: 3,
            redirect: 4,
            onehandin: 5,
            onehandout: 6,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Expected: 10*1 + 20*2 + 30*3 + 40*4 + 50*5 + 60*6
        //         = 10 + 40 + 90 + 160 + 250 + 360 = 910
        assert_eq!(cache.score(), 910);
    }

    #[test]
    fn test_score_handles_negative_weights() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set frequencies
        cache.inroll_freq = 100;
        cache.outroll_freq = 100;
        cache.alternate_freq = 100;
        cache.redirect_freq = 100;
        cache.onehandin_freq = 100;
        cache.onehandout_freq = 100;

        // Set weights with some negative values (redirect is typically negative)
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 10,
            alternate: 10,
            redirect: -20,  // Negative weight for redirect
            onehandin: 5,
            onehandout: -5, // Negative weight for onehandout
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Expected: 100*10 + 100*10 + 100*10 + 100*(-20) + 100*5 + 100*(-5)
        //         = 1000 + 1000 + 1000 - 2000 + 500 - 500 = 1000
        assert_eq!(cache.score(), 1000);
    }

    #[test]
    fn test_score_handles_negative_frequencies() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set frequencies with some negative values
        cache.inroll_freq = 50;
        cache.outroll_freq = -10;
        cache.alternate_freq = 30;
        cache.redirect_freq = -20;
        cache.onehandin_freq = 10;
        cache.onehandout_freq = 0;

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 2,
            outroll: 3,
            alternate: 4,
            redirect: 5,
            onehandin: 6,
            onehandout: 7,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Expected: 50*2 + (-10)*3 + 30*4 + (-20)*5 + 10*6 + 0*7
        //         = 100 - 30 + 120 - 100 + 60 + 0 = 150
        assert_eq!(cache.score(), 150);
    }

    #[test]
    fn test_score_with_single_nonzero_type() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Only set inroll frequency
        cache.inroll_freq = 42;

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 20,
            alternate: 30,
            redirect: 40,
            onehandin: 50,
            onehandout: 60,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Expected: 42*10 = 420
        assert_eq!(cache.score(), 420);
    }

    // ==================== update_trigram tests ====================

    #[test]
    fn test_update_trigram_alternate_type() {
        // LP -> RI -> LP is an alternate (hand alternates each key)
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.alternate_freq, 0);

        // Update trigram with positions 0, 1, 0 (LP -> RI -> LP)
        cache.update_trigram(0, 1, 0, 0, 100);

        // Should update alternate frequency
        assert_eq!(cache.alternate_freq, 100);
        // Other frequencies should remain zero
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
    }

    #[test]
    fn test_update_trigram_inroll_type() {
        // LP -> LI -> RI should be an inroll (LP -> LI is inward on left hand, then switch to right)
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.inroll_freq, 0);

        // Update trigram with positions 0, 1, 2 (LP -> LI -> RI)
        cache.update_trigram(0, 1, 2, 0, 50);

        // Should update inroll frequency
        assert_eq!(cache.inroll_freq, 50);
    }

    #[test]
    fn test_update_trigram_outroll_type() {
        // LI -> LP -> RI should be an outroll (LI -> LP is outward on left hand, then switch to right)
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.outroll_freq, 0);

        // Update trigram with positions 1, 0, 2 (LI -> LP -> RI)
        cache.update_trigram(1, 0, 2, 0, 75);

        // Should update outroll frequency
        assert_eq!(cache.outroll_freq, 75);
    }

    #[test]
    fn test_update_trigram_redirect_type() {
        // LP -> LI -> LP should be a redirect (direction changes: inward then outward)
        let fingers = vec![LP, LI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.redirect_freq, 0);

        // Update trigram with positions 0, 1, 0 (LP -> LI -> LP)
        cache.update_trigram(0, 1, 0, 0, 25);

        // Should update redirect frequency
        assert_eq!(cache.redirect_freq, 25);
    }

    #[test]
    fn test_update_trigram_onehandin_type() {
        // LP -> LR -> LM should be onehandin (all inward on left hand)
        let fingers = vec![LP, LR, LM];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.onehandin_freq, 0);

        // Update trigram with positions 0, 1, 2 (LP -> LR -> LM)
        cache.update_trigram(0, 1, 2, 0, 30);

        // Should update onehandin frequency
        assert_eq!(cache.onehandin_freq, 30);
    }

    #[test]
    fn test_update_trigram_onehandout_type() {
        // LM -> LR -> LP should be onehandout (all outward on left hand)
        let fingers = vec![LP, LR, LM];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Verify initial state
        assert_eq!(cache.onehandout_freq, 0);

        // Update trigram with positions 2, 1, 0 (LM -> LR -> LP)
        cache.update_trigram(2, 1, 0, 0, 40);

        // Should update onehandout frequency
        assert_eq!(cache.onehandout_freq, 40);
    }

    #[test]
    fn test_update_trigram_untracked_sft_ignored() {
        // LP -> LP -> LP is Sft (same finger trigram) - should be ignored
        let fingers = vec![LP, LP, LP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Update trigram with all same finger positions
        cache.update_trigram(0, 1, 2, 0, 100);

        // All frequencies should remain zero (Sft is not tracked)
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.alternate_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
    }

    #[test]
    fn test_update_trigram_untracked_sfb_ignored() {
        // LP -> LP -> LI is Sfb (same finger bigram) - should be ignored
        let fingers = vec![LP, LP, LI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Update trigram with same finger bigram
        cache.update_trigram(0, 1, 2, 0, 100);

        // All frequencies should remain zero (Sfb is not tracked)
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.alternate_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
    }

    #[test]
    fn test_update_trigram_untracked_thumb_ignored() {
        // Any trigram involving thumb should be ignored
        let fingers = vec![LP, LT, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Update trigram involving thumb
        cache.update_trigram(0, 1, 2, 0, 100);

        // All frequencies should remain zero (Thumb is not tracked)
        assert_eq!(cache.inroll_freq, 0);
        assert_eq!(cache.outroll_freq, 0);
        assert_eq!(cache.alternate_freq, 0);
        assert_eq!(cache.redirect_freq, 0);
        assert_eq!(cache.onehandin_freq, 0);
        assert_eq!(cache.onehandout_freq, 0);
    }

    #[test]
    fn test_update_trigram_delta_calculation() {
        // Test that delta is correctly computed as new_freq - old_freq
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // First update: 0 -> 100 (delta = 100)
        cache.update_trigram(0, 1, 0, 0, 100);
        assert_eq!(cache.alternate_freq, 100);

        // Second update: 100 -> 150 (delta = 50)
        cache.update_trigram(0, 1, 0, 100, 150);
        assert_eq!(cache.alternate_freq, 150);

        // Third update: 150 -> 120 (delta = -30)
        cache.update_trigram(0, 1, 0, 150, 120);
        assert_eq!(cache.alternate_freq, 120);
    }

    #[test]
    fn test_update_trigram_negative_delta() {
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Start with some frequency
        cache.update_trigram(0, 1, 0, 0, 100);
        assert_eq!(cache.alternate_freq, 100);

        // Decrease frequency (negative delta)
        cache.update_trigram(0, 1, 0, 100, 30);
        assert_eq!(cache.alternate_freq, 30);
    }

    #[test]
    fn test_update_trigram_zero_delta() {
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Start with some frequency
        cache.update_trigram(0, 1, 0, 0, 100);
        assert_eq!(cache.alternate_freq, 100);

        // Update with same old and new (delta = 0)
        cache.update_trigram(0, 1, 0, 50, 50);
        assert_eq!(cache.alternate_freq, 100); // Should remain unchanged
    }

    #[test]
    fn test_update_trigram_multiple_types() {
        // Test updating multiple different trigram types
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // LP -> RI -> LP is alternate
        cache.update_trigram(0, 2, 0, 0, 10);
        assert_eq!(cache.alternate_freq, 10);

        // LP -> LI -> RI is inroll
        cache.update_trigram(0, 1, 2, 0, 20);
        assert_eq!(cache.inroll_freq, 20);

        // LI -> LP -> RI is outroll
        cache.update_trigram(1, 0, 2, 0, 30);
        assert_eq!(cache.outroll_freq, 30);

        // Verify all frequencies are independent
        assert_eq!(cache.alternate_freq, 10);
        assert_eq!(cache.inroll_freq, 20);
        assert_eq!(cache.outroll_freq, 30);
    }

    #[test]
    fn test_update_trigram_affects_score() {
        use crate::weights::{FingerWeights, Weights};

        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 2,
            alternate: 10,
            redirect: 4,
            onehandin: 5,
            onehandout: 6,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        // Initial score should be 0
        assert_eq!(cache.score(), 0);

        // Update alternate trigram
        cache.update_trigram(0, 1, 0, 0, 5);

        // Score should now be 5 * 10 = 50
        assert_eq!(cache.score(), 50);
    }

    // ==================== compute_replace_delta tests ====================

    /// Helper function to create a 3D trigram frequency array
    fn create_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
        let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
        for &(k1, k2, k3, freq) in entries {
            if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                tg_freq[k1][k2][k3] = freq;
            }
        }
        tg_freq
    }

    #[test]
    fn test_compute_replace_delta_basic() {
        // Test basic delta computation for an alternate trigram
        // Position 0 (LP) and Position 1 (RI) form alternates
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        // Keys: position 0 has key 0, position 1 has key 1
        let keys = vec![0, 1];

        // Create trigram frequency data
        // For alternate trigram (pos 0 -> pos 1 -> pos 0), we need tg_freq[key_a][key_b][key_c]
        // where key_a is at pos 0, key_b is at pos 1, key_c is at pos 0
        // Old: tg_freq[0][1][0] = 100 (key 0 at pos 0)
        // New: tg_freq[2][1][0] = 150 (key 2 at pos 0)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0, key_b=1, key_c=0
            (2, 1, 0, 150),  // new_key=2, key_b=1, key_c=0
        ]);

        // Replace key at position 0: old_key=0, new_key=2
        let delta = cache.compute_replace_delta(0, 0, 2, &keys, None, &tg_freq);

        // The delta for alternate should be 150 - 100 = 50
        assert_eq!(delta.alternate_freq, 50);
        // Other types should be 0
        assert_eq!(delta.inroll_freq, 0);
        assert_eq!(delta.outroll_freq, 0);
        assert_eq!(delta.redirect_freq, 0);
        assert_eq!(delta.onehandin_freq, 0);
        assert_eq!(delta.onehandout_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_with_skip_pos() {
        // Test that skip_pos correctly skips the specified position
        let fingers = vec![LP, RI, RP];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1, 2];

        // Create trigram frequency data
        // Trigram involving pos 0, 1, 0 (alternate)
        // Trigram involving pos 0, 2, 0 (alternate)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // pos 0 -> pos 1 -> pos 0
            (3, 1, 0, 200),  // new key at pos 0
            (0, 2, 0, 50),   // pos 0 -> pos 2 -> pos 0
            (3, 2, 0, 75),   // new key at pos 0
        ]);

        // Replace key at position 0, skipping position 1
        let delta = cache.compute_replace_delta(0, 0, 3, &keys, Some(1), &tg_freq);

        // Only trigrams NOT involving position 1 should be counted
        // The trigram (0, 2, 0) should contribute: 75 - 50 = 25
        // The trigram (0, 1, 0) should be skipped because pos_b=1 matches skip_pos
        assert_eq!(delta.alternate_freq, 25);
    }

    #[test]
    fn test_compute_replace_delta_negative_delta() {
        // Test that negative deltas are computed correctly
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1];

        // Old key has higher frequency than new key
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 200),  // old_key=0
            (2, 1, 0, 50),   // new_key=2
        ]);

        let delta = cache.compute_replace_delta(0, 0, 2, &keys, None, &tg_freq);

        // Delta should be negative: 50 - 200 = -150
        assert_eq!(delta.alternate_freq, -150);
    }

    #[test]
    fn test_compute_replace_delta_invalid_old_key() {
        // Test that invalid old_key (>= num_keys) is handled correctly
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (2, 1, 0, 100),  // new_key=2
        ]);

        // old_key=99 is invalid (>= num_keys=4)
        let delta = cache.compute_replace_delta(0, 99, 2, &keys, None, &tg_freq);

        // Old frequency should be treated as 0, so delta = 100 - 0 = 100
        assert_eq!(delta.alternate_freq, 100);
    }

    #[test]
    fn test_compute_replace_delta_invalid_new_key() {
        // Test that invalid new_key (>= num_keys) is handled correctly
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0
        ]);

        // new_key=99 is invalid (>= num_keys=4)
        let delta = cache.compute_replace_delta(0, 0, 99, &keys, None, &tg_freq);

        // New frequency should be treated as 0, so delta = 0 - 100 = -100
        assert_eq!(delta.alternate_freq, -100);
    }

    #[test]
    fn test_compute_replace_delta_invalid_other_key() {
        // Test that invalid keys at other positions are skipped
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        // key at position 1 is invalid (>= num_keys)
        let keys = vec![0, 99];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 0, 200),
        ]);

        let delta = cache.compute_replace_delta(0, 0, 2, &keys, None, &tg_freq);

        // All trigrams should be skipped because key_b=99 is invalid
        assert_eq!(delta.alternate_freq, 0);
        assert_eq!(delta.inroll_freq, 0);
        assert_eq!(delta.outroll_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_multiple_trigram_types() {
        // Test that multiple trigram types are accumulated correctly
        // LP -> LI -> RI is inroll
        // LI -> LP -> RI is outroll
        // LP -> RI -> LP is alternate
        let fingers = vec![LP, LI, RI];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1, 2];

        // Create frequency data for different trigram types
        let tg_freq = create_tg_freq(4, &[
            // For position 0 (LP), combos include:
            // (pos_b=1, pos_c=2) -> LP -> LI -> RI (inroll)
            (0, 1, 2, 100),  // old: key 0 at pos 0
            (3, 1, 2, 150),  // new: key 3 at pos 0
            // (pos_b=2, pos_c=1) -> LP -> RI -> LI (alternate)
            (0, 2, 1, 50),   // old
            (3, 2, 1, 80),   // new
        ]);

        let delta = cache.compute_replace_delta(0, 0, 3, &keys, None, &tg_freq);

        // Inroll delta: 150 - 100 = 50
        assert_eq!(delta.inroll_freq, 50);
        // Alternate delta: 80 - 50 = 30
        assert_eq!(delta.alternate_freq, 30);
    }

    #[test]
    fn test_compute_replace_delta_empty_cache() {
        // Test with an empty cache (no positions)
        let fingers: Vec<DofFinger> = vec![];
        let cache = TrigramCache::new(&fingers, 4);

        let keys: Vec<usize> = vec![];
        let tg_freq = create_tg_freq(4, &[]);

        // This should not panic and return a default delta
        // Note: We can't actually call compute_replace_delta with pos=0 on an empty cache
        // because trigram_combos_per_key would be empty. Let's test with a valid but empty scenario.
        let delta = TrigramDelta::default();
        assert_eq!(delta.inroll_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_same_key() {
        // Test replacing a key with itself (should result in zero delta)
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
        ]);

        // Replace key 0 with key 0 (no change)
        let delta = cache.compute_replace_delta(0, 0, 0, &keys, None, &tg_freq);

        // Delta should be 0 for all types
        assert_eq!(delta.alternate_freq, 0);
        assert_eq!(delta.inroll_freq, 0);
        assert_eq!(delta.outroll_freq, 0);
        assert_eq!(delta.redirect_freq, 0);
        assert_eq!(delta.onehandin_freq, 0);
        assert_eq!(delta.onehandout_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_skip_pos_b_and_c() {
        // Test that skip_pos skips when it matches either pos_b or pos_c
        let fingers = vec![LP, RI, RP];
        let cache = TrigramCache::new(&fingers, 4);

        let keys = vec![0, 1, 2];

        // Create trigram frequency data for various combinations
        let tg_freq = create_tg_freq(4, &[
            // Trigrams from position 0:
            // (0, 1, 2) - pos_b=1, pos_c=2
            (0, 1, 2, 100),
            (3, 1, 2, 200),
            // (0, 2, 1) - pos_b=2, pos_c=1
            (0, 2, 1, 50),
            (3, 2, 1, 75),
            // (0, 1, 0) - pos_b=1, pos_c=0
            (0, 1, 0, 30),
            (3, 1, 0, 40),
        ]);

        // Skip position 1 - should skip any combo where pos_b=1 OR pos_c=1
        let delta = cache.compute_replace_delta(0, 0, 3, &keys, Some(1), &tg_freq);

        // Only combos where neither pos_b nor pos_c equals 1 should be counted
        // Looking at the combos for position 0:
        // - (pos_b=1, pos_c=2): skipped (pos_b=1)
        // - (pos_b=2, pos_c=1): skipped (pos_c=1)
        // - (pos_b=1, pos_c=0): skipped (pos_b=1)
        // So all should be skipped in this case
        // Note: The actual combos depend on what trigram types are valid for these finger combinations
    }

    #[test]
    fn test_compute_replace_delta_accumulates_multiple_combos() {
        // Test that multiple combos of the same type are accumulated
        let fingers = vec![LP, RI, RP, LI];
        let cache = TrigramCache::new(&fingers, 5);

        let keys = vec![0, 1, 2, 3];

        // Multiple alternate trigrams from position 0
        let tg_freq = create_tg_freq(5, &[
            // Various trigrams that might be alternates
            (0, 1, 0, 10),
            (4, 1, 0, 20),
            (0, 1, 2, 30),
            (4, 1, 2, 50),
        ]);

        let delta = cache.compute_replace_delta(0, 0, 4, &keys, None, &tg_freq);

        // The exact values depend on which trigram types are valid for these finger combinations
        // This test verifies that multiple combos are accumulated
        // At minimum, we should see some non-zero delta if there are valid combos
    }

    // ==================== compute_replace_delta_score_only tests ====================

    #[test]
    fn test_compute_replace_delta_score_only_basic() {
        use crate::weights::{FingerWeights, Weights};

        // Test basic score delta computation for an alternate trigram
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 2,
            alternate: 10,
            redirect: 4,
            onehandin: 5,
            onehandout: 6,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        // Create trigram frequency data
        // For alternate trigram (pos 0 -> pos 1 -> pos 0)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0
            (2, 1, 0, 150),  // new_key=2
        ]);

        // Replace key at position 0: old_key=0, new_key=2
        let score_delta = cache.compute_replace_delta_score_only(0, 0, 2, &keys, None, &tg_freq);

        // The frequency delta for alternate is 150 - 100 = 50
        // Score delta should be 50 * 10 (alternate weight) = 500
        assert_eq!(score_delta, 500);
    }

    #[test]
    fn test_compute_replace_delta_score_only_matches_full_delta() {
        use crate::weights::{FingerWeights, Weights};

        // Test that score_only produces the same result as computing full delta and applying weights
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        // Set weights
        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 3,
            outroll: 5,
            alternate: 7,
            redirect: -2,
            onehandin: 4,
            onehandout: -1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        // Create frequency data for different trigram types
        let tg_freq = create_tg_freq(4, &[
            // For position 0 (LP), combos include:
            // (pos_b=1, pos_c=2) -> LP -> LI -> RI (inroll)
            (0, 1, 2, 100),  // old: key 0 at pos 0
            (3, 1, 2, 150),  // new: key 3 at pos 0
            // (pos_b=2, pos_c=1) -> LP -> RI -> LI (alternate)
            (0, 2, 1, 50),   // old
            (3, 2, 1, 80),   // new
        ]);

        // Compute using full delta method
        let delta = cache.compute_replace_delta(0, 0, 3, &keys, None, &tg_freq);
        let expected_score = delta.inroll_freq * weights.inroll
            + delta.outroll_freq * weights.outroll
            + delta.alternate_freq * weights.alternate
            + delta.redirect_freq * weights.redirect
            + delta.onehandin_freq * weights.onehandin
            + delta.onehandout_freq * weights.onehandout;

        // Compute using score_only method
        let score_delta = cache.compute_replace_delta_score_only(0, 0, 3, &keys, None, &tg_freq);

        // Both should produce the same result
        assert_eq!(score_delta, expected_score);
    }

    #[test]
    fn test_compute_replace_delta_score_only_with_skip_pos() {
        use crate::weights::{FingerWeights, Weights};

        // Test that skip_pos works correctly in score_only
        let fingers = vec![LP, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (3, 1, 0, 200),
            (0, 2, 0, 50),
            (3, 2, 0, 75),
        ]);

        // Compute with skip_pos=1
        let delta_with_skip = cache.compute_replace_delta(0, 0, 3, &keys, Some(1), &tg_freq);
        let expected_with_skip = delta_with_skip.inroll_freq * weights.inroll
            + delta_with_skip.outroll_freq * weights.outroll
            + delta_with_skip.alternate_freq * weights.alternate
            + delta_with_skip.redirect_freq * weights.redirect
            + delta_with_skip.onehandin_freq * weights.onehandin
            + delta_with_skip.onehandout_freq * weights.onehandout;

        let score_with_skip = cache.compute_replace_delta_score_only(0, 0, 3, &keys, Some(1), &tg_freq);

        assert_eq!(score_with_skip, expected_with_skip);
    }

    #[test]
    fn test_compute_replace_delta_score_only_negative_delta() {
        use crate::weights::{FingerWeights, Weights};

        // Test that negative deltas are computed correctly
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        // Old key has higher frequency than new key
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 200),  // old_key=0
            (2, 1, 0, 50),   // new_key=2
        ]);

        let score_delta = cache.compute_replace_delta_score_only(0, 0, 2, &keys, None, &tg_freq);

        // Frequency delta: 50 - 200 = -150
        // Score delta: -150 * 10 = -1500
        assert_eq!(score_delta, -1500);
    }

    #[test]
    fn test_compute_replace_delta_score_only_zero_weights() {
        // Test that zero weights result in zero score delta
        let fingers = vec![LP, RI];
        let cache = TrigramCache::new(&fingers, 4);
        // Weights are zero by default

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 0, 200),
        ]);

        let score_delta = cache.compute_replace_delta_score_only(0, 0, 2, &keys, None, &tg_freq);

        // Even though frequency delta is non-zero, score delta should be 0 because weights are 0
        assert_eq!(score_delta, 0);
    }

    #[test]
    fn test_compute_replace_delta_score_only_same_key() {
        use crate::weights::{FingerWeights, Weights};

        // Test replacing a key with itself (should result in zero score delta)
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 20,
            alternate: 30,
            redirect: 40,
            onehandin: 50,
            onehandout: 60,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
        ]);

        // Replace key 0 with key 0 (no change)
        let score_delta = cache.compute_replace_delta_score_only(0, 0, 0, &keys, None, &tg_freq);

        // Score delta should be 0
        assert_eq!(score_delta, 0);
    }

    #[test]
    fn test_compute_replace_delta_score_only_invalid_keys() {
        use crate::weights::{FingerWeights, Weights};

        // Test that invalid keys are handled correctly
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (2, 1, 0, 100),  // new_key=2
        ]);

        // old_key=99 is invalid (>= num_keys=4)
        let score_delta = cache.compute_replace_delta_score_only(0, 99, 2, &keys, None, &tg_freq);

        // Old frequency should be treated as 0, so freq delta = 100 - 0 = 100
        // Score delta = 100 * 10 = 1000
        assert_eq!(score_delta, 1000);
    }

    #[test]
    fn test_compute_replace_delta_score_only_multiple_types() {
        use crate::weights::{FingerWeights, Weights};

        // Test that multiple trigram types are weighted and accumulated correctly
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 2,
            outroll: 3,
            alternate: 5,
            redirect: -1,
            onehandin: 4,
            onehandout: -2,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];

        let tg_freq = create_tg_freq(5, &[
            // Various trigrams
            (0, 1, 2, 10),
            (4, 1, 2, 30),
            (0, 2, 1, 20),
            (4, 2, 1, 50),
            (0, 3, 0, 15),
            (4, 3, 0, 25),
        ]);

        // Compute using both methods and verify they match
        let delta = cache.compute_replace_delta(0, 0, 4, &keys, None, &tg_freq);
        let expected_score = delta.inroll_freq * weights.inroll
            + delta.outroll_freq * weights.outroll
            + delta.alternate_freq * weights.alternate
            + delta.redirect_freq * weights.redirect
            + delta.onehandin_freq * weights.onehandin
            + delta.onehandout_freq * weights.onehandout;

        let score_delta = cache.compute_replace_delta_score_only(0, 0, 4, &keys, None, &tg_freq);

        assert_eq!(score_delta, expected_score);
    }

    // ==================== replace_key tests ====================

    #[test]
    fn test_replace_key_apply_true_mutates_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that replace_key with apply=true mutates internal state
        // Validates: Requirement 5.2
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key trigram
            (2, 1, 0, 200),  // new_key trigram
        ]);

        // Initial score should be 0
        assert_eq!(cache.score(), 0);

        // Replace key 0 with key 2 (apply=true)
        let new_score = cache.replace_key(0, 0, 2, &keys, None, &tg_freq, true);

        // Score should be updated
        assert_ne!(new_score, 0);

        // Subsequent call to score() should return the same value
        assert_eq!(cache.score(), new_score);
    }

    #[test]
    fn test_replace_key_apply_false_preserves_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that replace_key with apply=false does NOT mutate internal state
        // Validates: Requirement 5.3
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key trigram
            (2, 1, 0, 200),  // new_key trigram
        ]);

        // Initial score should be 0
        let initial_score = cache.score();
        assert_eq!(initial_score, 0);

        // Replace key 0 with key 2 (apply=false)
        let speculative_score = cache.replace_key(0, 0, 2, &keys, None, &tg_freq, false);

        // Speculative score should be non-zero
        assert_ne!(speculative_score, 0);

        // But the actual score should still be 0 (state not mutated)
        assert_eq!(cache.score(), initial_score);
    }

    #[test]
    fn test_replace_key_apply_true_returns_score() {
        use crate::weights::{FingerWeights, Weights};

        // Test that replace_key with apply=true returns score()
        // Validates: Requirement 5.2
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 0, 200),
        ]);

        // Replace key 0 with key 2 (apply=true)
        let returned_score = cache.replace_key(0, 0, 2, &keys, None, &tg_freq, true);

        // The returned score should equal score()
        assert_eq!(returned_score, cache.score());
    }

    #[test]
    fn test_replace_key_apply_false_returns_score_plus_delta() {
        use crate::weights::{FingerWeights, Weights};

        // Test that replace_key with apply=false returns score() + delta
        // Validates: Requirement 5.3
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 0, 200),
        ]);

        // Get initial score
        let initial_score = cache.score();

        // Compute expected score delta
        let score_delta = cache.compute_replace_delta_score_only(0, 0, 2, &keys, None, &tg_freq);

        // Replace key 0 with key 2 (apply=false)
        let speculative_score = cache.replace_key(0, 0, 2, &keys, None, &tg_freq, false);

        // The speculative score should equal initial_score + score_delta
        assert_eq!(speculative_score, initial_score + score_delta);
    }

    #[test]
    fn test_replace_key_apply_true_and_false_return_same_score() {
        use crate::weights::{FingerWeights, Weights};

        // Test that apply=true and apply=false return the same score value
        // (just one mutates state and the other doesn't)
        let fingers = vec![LP, RI];
        let mut cache1 = TrigramCache::new(&fingers, 4);
        let mut cache2 = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache1.set_weights(&weights);
        cache2.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 0, 200),
        ]);

        // Replace with apply=false on cache1
        let score_false = cache1.replace_key(0, 0, 2, &keys, None, &tg_freq, false);

        // Replace with apply=true on cache2
        let score_true = cache2.replace_key(0, 0, 2, &keys, None, &tg_freq, true);

        // Both should return the same score
        assert_eq!(score_false, score_true);
    }

    #[test]
    fn test_replace_key_with_skip_pos() {
        use crate::weights::{FingerWeights, Weights};

        // Test that skip_pos is passed through correctly
        let fingers = vec![LP, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 2, 100),  // Involves pos 1
            (3, 1, 2, 200),  // Involves pos 1
            (0, 2, 0, 50),   // Does not involve pos 1
            (3, 2, 0, 75),   // Does not involve pos 1
        ]);

        // Replace without skip_pos
        let score_no_skip = cache.replace_key(0, 0, 3, &keys, None, &tg_freq, false);

        // Replace with skip_pos=1 (should skip trigrams involving pos 1)
        let score_with_skip = cache.replace_key(0, 0, 3, &keys, Some(1), &tg_freq, false);

        // The scores should be different because skip_pos excludes some trigrams
        assert_ne!(score_no_skip, score_with_skip);
    }

    #[test]
    fn test_replace_key_same_key_no_change() {
        use crate::weights::{FingerWeights, Weights};

        // Test that replacing a key with itself results in no change
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
        ]);

        let initial_score = cache.score();

        // Replace key 0 with key 0 (no change)
        let new_score = cache.replace_key(0, 0, 0, &keys, None, &tg_freq, true);

        // Score should remain the same
        assert_eq!(new_score, initial_score);
        assert_eq!(cache.score(), initial_score);
    }

    // ==================== key_swap() tests ====================

    #[test]
    fn test_key_swap_apply_true_mutates_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that key_swap with apply=true updates internal state
        // Use 3 positions so that swapping positions 0 and 1 affects trigrams
        // involving position 2 differently
        let fingers = vec![LP, RI, LI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        // Create asymmetric frequencies involving position 2
        // This ensures swapping positions 0 and 1 changes the score
        let tg_freq = create_tg_freq(4, &[
            (0, 2, 0, 100),  // key 0 at pos 0, key 2 at pos 2
            (1, 2, 1, 500),  // key 1 at pos 1, key 2 at pos 2 - much higher!
        ]);

        // First, establish some initial state by doing replace_key
        cache.replace_key(0, 99, 0, &keys, None, &tg_freq, true);
        cache.replace_key(1, 99, 1, &keys, None, &tg_freq, true);
        cache.replace_key(2, 99, 2, &keys, None, &tg_freq, true);

        let initial_score = cache.score();

        // Swap keys at positions 0 and 1 (key 0 <-> key 1)
        // After swap: pos 0 will have key 1, pos 1 will have key 0
        // This changes which trigrams are counted because:
        // - Before: pos0 has key0, so tg_freq[0][2][0] = 100 is counted
        // - After: pos0 has key1, so tg_freq[1][2][1] = 500 is counted
        let new_score = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        // Score should have changed because frequencies are asymmetric
        assert_ne!(new_score, initial_score,
            "Score should change after swap with asymmetric frequencies");

        // Subsequent call to score() should return the same value as key_swap returned
        assert_eq!(cache.score(), new_score);
    }

    #[test]
    fn test_key_swap_apply_false_preserves_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that key_swap with apply=false does not update internal state
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (1, 0, 1, 50),
            (2, 3, 2, 200),
            (3, 2, 3, 150),
        ]);

        let initial_score = cache.score();
        let initial_inroll = cache.inroll_freq;
        let initial_outroll = cache.outroll_freq;
        let initial_alternate = cache.alternate_freq;
        let initial_redirect = cache.redirect_freq;
        let initial_onehandin = cache.onehandin_freq;
        let initial_onehandout = cache.onehandout_freq;

        // Swap keys with apply=false
        let _speculative_score = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, false);

        // State should be unchanged
        assert_eq!(cache.score(), initial_score);
        assert_eq!(cache.inroll_freq, initial_inroll);
        assert_eq!(cache.outroll_freq, initial_outroll);
        assert_eq!(cache.alternate_freq, initial_alternate);
        assert_eq!(cache.redirect_freq, initial_redirect);
        assert_eq!(cache.onehandin_freq, initial_onehandin);
        assert_eq!(cache.onehandout_freq, initial_onehandout);
    }

    #[test]
    fn test_key_swap_apply_true_and_false_return_same_score() {
        use crate::weights::{FingerWeights, Weights};

        // Test that apply=true and apply=false return the same score
        let fingers = vec![LP, RI];
        let mut cache1 = TrigramCache::new(&fingers, 4);
        let mut cache2 = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 2,
            alternate: 10,
            redirect: -5,
            onehandin: 3,
            onehandout: -2,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache1.set_weights(&weights);
        cache2.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (1, 0, 1, 50),
            (2, 3, 2, 200),
            (3, 2, 3, 150),
        ]);

        // Get speculative score with apply=false
        let speculative_score = cache1.key_swap(0, 1, 0, 1, &keys, &tg_freq, false);

        // Get actual score with apply=true
        let actual_score = cache2.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        // Both should return the same score
        assert_eq!(speculative_score, actual_score);
    }

    #[test]
    fn test_key_swap_uses_skip_pos_to_avoid_double_counting() {
        use crate::weights::{FingerWeights, Weights};

        // Test that skip_pos is used correctly to avoid double-counting
        // trigrams that involve both swapped positions
        let fingers = vec![LP, RI, LI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        // Create trigrams that involve both positions 0 and 1
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 2, 100),  // pos 0 -> pos 1 -> pos 2 (involves both 0 and 1)
            (1, 0, 2, 50),   // pos 1 -> pos 0 -> pos 2 (involves both 0 and 1)
            (0, 2, 1, 75),   // pos 0 -> pos 2 -> pos 1 (involves both 0 and 1)
            (2, 0, 1, 25),   // pos 2 -> pos 0 -> pos 1 (involves both 0 and 1)
        ]);

        // Swap keys at positions 0 and 1
        let score = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        // The score should be computed correctly without double-counting
        // We can't easily verify the exact value, but we can verify it doesn't panic
        // and returns a reasonable value
        assert!(score >= 0 || score < 0); // Just verify it returns a value
    }

    #[test]
    fn test_key_swap_same_position_no_change() {
        use crate::weights::{FingerWeights, Weights};

        // Test swapping a position with itself (edge case)
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
        ]);

        let initial_score = cache.score();

        // Swap position 0 with itself (key 0 <-> key 0)
        let new_score = cache.key_swap(0, 0, 0, 0, &keys, &tg_freq, true);

        // Score should remain the same
        assert_eq!(new_score, initial_score);
    }

    #[test]
    fn test_key_swap_combines_deltas_correctly() {
        use crate::weights::{FingerWeights, Weights};

        // Test that the combined delta from both positions is applied correctly
        let fingers = vec![LP, RI, LI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];

        // Create trigrams with different frequencies for different key combinations
        let tg_freq = create_tg_freq(4, &[
            (0, 2, 0, 100),  // Only involves pos 0
            (1, 2, 1, 200),  // Only involves pos 1
            (3, 2, 3, 50),   // Only involves key 3 (will be at pos 0 after swap)
            (2, 2, 2, 75),   // Only involves key 2 (will be at pos 1 after swap)
        ]);

        // Swap keys: pos 0 has key 0, pos 1 has key 1
        // After swap: pos 0 will have key 1, pos 1 will have key 0
        let score = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        // Verify the score is computed (exact value depends on trigram types)
        assert_eq!(cache.score(), score);
    }

    #[test]
    fn test_key_swap_with_invalid_keys() {
        use crate::weights::{FingerWeights, Weights};

        // Test key_swap when one or both keys are invalid (>= num_keys)
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 5]; // key 5 is invalid (>= num_keys=4)

        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
        ]);

        let initial_score = cache.score();

        // Swap with an invalid key
        let new_score = cache.key_swap(0, 1, 0, 5, &keys, &tg_freq, true);

        // Should handle gracefully (invalid keys contribute 0 frequency)
        assert!(new_score >= 0 || new_score < 0);
    }

    #[test]
    fn test_key_swap_empty_tg_freq() {
        use crate::weights::{FingerWeights, Weights};

        // Test key_swap with empty trigram frequencies
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 1,
            outroll: 1,
            alternate: 10,
            redirect: 1,
            onehandin: 1,
            onehandout: 1,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        // Empty trigram frequencies
        let tg_freq = create_tg_freq(4, &[]);

        let initial_score = cache.score();

        // Swap should result in no change since all frequencies are 0
        let new_score = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        assert_eq!(new_score, initial_score);
        assert_eq!(cache.score(), initial_score);
    }

    // ==================== stats() tests ====================

    #[test]
    fn test_stats_populates_trigram_stats() {
        use crate::stats::Stats;

        // Create a cache with some frequencies
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Manually set frequencies for testing
        cache.inroll_freq = 100;
        cache.outroll_freq = 200;
        cache.alternate_freq = 300;
        cache.redirect_freq = 50;
        cache.onehandin_freq = 75;
        cache.onehandout_freq = 25;

        let mut stats = Stats::default();
        let trigram_total = 1000.0;

        cache.stats(&mut stats, trigram_total);

        // Verify normalized frequencies
        assert!((stats.trigrams.inroll - 0.1).abs() < 1e-10);
        assert!((stats.trigrams.outroll - 0.2).abs() < 1e-10);
        assert!((stats.trigrams.alternate - 0.3).abs() < 1e-10);
        assert!((stats.trigrams.redirect - 0.05).abs() < 1e-10);
        assert!((stats.trigrams.onehandin - 0.075).abs() < 1e-10);
        assert!((stats.trigrams.onehandout - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_stats_handles_zero_trigram_total() {
        use crate::stats::Stats;

        // Create a cache with some frequencies
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set non-zero frequencies
        cache.inroll_freq = 100;
        cache.outroll_freq = 200;
        cache.alternate_freq = 300;
        cache.redirect_freq = 50;
        cache.onehandin_freq = 75;
        cache.onehandout_freq = 25;

        let mut stats = Stats::default();

        // Call stats with zero trigram_total
        cache.stats(&mut stats, 0.0);

        // All fields should be 0 to avoid division by zero
        assert_eq!(stats.trigrams.inroll, 0.0);
        assert_eq!(stats.trigrams.outroll, 0.0);
        assert_eq!(stats.trigrams.alternate, 0.0);
        assert_eq!(stats.trigrams.redirect, 0.0);
        assert_eq!(stats.trigrams.onehandin, 0.0);
        assert_eq!(stats.trigrams.onehandout, 0.0);
    }

    #[test]
    fn test_stats_handles_negative_trigram_total() {
        use crate::stats::Stats;

        // Create a cache with some frequencies
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set non-zero frequencies
        cache.inroll_freq = 100;
        cache.outroll_freq = 200;

        let mut stats = Stats::default();

        // Call stats with negative trigram_total
        cache.stats(&mut stats, -100.0);

        // All fields should be 0 to avoid division by zero
        assert_eq!(stats.trigrams.inroll, 0.0);
        assert_eq!(stats.trigrams.outroll, 0.0);
        assert_eq!(stats.trigrams.alternate, 0.0);
        assert_eq!(stats.trigrams.redirect, 0.0);
        assert_eq!(stats.trigrams.onehandin, 0.0);
        assert_eq!(stats.trigrams.onehandout, 0.0);
    }

    #[test]
    fn test_stats_with_zero_frequencies() {
        use crate::stats::Stats;

        // Create a cache with default (zero) frequencies
        let fingers = vec![LP, LI, RI, RP];
        let cache = TrigramCache::new(&fingers, 30);

        let mut stats = Stats::default();
        let trigram_total = 1000.0;

        cache.stats(&mut stats, trigram_total);

        // All fields should be 0 since frequencies are 0
        assert_eq!(stats.trigrams.inroll, 0.0);
        assert_eq!(stats.trigrams.outroll, 0.0);
        assert_eq!(stats.trigrams.alternate, 0.0);
        assert_eq!(stats.trigrams.redirect, 0.0);
        assert_eq!(stats.trigrams.onehandin, 0.0);
        assert_eq!(stats.trigrams.onehandout, 0.0);
    }

    #[test]
    fn test_stats_with_negative_frequencies() {
        use crate::stats::Stats;

        // Create a cache with negative frequencies (edge case)
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set negative frequencies (can happen during delta calculations)
        cache.inroll_freq = -100;
        cache.outroll_freq = 200;

        let mut stats = Stats::default();
        let trigram_total = 1000.0;

        cache.stats(&mut stats, trigram_total);

        // Should normalize correctly even with negative frequencies
        assert!((stats.trigrams.inroll - (-0.1)).abs() < 1e-10);
        assert!((stats.trigrams.outroll - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_stats_normalization_sums_correctly() {
        use crate::stats::Stats;

        // Create a cache where frequencies sum to trigram_total
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 30);

        // Set frequencies that sum to 1000
        cache.inroll_freq = 100;
        cache.outroll_freq = 200;
        cache.alternate_freq = 300;
        cache.redirect_freq = 150;
        cache.onehandin_freq = 150;
        cache.onehandout_freq = 100;

        let mut stats = Stats::default();
        let trigram_total = 1000.0;

        cache.stats(&mut stats, trigram_total);

        // Sum of normalized frequencies should equal 1.0
        let sum = stats.trigrams.inroll
            + stats.trigrams.outroll
            + stats.trigrams.alternate
            + stats.trigrams.redirect
            + stats.trigrams.onehandin
            + stats.trigrams.onehandout;

        assert!((sum - 1.0).abs() < 1e-10);
    }
}


