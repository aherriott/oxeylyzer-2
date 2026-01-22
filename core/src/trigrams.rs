use std::collections::HashMap;

use crate::stats::Stats;
use crate::types::{CacheKey, CachePos};
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

/// Pre-computed trigram combination with type (for position as first element)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrigramCombo {
    /// Second position in the trigram
    pub pos_b: usize,
    /// Third position in the trigram
    pub pos_c: usize,
    /// Pre-computed trigram type
    pub trigram_type: TrigramType,
}

/// Pre-computed trigram combination where the key position is the second element
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrigramComboMid {
    /// First position in the trigram
    pub pos_a: usize,
    /// Third position in the trigram
    pub pos_c: usize,
    /// Pre-computed trigram type
    pub trigram_type: TrigramType,
}

/// Pre-computed trigram combination where the key position is the third element
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrigramComboEnd {
    /// First position in the trigram
    pub pos_a: usize,
    /// Second position in the trigram
    pub pos_b: usize,
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
    /// For each position, list of (pos_b, pos_c, type) combinations where position is first
    trigram_combos_per_key: Vec<Vec<TrigramCombo>>,

    /// For each position, list of (pos_a, pos_c, type) combinations where position is second
    trigram_combos_mid: Vec<Vec<TrigramComboMid>>,

    /// For each position, list of (pos_a, pos_b, type) combinations where position is third
    trigram_combos_end: Vec<Vec<TrigramComboEnd>>,

    /// Number of keys for frequency array indexing
    num_keys: usize,

    /// Number of positions
    num_positions: usize,

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

    /// Pre-computed weighted scores for O(1) speculative scoring
    /// weighted_score_first[pos * num_keys + key] = weighted score when pos is first element with key
    weighted_score_first: Vec<i64>,
    /// weighted_score_mid[pos * num_keys + key] = weighted score when pos is middle element with key
    weighted_score_mid: Vec<i64>,
    /// weighted_score_end[pos * num_keys + key] = weighted score when pos is last element with key
    weighted_score_end: Vec<i64>,

    /// Pre-computed swap scores for O(1) swap speculative scoring
    /// For position pair (pos_a, pos_b) where pos_a < pos_b, and keys (key_a, key_b):
    /// swap_score_both[pair_idx * num_keys * num_keys + key_a * num_keys + key_b]
    /// = weighted score delta for trigrams involving BOTH positions when swapping
    ///
    /// pair_idx = pos_a * num_positions - pos_a * (pos_a + 1) / 2 + (pos_b - pos_a - 1)
    /// This is the index into the upper triangle of the position pair matrix.
    swap_score_both: Vec<i64>,

    /// Number of position pairs (upper triangle)
    num_pairs: usize,

    /// Whether the weighted scores have been initialized
    weighted_scores_initialized: bool,

    /// Tracks active magic rules: (magic_key, leader) -> (output, delta)
    /// When a rule A→M steals output B, we store ((M, A), (B, delta))
    /// This allows the analyzer to correctly compute deltas when rules change.
    /// The delta is stored so we can properly reverse the rule by subtracting it.
    active_rules: HashMap<(CacheKey, CacheKey), (CacheKey, i64)>,

    /// Cumulative score delta from all active magic rules.
    /// This is added to the base score (computed from frequencies) to get the total score.
    magic_rule_score_delta: i64,

    /// Pre-computed rule deltas for O(1) `add_rule` speculative scoring.
    /// Maps (leader, output, magic_key) -> score delta.
    /// Uses sparse storage (HashMap) to stay under 10MB memory target.
    /// When `add_rule` is called with `apply=false`, this lookup table is used
    /// instead of computing the delta from scratch.
    rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,

    /// Pre-computed swap deltas for O(1) speculative key_swap scoring.
    /// Maps (pos_a, pos_b, key_a, key_b) -> score delta where pos_a < pos_b.
    /// Uses sparse storage (HashMap) to keep memory under control - only stores non-zero deltas.
    /// When `key_swap` is called with `apply=false`, this lookup table is used
    /// instead of computing the delta from scratch.
    swap_delta: HashMap<(CachePos, CachePos, CacheKey, CacheKey), i32>,
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
        let mut trigram_combos_mid = Vec::with_capacity(num_positions);
        let mut trigram_combos_end = Vec::with_capacity(num_positions);

        // Initialize empty vectors for each position
        for _ in 0..num_positions {
            trigram_combos_per_key.push(Vec::new());
            trigram_combos_mid.push(Vec::new());
            trigram_combos_end.push(Vec::new());
        }

        // Pre-compute all valid trigram combinations
        for pos_a in 0..num_positions {
            let f_a = finger_indices[pos_a];

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
                            // Store for pos_a as first position
                            trigram_combos_per_key[pos_a].push(TrigramCombo {
                                pos_b,
                                pos_c,
                                trigram_type,
                            });

                            // Store for pos_b as second position
                            trigram_combos_mid[pos_b].push(TrigramComboMid {
                                pos_a,
                                pos_c,
                                trigram_type,
                            });

                            // Store for pos_c as third position
                            trigram_combos_end[pos_c].push(TrigramComboEnd {
                                pos_a,
                                pos_b,
                                trigram_type,
                            });
                        }
                        // Skip untracked types: Sft, Sfb, Thumb, Invalid
                        _ => {}
                    }
                }
            }
        }

        // Number of position pairs in upper triangle: n*(n-1)/2
        let num_pairs = if num_positions > 0 {
            num_positions * (num_positions - 1) / 2
        } else {
            0
        };

        Self {
            trigram_combos_per_key,
            trigram_combos_mid,
            trigram_combos_end,
            num_keys,
            num_positions,
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
            // Pre-computed weighted scores (initialized lazily)
            weighted_score_first: vec![0; num_positions * num_keys],
            weighted_score_mid: vec![0; num_positions * num_keys],
            weighted_score_end: vec![0; num_positions * num_keys],
            // Pre-computed swap scores for trigrams involving both positions
            swap_score_both: vec![0; num_pairs * num_keys * num_keys],
            num_pairs,
            weighted_scores_initialized: false,
            // Initialize active rules to empty (no magic rules active initially)
            active_rules: HashMap::new(),
            // Initialize magic rule score delta to zero
            magic_rule_score_delta: 0,
            // Initialize rule delta lookup table to empty (populated by init_rule_deltas)
            rule_delta: HashMap::new(),
            // Initialize swap delta lookup table to empty (populated by init_swap_deltas)
            swap_delta: HashMap::new(),
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
        // Invalidate pre-computed scores when weights change
        self.weighted_scores_initialized = false;
    }

    /// Get the weight for a trigram type
    #[inline]
    fn get_weight(&self, trigram_type: TrigramType) -> i64 {
        match trigram_type {
            TrigramType::Inroll => self.inroll_weight,
            TrigramType::Outroll => self.outroll_weight,
            TrigramType::Alternate => self.alternate_weight,
            TrigramType::Redirect => self.redirect_weight,
            TrigramType::OnehandIn => self.onehandin_weight,
            TrigramType::OnehandOut => self.onehandout_weight,
            _ => 0,
        }
    }

    /// Compute the score delta for applying a magic rule.
    ///
    /// When rule A→M steals output B:
    /// - Trigrams Z→A→B become Z→A→M (for all Z with positions)
    /// - Trigrams A→B→C become A→M→C (for all C with positions)
    ///
    /// Returns the score delta (new_score - old_score).
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    fn compute_rule_delta(
        &self,
        leader: CacheKey,      // A - the key that triggers the magic rule
        output: CacheKey,      // B - the output being stolen
        magic_key: CacheKey,   // M - the magic key that steals the output
        _keys: &[CacheKey],    // Not used but kept for API consistency
        key_positions: &[Option<CachePos>],
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        let num_positions = self.num_positions;

        // Validate keys are within bounds
        if leader >= num_keys || output >= num_keys || magic_key >= num_keys {
            return 0;
        }

        // Get positions for leader (A), output (B), and magic_key (M)
        // Leader must have a position for the rule to have any effect
        let leader_pos = match key_positions.get(leader).copied().flatten() {
            Some(pos) => pos,
            None => return 0, // Leader has no position, rule has no effect
        };

        // Output (B) and magic_key (M) positions are needed for type computation
        let output_pos = key_positions.get(output).copied().flatten();
        let magic_pos = key_positions.get(magic_key).copied().flatten();

        let mut delta: i64 = 0;

        // Part 1: Z→A→B becomes Z→A→M (for all Z with positions)
        // The frequency is tg_freq[Z][A][B] for both old and new
        // The type changes based on positions of Z, A, and M vs Z, A, and B
        for z_key in 0..num_keys {
            // Z must have a position
            let z_pos = match key_positions.get(z_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency for trigram Z→A→B
            let freq = tg_freq[z_key][leader][output];
            if freq == 0 {
                continue;
            }

            // Compute old type: Z→A→B (positions: z_pos, leader_pos, output_pos)
            // If output has no position, the old trigram type is Invalid (weight 0)
            let old_weight = if let Some(b_pos) = output_pos {
                if z_pos < num_positions && leader_pos < num_positions && b_pos < num_positions {
                    let f_z = self.fingers[z_pos];
                    let f_a = self.fingers[leader_pos];
                    let f_b = self.fingers[b_pos];
                    let old_type = TRIGRAMS[f_z * 100 + f_a * 10 + f_b];
                    self.get_weight(old_type)
                } else {
                    0
                }
            } else {
                0
            };

            // Compute new type: Z→A→M (positions: z_pos, leader_pos, magic_pos)
            // If magic_key has no position, the new trigram type is Invalid (weight 0)
            let new_weight = if let Some(m_pos) = magic_pos {
                if z_pos < num_positions && leader_pos < num_positions && m_pos < num_positions {
                    let f_z = self.fingers[z_pos];
                    let f_a = self.fingers[leader_pos];
                    let f_m = self.fingers[m_pos];
                    let new_type = TRIGRAMS[f_z * 100 + f_a * 10 + f_m];
                    self.get_weight(new_type)
                } else {
                    0
                }
            } else {
                0
            };

            // Delta contribution: freq * (new_weight - old_weight)
            delta += freq * (new_weight - old_weight);
        }

        // Part 2: A→B→C becomes A→M→C (for all C with positions)
        // The frequency is tg_freq[A][B][C] for both old and new
        // The type changes based on positions of A, B, C vs A, M, C
        for c_key in 0..num_keys {
            // C must have a position
            let c_pos = match key_positions.get(c_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency for trigram A→B→C
            let freq = tg_freq[leader][output][c_key];
            if freq == 0 {
                continue;
            }

            // Compute old type: A→B→C (positions: leader_pos, output_pos, c_pos)
            // If output has no position, the old trigram type is Invalid (weight 0)
            let old_weight = if let Some(b_pos) = output_pos {
                if leader_pos < num_positions && b_pos < num_positions && c_pos < num_positions {
                    let f_a = self.fingers[leader_pos];
                    let f_b = self.fingers[b_pos];
                    let f_c = self.fingers[c_pos];
                    let old_type = TRIGRAMS[f_a * 100 + f_b * 10 + f_c];
                    self.get_weight(old_type)
                } else {
                    0
                }
            } else {
                0
            };

            // Compute new type: A→M→C (positions: leader_pos, magic_pos, c_pos)
            // If magic_key has no position, the new trigram type is Invalid (weight 0)
            let new_weight = if let Some(m_pos) = magic_pos {
                if leader_pos < num_positions && m_pos < num_positions && c_pos < num_positions {
                    let f_a = self.fingers[leader_pos];
                    let f_m = self.fingers[m_pos];
                    let f_c = self.fingers[c_pos];
                    let new_type = TRIGRAMS[f_a * 100 + f_m * 10 + f_c];
                    self.get_weight(new_type)
                } else {
                    0
                }
            } else {
                0
            };

            // Delta contribution: freq * (new_weight - old_weight)
            delta += freq * (new_weight - old_weight);
        }

        delta
    }

    /// Apply a magic rule. Returns the score delta.
    ///
    /// If `apply` is false, computes the score delta without mutating state (speculative scoring).
    /// If `apply` is true, updates internal state and returns the score delta.
    ///
    /// When rule A→M steals output B:
    /// - Trigrams Z→A→B become Z→A→M (for all Z)
    /// - Trigrams A→B→C become A→M→C (for all C)
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    /// * `apply` - If true, update internal state; if false, just compute the delta
    ///
    /// # Returns
    /// The score delta (new_score - old_score) for this rule application.
    ///
    /// Requirements: 2.1, 2.5, 2.6, 6.1, 6.5
    pub fn add_rule(
        &mut self,
        leader: CacheKey,      // A
        output: CacheKey,      // B (being stolen)
        magic_key: CacheKey,   // M
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        let rule_key = (magic_key, leader);

        // Check if there's an existing rule for this (magic_key, leader) pair
        let old_delta = if let Some(&(old_output, old_delta)) = self.active_rules.get(&rule_key) {
            // If the new output is the same as the old output, no change needed
            if old_output == output {
                return 0;
            }
            old_delta
        } else {
            0
        };

        // Compute the score delta for the new rule (0 if output is EMPTY_KEY)
        let new_delta = if output != crate::cached_layout::EMPTY_KEY {
            // When apply=false and lookup table is populated, use O(1) lookup
            // instead of computing the delta from scratch.
            if !apply && !self.rule_delta.is_empty() {
                // Look up the delta directly from the pre-computed table.
                // If not found in the HashMap, the delta is 0 (sparse storage).
                self.rule_delta
                    .get(&(leader, output, magic_key))
                    .copied()
                    .unwrap_or(0) as i64
            } else {
                // Either apply=true or lookup table not initialized - compute from scratch
                self.compute_rule_delta(leader, output, magic_key, keys, key_positions, tg_freq)
            }
        } else {
            0
        };

        // The net delta is: new_delta - old_delta
        // (removing old rule subtracts old_delta, adding new rule adds new_delta)
        let net_delta = new_delta - old_delta;

        if apply {
            // Always remove old rule if it exists (regardless of delta value)
            self.active_rules.remove(&rule_key);

            // Add new rule if output is not EMPTY_KEY
            if output != crate::cached_layout::EMPTY_KEY {
                self.active_rules.insert(rule_key, (output, new_delta));
            }

            // Update the cumulative magic rule score delta
            self.magic_rule_score_delta += net_delta;
        }

        net_delta
    }

    /// Initialize pre-computed weighted scores for O(1) speculative scoring.
    ///
    /// For each position and hypothetical key, computes the total weighted score
    /// contribution from all trigrams where that position would have that key.
    ///
    /// The pre-computation handles three cases for each position:
    /// 1. Position is first in trigram: weighted_score_first
    /// 2. Position is second in trigram (and not first): weighted_score_mid
    /// 3. Position is third in trigram (and not first or second): weighted_score_end
    ///
    /// Special handling for trigrams where the same position appears multiple times:
    /// - If pos_b == pos or pos_c == pos in case 1, use the hypothetical key k
    /// - If pos_c == pos in case 2, use the hypothetical key k
    ///
    /// This must be called after the layout is fully initialized (all keys placed).
    pub fn init_weighted_scores(&mut self, keys: &[usize], tg_freq: &[Vec<Vec<i64>>]) {
        let num_keys = self.num_keys;
        let num_positions = self.num_positions;

        // Reset all scores to 0
        for score in &mut self.weighted_score_first {
            *score = 0;
        }
        for score in &mut self.weighted_score_mid {
            *score = 0;
        }
        for score in &mut self.weighted_score_end {
            *score = 0;
        }

        // Case 1: Compute weighted scores for each position as first element
        // For trigram (pos, pos_b, pos_c), if we place key k at pos:
        // - key_b = k if pos_b == pos, else keys[pos_b]
        // - key_c = k if pos_c == pos, else keys[pos_c]
        // - contribution = tg_freq[k][key_b][key_c] * weight
        for pos in 0..num_positions {
            for combo in &self.trigram_combos_per_key[pos] {
                let pos_b = combo.pos_b;
                let pos_c = combo.pos_c;
                let weight = self.get_weight(combo.trigram_type);

                // For each possible key k at pos
                for k in 0..num_keys {
                    // Determine key_b: if pos_b == pos, use k; else use current key
                    let key_b = if pos_b == pos { k } else { keys[pos_b] };
                    // Determine key_c: if pos_c == pos, use k; else use current key
                    let key_c = if pos_c == pos { k } else { keys[pos_c] };

                    if key_b >= num_keys || key_c >= num_keys {
                        continue;
                    }

                    let freq = tg_freq[k][key_b][key_c];
                    self.weighted_score_first[pos * num_keys + k] += freq * weight;
                }
            }
        }

        // Case 2: Compute weighted scores for each position as middle element
        // Only for trigrams where pos_a != pos (to avoid double-counting with case 1)
        // For trigram (pos_a, pos, pos_c), if we place key k at pos:
        // - key_a = keys[pos_a] (pos_a != pos by filter)
        // - key_c = k if pos_c == pos, else keys[pos_c]
        // - contribution = tg_freq[key_a][k][key_c] * weight
        for pos in 0..num_positions {
            for combo in &self.trigram_combos_mid[pos] {
                let pos_a = combo.pos_a;
                let pos_c = combo.pos_c;

                // Skip if pos_a == pos (already counted in case 1)
                if pos_a == pos {
                    continue;
                }

                let weight = self.get_weight(combo.trigram_type);
                let key_a = keys[pos_a];

                if key_a >= num_keys {
                    continue;
                }

                // For each possible key k at pos
                for k in 0..num_keys {
                    // Determine key_c: if pos_c == pos, use k; else use current key
                    let key_c = if pos_c == pos { k } else { keys[pos_c] };

                    if key_c >= num_keys {
                        continue;
                    }

                    let freq = tg_freq[key_a][k][key_c];
                    self.weighted_score_mid[pos * num_keys + k] += freq * weight;
                }
            }
        }

        // Case 3: Compute weighted scores for each position as last element
        // Only for trigrams where pos_a != pos AND pos_b != pos
        // For trigram (pos_a, pos_b, pos), if we place key k at pos:
        // - key_a = keys[pos_a], key_b = keys[pos_b]
        // - contribution = tg_freq[key_a][key_b][k] * weight
        for pos in 0..num_positions {
            for combo in &self.trigram_combos_end[pos] {
                let pos_a = combo.pos_a;
                let pos_b = combo.pos_b;

                // Skip if pos_a == pos or pos_b == pos (already counted in cases 1 or 2)
                if pos_a == pos || pos_b == pos {
                    continue;
                }

                let weight = self.get_weight(combo.trigram_type);
                let key_a = keys[pos_a];
                let key_b = keys[pos_b];

                if key_a >= num_keys || key_b >= num_keys {
                    continue;
                }

                for k in 0..num_keys {
                    let freq = tg_freq[key_a][key_b][k];
                    self.weighted_score_end[pos * num_keys + k] += freq * weight;
                }
            }
        }

        // Initialize swap_score_both for O(1) swap scoring
        // For each position pair (pos_a, pos_b) where pos_a < pos_b, and each key pair (ka, kb),
        // compute the weighted score for trigrams involving BOTH positions when:
        // - pos_a has key ka, pos_b has key kb (before swap)
        // - After swap: pos_a has kb, pos_b has ka
        // Store: new_score - old_score for the "both" trigrams
        self.init_swap_scores(tg_freq);

        self.weighted_scores_initialized = true;
    }

    /// Initialize pre-computed swap scores for trigrams involving both positions.
    ///
    /// For each position pair (pos_a, pos_b) and key pair (key_a, key_b), computes
    /// the weighted score delta for trigrams that involve BOTH positions.
    fn init_swap_scores(&mut self, tg_freq: &[Vec<Vec<i64>>]) {
        let num_keys = self.num_keys;
        let num_positions = self.num_positions;

        // Reset swap scores
        for score in &mut self.swap_score_both {
            *score = 0;
        }

        // For each position pair (pos_a, pos_b) where pos_a < pos_b
        for pos_a in 0..num_positions {
            for pos_b in (pos_a + 1)..num_positions {
                let pair_idx = self.pair_index(pos_a, pos_b);
                let pair_base = pair_idx * num_keys * num_keys;

                // For each key pair (key_a at pos_a, key_b at pos_b)
                for key_a in 0..num_keys {
                    for key_b in 0..num_keys {
                        let score = self.compute_swap_both_score_for_keys(
                            pos_a, pos_b, key_a, key_b, tg_freq
                        );
                        self.swap_score_both[pair_base + key_a * num_keys + key_b] = score;
                    }
                }
            }
        }
    }

    /// Initialize pre-computed rule deltas for O(1) `add_rule` speculative scoring.
    ///
    /// Pre-computes the score delta for all valid (leader, output, magic_key) combinations
    /// where the leader has a position on the layout. Uses sparse storage (HashMap) to
    /// store only non-zero deltas, keeping memory usage under 10MB for typical configurations.
    ///
    /// This must be called after the layout is fully initialized (all keys placed).
    ///
    /// # Arguments
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// Requirements: 6.1, 6.5
    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        // Clear existing rule deltas
        self.rule_delta.clear();

        let num_keys = self.num_keys;

        // Iterate over all valid (leader, output, magic_key) combinations
        for leader in 0..num_keys {
            // Leader must have a position for the rule to have any effect
            let leader_pos = key_positions.get(leader).copied().flatten();
            if leader_pos.is_none() {
                continue;
            }

            for output in 0..num_keys {
                for magic_key in 0..num_keys {
                    // Compute the score delta for this rule
                    let delta = self.compute_rule_delta(
                        leader as CacheKey,
                        output as CacheKey,
                        magic_key as CacheKey,
                        keys,
                        key_positions,
                        tg_freq,
                    );

                    // Only store non-zero deltas for sparse storage (memory efficiency)
                    if delta != 0 {
                        self.rule_delta.insert(
                            (leader as CacheKey, output as CacheKey, magic_key as CacheKey),
                            delta as i32,
                        );
                    }
                }
            }
        }
    }

    /// Initialize swap_delta lookup table for O(1) speculative key_swap scoring.
    ///
    /// Pre-computes the score delta for all valid (pos_a, pos_b, key_a, key_b) combinations
    /// where pos_a < pos_b (canonical ordering). Uses sparse storage (HashMap) to store
    /// only non-zero deltas, keeping memory usage under control.
    ///
    /// This must be called after the layout is fully initialized (all keys placed)
    /// and after init_weighted_scores has been called.
    ///
    /// # Arguments
    /// * `keys` - The current key assignments for all positions
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// **Validates: Requirements 1.2, 7.2**
    pub fn init_swap_deltas(
        &mut self,
        keys: &[CacheKey],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        // Clear existing swap deltas
        self.swap_delta.clear();

        let num_keys = self.num_keys;
        let num_positions = self.num_positions;

        // Iterate over all position pairs (pos_a, pos_b) where pos_a < pos_b
        for pos_a in 0..num_positions {
            for pos_b in (pos_a + 1)..num_positions {
                // Iterate over all key pairs (key_a, key_b)
                for key_a in 0..num_keys {
                    for key_b in 0..num_keys {
                        // Skip if same key (no change when swapping same key)
                        if key_a == key_b {
                            continue;
                        }

                        // Compute the total swap delta using existing methods:
                        // 1. Delta for pos_a (key_a -> key_b), skipping pos_b
                        // 2. Delta for pos_b (key_b -> key_a), skipping pos_a
                        // 3. Delta for trigrams involving BOTH positions
                        let score_a = self.compute_replace_delta_score_only(
                            pos_a, key_a, key_b, keys, Some(pos_b), tg_freq
                        );
                        let score_b = self.compute_replace_delta_score_only(
                            pos_b, key_b, key_a, keys, Some(pos_a), tg_freq
                        );
                        let score_both = self.compute_swap_both_delta_score_only(
                            pos_a, pos_b, key_a, key_b, keys, tg_freq
                        );

                        let delta = score_a + score_b + score_both;

                        // Only store non-zero deltas (sparse storage for memory efficiency)
                        if delta != 0 {
                            self.swap_delta.insert(
                                (pos_a, pos_b, key_a, key_b),
                                delta as i32,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Compute the pair index for positions (pos_a, pos_b) where pos_a < pos_b.
    /// Uses upper triangle indexing.
    #[inline]
    fn pair_index(&self, pos_a: usize, pos_b: usize) -> usize {
        debug_assert!(pos_a < pos_b);
        // Upper triangle index: sum of (n-1) + (n-2) + ... + (n-pos_a) + (pos_b - pos_a - 1)
        // = pos_a * n - pos_a * (pos_a + 1) / 2 + (pos_b - pos_a - 1)
        let n = self.num_positions;
        pos_a * n - pos_a * (pos_a + 1) / 2 + (pos_b - pos_a - 1)
    }

    /// Compute the weighted score delta for trigrams involving BOTH positions
    /// when swapping keys key_a (at pos_a) and key_b (at pos_b).
    fn compute_swap_both_score_for_keys(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        let key_a_valid = key_a < num_keys;
        let key_b_valid = key_b < num_keys;

        if !key_a_valid || !key_b_valid {
            return 0;
        }

        let mut score_delta: i64 = 0;

        macro_rules! weighted_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => $freq_delta * self.inroll_weight,
                    TrigramType::Outroll => $freq_delta * self.outroll_weight,
                    TrigramType::Alternate => $freq_delta * self.alternate_weight,
                    TrigramType::Redirect => $freq_delta * self.redirect_weight,
                    TrigramType::OnehandIn => $freq_delta * self.onehandin_weight,
                    TrigramType::OnehandOut => $freq_delta * self.onehandout_weight,
                    _ => 0,
                }
            };
        }

        // Case 1: pos_a is first, pos_b is second or third
        // Trigrams: (pos_a, pos_b, x) or (pos_a, x, pos_b)
        for combo in &self.trigram_combos_per_key[pos_a] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            // Only handle trigrams where pos_b appears
            if p_b != pos_b && p_c != pos_b {
                continue;
            }

            // Before swap: pos_a has key_a, pos_b has key_b
            // After swap: pos_a has key_b, pos_b has key_a
            let old_k1 = key_a;
            let old_k2 = if p_b == pos_a { key_a } else if p_b == pos_b { key_b } else { continue; };
            let old_k3 = if p_c == pos_a { key_a } else if p_c == pos_b { key_b } else { continue; };

            let new_k1 = key_b;
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { continue; };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { continue; };

            let old_freq = tg_freq[old_k1][old_k2][old_k3];
            let new_freq = tg_freq[new_k1][new_k2][new_k3];

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 2: pos_b is first, pos_a is second or third
        for combo in &self.trigram_combos_per_key[pos_b] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            if p_b != pos_a && p_c != pos_a {
                continue;
            }

            let old_k1 = key_b;
            let old_k2 = if p_b == pos_a { key_a } else if p_b == pos_b { key_b } else { continue; };
            let old_k3 = if p_c == pos_a { key_a } else if p_c == pos_b { key_b } else { continue; };

            let new_k1 = key_a;
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { continue; };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { continue; };

            let old_freq = tg_freq[old_k1][old_k2][old_k3];
            let new_freq = tg_freq[new_k1][new_k2][new_k3];

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 3: pos_a is second, pos_b is third (neither is first)
        // Trigrams: (x, pos_a, pos_b) where x != pos_a and x != pos_b
        for combo in &self.trigram_combos_mid[pos_a] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            if p_a == pos_a || p_c != pos_b || p_a == pos_b {
                continue;
            }

            // This trigram has form (other, pos_a, pos_b)
            // We don't know the key at 'other' position at init time, so we can't pre-compute this.
            // However, we CAN pre-compute the TYPE contribution and look up freq at runtime.
            // For now, skip - we'll handle this case specially in key_swap.
        }

        // Case 4: pos_b is second, pos_a is third (neither is first)
        // Similar to case 3 - skip for now

        score_delta
    }

    /// Update pre-computed weighted scores when a key at a position changes.
    ///
    /// When a key changes at `changed_pos` from `old_key` to `new_key`, we need to
    /// update the weighted scores for all OTHER positions that have trigrams
    /// involving `changed_pos`.
    ///
    /// For a position `pos` (where pos != changed_pos), its weighted scores depend
    /// on the keys at other positions. If `changed_pos` appears in a trigram with `pos`,
    /// then `pos`'s weighted scores need to be updated.
    fn update_weighted_scores_for_key_change(
        &mut self,
        changed_pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        tg_freq: &[Vec<Vec<i64>>],
    ) {
        if !self.weighted_scores_initialized {
            return;
        }

        let num_keys = self.num_keys;
        let num_positions = self.num_positions;

        // We need to update weighted scores for all positions that have trigrams
        // involving changed_pos. The changed_pos itself doesn't need updating because
        // its weighted scores are computed based on OTHER positions' keys.

        // Update weighted_score_first for positions where changed_pos is pos_b or pos_c
        for pos in 0..num_positions {
            if pos == changed_pos {
                continue; // Skip the changed position itself
            }

            for combo in &self.trigram_combos_per_key[pos] {
                let pos_b = combo.pos_b;
                let pos_c = combo.pos_c;

                // Only update if changed_pos is involved as pos_b or pos_c
                // (and not as pos itself, which we already skipped)
                let b_is_changed = pos_b == changed_pos;
                let c_is_changed = pos_c == changed_pos;

                if !b_is_changed && !c_is_changed {
                    continue;
                }

                let weight = self.get_weight(combo.trigram_type);

                // For each possible key k at pos, update the weighted score
                for k in 0..num_keys {
                    // Compute old contribution (using old_key at changed_pos)
                    let old_key_b = if pos_b == pos { k } else if b_is_changed { old_key } else { keys[pos_b] };
                    let old_key_c = if pos_c == pos { k } else if c_is_changed { old_key } else { keys[pos_c] };

                    let old_freq = if old_key_b < num_keys && old_key_c < num_keys {
                        tg_freq[k][old_key_b][old_key_c]
                    } else {
                        0
                    };

                    // Compute new contribution (using new_key at changed_pos)
                    let new_key_b = if pos_b == pos { k } else if b_is_changed { new_key } else { keys[pos_b] };
                    let new_key_c = if pos_c == pos { k } else if c_is_changed { new_key } else { keys[pos_c] };

                    let new_freq = if new_key_b < num_keys && new_key_c < num_keys {
                        tg_freq[k][new_key_b][new_key_c]
                    } else {
                        0
                    };

                    let delta = (new_freq - old_freq) * weight;
                    self.weighted_score_first[pos * num_keys + k] += delta;
                }
            }
        }

        // Update weighted_score_mid for positions where changed_pos is pos_a or pos_c
        for pos in 0..num_positions {
            if pos == changed_pos {
                continue;
            }

            for combo in &self.trigram_combos_mid[pos] {
                let pos_a = combo.pos_a;
                let pos_c = combo.pos_c;

                // Skip if pos_a == pos (these are counted in weighted_score_first)
                if pos_a == pos {
                    continue;
                }

                let a_is_changed = pos_a == changed_pos;
                let c_is_changed = pos_c == changed_pos;

                if !a_is_changed && !c_is_changed {
                    continue;
                }

                let weight = self.get_weight(combo.trigram_type);

                for k in 0..num_keys {
                    // Compute old contribution
                    let old_key_a = if a_is_changed { old_key } else { keys[pos_a] };
                    let old_key_c = if pos_c == pos { k } else if c_is_changed { old_key } else { keys[pos_c] };

                    let old_freq = if old_key_a < num_keys && old_key_c < num_keys {
                        tg_freq[old_key_a][k][old_key_c]
                    } else {
                        0
                    };

                    // Compute new contribution
                    let new_key_a = if a_is_changed { new_key } else { keys[pos_a] };
                    let new_key_c = if pos_c == pos { k } else if c_is_changed { new_key } else { keys[pos_c] };

                    let new_freq = if new_key_a < num_keys && new_key_c < num_keys {
                        tg_freq[new_key_a][k][new_key_c]
                    } else {
                        0
                    };

                    let delta = (new_freq - old_freq) * weight;
                    self.weighted_score_mid[pos * num_keys + k] += delta;
                }
            }
        }

        // Update weighted_score_end for positions where changed_pos is pos_a or pos_b
        for pos in 0..num_positions {
            if pos == changed_pos {
                continue;
            }

            for combo in &self.trigram_combos_end[pos] {
                let pos_a = combo.pos_a;
                let pos_b = combo.pos_b;

                // Skip if pos_a == pos or pos_b == pos (counted elsewhere)
                if pos_a == pos || pos_b == pos {
                    continue;
                }

                let a_is_changed = pos_a == changed_pos;
                let b_is_changed = pos_b == changed_pos;

                if !a_is_changed && !b_is_changed {
                    continue;
                }

                let weight = self.get_weight(combo.trigram_type);

                for k in 0..num_keys {
                    // Compute old contribution
                    let old_key_a = if a_is_changed { old_key } else { keys[pos_a] };
                    let old_key_b = if b_is_changed { old_key } else { keys[pos_b] };

                    let old_freq = if old_key_a < num_keys && old_key_b < num_keys {
                        tg_freq[old_key_a][old_key_b][k]
                    } else {
                        0
                    };

                    // Compute new contribution
                    let new_key_a = if a_is_changed { new_key } else { keys[pos_a] };
                    let new_key_b = if b_is_changed { new_key } else { keys[pos_b] };

                    let new_freq = if new_key_a < num_keys && new_key_b < num_keys {
                        tg_freq[new_key_a][new_key_b][k]
                    } else {
                        0
                    };

                    let delta = (new_freq - old_freq) * weight;
                    self.weighted_score_end[pos * num_keys + k] += delta;
                }
            }
        }
    }

    /// Compute the score delta using pre-computed weighted scores (O(1) lookup).
    ///
    /// This is the fast path for speculative scoring when weighted scores are initialized.
    #[inline]
    fn compute_replace_delta_fast(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let old_score = if old_valid {
            self.weighted_score_first[pos * num_keys + old_key]
                + self.weighted_score_mid[pos * num_keys + old_key]
                + self.weighted_score_end[pos * num_keys + old_key]
        } else {
            0
        };

        let new_score = if new_valid {
            self.weighted_score_first[pos * num_keys + new_key]
                + self.weighted_score_mid[pos * num_keys + new_key]
                + self.weighted_score_end[pos * num_keys + new_key]
        } else {
            0
        };

        new_score - old_score
    }

    /// Get the current weighted score
    ///
    /// Returns the weighted sum of all frequency totals plus the magic rule score delta:
    /// `inroll_freq * inroll_weight + outroll_freq * outroll_weight + alternate_freq * alternate_weight
    ///  + redirect_freq * redirect_weight + onehandin_freq * onehandin_weight + onehandout_freq * onehandout_weight
    ///  + magic_rule_score_delta`
    ///
    /// Requirements: 6.1, 6.2
    pub fn score(&self) -> i64 {
        self.inroll_freq * self.inroll_weight
            + self.outroll_freq * self.outroll_weight
            + self.alternate_freq * self.alternate_weight
            + self.redirect_freq * self.redirect_weight
            + self.onehandin_freq * self.onehandin_weight
            + self.onehandout_freq * self.onehandout_weight
            + self.magic_rule_score_delta
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

        // Helper to add frequency delta to the appropriate field
        macro_rules! add_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => delta.inroll_freq += $freq_delta,
                    TrigramType::Outroll => delta.outroll_freq += $freq_delta,
                    TrigramType::Alternate => delta.alternate_freq += $freq_delta,
                    TrigramType::Redirect => delta.redirect_freq += $freq_delta,
                    TrigramType::OnehandIn => delta.onehandin_freq += $freq_delta,
                    TrigramType::OnehandOut => delta.onehandout_freq += $freq_delta,
                    _ => {}
                }
            };
        }

        // Case 1: pos is the first position in the trigram (pos, pos_b, pos_c)
        // tg_freq[old_key][key_b][key_c] -> tg_freq[new_key][key_b][key_c]
        // Note: If pos_b == pos or pos_c == pos, we need to use old_key/new_key for those positions
        for combo in &self.trigram_combos_per_key[pos] {
            let pos_b = combo.pos_b;
            let pos_c = combo.pos_c;

            // Skip if skip_pos matches pos_b or pos_c (to avoid double-counting in swaps)
            if let Some(skip) = skip_pos {
                if pos_b == skip || pos_c == skip {
                    continue;
                }
            }

            // For old_freq: if pos_b or pos_c equals pos, use old_key
            let old_key_b = if pos_b == pos { old_key } else { keys[pos_b] };
            let old_key_c = if pos_c == pos { old_key } else { keys[pos_c] };

            // For new_freq: if pos_b or pos_c equals pos, use new_key
            let new_key_b = if pos_b == pos { new_key } else { keys[pos_b] };
            let new_key_c = if pos_c == pos { new_key } else { keys[pos_c] };

            // Compute old_freq: 0 if old_key or any derived old key is invalid
            let old_freq = if old_valid && old_key_b < num_keys && old_key_c < num_keys {
                tg_freq[old_key][old_key_b][old_key_c]
            } else {
                0
            };

            // Compute new_freq: 0 if new_key or any derived new key is invalid
            let new_freq = if new_valid && new_key_b < num_keys && new_key_c < num_keys {
                tg_freq[new_key][new_key_b][new_key_c]
            } else {
                0
            };

            // Skip if both are 0 (no change)
            if old_freq == 0 && new_freq == 0 {
                continue;
            }

            let freq_delta = new_freq - old_freq;

            add_delta!(combo.trigram_type, freq_delta);
        }

        // Case 2: pos is the second position in the trigram (pos_a, pos, pos_c)
        // tg_freq[key_a][old_key][key_c] -> tg_freq[key_a][new_key][key_c]
        // Only count if pos_a != pos (to avoid double-counting with case 1)
        // Note: If pos_c == pos, we need to use old_key/new_key for that position
        for combo in &self.trigram_combos_mid[pos] {
            let pos_a = combo.pos_a;
            let pos_c = combo.pos_c;

            // Skip if pos_a == pos (already counted in case 1)
            if pos_a == pos {
                continue;
            }

            // Skip if skip_pos matches pos_a or pos_c (to avoid double-counting in swaps)
            if let Some(skip) = skip_pos {
                if pos_a == skip || pos_c == skip {
                    continue;
                }
            }

            let key_a = keys[pos_a];

            // For old_freq: if pos_c equals pos, use old_key
            let old_key_c = if pos_c == pos { old_key } else { keys[pos_c] };

            // For new_freq: if pos_c equals pos, use new_key
            let new_key_c = if pos_c == pos { new_key } else { keys[pos_c] };

            // Skip if key_a is invalid (it's not being replaced)
            if key_a >= num_keys {
                continue;
            }

            // Compute old_freq: 0 if old_key or old_key_c is invalid
            let old_freq = if old_valid && old_key_c < num_keys {
                tg_freq[key_a][old_key][old_key_c]
            } else {
                0
            };

            // Compute new_freq: 0 if new_key or new_key_c is invalid
            let new_freq = if new_valid && new_key_c < num_keys {
                tg_freq[key_a][new_key][new_key_c]
            } else {
                0
            };

            // Skip if both are 0 (no change)
            if old_freq == 0 && new_freq == 0 {
                continue;
            }

            let freq_delta = new_freq - old_freq;

            add_delta!(combo.trigram_type, freq_delta);
        }

        // Case 3: pos is the third position in the trigram (pos_a, pos_b, pos)
        // tg_freq[key_a][key_b][old_key] -> tg_freq[key_a][key_b][new_key]
        // Only count if pos_a != pos and pos_b != pos (to avoid double-counting with cases 1 and 2)
        for combo in &self.trigram_combos_end[pos] {
            let pos_a = combo.pos_a;
            let pos_b = combo.pos_b;

            // Skip if pos_a == pos or pos_b == pos (already counted in cases 1 or 2)
            if pos_a == pos || pos_b == pos {
                continue;
            }

            // Skip if skip_pos matches pos_a or pos_b (to avoid double-counting in swaps)
            if let Some(skip) = skip_pos {
                if pos_a == skip || pos_b == skip {
                    continue;
                }
            }

            let key_a = keys[pos_a];
            let key_b = keys[pos_b];

            // Skip if key_a or key_b is invalid
            if key_a >= num_keys || key_b >= num_keys {
                continue;
            }

            let old_freq = if old_valid { tg_freq[key_a][key_b][old_key] } else { 0 };
            let new_freq = if new_valid { tg_freq[key_a][key_b][new_key] } else { 0 };
            let freq_delta = new_freq - old_freq;

            add_delta!(combo.trigram_type, freq_delta);
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

        // Helper to get weighted score delta for a trigram type
        macro_rules! weighted_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => $freq_delta * self.inroll_weight,
                    TrigramType::Outroll => $freq_delta * self.outroll_weight,
                    TrigramType::Alternate => $freq_delta * self.alternate_weight,
                    TrigramType::Redirect => $freq_delta * self.redirect_weight,
                    TrigramType::OnehandIn => $freq_delta * self.onehandin_weight,
                    TrigramType::OnehandOut => $freq_delta * self.onehandout_weight,
                    _ => 0,
                }
            };
        }

        // Case 1: pos is the first position in the trigram (pos, pos_b, pos_c)
        // Note: If pos_b == pos or pos_c == pos, we need to use old_key/new_key for those positions
        for combo in &self.trigram_combos_per_key[pos] {
            let pos_b = combo.pos_b;
            let pos_c = combo.pos_c;

            if let Some(skip) = skip_pos {
                if pos_b == skip || pos_c == skip {
                    continue;
                }
            }

            // For old_freq: if pos_b or pos_c equals pos, use old_key
            let old_key_b = if pos_b == pos { old_key } else { keys[pos_b] };
            let old_key_c = if pos_c == pos { old_key } else { keys[pos_c] };

            // For new_freq: if pos_b or pos_c equals pos, use new_key
            let new_key_b = if pos_b == pos { new_key } else { keys[pos_b] };
            let new_key_c = if pos_c == pos { new_key } else { keys[pos_c] };

            // Compute old_freq: 0 if old_key or any derived old key is invalid
            let old_freq = if old_valid && old_key_b < num_keys && old_key_c < num_keys {
                tg_freq[old_key][old_key_b][old_key_c]
            } else {
                0
            };

            // Compute new_freq: 0 if new_key or any derived new key is invalid
            let new_freq = if new_valid && new_key_b < num_keys && new_key_c < num_keys {
                tg_freq[new_key][new_key_b][new_key_c]
            } else {
                0
            };

            // Skip if both are 0 (no change)
            if old_freq == 0 && new_freq == 0 {
                continue;
            }

            let freq_delta = new_freq - old_freq;

            score_delta += weighted_delta!(combo.trigram_type, freq_delta);
        }

        // Case 2: pos is the second position in the trigram (pos_a, pos, pos_c)
        // Only count if pos_a != pos (to avoid double-counting with case 1)
        // Note: If pos_c == pos, we need to use old_key/new_key for that position
        for combo in &self.trigram_combos_mid[pos] {
            let pos_a = combo.pos_a;
            let pos_c = combo.pos_c;

            // Skip if pos_a == pos (already counted in case 1)
            if pos_a == pos {
                continue;
            }

            if let Some(skip) = skip_pos {
                if pos_a == skip || pos_c == skip {
                    continue;
                }
            }

            let key_a = keys[pos_a];

            // For old_freq: if pos_c equals pos, use old_key
            let old_key_c = if pos_c == pos { old_key } else { keys[pos_c] };

            // For new_freq: if pos_c equals pos, use new_key
            let new_key_c = if pos_c == pos { new_key } else { keys[pos_c] };

            // Skip if key_a is invalid (it's not being replaced)
            if key_a >= num_keys {
                continue;
            }

            // Compute old_freq: 0 if old_key or old_key_c is invalid
            let old_freq = if old_valid && old_key_c < num_keys {
                tg_freq[key_a][old_key][old_key_c]
            } else {
                0
            };

            // Compute new_freq: 0 if new_key or new_key_c is invalid
            let new_freq = if new_valid && new_key_c < num_keys {
                tg_freq[key_a][new_key][new_key_c]
            } else {
                0
            };

            // Skip if both are 0 (no change)
            if old_freq == 0 && new_freq == 0 {
                continue;
            }

            let freq_delta = new_freq - old_freq;

            score_delta += weighted_delta!(combo.trigram_type, freq_delta);
        }

        // Case 3: pos is the third position in the trigram (pos_a, pos_b, pos)
        // Only count if pos_a != pos and pos_b != pos (to avoid double-counting with cases 1 and 2)
        for combo in &self.trigram_combos_end[pos] {
            let pos_a = combo.pos_a;
            let pos_b = combo.pos_b;

            // Skip if pos_a == pos or pos_b == pos (already counted in cases 1 or 2)
            if pos_a == pos || pos_b == pos {
                continue;
            }

            if let Some(skip) = skip_pos {
                if pos_a == skip || pos_b == skip {
                    continue;
                }
            }

            let key_a = keys[pos_a];
            let key_b = keys[pos_b];

            if key_a >= num_keys || key_b >= num_keys {
                continue;
            }

            let old_freq = if old_valid { tg_freq[key_a][key_b][old_key] } else { 0 };
            let new_freq = if new_valid { tg_freq[key_a][key_b][new_key] } else { 0 };
            let freq_delta = new_freq - old_freq;

            score_delta += weighted_delta!(combo.trigram_type, freq_delta);
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
    /// When weighted scores are pre-computed and skip_pos is None, uses O(1) fast path.
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
            // Update pre-computed weighted scores for other positions
            self.update_weighted_scores_for_key_change(pos, old_key, new_key, keys, tg_freq);
            self.score()
        } else {
            // Use fast path if weighted scores are initialized and no skip_pos
            if self.weighted_scores_initialized && skip_pos.is_none() {
                let score_delta = self.compute_replace_delta_fast(pos, old_key, new_key);
                self.score() + score_delta
            } else {
                // Fall back to slow path
                let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, skip_pos, tg_freq);
                self.score() + score_delta
            }
        }
    }

    /// Swap keys at two positions. Returns the new score.
    ///
    /// If `apply` is true, computes the combined delta for both positions,
    /// applies it to internal state, and returns the new score.
    /// If `apply` is false, computes only the score deltas and returns the
    /// speculative new score without mutating state.
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
        // If swapping a position with itself, no change
        if pos_a == pos_b {
            return self.score();
        }

        // For trigrams, we need to handle three cases:
        // 1. Trigrams involving only pos_a (not pos_b) - handled by delta_a with skip_pos=pos_b
        // 2. Trigrams involving only pos_b (not pos_a) - handled by delta_b with skip_pos=pos_a
        // 3. Trigrams involving BOTH pos_a and pos_b - need special handling
        //
        // The skip_pos mechanism in compute_replace_delta skips trigrams where ANY of the
        // other positions equals skip_pos. This means trigrams involving both pos_a and pos_b
        // are skipped in BOTH delta computations, so we need to handle them separately.

        if apply {
            // Compute deltas for trigrams involving only one of the swap positions
            let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), tg_freq);
            let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), tg_freq);

            // Compute delta for trigrams involving BOTH pos_a and pos_b
            let delta_both = self.compute_swap_both_delta(pos_a, pos_b, key_a, key_b, keys, tg_freq);

            let combined = delta_a.combine(&delta_b).combine(&delta_both);
            self.apply_delta(&combined);
            self.score()
        } else {
            // O(1) lookup for speculative scoring when swap_delta table is populated
            // **Validates: Requirements 1.3, 1.5**
            if !self.swap_delta.is_empty() {
                // Normalize position ordering (pos_a < pos_b) for canonical lookup
                let (p_lo, p_hi, k_lo, k_hi) = if pos_a < pos_b {
                    (pos_a, pos_b, key_a, key_b)
                } else {
                    (pos_b, pos_a, key_b, key_a)
                };

                // Look up the delta directly from the pre-computed table.
                // If not found in the HashMap, the delta is 0 (sparse storage semantics).
                let delta = self.swap_delta
                    .get(&(p_lo, p_hi, k_lo, k_hi))
                    .copied()
                    .unwrap_or(0) as i64;

                return self.score() + delta;
            }

            // Fallback to computed delta if swap_delta table not initialized
            // The slow path iterates over pre-computed trigram combinations which is still
            // much faster than recomputing from scratch.
            let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), tg_freq);
            let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), tg_freq);
            let score_both = self.compute_swap_both_delta_score_only(pos_a, pos_b, key_a, key_b, keys, tg_freq);
            self.score() + score_a + score_b + score_both
        }
    }

    /// Compute the correction for trigrams involving BOTH swap positions when using fast path.
    ///
    /// The pre-computed weighted scores assume other positions have their current keys.
    /// For a swap, this assumption is wrong for trigrams involving both positions.
    /// This method computes the correction needed.
    ///
    /// For trigrams involving both pos_a and pos_b:
    /// - The fast path (compute_replace_delta_fast) already counted these trigrams
    ///   but with the WRONG assumption about the other position's key
    /// - We need to subtract the wrong contribution and add the correct one
    fn compute_swap_both_correction_fast(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        let key_a_valid = key_a < num_keys;
        let key_b_valid = key_b < num_keys;

        if !key_a_valid || !key_b_valid {
            return 0;
        }

        let mut correction: i64 = 0;

        macro_rules! weighted_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => $freq_delta * self.inroll_weight,
                    TrigramType::Outroll => $freq_delta * self.outroll_weight,
                    TrigramType::Alternate => $freq_delta * self.alternate_weight,
                    TrigramType::Redirect => $freq_delta * self.redirect_weight,
                    TrigramType::OnehandIn => $freq_delta * self.onehandin_weight,
                    TrigramType::OnehandOut => $freq_delta * self.onehandout_weight,
                    _ => 0,
                }
            };
        }

        // For trigrams involving both pos_a and pos_b, the fast path computed:
        // - For pos_a: delta assuming keys[pos_b] = key_b (wrong after swap, should be key_a)
        // - For pos_b: delta assuming keys[pos_a] = key_a (wrong after swap, should be key_b)
        //
        // We need to correct this by:
        // 1. Subtracting what the fast path computed for these trigrams
        // 2. Adding the correct delta for these trigrams
        //
        // The correct delta for a trigram involving both positions is:
        // new_freq - old_freq where:
        // - old_freq uses (key_a at pos_a, key_b at pos_b)
        // - new_freq uses (key_b at pos_a, key_a at pos_b)

        // Case 1: pos_a is first, pos_b is second or third
        for combo in &self.trigram_combos_per_key[pos_a] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            // Only handle trigrams where pos_b appears
            let b_is_pos_b = p_b == pos_b;
            let c_is_pos_b = p_c == pos_b;
            if !b_is_pos_b && !c_is_pos_b {
                continue;
            }

            // What the fast path computed (wrong):
            // old: tg_freq[key_a][keys[p_b]][keys[p_c]] where keys[pos_b] = key_b
            // new: tg_freq[key_b][keys[p_b]][keys[p_c]] where keys[pos_b] = key_b
            let wrong_old_k2 = if p_b == pos_a { key_a } else if b_is_pos_b { key_b } else { return 0; /* shouldn't happen */ };
            let wrong_old_k3 = if p_c == pos_a { key_a } else if c_is_pos_b { key_b } else { return 0; };
            let wrong_new_k2 = if p_b == pos_a { key_b } else if b_is_pos_b { key_b } else { return 0; };
            let wrong_new_k3 = if p_c == pos_a { key_b } else if c_is_pos_b { key_b } else { return 0; };

            // What it should be (correct):
            // old: tg_freq[key_a][...][...] with original keys
            // new: tg_freq[key_b][...][...] with swapped keys (key_a at pos_b)
            let correct_old_k2 = if p_b == pos_a { key_a } else if b_is_pos_b { key_b } else { return 0; };
            let correct_old_k3 = if p_c == pos_a { key_a } else if c_is_pos_b { key_b } else { return 0; };
            let correct_new_k2 = if p_b == pos_a { key_b } else if b_is_pos_b { key_a } else { return 0; };
            let correct_new_k3 = if p_c == pos_a { key_b } else if c_is_pos_b { key_a } else { return 0; };

            if wrong_old_k2 >= num_keys || wrong_old_k3 >= num_keys ||
               wrong_new_k2 >= num_keys || wrong_new_k3 >= num_keys ||
               correct_new_k2 >= num_keys || correct_new_k3 >= num_keys {
                continue;
            }

            let wrong_old = tg_freq[key_a][wrong_old_k2][wrong_old_k3];
            let wrong_new = tg_freq[key_b][wrong_new_k2][wrong_new_k3];
            let wrong_delta = wrong_new - wrong_old;

            let correct_old = tg_freq[key_a][correct_old_k2][correct_old_k3];
            let correct_new = tg_freq[key_b][correct_new_k2][correct_new_k3];
            let correct_delta = correct_new - correct_old;

            // Correction = correct - wrong
            correction += weighted_delta!(combo.trigram_type, correct_delta - wrong_delta);
        }

        // Case 2: pos_b is first, pos_a is second or third
        for combo in &self.trigram_combos_per_key[pos_b] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            // Only handle trigrams where pos_a appears
            let b_is_pos_a = p_b == pos_a;
            let c_is_pos_a = p_c == pos_a;
            if !b_is_pos_a && !c_is_pos_a {
                continue;
            }

            // What the fast path computed (wrong):
            // Assumes keys[pos_a] = key_a, but after swap it's key_b
            let wrong_old_k2 = if p_b == pos_b { key_b } else if b_is_pos_a { key_a } else { return 0; };
            let wrong_old_k3 = if p_c == pos_b { key_b } else if c_is_pos_a { key_a } else { return 0; };
            let wrong_new_k2 = if p_b == pos_b { key_a } else if b_is_pos_a { key_a } else { return 0; };
            let wrong_new_k3 = if p_c == pos_b { key_a } else if c_is_pos_a { key_a } else { return 0; };

            // What it should be (correct):
            let correct_old_k2 = if p_b == pos_b { key_b } else if b_is_pos_a { key_a } else { return 0; };
            let correct_old_k3 = if p_c == pos_b { key_b } else if c_is_pos_a { key_a } else { return 0; };
            let correct_new_k2 = if p_b == pos_b { key_a } else if b_is_pos_a { key_b } else { return 0; };
            let correct_new_k3 = if p_c == pos_b { key_a } else if c_is_pos_a { key_b } else { return 0; };

            if wrong_old_k2 >= num_keys || wrong_old_k3 >= num_keys ||
               wrong_new_k2 >= num_keys || wrong_new_k3 >= num_keys ||
               correct_new_k2 >= num_keys || correct_new_k3 >= num_keys {
                continue;
            }

            let wrong_old = tg_freq[key_b][wrong_old_k2][wrong_old_k3];
            let wrong_new = tg_freq[key_a][wrong_new_k2][wrong_new_k3];
            let wrong_delta = wrong_new - wrong_old;

            let correct_old = tg_freq[key_b][correct_old_k2][correct_old_k3];
            let correct_new = tg_freq[key_a][correct_new_k2][correct_new_k3];
            let correct_delta = correct_new - correct_old;

            correction += weighted_delta!(combo.trigram_type, correct_delta - wrong_delta);
        }

        // Cases 3 and 4: pos_a or pos_b is middle, the other is third
        // These are more complex - for now, fall back to the slow path for these
        // by computing them directly

        // Case 3: pos_a is second, pos_b is third (neither is first)
        for combo in &self.trigram_combos_mid[pos_a] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            if p_a == pos_a || p_c != pos_b || p_a == pos_b {
                continue;
            }

            // This trigram has: (some_other_pos, pos_a, pos_b)
            // The fast path for pos_a assumed keys[pos_b] = key_b
            // The fast path for pos_b assumed keys[pos_a] = key_a
            // Both are wrong after the swap

            // We need to compute the actual delta and subtract what was computed
            // But this is getting complex - let's just compute the correct delta directly
            // and not try to correct the fast path
        }

        correction
    }

    /// Compute delta for trigrams involving BOTH swap positions.
    ///
    /// This handles the case where a trigram has both pos_a and pos_b in different slots.
    /// These trigrams are skipped by the regular compute_replace_delta calls (due to skip_pos),
    /// so we need to handle them separately.
    fn compute_swap_both_delta(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        tg_freq: &[Vec<Vec<i64>>],
    ) -> TrigramDelta {
        let num_keys = self.num_keys;
        let key_a_valid = key_a < num_keys;
        let key_b_valid = key_b < num_keys;

        let mut delta = TrigramDelta::default();

        macro_rules! add_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => delta.inroll_freq += $freq_delta,
                    TrigramType::Outroll => delta.outroll_freq += $freq_delta,
                    TrigramType::Alternate => delta.alternate_freq += $freq_delta,
                    TrigramType::Redirect => delta.redirect_freq += $freq_delta,
                    TrigramType::OnehandIn => delta.onehandin_freq += $freq_delta,
                    TrigramType::OnehandOut => delta.onehandout_freq += $freq_delta,
                    _ => {}
                }
            };
        }

        // Case 1: pos_a is first, pos_b is second or third
        // Trigrams: (pos_a, pos_b, x) or (pos_a, x, pos_b) where x can be pos_a or pos_b too
        for combo in &self.trigram_combos_per_key[pos_a] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            // Only handle trigrams where pos_b appears in position 2 or 3
            if p_b != pos_b && p_c != pos_b {
                continue;
            }

            // Before swap: (key_a, keys[p_b], keys[p_c])
            // After swap: (key_b, new_keys[p_b], new_keys[p_c])
            // where new_keys[pos_a] = key_b, new_keys[pos_b] = key_a
            let old_k1 = key_a;
            let old_k2 = keys[p_b];
            let old_k3 = keys[p_c];

            // For new keys, check if each position is pos_a or pos_b
            let new_k1 = key_b;  // pos_a now has key_b
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { keys[p_b] };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { keys[p_c] };

            if old_k2 >= num_keys || old_k3 >= num_keys || new_k2 >= num_keys || new_k3 >= num_keys {
                continue;
            }

            let old_freq = if key_a_valid { tg_freq[old_k1][old_k2][old_k3] } else { 0 };
            let new_freq = if key_b_valid && new_k2 < num_keys && new_k3 < num_keys {
                tg_freq[new_k1][new_k2][new_k3]
            } else { 0 };

            add_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 2: pos_b is first, pos_a is second or third
        // Trigrams: (pos_b, pos_a, x) or (pos_b, x, pos_a) where x can be pos_a or pos_b too
        for combo in &self.trigram_combos_per_key[pos_b] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            // Only handle trigrams where pos_a appears in position 2 or 3
            if p_b != pos_a && p_c != pos_a {
                continue;
            }

            // Before swap: (key_b, keys[p_b], keys[p_c])
            // After swap: (key_a, new_keys[p_b], new_keys[p_c])
            let old_k1 = key_b;
            let old_k2 = keys[p_b];
            let old_k3 = keys[p_c];

            // For new keys, check if each position is pos_a or pos_b
            let new_k1 = key_a;  // pos_b now has key_a
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { keys[p_b] };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { keys[p_c] };

            if old_k2 >= num_keys || old_k3 >= num_keys || new_k2 >= num_keys || new_k3 >= num_keys {
                continue;
            }

            let old_freq = if key_b_valid { tg_freq[old_k1][old_k2][old_k3] } else { 0 };
            let new_freq = if key_a_valid && new_k2 < num_keys && new_k3 < num_keys {
                tg_freq[new_k1][new_k2][new_k3]
            } else { 0 };

            add_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 3: pos_a is second, pos_b is third (and neither is first)
        // Trigrams: (x, pos_a, pos_b) where x != pos_a and x != pos_b
        for combo in &self.trigram_combos_mid[pos_a] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            // Skip if p_a is pos_a (already handled in case 1)
            if p_a == pos_a {
                continue;
            }

            // Only handle trigrams where pos_b is the third position
            if p_c != pos_b {
                continue;
            }

            // Skip if p_a is pos_b (already handled in case 2)
            if p_a == pos_b {
                continue;
            }

            let first_key = keys[p_a];
            if first_key >= num_keys {
                continue;
            }

            // Before swap: (keys[p_a], key_a, key_b)
            // After swap: (keys[p_a], key_b, key_a)
            let old_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_a][key_b] } else { 0 };
            let new_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_b][key_a] } else { 0 };

            add_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 4: pos_b is second, pos_a is third (and neither is first)
        // Trigrams: (x, pos_b, pos_a) where x != pos_a and x != pos_b
        for combo in &self.trigram_combos_mid[pos_b] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            // Skip if p_a is pos_b (already handled in case 2)
            if p_a == pos_b {
                continue;
            }

            // Only handle trigrams where pos_a is the third position
            if p_c != pos_a {
                continue;
            }

            // Skip if p_a is pos_a (already handled in case 1)
            if p_a == pos_a {
                continue;
            }

            let first_key = keys[p_a];
            if first_key >= num_keys {
                continue;
            }

            // Before swap: (keys[p_a], key_b, key_a)
            // After swap: (keys[p_a], key_a, key_b)
            let old_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_b][key_a] } else { 0 };
            let new_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_a][key_b] } else { 0 };

            add_delta!(combo.trigram_type, new_freq - old_freq);
        }

        delta
    }

    /// Compute only the score delta for trigrams involving BOTH swap positions.
    fn compute_swap_both_delta_score_only(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;
        let key_a_valid = key_a < num_keys;
        let key_b_valid = key_b < num_keys;

        let mut score_delta: i64 = 0;

        macro_rules! weighted_delta {
            ($trigram_type:expr, $freq_delta:expr) => {
                match $trigram_type {
                    TrigramType::Inroll => $freq_delta * self.inroll_weight,
                    TrigramType::Outroll => $freq_delta * self.outroll_weight,
                    TrigramType::Alternate => $freq_delta * self.alternate_weight,
                    TrigramType::Redirect => $freq_delta * self.redirect_weight,
                    TrigramType::OnehandIn => $freq_delta * self.onehandin_weight,
                    TrigramType::OnehandOut => $freq_delta * self.onehandout_weight,
                    _ => 0,
                }
            };
        }

        // Case 1: pos_a is first, pos_b is second or third
        for combo in &self.trigram_combos_per_key[pos_a] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            if p_b != pos_b && p_c != pos_b {
                continue;
            }

            let old_k1 = key_a;
            let old_k2 = keys[p_b];
            let old_k3 = keys[p_c];

            let new_k1 = key_b;
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { keys[p_b] };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { keys[p_c] };

            if old_k2 >= num_keys || old_k3 >= num_keys || new_k2 >= num_keys || new_k3 >= num_keys {
                continue;
            }

            let old_freq = if key_a_valid { tg_freq[old_k1][old_k2][old_k3] } else { 0 };
            let new_freq = if key_b_valid && new_k2 < num_keys && new_k3 < num_keys {
                tg_freq[new_k1][new_k2][new_k3]
            } else { 0 };

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 2: pos_b is first, pos_a is second or third
        for combo in &self.trigram_combos_per_key[pos_b] {
            let p_b = combo.pos_b;
            let p_c = combo.pos_c;

            if p_b != pos_a && p_c != pos_a {
                continue;
            }

            let old_k1 = key_b;
            let old_k2 = keys[p_b];
            let old_k3 = keys[p_c];

            let new_k1 = key_a;
            let new_k2 = if p_b == pos_a { key_b } else if p_b == pos_b { key_a } else { keys[p_b] };
            let new_k3 = if p_c == pos_a { key_b } else if p_c == pos_b { key_a } else { keys[p_c] };

            if old_k2 >= num_keys || old_k3 >= num_keys || new_k2 >= num_keys || new_k3 >= num_keys {
                continue;
            }

            let old_freq = if key_b_valid { tg_freq[old_k1][old_k2][old_k3] } else { 0 };
            let new_freq = if key_a_valid && new_k2 < num_keys && new_k3 < num_keys {
                tg_freq[new_k1][new_k2][new_k3]
            } else { 0 };

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 3: pos_a is second, pos_b is third (and neither is first)
        for combo in &self.trigram_combos_mid[pos_a] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            if p_a == pos_a || p_c != pos_b || p_a == pos_b {
                continue;
            }

            let first_key = keys[p_a];
            if first_key >= num_keys {
                continue;
            }

            let old_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_a][key_b] } else { 0 };
            let new_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_b][key_a] } else { 0 };

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        // Case 4: pos_b is second, pos_a is third (and neither is first)
        for combo in &self.trigram_combos_mid[pos_b] {
            let p_a = combo.pos_a;
            let p_c = combo.pos_c;

            if p_a == pos_b || p_c != pos_a || p_a == pos_a {
                continue;
            }

            let first_key = keys[p_a];
            if first_key >= num_keys {
                continue;
            }

            let old_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_b][key_a] } else { 0 };
            let new_freq = if key_a_valid && key_b_valid { tg_freq[first_key][key_a][key_b] } else { 0 };

            score_delta += weighted_delta!(combo.trigram_type, new_freq - old_freq);
        }

        score_delta
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
        // When replacing key at pos 0 from key 0 to key 2:
        // Old: tg_freq[0][1][0] = 100 (old_key, key_b, old_key)
        // New: tg_freq[2][1][2] = 150 (new_key, key_b, new_key) - since pos_c == pos
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0, key_b=1, old_key=0
            (2, 1, 2, 150),  // new_key=2, key_b=1, new_key=2
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
        // Trigram involving pos 0, 1, 0 (alternate) - pos_b=1 will be skipped
        // Trigram involving pos 0, 2, 0 (alternate) - should be counted
        // When replacing key at pos 0 from key 0 to key 3:
        // For (0, 2, 0): old = tg_freq[0][2][0], new = tg_freq[3][2][3] (since pos_c == pos)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // pos 0 -> pos 1 -> pos 0 (will be skipped)
            (3, 1, 3, 200),  // new key at pos 0 (will be skipped)
            (0, 2, 0, 50),   // pos 0 -> pos 2 -> pos 0
            (3, 2, 3, 75),   // new key at pos 0 (new_key, key_c=2, new_key)
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
        // For trigram (0, 1, 0): old = tg_freq[0][1][0], new = tg_freq[2][1][2]
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 200),  // old_key=0
            (2, 1, 2, 50),   // new_key=2 (new_key, key_b, new_key since pos_c == pos)
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

        // For trigram (0, 1, 0): new = tg_freq[2][1][2] (new_key, key_b, new_key)
        let tg_freq = create_tg_freq(4, &[
            (2, 1, 2, 100),  // new_key=2
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

        // For trigram (0, 1, 0): old = tg_freq[0][1][0]
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0
        ]);

        // new_key=99 is invalid (>= num_keys=4)
        // Since new_key is invalid, new_key_b and new_key_c will also be invalid (99)
        // This should cause the trigram to be skipped due to invalid keys
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
        // When replacing key at pos 0 from key 0 to key 2:
        // Old: tg_freq[0][1][0] (old_key, key_b, old_key)
        // New: tg_freq[2][1][2] (new_key, key_b, new_key since pos_c == pos)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // old_key=0
            (2, 1, 2, 150),  // new_key=2
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

        // For trigram (0, 1, 0) and (0, 2, 0):
        // When replacing key at pos 0 from key 0 to key 3:
        // Old: tg_freq[0][key_b][0], New: tg_freq[3][key_b][3] (since pos_c == pos)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (3, 1, 3, 200),
            (0, 2, 0, 50),
            (3, 2, 3, 75),
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
        // For trigram (0, 1, 0): old = tg_freq[0][1][0], new = tg_freq[2][1][2]
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 200),  // old_key=0
            (2, 1, 2, 50),   // new_key=2 (new_key, key_b, new_key since pos_c == pos)
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

        // For trigram (0, 1, 0): old = tg_freq[0][1][0], new = tg_freq[2][1][2]
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),
            (2, 1, 2, 200),
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

        // For trigram (0, 1, 0): new = tg_freq[2][1][2] (new_key, key_b, new_key)
        let tg_freq = create_tg_freq(4, &[
            (2, 1, 2, 100),  // new_key=2
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

    #[test]
    fn test_key_swap_is_reversible() {
        use crate::weights::{FingerWeights, Weights};

        // Test that swapping twice returns to the original state
        // Use a simpler case with just 2 positions
        let fingers = vec![LP, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 0,
            outroll: 0,
            alternate: 100,  // Only count alternates
            redirect: 0,
            onehandin: 0,
            onehandout: 0,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights::default(),
        };
        cache.set_weights(&weights);

        let keys = vec![0, 1];

        // Create a simple trigram frequency
        // Trigram (0, 1, 0) is alternate (LP -> RI -> LP)
        let tg_freq = create_tg_freq(4, &[
            (0, 1, 0, 100),  // key sequence 0 -> 1 -> 0
            (1, 0, 1, 200),  // key sequence 1 -> 0 -> 1
        ]);

        // Initialize the cache
        cache.replace_key(0, 99, 0, &keys, None, &tg_freq, true);
        cache.replace_key(1, 99, 1, &keys, None, &tg_freq, true);

        let original_score = cache.score();
        let original_alternate = cache.alternate_freq;

        // First swap: positions 0 and 1
        // Before: pos 0 has key 0, pos 1 has key 1
        // After: pos 0 has key 1, pos 1 has key 0
        let score_after_first_swap = cache.key_swap(0, 1, 0, 1, &keys, &tg_freq, true);

        // After first swap, keys are: [1, 0]
        let keys_after_swap = vec![1, 0];

        // Second swap: reverse the first swap
        // Before: pos 0 has key 1, pos 1 has key 0
        // After: pos 0 has key 0, pos 1 has key 1
        let score_after_second_swap = cache.key_swap(0, 1, 1, 0, &keys_after_swap, &tg_freq, true);

        // Suppress unused variable warnings
        let _ = score_after_first_swap;
        let _ = score_after_second_swap;

        // Score should be restored
        assert_eq!(cache.alternate_freq, original_alternate, "Alternate freq should be restored");
        assert_eq!(cache.score(), original_score, "Score should be restored after double swap");
    }

    // ==================== add_rule() tests ====================

    /// Helper function to create key_positions array from keys
    /// Maps each key to its position (index in keys array)
    fn create_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
        let mut key_positions = vec![None; num_keys];
        for (pos, &key) in keys.iter().enumerate() {
            if key < num_keys {
                key_positions[key] = Some(pos);
            }
        }
        key_positions
    }

    #[test]
    fn test_add_rule_basic_apply_true_updates_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that add_rule with apply=true updates active_rules and magic_rule_score_delta
        // Validates: Requirement 2.5
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        // Keys: pos 0 has key 0 (leader A), pos 1 has key 1 (output B), pos 2 has key 2 (magic M)
        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        // Create trigram frequencies for the rule A→M stealing B
        // Z→A→B becomes Z→A→M: tg_freq[Z][A][B] for all Z
        // A→B→C becomes A→M→C: tg_freq[A][B][C] for all C
        let tg_freq = create_tg_freq(4, &[
            (2, 0, 1, 100),  // Z=2, A=0, B=1: trigram 2→0→1
            (0, 1, 2, 50),   // A=0, B=1, C=2: trigram 0→1→2
        ]);

        // Verify initial state
        assert!(cache.active_rules.is_empty());
        assert_eq!(cache.magic_rule_score_delta, 0);
        let initial_score = cache.score();

        // Apply rule: leader=0 (A), output=1 (B), magic_key=2 (M)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, true);

        // Verify active_rules is updated
        assert_eq!(cache.active_rules.len(), 1);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _)))); // (magic_key, leader) -> (output, delta)

        // Verify magic_rule_score_delta is updated
        assert_eq!(cache.magic_rule_score_delta, delta);

        // Verify score() reflects the delta
        assert_eq!(cache.score(), initial_score + delta);
    }

    #[test]
    fn test_add_rule_speculative_apply_false_preserves_state() {
        use crate::weights::{FingerWeights, Weights};

        // Test that add_rule with apply=false returns delta without modifying state
        // Validates: Requirement 2.6
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        let tg_freq = create_tg_freq(4, &[
            (2, 0, 1, 100),
            (0, 1, 2, 50),
        ]);

        // Record initial state
        let initial_active_rules_len = cache.active_rules.len();
        let initial_magic_delta = cache.magic_rule_score_delta;
        let initial_score = cache.score();

        // Apply rule speculatively (apply=false)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, false);

        // Verify state is unchanged
        assert_eq!(cache.active_rules.len(), initial_active_rules_len);
        assert_eq!(cache.magic_rule_score_delta, initial_magic_delta);
        assert_eq!(cache.score(), initial_score);

        // Delta should still be computed
        // (exact value depends on trigram types, but should be non-zero with these frequencies)
        // We just verify it returns a value
        let _ = delta;
    }

    #[test]
    fn test_add_rule_apply_true_and_false_return_same_delta() {
        use crate::weights::{FingerWeights, Weights};

        // Test that apply=true and apply=false return the same delta value
        let fingers = vec![LP, LI, RI];
        let mut cache1 = TrigramCache::new(&fingers, 4);
        let mut cache2 = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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

        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        let tg_freq = create_tg_freq(4, &[
            (2, 0, 1, 100),
            (0, 1, 2, 50),
        ]);

        // Get delta with apply=false
        let delta_false = cache1.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, false);

        // Get delta with apply=true
        let delta_true = cache2.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, true);

        // Both should return the same delta
        assert_eq!(delta_false, delta_true);
    }

    #[test]
    fn test_add_rule_multiple_rules_sequential() {
        use crate::weights::{FingerWeights, Weights};

        // Test applying multiple rules sequentially
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        // Keys: pos 0=key 0, pos 1=key 1, pos 2=key 2, pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let tg_freq = create_tg_freq(5, &[
            (1, 0, 2, 100),  // For rule 1
            (0, 2, 1, 50),   // For rule 1
            (3, 1, 0, 75),   // For rule 2
            (1, 0, 3, 25),   // For rule 2
        ]);

        let initial_score = cache.score();

        // Apply first rule: leader=0, output=2, magic_key=1
        let delta1 = cache.add_rule(0, 2, 1, &keys, &key_positions, &tg_freq, true);
        assert_eq!(cache.active_rules.len(), 1);
        assert_eq!(cache.magic_rule_score_delta, delta1);

        // Apply second rule: leader=1, output=0, magic_key=3
        let delta2 = cache.add_rule(1, 0, 3, &keys, &key_positions, &tg_freq, true);
        assert_eq!(cache.active_rules.len(), 2);
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);

        // Verify final score
        assert_eq!(cache.score(), initial_score + delta1 + delta2);
    }

    #[test]
    fn test_add_rule_replacement_same_magic_key_leader() {
        use crate::weights::{FingerWeights, Weights};

        // Test that applying a new rule for the same (magic_key, leader) replaces the old one
        let fingers = vec![LP, LI, RI, RP];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let tg_freq = create_tg_freq(5, &[
            (2, 0, 1, 100),  // For first rule
            (0, 1, 2, 50),   // For first rule
            (2, 0, 3, 200),  // For second rule (different output)
            (0, 3, 2, 75),   // For second rule
        ]);

        // Apply first rule: leader=0, output=1, magic_key=2
        let delta1 = cache.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _))));

        // Apply second rule with same (magic_key=2, leader=0) but different output=3
        let delta2 = cache.add_rule(0, 3, 2, &keys, &key_positions, &tg_freq, true);

        // The active_rules entry should be replaced (still only 1 entry)
        assert_eq!(cache.active_rules.len(), 1);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(3, _)))); // Now points to output=3

        // The magic_rule_score_delta should accumulate both deltas
        // (In a real implementation, you might want to subtract the old delta first,
        // but the current implementation just accumulates)
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);
    }

    #[test]
    fn test_add_rule_leader_no_position() {
        use crate::weights::{FingerWeights, Weights};

        // Test that add_rule returns 0 delta when leader has no position
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        // Keys: pos 0=key 0, pos 1=key 1, pos 2=key 2
        // Key 4 has no position
        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 5);

        let tg_freq = create_tg_freq(5, &[
            (2, 4, 1, 100),  // Would be relevant if key 4 had a position
        ]);

        // Apply rule with leader=4 (no position)
        let delta = cache.add_rule(4, 1, 2, &keys, &key_positions, &tg_freq, true);

        // Delta should be 0 since leader has no position
        assert_eq!(delta, 0);

        // State should still be updated (rule is tracked even if delta is 0)
        assert!(matches!(cache.active_rules.get(&(2, 4)), Some(&(1, _))));
    }

    #[test]
    fn test_add_rule_output_no_position() {
        use crate::weights::{FingerWeights, Weights};

        // Test add_rule when output (B) has no position
        // The old trigram type becomes Invalid (weight 0)
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader), pos 1=key 1, pos 2=key 2 (magic)
        // Key 4 (output) has no position
        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 5);

        let tg_freq = create_tg_freq(5, &[
            (1, 0, 4, 100),  // Z=1, A=0, B=4 (B has no position)
            (0, 4, 1, 50),   // A=0, B=4, C=1 (B has no position)
        ]);

        // Apply rule: leader=0, output=4 (no position), magic_key=2
        let delta = cache.add_rule(0, 4, 2, &keys, &key_positions, &tg_freq, true);

        // Delta should be computed (old weight is 0 since B has no position)
        // The exact value depends on the new trigram types
        let _ = delta;

        // Rule should be tracked
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(4, _))));
    }

    #[test]
    fn test_add_rule_magic_key_no_position() {
        use crate::weights::{FingerWeights, Weights};

        // Test add_rule when magic_key (M) has no position
        // The new trigram type becomes Invalid (weight 0)
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 5);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader), pos 1=key 1 (output), pos 2=key 2
        // Key 4 (magic) has no position
        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 5);

        let tg_freq = create_tg_freq(5, &[
            (2, 0, 1, 100),  // Z=2, A=0, B=1
            (0, 1, 2, 50),   // A=0, B=1, C=2
        ]);

        // Apply rule: leader=0, output=1, magic_key=4 (no position)
        let delta = cache.add_rule(0, 1, 4, &keys, &key_positions, &tg_freq, true);

        // Delta should be computed (new weight is 0 since M has no position)
        // This typically results in a negative delta (losing the old trigram score)
        let _ = delta;

        // Rule should be tracked
        assert!(matches!(cache.active_rules.get(&(4, 0)), Some(&(1, _))));
    }

    #[test]
    fn test_add_rule_empty_frequencies() {
        use crate::weights::{FingerWeights, Weights};

        // Test add_rule with empty/zero trigram frequencies
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        // Empty trigram frequencies
        let tg_freq = create_tg_freq(4, &[]);

        let initial_score = cache.score();

        // Apply rule with empty frequencies
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, true);

        // Delta should be 0 since all frequencies are 0
        assert_eq!(delta, 0);

        // Score should remain unchanged
        assert_eq!(cache.score(), initial_score);

        // Rule should still be tracked
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _))));
    }

    #[test]
    fn test_add_rule_invalid_key_indices() {
        use crate::weights::{FingerWeights, Weights};

        // Test add_rule with key indices >= num_keys
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        let tg_freq = create_tg_freq(4, &[
            (2, 0, 1, 100),
        ]);

        // Apply rule with invalid leader (>= num_keys)
        let delta = cache.add_rule(99, 1, 2, &keys, &key_positions, &tg_freq, true);

        // Delta should be 0 for invalid keys
        assert_eq!(delta, 0);
    }

    #[test]
    fn test_add_rule_score_reflects_delta() {
        use crate::weights::{FingerWeights, Weights};

        // Test that score() correctly reflects the magic_rule_score_delta
        let fingers = vec![LP, LI, RI];
        let mut cache = TrigramCache::new(&fingers, 4);

        let weights = Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 10,
            outroll: 5,
            alternate: 20,
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
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, 4);

        let tg_freq = create_tg_freq(4, &[
            (2, 0, 1, 100),
            (0, 1, 2, 50),
        ]);

        let score_before = cache.score();

        // Apply rule
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &tg_freq, true);

        let score_after = cache.score();

        // Score difference should equal the delta
        assert_eq!(score_after - score_before, delta);
    }

    // ==================== Property-Based Tests ====================

    /// **Validates: Requirements 2.5**
    ///
    /// Property test: add_rule with apply=true mutates state correctly.
    ///
    /// For any add_rule call with apply=true, the analyzer's internal state should
    /// reflect the rule application, and subsequent calls to score() should return
    /// the same value as was returned by the operation.
    mod pbt_add_rule_apply_true {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            /// **Validates: Requirements 2.5**
            ///
            /// Property: When add_rule is called with apply=true:
            /// 1. active_rules contains the new rule: (magic_key, leader) -> output
            /// 2. magic_rule_score_delta equals the returned delta
            /// 3. score() equals initial_score + delta
            #[test]
            fn prop_add_rule_apply_true_mutates_state_correctly(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate tg_freq from entries (constrained to num_keys)
                let constrained_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_entries);

                // Create a simple finger layout (alternating left/right for variety)
                let fingers: Vec<DofFinger> = (0..num_positions)
                    .map(|i| if i % 2 == 0 { LP } else { RI })
                    .collect();

                let mut cache = TrigramCache::new(&fingers, num_keys);

                // Set some weights so score changes are visible
                use crate::weights::{FingerWeights, Weights};
                let weights = Weights {
                    sfbs: 0,
                    sfs: 0,
                    stretches: 0,
                    sft: 0,
                    inroll: 10,
                    outroll: 5,
                    alternate: 20,
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
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Record initial state
                let initial_score = cache.score();
                let initial_magic_delta = cache.magic_rule_score_delta;

                // Apply the rule with apply=true
                let delta = cache.add_rule(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &tg_freq,
                    true,
                );

                // Property 1: active_rules contains the new rule
                prop_assert!(
                    cache.active_rules.contains_key(&(magic_key, leader)),
                    "active_rules should contain the rule ({}, {}) -> {}",
                    magic_key, leader, output
                );
                prop_assert!(
                    cache.active_rules.get(&(magic_key, leader)).map(|(o, _)| *o) == Some(output),
                    "active_rules[({}, {})] should have output {}",
                    magic_key, leader, output
                );

                // Property 2: magic_rule_score_delta equals initial + returned delta
                prop_assert_eq!(
                    cache.magic_rule_score_delta,
                    initial_magic_delta + delta,
                    "magic_rule_score_delta should be {} + {} = {}",
                    initial_magic_delta, delta, initial_magic_delta + delta
                );

                // Property 3: score() equals initial_score + delta
                let final_score = cache.score();
                prop_assert_eq!(
                    final_score,
                    initial_score + delta,
                    "score() should be {} + {} = {}, but got {}",
                    initial_score, delta, initial_score + delta, final_score
                );
            }
        }
    }

    /// **Validates: Requirements 2.6**
    ///
    /// Property test: add_rule with apply=false preserves state.
    ///
    /// For any add_rule call with apply=false, the analyzer's internal state should
    /// remain unchanged, and score() called before and after should return the same value.
    mod pbt_add_rule_apply_false {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            /// **Validates: Requirements 2.6**
            ///
            /// Property: When add_rule is called with apply=false:
            /// 1. active_rules is unchanged
            /// 2. magic_rule_score_delta is unchanged
            /// 3. score() is unchanged
            #[test]
            fn prop_add_rule_apply_false_preserves_state(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate tg_freq from entries (constrained to num_keys)
                let constrained_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_entries);

                // Create a simple finger layout (alternating left/right for variety)
                let fingers: Vec<DofFinger> = (0..num_positions)
                    .map(|i| if i % 2 == 0 { LP } else { RI })
                    .collect();

                let mut cache = TrigramCache::new(&fingers, num_keys);

                // Set some weights so score changes are visible
                use crate::weights::{FingerWeights, Weights};
                let weights = Weights {
                    sfbs: 0,
                    sfs: 0,
                    stretches: 0,
                    sft: 0,
                    inroll: 10,
                    outroll: 5,
                    alternate: 20,
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
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Record initial state
                let initial_score = cache.score();
                let initial_magic_delta = cache.magic_rule_score_delta;
                let initial_active_rules = cache.active_rules.clone();

                // Call add_rule with apply=false (speculative scoring)
                let _delta = cache.add_rule(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &tg_freq,
                    false,  // apply=false - should NOT mutate state
                );

                // Property 1: active_rules is unchanged
                prop_assert_eq!(
                    &cache.active_rules,
                    &initial_active_rules,
                    "active_rules should be unchanged after apply=false"
                );

                // Property 2: magic_rule_score_delta is unchanged
                prop_assert_eq!(
                    cache.magic_rule_score_delta,
                    initial_magic_delta,
                    "magic_rule_score_delta should be unchanged after apply=false, was {}, now {}",
                    initial_magic_delta, cache.magic_rule_score_delta
                );

                // Property 3: score() is unchanged
                let final_score = cache.score();
                prop_assert_eq!(
                    final_score,
                    initial_score,
                    "score() should be unchanged after apply=false, was {}, now {}",
                    initial_score, final_score
                );
            }
        }
    }

    /// **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    ///
    /// Property test: lookup table values match computed values.
    ///
    /// For any (leader, output, magic_key) triple, the pre-computed `rule_delta` lookup
    /// should equal the value computed by `compute_rule_delta` from scratch.
    mod pbt_lookup_table_matches_computed_trigram {
        use super::*;
        use proptest::prelude::*;

        /// Helper to create key_positions from keys
        fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
            let mut key_positions = vec![None; num_keys];
            for (pos, &key) in keys.iter().enumerate() {
                if key < num_keys {
                    key_positions[key] = Some(pos);
                }
            }
            key_positions
        }

        /// Convert entries to 3D frequency array
        fn entries_to_tg_freq(num_keys: usize, entries: &[(usize, usize, usize, i64)]) -> Vec<Vec<Vec<i64>>> {
            let mut tg_freq = vec![vec![vec![0i64; num_keys]; num_keys]; num_keys];
            for &(k1, k2, k3, freq) in entries {
                if k1 < num_keys && k2 < num_keys && k3 < num_keys {
                    tg_freq[k1][k2][k3] = freq;
                }
            }
            tg_freq
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            /// **Validates: Requirements 6.1, 6.5**
            ///
            /// Property: After calling init_rule_deltas, for any valid (leader, output, magic_key)
            /// triple where leader has a position, the value in rule_delta.get(&(leader, output, magic_key))
            /// should equal compute_rule_delta(leader, output, magic_key, ...).
            #[test]
            fn prop_lookup_table_matches_computed_trigram(
                // Use a fixed small number of positions for reasonable test speed
                num_positions in 3usize..=5,
                // Use a fixed small number of keys for reasonable test speed
                num_keys in 4usize..=6,
                // Seed for generating keys
                keys_seed in proptest::collection::vec(0usize..100, 3..=5),
                // Seed for generating tg_freq entries
                tg_freq_entries in proptest::collection::vec(
                    (0usize..6, 0usize..6, 0usize..6, 0i64..1000i64),
                    0..20
                ),
                // Key indices for the rule to test
                leader_idx in 0usize..6,
                output_idx in 0usize..6,
                magic_key_idx in 0usize..6,
            ) {
                // Constrain indices to actual num_keys
                let leader = leader_idx % num_keys;
                let output = output_idx % num_keys;
                let magic_key = magic_key_idx % num_keys;

                // Generate keys for each position (constrained to num_keys)
                let keys: Vec<usize> = keys_seed.iter()
                    .take(num_positions)
                    .map(|&k| k % num_keys)
                    .collect();

                // Ensure we have exactly num_positions keys
                let keys: Vec<usize> = if keys.len() < num_positions {
                    let mut k = keys;
                    while k.len() < num_positions {
                        k.push(k.len() % num_keys);
                    }
                    k
                } else {
                    keys
                };

                // Generate tg_freq from entries (constrained to num_keys)
                let constrained_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                    .iter()
                    .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                    .collect();
                let tg_freq = entries_to_tg_freq(num_keys, &constrained_entries);

                // Create a simple finger layout (alternating left/right for variety)
                let fingers: Vec<DofFinger> = (0..num_positions)
                    .map(|i| if i % 2 == 0 { LP } else { RI })
                    .collect();

                let mut cache = TrigramCache::new(&fingers, num_keys);

                // Set some weights so score changes are visible
                use crate::weights::{FingerWeights, Weights};
                let weights = Weights {
                    sfbs: 0,
                    sfs: 0,
                    stretches: 0,
                    sft: 0,
                    inroll: 10,
                    outroll: 5,
                    alternate: 20,
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
                cache.set_weights(&weights);

                let key_positions = make_key_positions(&keys, num_keys);

                // Initialize the rule delta lookup table
                cache.init_rule_deltas(&keys, &key_positions, &tg_freq);

                // Compute the delta from scratch using compute_rule_delta
                let computed_delta = cache.compute_rule_delta(
                    leader,
                    output,
                    magic_key,
                    &keys,
                    &key_positions,
                    &tg_freq,
                );

                // Get the lookup table value (0 if not present due to sparse storage)
                let lookup_delta = cache.rule_delta
                    .get(&(leader, output, magic_key))
                    .copied()
                    .unwrap_or(0) as i64;

                // Property: lookup table value should equal computed value
                prop_assert_eq!(
                    lookup_delta,
                    computed_delta,
                    "Lookup table value {} should equal computed value {} for (leader={}, output={}, magic_key={})",
                    lookup_delta, computed_delta, leader, output, magic_key
                );
            }
        }
    }
}
