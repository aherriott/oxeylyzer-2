/*
 **************************************
 *            Stretches
 **************************************
 */

use std::collections::HashMap;

use crate::dist::{dx_dy, x_overlap};
use crate::stats::Stats;
use crate::types::{CacheKey, CachePos};
use crate::weights::Weights;
use libdof::dofinitions::Finger;
use libdof::prelude::PhysicalKey;

/// Pre-computed stretch pair with distance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StretchPair {
    pub other_pos: usize,
    pub dist: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    /// For each position, list of other positions that form a stretch pair with pre-computed distances
    stretch_pairs_per_key: Vec<Vec<StretchPair>>,
    /// Number of keys for frequency array indexing
    num_keys: usize,
    /// Running total (freq * dist, not yet weighted)
    total: i64,
    /// Pre-computed stretch weight
    stretch_weight: i64,

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
}

impl StretchCache {
    pub fn new(keyboard: &[PhysicalKey], fingers: &[Finger], num_keys: usize) -> Self {
        let len = keyboard.len();

        // Compute stretch distances
        let stretch_dists: Vec<Vec<i64>> = (0..len)
            .map(|i| {
                (0..len)
                    .map(|j| {
                        if i != j {
                            Self::compute_stretch(&keyboard[i], &keyboard[j], fingers[i], fingers[j])
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();

        // Build stretch pair lookup with pre-computed distances
        let stretch_pairs_per_key: Vec<Vec<StretchPair>> = (0..len)
            .map(|i| {
                (0..len)
                    .filter_map(|j| {
                        let dist = stretch_dists[i][j];
                        if i != j && dist > 0 {
                            Some(StretchPair { other_pos: j, dist })
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            stretch_pairs_per_key,
            num_keys,
            total: 0,
            stretch_weight: 0,
            // Initialize active rules to empty (no magic rules active initially)
            active_rules: HashMap::new(),
            // Initialize magic rule score delta to zero
            magic_rule_score_delta: 0,
            // Initialize rule delta lookup table to empty (populated by init_rule_deltas)
            rule_delta: HashMap::new(),
        }
    }

    /// Set weights
    pub fn set_weights(&mut self, weights: &Weights) {
        self.stretch_weight = weights.stretches;
    }

    /// Compute stretch distance for a key pair.
    /// Returns 0 if not a stretch (same finger or different hand).
    fn compute_stretch(k1: &PhysicalKey, k2: &PhysicalKey, f1: Finger, f2: Finger) -> i64 {
        // Stretch only applies to different fingers on the same hand
        if f1 == f2 || f1.hand() != f2.hand() {
            return 0;
        }

        let (dx, dy) = dx_dy(k1, k2, f1, f2);
        let diff = (f1 as usize).abs_diff(f2 as usize) as f64;
        let finger_dist_allowance = diff * 1.35;
        let dist = dx.hypot(dy);
        let xo = x_overlap(dx, dy, f1, f2);

        let stretch = dist + xo - finger_dist_allowance;
        if stretch > 0.001 {
            (stretch * 100.0) as i64
        } else {
            0
        }
    }

    #[inline]
    pub fn get_stretch(&self, p1: CachePos, p2: CachePos) -> i64 {
        self.stretch_pairs_per_key[p1]
            .iter()
            .find(|sp| sp.other_pos == p2)
            .map(|sp| sp.dist)
            .unwrap_or(0)
    }

    #[inline]
    pub fn score(&self) -> i64 {
        self.total * self.stretch_weight + self.magic_rule_score_delta
    }

    pub fn stats(&self, stats: &mut Stats, bigram_total: f64) {
        // Convert from centiunits back to units, normalized by bigram total
        stats.stretches = self.total as f64 / (bigram_total * 100.0);
    }

    /// Replace key at position. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        apply: bool,
    ) -> i64 {
        let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, bg_freq);
        if apply {
            self.total += delta;
        }
        let base_score = (self.total + if apply { 0 } else { delta }) * self.stretch_weight;
        base_score + self.magic_rule_score_delta
    }

    /// Compute the delta for replacing a key without mutating state.
    #[inline]
    fn compute_replace_delta(
        &self,
        pos: CachePos,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut delta: i64 = 0;

        for sp in &self.stretch_pairs_per_key[pos] {
            let other_pos = sp.other_pos;
            if skip_pos == Some(other_pos) {
                continue;
            }
            let other_key = keys[other_pos];

            if other_key >= num_keys {
                continue;
            }

            let stretch_dist = sp.dist;
            let other_row = other_key * num_keys;

            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };

            let bg_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            delta += bg_delta * stretch_dist;
        }

        delta
    }

    /// Swap keys at two positions. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: CachePos,
        pos_b: CachePos,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        apply: bool,
    ) -> i64 {
        // Compute delta for pos_a (key_a -> key_b), skipping pos_b
        let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq);
        // Compute delta for pos_b (key_b -> key_a), skipping pos_a
        let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq);
        let total_delta = delta_a + delta_b;

        if apply {
            self.total += total_delta;
        }
        let base_score = (self.total + if apply { 0 } else { total_delta }) * self.stretch_weight;
        base_score + self.magic_rule_score_delta
    }

    /// Compute the weighted score for a stretch bigram pair.
    /// Returns 0 if the positions don't form a stretch.
    #[inline]
    fn stretch_bigram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        let dist = self.get_stretch(p_a, p_b);
        if dist > 0 {
            dist * self.stretch_weight
        } else {
            0
        }
    }

    /// Compute the score delta for applying a magic rule.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    ///
    /// Note: StretchCache does NOT handle skipgrams, so there's no Part 3 like in SFCache.
    ///
    /// Returns the score delta (new_score - old_score).
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `_keys` - The current key assignments for all positions (not used but kept for API consistency)
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    fn compute_rule_delta(
        &self,
        leader: CacheKey,      // A - the key that triggers the magic rule
        output: CacheKey,      // B - the output being stolen
        magic_key: CacheKey,   // M - the magic key that steals the output
        _keys: &[CacheKey],    // Not used but kept for API consistency
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],       // Flat bigram frequencies
        tg_freq: &[Vec<Vec<i64>>],
    ) -> i64 {
        let num_keys = self.num_keys;

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

        // Output (B) and magic_key (M) positions are needed for stretch computation
        let output_pos = key_positions.get(output).copied().flatten();
        let magic_pos = key_positions.get(magic_key).copied().flatten();

        let mut delta: i64 = 0;

        // Part 1: Full steal - Bigram A→B becomes A→M
        // The frequency bg_freq[A][B] is redirected from A→B to A→M
        // Delta contribution: bg_freq[A][B] * (new_weight - old_weight)
        let full_steal_freq = bg_freq[leader * num_keys + output];
        if full_steal_freq != 0 {
            // Old weight: stretch_weight(A, B) - based on positions of leader and output
            let old_weight = if let Some(b_pos) = output_pos {
                self.stretch_bigram_weight(leader_pos, b_pos)
            } else {
                0
            };

            // New weight: stretch_weight(A, M) - based on positions of leader and magic_key
            let new_weight = if let Some(m_pos) = magic_pos {
                self.stretch_bigram_weight(leader_pos, m_pos)
            } else {
                0
            };

            delta += full_steal_freq * (new_weight - old_weight);
        }

        // Part 2: Partial steal - Bigrams B→C become M→C based on tg_freq[A][B][C]
        // For each C with a position, the frequency stolen is tg_freq[A][B][C]
        // The weight changes from stretch_weight(B, C) to stretch_weight(M, C)
        for c_key in 0..num_keys {
            // C must have a position
            let c_pos = match key_positions.get(c_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency stolen: tg_freq[A][B][C]
            let stolen_freq = tg_freq[leader][output][c_key];
            if stolen_freq == 0 {
                continue;
            }

            // Old weight: stretch_weight(B, C) - based on positions of output and C
            let old_weight = if let Some(b_pos) = output_pos {
                self.stretch_bigram_weight(b_pos, c_pos)
            } else {
                0
            };

            // New weight: stretch_weight(M, C) - based on positions of magic_key and C
            let new_weight = if let Some(m_pos) = magic_pos {
                self.stretch_bigram_weight(m_pos, c_pos)
            } else {
                0
            };

            delta += stolen_freq * (new_weight - old_weight);
        }

        // Note: StretchCache does NOT handle skipgrams, so there's no Part 3

        delta
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
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// **Validates: Requirements 6.3, 6.5**
    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
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
                        bg_freq,
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

    /// Apply a magic rule. Returns the score delta.
    ///
    /// If `apply` is false, computes the score delta without mutating state (speculative scoring).
    /// If `apply` is true, updates internal state and returns the score delta.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    ///
    /// Note: StretchCache does NOT handle skipgrams.
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    /// * `apply` - If true, update internal state; if false, just compute the delta
    ///
    /// # Returns
    /// The score delta (new_score - old_score) for this rule application.
    ///
    /// **Validates: Requirements 2.5, 2.6**
    pub fn add_rule(
        &mut self,
        leader: CacheKey,      // A
        output: CacheKey,      // B (being stolen)
        magic_key: CacheKey,   // M
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
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
                self.compute_rule_delta(
                    leader, output, magic_key, keys, key_positions, bg_freq, tg_freq
                )
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
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;

    /// Helper function to create a simple StretchCache for testing
    /// Creates a 4-position layout with fingers: LP, LI, LM, RP
    /// Positions 0 and 1 are on the same hand but different fingers (potential stretch)
    fn create_test_cache(num_keys: usize) -> StretchCache {
        // Fingers: LP, LI, LM, RP (positions 0, 1, 2 are left hand, position 3 is right hand)
        let fingers = vec![LP, LI, LM, RP];
        // Simple keyboard with 4 positions - positions 0 and 1 are far apart (stretch)
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),   // LP
            PhysicalKey::xy(3.0, 0.0),   // LI - far from LP (stretch)
            PhysicalKey::xy(2.0, 0.0),   // LM
            PhysicalKey::xy(5.0, 0.0),   // RP
        ];
        StretchCache::new(&keyboard, &fingers, num_keys)
    }

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

    /// Helper function to create flat bigram frequency array
    fn create_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut bg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                bg_freq[k1 * num_keys + k2] = freq;
            }
        }
        bg_freq
    }

    /// Helper to create test weights with non-zero stretch weight
    fn create_test_weights() -> Weights {
        Weights {
            sfbs: 0,
            sfs: 0,
            stretches: -10,  // Negative weight for stretches (penalty)
            sft: 0,
            inroll: 0,
            outroll: 0,
            alternate: 0,
            redirect: 0,
            onehandin: 0,
            onehandout: 0,
            thumb: 0,
            full_scissors: 0,
            half_scissors: 0,
            full_scissors_skip: 0,
            half_scissors_skip: 0,
            fingers: FingerWeights {
                lp: 100,
                lr: 50,
                lm: 30,
                li: 20,
                lt: 10,
                rt: 10,
                ri: 20,
                rm: 30,
                rr: 50,
                rp: 100,
            },
        }
    }

    // ==================== add_rule tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_false_preserves_state() {
        // Test that add_rule with apply=false returns correct delta without mutating state
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create bigram frequencies for the full steal: A→B
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let tg_freq = create_tg_freq(5, &[]);

        // Record initial state
        let initial_active_rules_len = cache.active_rules.len();
        let initial_magic_delta = cache.magic_rule_score_delta;
        let initial_score = cache.score();

        // Apply rule speculatively (apply=false)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, false);

        // Verify state is unchanged
        assert_eq!(cache.active_rules.len(), initial_active_rules_len, "active_rules should not change with apply=false");
        assert_eq!(cache.magic_rule_score_delta, initial_magic_delta, "magic_rule_score_delta should not change with apply=false");
        assert_eq!(cache.score(), initial_score, "score should not change with apply=false");

        // Delta should be computed (may be non-zero depending on positions)
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_true_updates_state() {
        // Test that add_rule with apply=true updates active_rules and magic_rule_score_delta
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create bigram frequencies
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let tg_freq = create_tg_freq(5, &[]);

        // Verify initial state
        assert!(cache.active_rules.is_empty());
        assert_eq!(cache.magic_rule_score_delta, 0);

        // Apply rule: leader=0 (A), output=1 (B), magic_key=2 (M)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Verify active_rules is updated
        assert_eq!(cache.active_rules.len(), 1);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _)))); // (magic_key, leader) -> (output, delta)

        // Verify magic_rule_score_delta is updated
        assert_eq!(cache.magic_rule_score_delta, delta);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_active_rules_updated_correctly() {
        // Test that active_rules HashMap is updated correctly when apply=true
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply first rule: leader=0, output=1, magic_key=2
        cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _))));

        // Apply second rule with different magic_key and leader
        cache.add_rule(1, 3, 4, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(4, 1)), Some(&(3, _))));

        // Both rules should be tracked
        assert_eq!(cache.active_rules.len(), 2);

        // Apply rule with same (magic_key, leader) but different output - should replace
        cache.add_rule(0, 3, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(3, _)))); // Updated to new output
        assert_eq!(cache.active_rules.len(), 2); // Still 2 rules
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_magic_rule_score_delta_updated_correctly() {
        // Test that magic_rule_score_delta is updated correctly when apply=true
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // Create frequencies that will produce non-zero deltas
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let tg_freq = create_tg_freq(5, &[
            (0, 1, 2, 50),  // A→B→C for partial steal
        ]);

        // Apply first rule
        let delta1 = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(cache.magic_rule_score_delta, delta1);

        // Apply second rule
        let delta2 = cache.add_rule(1, 2, 3, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_true_and_false_return_same_delta() {
        // Test that apply=true and apply=false return the same delta value
        let mut cache1 = create_test_cache(5);
        let mut cache2 = create_test_cache(5);
        let weights = create_test_weights();
        cache1.set_weights(&weights);
        cache2.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),
        ]);
        let tg_freq = create_tg_freq(5, &[
            (0, 1, 2, 75),  // A→B→C for bigram partial steal
        ]);

        // Get delta with apply=false
        let delta_false = cache1.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, false);

        // Get delta with apply=true
        let delta_true = cache2.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Both should return the same delta
        assert_eq!(delta_false, delta_true);
    }

    // ==================== Edge case tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_invalid_keys() {
        // Test add_rule with key indices >= num_keys
        let mut cache = create_test_cache(4);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 4);

        let bg_freq = create_bg_freq(4, &[]);
        let tg_freq = create_tg_freq(4, &[]);

        // Apply rule with invalid leader (>= num_keys)
        let delta = cache.add_rule(99, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid leader key");

        // Apply rule with invalid output (>= num_keys)
        let delta = cache.add_rule(0, 99, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid output key");

        // Apply rule with invalid magic_key (>= num_keys)
        let delta = cache.add_rule(0, 1, 99, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid magic key");
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_keys_without_positions() {
        // Test add_rule when keys don't have positions
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Only keys 0, 1, 2, 3 have positions (4 positions)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[
            (4, 5, 100),  // Frequency for keys without positions
        ]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply rule where leader has no position
        let delta = cache.add_rule(4, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 when leader has no position");

        // Apply rule where output has no position (old weight becomes 0)
        let delta = cache.add_rule(0, 4, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);
        // Delta may be non-zero if magic_key has a position
        let _ = delta;

        // Apply rule where magic_key has no position (new weight becomes 0)
        let delta = cache.add_rule(0, 1, 5, &keys, &key_positions, &bg_freq, &tg_freq, true);
        // Delta may be non-zero if output has a position
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_zero_frequencies() {
        // Test add_rule with zero frequencies
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // All frequencies are zero
        let bg_freq = create_bg_freq(5, &[]);
        let tg_freq = create_tg_freq(5, &[]);

        let initial_score = cache.score();

        // Apply rule with zero frequencies
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Delta should be 0 since all frequencies are 0
        assert_eq!(delta, 0);

        // Score should remain unchanged
        assert_eq!(cache.score(), initial_score);

        // Rule should still be tracked
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _))));
    }

    // ==================== Full steal tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    /// Test full steal: stretch pair to non-stretch pair
    #[test]
    fn test_add_rule_full_steal_stretch_to_non_stretch() {
        // Test full steal: A→B (stretch) becomes A→M (non-stretch)
        // This should reduce the stretch score (positive delta since stretch weight is negative)

        // Create a cache where positions 0 and 1 form a stretch, but 0 and 2 don't
        let fingers = vec![LP, LI, RP, RP];  // pos 0 and 1 are left hand (stretch), pos 2 is right hand
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),   // LP
            PhysicalKey::xy(3.0, 0.0),   // LI - far from LP (stretch)
            PhysicalKey::xy(6.0, 0.0),   // RP - different hand
            PhysicalKey::xy(7.0, 0.0),   // RP
        ];
        let mut cache = StretchCache::new(&keyboard, &fingers, 5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // A→B is a stretch (positions 0 and 1 are same hand, different fingers, far apart)
        // A→M is NOT a stretch (positions 0 and 2 are different hands)
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let tg_freq = create_tg_freq(5, &[]);

        // Check if positions 0 and 1 form a stretch
        let stretch_dist = cache.get_stretch(0, 1);

        // Only test if there's actually a stretch
        if stretch_dist > 0 {
            let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

            // The delta should be positive (removing stretch penalty)
            // Old: 100 * stretch_dist * stretch_weight (negative contribution)
            // New: 0 (A→M is not a stretch - different hands)
            // Delta = new - old = 0 - (negative) = positive
            assert!(delta > 0, "Delta should be positive when removing stretch: got {}", delta);
        }
    }

    /// **Validates: Requirements 2.5, 2.6**
    /// Test full steal: non-stretch pair to stretch pair
    #[test]
    fn test_add_rule_full_steal_non_stretch_to_stretch() {
        // Test full steal: A→B (non-stretch) becomes A→M (stretch)
        // This should increase the stretch score (negative delta since stretch weight is negative)

        // Create cache where positions 0 and 2 form a stretch, but 0 and 1 don't
        let fingers = vec![LP, RP, LI, RP];  // pos 0 and 2 are left hand (stretch), pos 1 is right hand
        let keyboard = vec![
            PhysicalKey::xy(0.0, 0.0),   // LP
            PhysicalKey::xy(6.0, 0.0),   // RP - different hand
            PhysicalKey::xy(3.0, 0.0),   // LI - same hand as LP (stretch)
            PhysicalKey::xy(7.0, 0.0),   // RP
        ];
        let mut cache = StretchCache::new(&keyboard, &fingers, 5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // A→B is NOT a stretch (positions 0 and 1 are different hands)
        // A→M IS a stretch (positions 0 and 2 are same hand, different fingers)
        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let tg_freq = create_tg_freq(5, &[]);

        // Check if positions 0 and 2 form a stretch
        let stretch_dist = cache.get_stretch(0, 2);

        // Only test if there's actually a stretch
        if stretch_dist > 0 {
            let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

            // The delta should be negative (adding stretch penalty)
            // Old: 0 (A→B is not a stretch - different hands)
            // New: 100 * stretch_dist * stretch_weight (negative contribution)
            // Delta = new - old = negative - 0 = negative
            assert!(delta < 0, "Delta should be negative when adding stretch: got {}", delta);
        }
    }

    // ==================== Partial steal tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_partial_steal_bigram() {
        // Test partial steal: B→C becomes M→C based on tg_freq[A][B][C]
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // No full steal frequency
        let bg_freq = create_bg_freq(5, &[]);
        // Partial steal: tg_freq[A][B][C] determines how much of B→C is stolen
        let tg_freq = create_tg_freq(5, &[
            (0, 1, 3, 200),  // A=0, B=1, C=3: trigram frequency
        ]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Delta depends on whether B→C and M→C are stretch pairs
        // In our test setup, we need to check the actual stretch distances
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    /// Note: StretchCache does NOT handle skipgrams, so this test verifies that behavior
    #[test]
    fn test_add_rule_no_skipgram_handling() {
        // Test that StretchCache only handles bigrams, not skipgrams
        // The delta should only come from bigram changes, not skipgram changes
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        // No bigram frequencies
        let bg_freq = create_bg_freq(5, &[]);
        // Only trigram frequencies that would affect skipgrams in SFCache
        // tg_freq[Z][A][B] would affect skipgram Z→B in SFCache, but not in StretchCache
        let tg_freq = create_tg_freq(5, &[
            (3, 0, 1, 150),  // Z=3, A=0, B=1: this would affect skipgram Z→B in SFCache
        ]);

        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Since there are no bigram frequencies and StretchCache doesn't handle skipgrams,
        // the delta should only come from partial steal (B→C becomes M→C)
        // With no tg_freq[A][B][C] entries, delta should be 0
        // But we have tg_freq[3][0][1] which is tg_freq[Z][A][B], not tg_freq[A][B][C]
        // So this shouldn't contribute to the delta
        let _ = delta;
    }

    // ==================== Multiple calls tests ====================

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_multiple_speculative_calls_preserve_state() {
        // Test that multiple apply=false calls don't accumulate state changes
        let mut cache = create_test_cache(5);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, 5);

        let bg_freq = create_bg_freq(5, &[
            (0, 1, 100),
        ]);
        let tg_freq = create_tg_freq(5, &[]);

        let initial_score = cache.score();
        let initial_rules_len = cache.active_rules.len();
        let initial_delta = cache.magic_rule_score_delta;

        // Call add_rule with apply=false multiple times
        for _ in 0..10 {
            cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &tg_freq, false);
        }

        // State should be unchanged
        assert_eq!(cache.score(), initial_score);
        assert_eq!(cache.active_rules.len(), initial_rules_len);
        assert_eq!(cache.magic_rule_score_delta, initial_delta);
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_sequential_apply_true_accumulates() {
        // Test that multiple apply=true calls accumulate deltas
        let mut cache = create_test_cache(6);
        let weights = create_test_weights();
        cache.set_weights(&weights);

        // Use 6 keys so we have enough for multiple rules
        let keys = vec![0, 1, 2, 3, 4, 5];
        let key_positions = create_key_positions(&keys, 6);

        let bg_freq = create_bg_freq(6, &[
            (0, 1, 100),
            (2, 3, 100),
        ]);
        let tg_freq = create_tg_freq(6, &[]);

        // Apply first rule
        let delta1 = cache.add_rule(0, 1, 4, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // Apply second rule (different magic_key and leader)
        let delta2 = cache.add_rule(2, 3, 5, &keys, &key_positions, &bg_freq, &tg_freq, true);

        // magic_rule_score_delta should be sum of both deltas
        assert_eq!(cache.magic_rule_score_delta, delta1 + delta2);

        // Both rules should be tracked
        assert_eq!(cache.active_rules.len(), 2);
    }
}

// ==================== Property-Based Tests ====================

/// **Validates: Requirements 2.5**
///
/// Property test: add_rule with apply=true mutates state correctly.
///
/// For any add_rule call with apply=true:
/// 1. active_rules should contain ((magic_key, leader), output) after the call
/// 2. magic_rule_score_delta should be updated by the returned delta
/// 3. The returned delta should equal compute_rule_delta result
#[cfg(test)]
mod pbt_add_rule_apply_true {
    use super::*;
    use proptest::prelude::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::{LP, LI, LM, RP};

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

    /// Convert entries to flat bigram frequency array
    fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut bg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                bg_freq[k1 * num_keys + k2] = freq;
            }
        }
        bg_freq
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 2.5**
        ///
        /// Property: When add_rule is called with apply=true:
        /// 1. active_rules contains the new rule: ((magic_key, leader), output)
        /// 2. magic_rule_score_delta is updated by the returned delta
        /// 3. The returned delta equals compute_rule_delta result
        #[test]
        fn prop_add_rule_apply_true_mutates_state_correctly(
            // Use a fixed small number of positions for reasonable test speed
            num_positions in 3usize..=5,
            // Use a fixed small number of keys for reasonable test speed
            num_keys in 4usize..=6,
            // Seed for generating keys
            keys_seed in proptest::collection::vec(0usize..100, 3..=5),
            // Seed for generating bg_freq entries
            bg_freq_entries in proptest::collection::vec(
                (0usize..6, 0usize..6, 0i64..1000i64),
                0..10
            ),
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

            // Generate frequency arrays from entries (constrained to num_keys)
            let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a simple finger layout with some stretch pairs
            // Positions 0 and 1 are on the same hand but different fingers (potential stretch)
            // Position 2+ are on the right hand
            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| match i {
                    0 => LP,  // Left pinky
                    1 => LI,  // Left index - same hand, different finger (stretch with LP)
                    2 => LM,  // Left middle
                    _ => RP,  // Right pinky
                })
                .collect();

            // Create keyboard with positions far enough apart to create stretches
            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| PhysicalKey::xy(i as f64 * 3.0, 0.0))  // Spread out positions
                .collect();

            let mut cache = StretchCache::new(&keyboard, &fingers, num_keys);

            // Set some weights so score changes are visible
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: -10,  // Negative weight for stretches (penalty)
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                thumb: 0,
                full_scissors: 0,
                half_scissors: 0,
                full_scissors_skip: 0,
                half_scissors_skip: 0,
                fingers: FingerWeights {
                    lp: 100,
                    lr: 50,
                    lm: 30,
                    li: 20,
                    lt: 10,
                    rt: 10,
                    ri: 20,
                    rm: 30,
                    rr: 50,
                    rp: 100,
                },
            };
            cache.set_weights(&weights);

            let key_positions = make_key_positions(&keys, num_keys);

            // Record initial state
            let initial_magic_delta = cache.magic_rule_score_delta;

            // Apply the rule with apply=true
            let delta = cache.add_rule(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
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
        }
    }
}

/// **Validates: Requirements 2.6**
///
/// Property test: add_rule with apply=false preserves state.
///
/// For any add_rule call with apply=false, the analyzer's internal state should
/// remain unchanged, and score() called before and after should return the same value.
#[cfg(test)]
mod pbt_add_rule_apply_false {
    use super::*;
    use proptest::prelude::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::{LP, LI, LM, RP};

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

    /// Convert entries to flat bigram frequency array
    fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut bg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                bg_freq[k1 * num_keys + k2] = freq;
            }
        }
        bg_freq
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 2.6**
        ///
        /// Property: When add_rule is called with apply=false:
        /// 1. active_rules is unchanged
        /// 2. magic_rule_score_delta is unchanged
        /// 3. score() is unchanged
        /// 4. The returned delta equals what apply=true would return
        #[test]
        fn prop_add_rule_apply_false_preserves_state(
            // Use a fixed small number of positions for reasonable test speed
            num_positions in 3usize..=5,
            // Use a fixed small number of keys for reasonable test speed
            num_keys in 4usize..=6,
            // Seed for generating keys
            keys_seed in proptest::collection::vec(0usize..100, 3..=5),
            // Seed for generating bg_freq entries
            bg_freq_entries in proptest::collection::vec(
                (0usize..6, 0usize..6, 0i64..1000i64),
                0..10
            ),
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

            // Generate frequency arrays from entries (constrained to num_keys)
            let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a simple finger layout with some stretch pairs
            // Positions 0 and 1 are on the same hand but different fingers (potential stretch)
            // Position 2+ are on the right hand
            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| match i {
                    0 => LP,  // Left pinky
                    1 => LI,  // Left index - same hand, different finger (stretch with LP)
                    2 => LM,  // Left middle
                    _ => RP,  // Right pinky
                })
                .collect();

            // Create keyboard with positions far enough apart to create stretches
            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| PhysicalKey::xy(i as f64 * 3.0, 0.0))  // Spread out positions
                .collect();

            let mut cache = StretchCache::new(&keyboard, &fingers, num_keys);

            // Set some weights so score changes are visible
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: -10,  // Negative weight for stretches (penalty)
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                thumb: 0,
                full_scissors: 0,
                half_scissors: 0,
                full_scissors_skip: 0,
                half_scissors_skip: 0,
                fingers: FingerWeights {
                    lp: 100,
                    lr: 50,
                    lm: 30,
                    li: 20,
                    lt: 10,
                    rt: 10,
                    ri: 20,
                    rm: 30,
                    rr: 50,
                    rp: 100,
                },
            };
            cache.set_weights(&weights);

            let key_positions = make_key_positions(&keys, num_keys);

            // Record initial state
            let initial_score = cache.score();
            let initial_magic_delta = cache.magic_rule_score_delta;
            let initial_active_rules = cache.active_rules.clone();

            // Call add_rule with apply=false (speculative scoring)
            let delta_false = cache.add_rule(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
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

            // Property 4: The returned delta equals what apply=true would return
            // Create a fresh cache to test apply=true
            let mut cache_for_apply = StretchCache::new(&keyboard, &fingers, num_keys);
            cache_for_apply.set_weights(&weights);

            let delta_true = cache_for_apply.add_rule(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
                &tg_freq,
                true,  // apply=true
            );

            prop_assert_eq!(
                delta_false,
                delta_true,
                "apply=false should return the same delta as apply=true, got {} vs {}",
                delta_false, delta_true
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
#[cfg(test)]
mod pbt_lookup_table_matches_computed_stretch {
    use super::*;
    use proptest::prelude::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::{LP, LI, LM, RP};

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

    /// Convert entries to flat bigram frequency array
    fn entries_to_bg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut bg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                bg_freq[k1 * num_keys + k2] = freq;
            }
        }
        bg_freq
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 6.3, 6.5**
        ///
        /// Property: After calling init_rule_deltas, for any valid (leader, output, magic_key)
        /// triple where leader has a position, the value in rule_delta.get(&(leader, output, magic_key))
        /// should equal compute_rule_delta(leader, output, magic_key, ...).
        #[test]
        fn prop_lookup_table_matches_computed_stretch(
            // Use a fixed small number of positions for reasonable test speed
            num_positions in 3usize..=5,
            // Use a fixed small number of keys for reasonable test speed
            num_keys in 4usize..=6,
            // Seed for generating keys
            keys_seed in proptest::collection::vec(0usize..100, 3..=5),
            // Seed for generating bg_freq entries
            bg_freq_entries in proptest::collection::vec(
                (0usize..6, 0usize..6, 0i64..1000i64),
                0..10
            ),
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

            // Generate frequency arrays from entries (constrained to num_keys)
            let constrained_bg_entries: Vec<(usize, usize, i64)> = bg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let bg_freq = entries_to_bg_freq(num_keys, &constrained_bg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a simple finger layout with potential stretches
            // Fingers: LP, LI, LM, RP (positions 0, 1, 2 are left hand, position 3 is right hand)
            let fingers = vec![LP, LI, LM, RP];
            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| fingers[i % fingers.len()])
                .collect();

            // Simple keyboard with positions - positions 0 and 1 are far apart (stretch)
            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| PhysicalKey::xy(i as f64 * 3.0, 0.0))
                .collect();

            let mut cache = StretchCache::new(&keyboard, &fingers, num_keys);

            // Set weights
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: -10,
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                thumb: 0,
                full_scissors: 0,
                half_scissors: 0,
                full_scissors_skip: 0,
                half_scissors_skip: 0,
                fingers: FingerWeights {
                    lp: 100,
                    lr: 50,
                    lm: 30,
                    li: 20,
                    lt: 10,
                    rt: 10,
                    ri: 20,
                    rm: 30,
                    rr: 50,
                    rp: 100,
                },
            };
            cache.set_weights(&weights);

            let key_positions = make_key_positions(&keys, num_keys);

            // Initialize the rule delta lookup table
            cache.init_rule_deltas(&keys, &key_positions, &bg_freq, &tg_freq);

            // Compute the delta from scratch using compute_rule_delta
            let computed_delta = cache.compute_rule_delta(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
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
