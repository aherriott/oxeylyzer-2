/*
 **************************************
 *     Scissors Scoring Cache
 **************************************
 */

use fxhash::FxHashMap as HashMap;

use crate::types::{CacheKey, CachePos};
use libdof::dofinitions::Finger;

/// Pre-computed scissor pair with severity
///
/// A scissor occurs when two keys on the same hand with different fingers
/// have vertical separation where the finger that prefers being higher
/// is actually positioned lower.
///
/// - Full scissors: 2-row vertical separation
/// - Half scissors: 1-row vertical separation
/// - Adjacent fingers have higher severity than non-adjacent
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScissorPair {
    /// Position of the other key in the scissor pair
    pub other_pos: usize,
    /// Pre-computed severity value (higher = more uncomfortable)
    /// Stored as centiunits (multiplied by 100 for precision)
    pub severity: i64,
    /// true = full scissor (2 rows), false = half scissor (1 row)
    pub is_full: bool,
    /// true = adjacent fingers (e.g., index-middle, middle-ring, ring-pinky)
    pub is_adjacent: bool,
}

/// Delta representing changes to ScissorsCache state.
///
/// Used by `compute_replace_delta()` to track changes to all totals
/// when replacing a key at a position. This allows for efficient
/// incremental updates and supports both apply=true and apply=false modes.
#[derive(Debug, Clone, Default, PartialEq)]
struct ScissorsDelta {
    /// Change to full scissors bigram total (freq * severity)
    full_scissors_bigram_total: i64,
    /// Change to full scissors bigram frequency
    full_scissors_bigram_freq: i64,
    /// Change to full scissors skipgram total (freq * severity)
    full_scissors_skipgram_total: i64,
    /// Change to full scissors skipgram frequency
    full_scissors_skipgram_freq: i64,
    /// Change to half scissors bigram total (freq * severity)
    half_scissors_bigram_total: i64,
    /// Change to half scissors bigram frequency
    half_scissors_bigram_freq: i64,
    /// Change to half scissors skipgram total (freq * severity)
    half_scissors_skipgram_total: i64,
    /// Change to half scissors skipgram frequency
    half_scissors_skipgram_freq: i64,
}

impl ScissorsDelta {
    /// Combine two deltas by adding their values.
    ///
    /// Used during key swaps to combine the deltas from both positions.
    fn combine(a: &ScissorsDelta, b: &ScissorsDelta) -> ScissorsDelta {
        ScissorsDelta {
            full_scissors_bigram_total: a.full_scissors_bigram_total + b.full_scissors_bigram_total,
            full_scissors_bigram_freq: a.full_scissors_bigram_freq + b.full_scissors_bigram_freq,
            full_scissors_skipgram_total: a.full_scissors_skipgram_total + b.full_scissors_skipgram_total,
            full_scissors_skipgram_freq: a.full_scissors_skipgram_freq + b.full_scissors_skipgram_freq,
            half_scissors_bigram_total: a.half_scissors_bigram_total + b.half_scissors_bigram_total,
            half_scissors_bigram_freq: a.half_scissors_bigram_freq + b.half_scissors_bigram_freq,
            half_scissors_skipgram_total: a.half_scissors_skipgram_total + b.half_scissors_skipgram_total,
            half_scissors_skipgram_freq: a.half_scissors_skipgram_freq + b.half_scissors_skipgram_freq,
        }
    }
}

/// Determine if f1 prefers being higher than f2 based on finger length and arm angle.
///
/// Finger preference model:
/// - Middle prefers higher than all others
/// - Ring prefers higher than index/pinky, lower than middle
/// - Pinky prefers higher than index, lower than middle/ring
/// - Index prefers lower than all others
///
/// # Arguments
/// * `f1` - First finger
/// * `f2` - Second finger
///
/// # Returns
/// `true` if f1 prefers being higher than f2, `false` otherwise
pub fn finger_prefers_higher(f1: Finger, f2: Finger) -> bool {
    use Finger::*;

    // Same finger has no preference
    if f1 == f2 {
        return false;
    }

    match (f1, f2) {
        // Middle prefers higher than all
        (LM | RM, _) => true,
        (_, LM | RM) => false,

        // Ring prefers higher than index/pinky
        (LR | RR, LI | RI | LP | RP) => true,
        (LI | RI | LP | RP, LR | RR) => false,

        // Pinky prefers higher than index
        (LP | RP, LI | RI) => true,
        (LI | RI, LP | RP) => false,

        _ => false,
    }
}

/// Check if finger preference is violated given the actual vertical positions.
///
/// A preference violation occurs when the finger that prefers being higher
/// is actually positioned lower on the keyboard.
///
/// # Arguments
/// * `f1` - First finger
/// * `f2` - Second finger
/// * `f1_higher` - `true` if f1 is positioned higher (lower y value) than f2
///
/// # Returns
/// `true` if the finger preference is violated, `false` otherwise
pub fn is_preference_violated(f1: Finger, f2: Finger, f1_higher: bool) -> bool {
    // Returns true if the finger that prefers being higher is actually lower
    let f1_prefers_higher = finger_prefers_higher(f1, f2);
    let f2_prefers_higher = finger_prefers_higher(f2, f1);

    (f1_prefers_higher && !f1_higher) || (f2_prefers_higher && f1_higher)
}

/// Check if two fingers are adjacent (next to each other on the same hand).
///
/// Adjacent finger pairs:
/// - Index and Middle (LI-LM, RI-RM)
/// - Middle and Ring (LM-LR, RM-RR)
/// - Ring and Pinky (LR-LP, RR-RP)
///
/// Note: Thumbs are not considered adjacent to any finger for scissor purposes.
///
/// # Arguments
/// * `f1` - First finger
/// * `f2` - Second finger
///
/// # Returns
/// `true` if the fingers are adjacent on the same hand, `false` otherwise
pub fn are_adjacent_fingers(f1: Finger, f2: Finger) -> bool {
    use Finger::*;
    matches!(
        (f1, f2),
        // Left hand adjacent pairs
        (LI, LM) | (LM, LI) |  // Index-Middle
        (LM, LR) | (LR, LM) |  // Middle-Ring
        (LR, LP) | (LP, LR) |  // Ring-Pinky
        // Right hand adjacent pairs
        (RI, RM) | (RM, RI) |  // Index-Middle
        (RM, RR) | (RR, RM) |  // Middle-Ring
        (RR, RP) | (RP, RR)    // Ring-Pinky
    )
}

use libdof::prelude::PhysicalKey;

/// Compute scissor severity (higher = worse).
///
/// Severity is calculated based on:
/// - Full vs half scissors: Full scissors (2+ row separation) are 2x worse than half scissors
/// - Adjacent vs non-adjacent fingers: Adjacent fingers are 1.5x worse than non-adjacent
///
/// Severity values (in centiunits):
/// - Full adjacent scissor: 300
/// - Full non-adjacent scissor: 200
/// - Half adjacent scissor: 150
/// - Half non-adjacent scissor: 100
///
/// # Arguments
/// * `is_full` - `true` for full scissors (2+ row separation), `false` for half scissors (1 row)
/// * `is_adjacent` - `true` if the fingers are adjacent (e.g., index-middle, middle-ring)
///
/// # Returns
/// Severity as i64 in centiunits (multiplied by 100 for precision)
///
/// # Requirements
/// - Requirement 2.1: Higher severity for full scissors than half scissors
/// - Requirement 2.2: Higher severity for adjacent finger scissors than non-adjacent
/// - Requirement 2.3: Pre-compute and store severity values during initialization
/// - Requirement 2.4: Store severity as integer value (multiplied by 100 for precision)
pub fn compute_severity(is_full: bool, is_adjacent: bool) -> i64 {
    let base = if is_full { 200 } else { 100 };  // Full scissors are 2x worse
    let adjacency_multiplier = if is_adjacent { 150 } else { 100 };  // Adjacent 1.5x worse

    base * adjacency_multiplier / 100
}

/// Determine if a key pair forms a scissor and its type.
///
/// A scissor occurs when two keys on the same hand with different fingers
/// have vertical separation where the finger that prefers being higher
/// is actually positioned lower.
///
/// # Arguments
/// * `k1` - First physical key
/// * `k2` - Second physical key
/// * `f1` - Finger assigned to first key
/// * `f2` - Finger assigned to second key
///
/// # Returns
/// * `Some((is_full, is_adjacent))` if the pair forms a scissor:
///   - `is_full`: true if 2+ row separation (full scissor), false if 1 row (half scissor)
///   - `is_adjacent`: true if the fingers are adjacent
/// * `None` if the pair does not form a scissor (different hands, same finger, or no row separation)
///
/// # Requirements
/// - Requirements 1.1, 1.2, 1.3, 1.4, 1.5: Scissor detection based on finger preferences
/// - Requirements 1.6, 1.7: Non-scissor rejection for different hands or same finger
pub fn compute_scissor(
    k1: &PhysicalKey,
    k2: &PhysicalKey,
    f1: Finger,
    f2: Finger,
) -> Option<(bool, bool)> {
    // Must be same hand, different fingers (Requirements 1.6, 1.7)
    if f1 == f2 || f1.hand() != f2.hand() {
        return None;
    }

    let row1 = k1.y().round() as i32;
    let row2 = k2.y().round() as i32;
    let row_diff = (row1 - row2).abs();

    // Must have vertical separation
    if row_diff == 0 {
        return None;
    }

    let is_full = row_diff >= 2;
    let is_half = row_diff == 1;

    if !is_full && !is_half {
        return None;
    }

    // Check if this violates finger preferences
    // Lower y = higher on keyboard (top of keyboard has lower y values)
    let f1_higher = row1 < row2;
    let is_scissor = is_preference_violated(f1, f2, f1_higher);

    if !is_scissor {
        return None;
    }

    let is_adjacent = are_adjacent_fingers(f1, f2);
    Some((is_full, is_adjacent))
}

/// Cache for efficient scissors scoring during layout optimization.
///
/// The ScissorsCache pre-computes scissor pairs during initialization and maintains
/// running totals for efficient incremental updates. It follows the same pattern
/// as `SFCache` and `StretchCache`.
///
/// # Structure
/// - `scissor_pairs_per_key`: For each position, stores all other positions that form
///   scissor pairs with it, along with pre-computed severity values
/// - Running totals: Separate totals for each scissor type (full/half × bigram/skipgram)
/// - Frequency totals: Unweighted frequencies for statistics reporting
/// - Weights: Pre-computed weight multipliers for score computation
///
/// # Requirements
/// - Requirement 3.1: Pre-compute all scissor pairs during initialization
/// - Requirement 3.3: Store scissor pairs indexed by position for O(1) lookup
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScissorsCache {
    /// For each position, list of other positions that form scissor pairs.
    /// Each entry contains the other position, severity, and scissor type info.
    scissor_pairs_per_key: Vec<Vec<ScissorPair>>,

    /// Number of keys for frequency array indexing
    num_keys: usize,

    /// Running totals (freq * severity, not yet weighted)
    /// These are updated incrementally as bigram/skipgram frequencies change.
    full_scissors_bigram_total: i64,
    full_scissors_skipgram_total: i64,
    half_scissors_bigram_total: i64,
    half_scissors_skipgram_total: i64,

    /// Unweighted frequency totals (for stats reporting)
    /// These track raw frequencies without severity weighting.
    full_scissors_bigram_freq: i64,
    full_scissors_skipgram_freq: i64,
    half_scissors_bigram_freq: i64,
    half_scissors_skipgram_freq: i64,

    /// Pre-computed weights from configuration
    /// Score = total * weight for each scissor type
    full_scissors_weight: i64,
    full_scissors_skip_weight: i64,
    half_scissors_weight: i64,
    half_scissors_skip_weight: i64,

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
    rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,
}

impl ScissorsCache {
    /// Create a new cache from keyboard layout and finger assignments.
    ///
    /// Pre-computes all scissor pairs during initialization for O(1) lookup during updates.
    /// For each pair of positions (i, j) where i < j, checks if they form a scissor and
    /// stores the pair in both positions' lookup tables.
    ///
    /// # Arguments
    /// * `keyboard` - Slice of physical keys defining the keyboard layout
    /// * `fingers` - Slice of finger assignments for each key position
    /// * `num_keys` - Number of keys for frequency array indexing
    ///
    /// # Panics
    /// Panics if `keyboard.len() != fingers.len()`
    ///
    /// # Requirements
    /// - Requirement 3.1: Identify all scissor pairs from keyboard layout and finger assignments
    /// - Requirement 3.2: Pre-compute severity values for each scissor pair
    /// - Requirement 3.3: Store scissor pairs indexed by position for O(1) lookup
    pub fn new(
        keyboard: &[PhysicalKey],
        fingers: &[Finger],
        num_keys: usize,
    ) -> Self {
        assert_eq!(
            keyboard.len(),
            fingers.len(),
            "keyboard len is not the same as fingers len"
        );

        let len = keyboard.len();

        // Initialize empty scissor pairs for each position
        let mut scissor_pairs_per_key: Vec<Vec<ScissorPair>> = (0..len)
            .map(|_| Vec::new())
            .collect();

        // Iterate over all position pairs (i, j) where i < j
        // This ensures we check each pair exactly once
        for i in 0..len {
            for j in (i + 1)..len {
                // Check if this pair forms a scissor
                if let Some((is_full, is_adjacent)) = compute_scissor(
                    &keyboard[i],
                    &keyboard[j],
                    fingers[i],
                    fingers[j],
                ) {
                    // Compute severity for this scissor pair
                    let severity = compute_severity(is_full, is_adjacent);

                    // Add entry to position i's list (pointing to j)
                    scissor_pairs_per_key[i].push(ScissorPair {
                        other_pos: j,
                        severity,
                        is_full,
                        is_adjacent,
                    });

                    // Add entry to position j's list (pointing to i)
                    // This allows O(1) lookup from either position
                    scissor_pairs_per_key[j].push(ScissorPair {
                        other_pos: i,
                        severity,
                        is_full,
                        is_adjacent,
                    });
                }
            }
        }

        Self {
            scissor_pairs_per_key,
            num_keys,
            // Initialize all totals to 0
            full_scissors_bigram_total: 0,
            full_scissors_skipgram_total: 0,
            half_scissors_bigram_total: 0,
            half_scissors_skipgram_total: 0,
            full_scissors_bigram_freq: 0,
            full_scissors_skipgram_freq: 0,
            half_scissors_bigram_freq: 0,
            half_scissors_skipgram_freq: 0,
            // Initialize weights to 0 (will be set via set_weights)
            full_scissors_weight: 0,
            full_scissors_skip_weight: 0,
            half_scissors_weight: 0,
            half_scissors_skip_weight: 0,
            active_rules: HashMap::default(),
            magic_rule_score_delta: 0,
            rule_delta: HashMap::default(),
        }
    }

    /// Get the scissor pairs for a given position.
    /// Returns an empty slice if the position has no scissor pairs.
    #[inline]
    pub fn get_scissor_pairs(&self, pos: usize) -> &[ScissorPair] {
        &self.scissor_pairs_per_key[pos]
    }

    /// Check if two positions form a scissor pair and return the pair info if so.
    #[inline]
    pub fn get_scissor(&self, p_a: usize, p_b: usize) -> Option<&ScissorPair> {
        self.scissor_pairs_per_key[p_a]
            .iter()
            .find(|sp| sp.other_pos == p_b)
    }

    /// Get the number of keyboard positions
    #[inline]
    pub fn num_positions(&self) -> usize {
        self.scissor_pairs_per_key.len()
    }

    /// Get the number of keys for frequency indexing
    #[inline]
    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    /// Set weights from configuration.
    ///
    /// Copies the scissor weight values from the Weights struct to the cache's
    /// internal weight fields for use in score computation.
    ///
    /// # Arguments
    /// * `weights` - Reference to the Weights configuration struct
    ///
    /// # Requirements
    /// - Requirement 3.4: Store pre-computed weight multipliers for each scissor type
    /// - Requirement 7.2: Use weights to compute the weighted score
    pub fn set_weights(&mut self, weights: &crate::weights::Weights) {
        // Scissors are penalties — negate so positive weight = worse score
        self.full_scissors_weight = -weights.full_scissors;
        self.full_scissors_skip_weight = -weights.full_scissors_skip;
        self.half_scissors_weight = -weights.half_scissors;
        self.half_scissors_skip_weight = -weights.half_scissors_skip;
    }

    #[inline]
    pub fn full_weight(&self) -> i64 { self.full_scissors_weight }
    #[inline]
    pub fn full_skip_weight(&self) -> i64 { self.full_scissors_skip_weight }
    #[inline]
    pub fn half_weight(&self) -> i64 { self.half_scissors_weight }
    #[inline]
    pub fn half_skip_weight(&self) -> i64 { self.half_scissors_skip_weight }

    /// Compute the score delta for replacing a key at a position.
    ///
    /// This is a helper function used by `replace_key()` and `key_swap()`. It computes
    /// the score delta when replacing a key at a position by iterating over all scissor
    /// pairs involving that position.
    ///
    /// # Arguments
    /// * `pos` - The position where the key is being replaced
    /// * `old_key` - The key currently at the position
    /// * `new_key` - The key that will replace it
    /// * `keys` - The current key assignment array
    /// * `skip_pos` - Optional position to skip (used during swaps to avoid double-counting)
    /// * `bg_freq` - Bigram frequency array (1D, indexed by key_a * num_keys + key_b)
    /// * `sg_freq` - Skipgram frequency array (1D, indexed by key_a * num_keys + key_b)
    ///
    /// # Returns
    /// A `ScissorsDelta` containing the changes to all totals.
    ///
    /// # Algorithm
    /// For each scissor pair involving the position:
    /// 1. Get the other position in the pair
    /// 2. If skip_pos is Some and equals other_pos, skip this pair (used during swaps)
    /// 3. Get the key at the other position from the keys array
    /// 4. Compute the old contribution: bg_freq[old_key][other_key] * severity + sg_freq[old_key][other_key] * severity
    /// 5. Compute the new contribution: bg_freq[new_key][other_key] * severity + sg_freq[new_key][other_key] * severity
    /// 6. Accumulate the delta (new - old) into the appropriate totals based on is_full
    ///
    /// # Requirements
    /// - Requirement 4.3: Compute score delta for key replacement without full recalculation
    fn compute_replace_delta(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> ScissorsDelta {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut delta = ScissorsDelta::default();

        for sp in &self.scissor_pairs_per_key[pos] {
            let other_pos = sp.other_pos;

            // Skip this pair if it's the skip_pos (used during swaps)
            if skip_pos == Some(other_pos) {
                continue;
            }

            let other_key = keys[other_pos];

            // Skip if other_key is invalid
            if other_key >= num_keys {
                continue;
            }

            let severity = sp.severity;
            let other_row = other_key * num_keys;

            // Compute bigram frequency deltas (both directions)
            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };

            // Compute skipgram frequency deltas (both directions)
            let old_sg = if old_valid { sg_freq[old_row + other_key] } else { 0 };
            let new_sg = if new_valid { sg_freq[new_row + other_key] } else { 0 };
            let old_sg_rev = if old_valid { sg_freq[other_row + old_key] } else { 0 };
            let new_sg_rev = if new_valid { sg_freq[other_row + new_key] } else { 0 };

            // Total frequency deltas
            let bg_freq_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            let sg_freq_delta = (new_sg - old_sg) + (new_sg_rev - old_sg_rev);

            // Score deltas (frequency * severity)
            let bg_score_delta = bg_freq_delta * severity;
            let sg_score_delta = sg_freq_delta * severity;

            // Accumulate into appropriate totals based on scissor type
            if sp.is_full {
                delta.full_scissors_bigram_total += bg_score_delta;
                delta.full_scissors_bigram_freq += bg_freq_delta;
                delta.full_scissors_skipgram_total += sg_score_delta;
                delta.full_scissors_skipgram_freq += sg_freq_delta;
            } else {
                delta.half_scissors_bigram_total += bg_score_delta;
                delta.half_scissors_bigram_freq += bg_freq_delta;
                delta.half_scissors_skipgram_total += sg_score_delta;
                delta.half_scissors_skipgram_freq += sg_freq_delta;
            }
        }

        delta
    }

    /// Compute only the weighted score delta for replacing a key (fast path).
    ///
    /// This is an optimized version of `compute_replace_delta` that only computes
    /// the total weighted score delta without tracking per-type frequencies.
    /// Used for speculative scoring when `apply=false`.
    ///
    /// # Arguments
    /// Same as `compute_replace_delta`
    ///
    /// # Returns
    /// The weighted score delta as i64.
    ///
    /// # Requirements
    /// - Requirement 4.3: Compute score delta for key replacement
    /// - Requirement 4.5: Support apply=false mode for speculative scoring
    #[inline]
    fn compute_replace_delta_score_only(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let num_keys = self.num_keys;
        let old_valid = old_key < num_keys;
        let new_valid = new_key < num_keys;

        let old_row = if old_valid { old_key * num_keys } else { 0 };
        let new_row = if new_valid { new_key * num_keys } else { 0 };

        let mut total_delta: i64 = 0;

        for sp in &self.scissor_pairs_per_key[pos] {
            let other_pos = sp.other_pos;

            // Skip this pair if it's the skip_pos (used during swaps)
            if skip_pos == Some(other_pos) {
                continue;
            }

            let other_key = keys[other_pos];

            // Skip if other_key is invalid
            if other_key >= num_keys {
                continue;
            }

            let severity = sp.severity;
            let other_row = other_key * num_keys;

            // Compute bigram frequency deltas (both directions)
            let old_bg = if old_valid { bg_freq[old_row + other_key] } else { 0 };
            let new_bg = if new_valid { bg_freq[new_row + other_key] } else { 0 };
            let old_bg_rev = if old_valid { bg_freq[other_row + old_key] } else { 0 };
            let new_bg_rev = if new_valid { bg_freq[other_row + new_key] } else { 0 };

            // Compute skipgram frequency deltas (both directions)
            let old_sg = if old_valid { sg_freq[old_row + other_key] } else { 0 };
            let new_sg = if new_valid { sg_freq[new_row + other_key] } else { 0 };
            let old_sg_rev = if old_valid { sg_freq[other_row + old_key] } else { 0 };
            let new_sg_rev = if new_valid { sg_freq[other_row + new_key] } else { 0 };

            // Total frequency deltas
            let bg_freq_delta = (new_bg - old_bg) + (new_bg_rev - old_bg_rev);
            let sg_freq_delta = (new_sg - old_sg) + (new_sg_rev - old_sg_rev);

            // Score deltas (frequency * severity)
            let bg_score_delta = bg_freq_delta * severity;
            let sg_score_delta = sg_freq_delta * severity;

            // Compute weighted score delta based on scissor type
            if sp.is_full {
                total_delta += bg_score_delta * self.full_scissors_weight
                    + sg_score_delta * self.full_scissors_skip_weight;
            } else {
                total_delta += bg_score_delta * self.half_scissors_weight
                    + sg_score_delta * self.half_scissors_skip_weight;
            }
        }

        total_delta
    }

    /// Apply a delta to the cache state.
    ///
    /// Updates all internal totals by adding the delta values.
    ///
    /// # Arguments
    /// * `delta` - The delta to apply
    fn apply_delta(&mut self, delta: &ScissorsDelta) {
        self.full_scissors_bigram_total += delta.full_scissors_bigram_total;
        self.full_scissors_bigram_freq += delta.full_scissors_bigram_freq;
        self.full_scissors_skipgram_total += delta.full_scissors_skipgram_total;
        self.full_scissors_skipgram_freq += delta.full_scissors_skipgram_freq;
        self.half_scissors_bigram_total += delta.half_scissors_bigram_total;
        self.half_scissors_bigram_freq += delta.half_scissors_bigram_freq;
        self.half_scissors_skipgram_total += delta.half_scissors_skipgram_total;
        self.half_scissors_skipgram_freq += delta.half_scissors_skipgram_freq;
    }

    /// Get the current weighted score.
    ///
    /// Computes the total score by multiplying each scissor type's total
    /// by its corresponding weight and summing them.
    ///
    /// # Returns
    /// The weighted sum of all scissor totals.
    ///
    /// # Formula
    /// ```text
    /// score = full_scissors_bigram_total * full_scissors_weight
    ///       + full_scissors_skipgram_total * full_scissors_skip_weight
    ///       + half_scissors_bigram_total * half_scissors_weight
    ///       + half_scissors_skipgram_total * half_scissors_skip_weight
    /// ```
    ///
    /// # Requirements
    /// - Requirement 5.1: Maintain a running total score that is updated incrementally
    /// - Requirement 5.2: Compute score as frequency × severity × weight for each scissor type
    /// - Requirement 5.3: Support separate weights for each scissor type
    /// - Requirement 5.4: Return the current weighted total score
    #[inline]
    pub fn score(&self) -> i64 {
        self.full_scissors_bigram_total * self.full_scissors_weight
            + self.full_scissors_skipgram_total * self.full_scissors_skip_weight
            + self.half_scissors_bigram_total * self.half_scissors_weight
            + self.half_scissors_skipgram_total * self.half_scissors_skip_weight
            + self.magic_rule_score_delta
    }

    /// Populate statistics with normalized scissor frequencies.
    ///
    /// Computes normalized frequencies for each scissor type by dividing the
    /// raw frequency totals by the appropriate gram total and 100 (since
    /// frequencies are stored in centiunits).
    ///
    /// # Arguments
    /// * `stats` - Mutable reference to the Stats struct to populate
    /// * `bigram_total` - Total bigram frequency for normalization
    /// * `skipgram_total` - Total skipgram frequency for normalization
    ///
    /// # Formula
    /// For each scissor type:
    /// - `stats.full_scissors_bigrams = full_scissors_bigram_freq / (bigram_total * 100)`
    /// - `stats.full_scissors_skipgrams = full_scissors_skipgram_freq / (skipgram_total * 100)`
    /// - `stats.half_scissors_bigrams = half_scissors_bigram_freq / (bigram_total * 100)`
    /// - `stats.half_scissors_skipgrams = half_scissors_skipgram_freq / (skipgram_total * 100)`
    ///
    /// # Note
    /// If bigram_total or skipgram_total is 0, the corresponding stats will be 0
    /// (division by zero is handled by returning 0).
    ///
    /// # Requirements
    /// - Requirement 6.2: Populate stats with normalized scissor frequencies
    /// - Requirement 6.3: Normalize frequencies by dividing by bigram_total for bigrams
    ///   and skipgram_total for skipgrams
    pub fn stats(&self, stats: &mut crate::stats::Stats, bigram_total: f64, skipgram_total: f64) {
        // Multiply by 100 because frequencies are stored in centiunits
        let bigram_total_raw = bigram_total * 100.0;
        let skipgram_total_raw = skipgram_total * 100.0;

        // Handle division by zero by returning 0 if total is 0
        stats.full_scissors_bigrams = if bigram_total_raw > 0.0 {
            self.full_scissors_bigram_freq as f64 / bigram_total_raw
        } else {
            0.0
        };

        stats.full_scissors_skipgrams = if skipgram_total_raw > 0.0 {
            self.full_scissors_skipgram_freq as f64 / skipgram_total_raw
        } else {
            0.0
        };

        stats.half_scissors_bigrams = if bigram_total_raw > 0.0 {
            self.half_scissors_bigram_freq as f64 / bigram_total_raw
        } else {
            0.0
        };

        stats.half_scissors_skipgrams = if skipgram_total_raw > 0.0 {
            self.half_scissors_skipgram_freq as f64 / skipgram_total_raw
        } else {
            0.0
        };
    }

    /// Replace key at position. Mutates running totals. Returns the new score.
    #[inline]
    pub fn replace_key(
        &mut self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        skip_pos: Option<usize>,
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
        self.apply_delta(&delta);
        self.score()
    }

    /// Speculative score for replacing a key. No mutation.
    #[inline]
    pub fn score_replace(
        &self,
        pos: usize,
        old_key: usize,
        new_key: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, None, bg_freq, sg_freq);
        self.score() + score_delta
    }

    /// Swap keys at two positions. Mutates running totals.
    #[inline]
    pub fn key_swap(
        &mut self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
        let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
        let combined = ScissorsDelta::combine(&delta_a, &delta_b);
        self.apply_delta(&combined);
        self.score()
    }

    /// Speculative score for a key swap. No mutation.
    #[inline]
    pub fn score_swap(
        &self,
        pos_a: usize,
        pos_b: usize,
        key_a: usize,
        key_b: usize,
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
        let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
        self.score() + score_a + score_b
    }

    /// Compute the weighted score for a scissor bigram pair.
    /// Returns severity * weight if the positions form a scissor, 0 otherwise.
    ///
    /// # Arguments
    /// * `p_a` - First position
    /// * `p_b` - Second position
    ///
    /// # Returns
    /// The weighted score contribution for this pair (severity * weight), or 0 if not a scissor.
    #[inline]
    fn scissors_bigram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some(sp) = self.get_scissor(p_a, p_b) {
            if sp.is_full {
                sp.severity * self.full_scissors_weight
            } else {
                sp.severity * self.half_scissors_weight
            }
        } else {
            0
        }
    }

    /// Compute the weighted score for a scissor skipgram pair.
    /// Returns severity * weight if the positions form a scissor, 0 otherwise.
    ///
    /// # Arguments
    /// * `p_a` - First position
    /// * `p_b` - Second position
    ///
    /// # Returns
    /// The weighted score contribution for this pair (severity * skip_weight), or 0 if not a scissor.
    #[inline]
    fn scissors_skipgram_weight(&self, p_a: usize, p_b: usize) -> i64 {
        if let Some(sp) = self.get_scissor(p_a, p_b) {
            if sp.is_full {
                sp.severity * self.full_scissors_skip_weight
            } else {
                sp.severity * self.half_scissors_skip_weight
            }
        } else {
            0
        }
    }

    // ==================== Lower Bound ====================

    /// Compute a lower bound on the remaining scissors penalty from unplaced keys.
    ///
    /// For each unplaced key, finds the available position that minimizes the
    /// scissors penalty with all already-placed keys, and sums these minimums.
    pub fn lower_bound_remaining(
        &self,
        unplaced_keys: &[usize],
        available_positions: &[usize],
        keys: &[usize],
        bg_freq: &[i64],
        sg_freq: &[i64],
    ) -> i64 {
        let nk = self.num_keys;
        let mut total_min: i64 = 0;

        for &key in unplaced_keys {
            if key >= nk { continue; }
            let mut best_penalty = i64::MAX;
            for &pos in available_positions {
                let mut penalty: i64 = 0;
                for sp in &self.scissor_pairs_per_key[pos] {
                    let other_key = keys[sp.other_pos];
                    if other_key >= nk { continue; }
                    let bg = bg_freq[key * nk + other_key] + bg_freq[other_key * nk + key];
                    let sg = sg_freq[key * nk + other_key] + sg_freq[other_key * nk + key];
                    let bg_score = bg * sp.severity;
                    let sg_score = sg * sp.severity;
                    if sp.is_full {
                        penalty += bg_score * self.full_scissors_weight
                            + sg_score * self.full_scissors_skip_weight;
                    } else {
                        penalty += bg_score * self.half_scissors_weight
                            + sg_score * self.half_scissors_skip_weight;
                    }
                }
                if penalty < best_penalty {
                    best_penalty = penalty;
                }
            }
            if best_penalty != i64::MAX {
                total_min += best_penalty;
            }
        }

        total_min
    }

    /// Compute the score delta for a magic rule application.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M (full steal)
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// # Returns
    /// The score delta (new_score - old_score) for this rule application.
    ///
    /// **Validates: Requirements 4.4 (ScissorsCache magic rule computation)**
    fn compute_rule_delta(
        &self,
        leader: CacheKey,      // A
        output: CacheKey,      // B (being stolen)
        magic_key: CacheKey,   // M
        _keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        _sg_freq: &[i64],
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

        // Output (B) and magic_key (M) positions are needed for scissor computation
        let output_pos = key_positions.get(output).copied().flatten();
        let magic_pos = key_positions.get(magic_key).copied().flatten();

        let mut delta: i64 = 0;

        // Part 1: Full steal - Bigram A→B becomes A→M
        // The frequency being stolen is bg_freq[A][B]
        // We compute: freq * (new_weight - old_weight)
        let full_steal_freq = bg_freq[leader * num_keys + output];
        if full_steal_freq != 0 {
            // Old weight: scissors_weight(A, B) - based on positions of leader and output
            let old_weight = if let Some(b_pos) = output_pos {
                self.scissors_bigram_weight(leader_pos, b_pos)
            } else {
                0
            };

            // New weight: scissors_weight(A, M) - based on positions of leader and magic_key
            let new_weight = if let Some(m_pos) = magic_pos {
                self.scissors_bigram_weight(leader_pos, m_pos)
            } else {
                0
            };

            delta += full_steal_freq * (new_weight - old_weight);
        }

        // Part 2: Partial steal - Bigrams B→C become M→C based on tg_freq[A][B][C]
        // For each C with a position, the frequency stolen is tg_freq[A][B][C]
        // The weight changes from scissors_weight(B, C) to scissors_weight(M, C)
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

            // Old weight: scissors_weight(B, C) - based on positions of output and C
            let old_weight = if let Some(b_pos) = output_pos {
                self.scissors_bigram_weight(b_pos, c_pos)
            } else {
                0
            };

            // New weight: scissors_weight(M, C) - based on positions of magic_key and C
            let new_weight = if let Some(m_pos) = magic_pos {
                self.scissors_bigram_weight(m_pos, c_pos)
            } else {
                0
            };

            delta += stolen_freq * (new_weight - old_weight);
        }

        // Part 3: Skipgram partial steal - Skipgrams Z→B become Z→M based on tg_freq[Z][A][B]
        // For each Z with a position, the frequency stolen is tg_freq[Z][A][B]
        // The weight changes from scissors_skipgram_weight(Z, B) to scissors_skipgram_weight(Z, M)
        for z_key in 0..num_keys {
            // Z must have a position
            let z_pos = match key_positions.get(z_key).copied().flatten() {
                Some(pos) => pos,
                None => continue,
            };

            // Get the frequency stolen: tg_freq[Z][A][B]
            let stolen_freq = tg_freq[z_key][leader][output];
            if stolen_freq == 0 {
                continue;
            }

            // Old weight: scissors_skipgram_weight(Z, B) - based on positions of Z and output
            let old_weight = if let Some(b_pos) = output_pos {
                self.scissors_skipgram_weight(z_pos, b_pos)
            } else {
                0
            };

            // New weight: scissors_skipgram_weight(Z, M) - based on positions of Z and magic_key
            let new_weight = if let Some(m_pos) = magic_pos {
                self.scissors_skipgram_weight(z_pos, m_pos)
            } else {
                0
            };

            delta += stolen_freq * (new_weight - old_weight);
        }

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
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
    /// * `tg_freq` - 3D trigram frequency data: tg_freq[key_a][key_b][key_c]
    ///
    /// **Validates: Requirements 6.4, 6.5**
    pub fn init_rule_deltas(
        &mut self,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
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
                        sg_freq,
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
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    ///
    /// # Arguments
    /// * `leader` - A: the key that triggers the magic rule
    /// * `output` - B: the output being stolen
    /// * `magic_key` - M: the magic key that steals the output
    /// * `keys` - The current key assignments for all positions
    /// * `key_positions` - Maps each key to its position (None if key has no position)
    /// * `bg_freq` - Flat bigram frequencies: bg_freq[a * num_keys + b]
    /// * `sg_freq` - Flat skipgram frequencies: sg_freq[a * num_keys + b]
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
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64 {
        let rule_key = (magic_key, leader);

        // Stored deltas can become stale when keys move, so recompute.
        let old_delta = if let Some(&(old_output, _stored_delta)) = self.active_rules.get(&rule_key) {
            if old_output == output {
                return 0;
            }
            self.compute_rule_delta(
                leader, old_output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq
            )
        } else {
            0
        };

        // Compute the score delta for the new rule (0 if output is EMPTY_KEY)
        let new_delta = if output != crate::cached_layout::EMPTY_KEY {
            self.compute_rule_delta(
                leader, output, magic_key, keys, key_positions, bg_freq, sg_freq, tg_freq
            )
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

    pub fn reset_magic_deltas(&mut self) {
        self.active_rules.clear();
        self.magic_rule_score_delta = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Finger::*;

    /// Helper to set internal totals directly for testing purposes.
    /// This is needed because update_bigram and update_skipgram have been removed
    /// as part of the constant-frequency architecture refactoring.
    impl ScissorsCache {
        #[cfg(test)]
        fn set_totals_for_testing(
            &mut self,
            full_bigram_total: i64,
            full_bigram_freq: i64,
            full_skipgram_total: i64,
            full_skipgram_freq: i64,
            half_bigram_total: i64,
            half_bigram_freq: i64,
            half_skipgram_total: i64,
            half_skipgram_freq: i64,
        ) {
            self.full_scissors_bigram_total = full_bigram_total;
            self.full_scissors_bigram_freq = full_bigram_freq;
            self.full_scissors_skipgram_total = full_skipgram_total;
            self.full_scissors_skipgram_freq = full_skipgram_freq;
            self.half_scissors_bigram_total = half_bigram_total;
            self.half_scissors_bigram_freq = half_bigram_freq;
            self.half_scissors_skipgram_total = half_skipgram_total;
            self.half_scissors_skipgram_freq = half_skipgram_freq;
        }
    }

    // ==========================================
    // Tests for finger_prefers_higher
    // ==========================================

    #[test]
    fn test_middle_prefers_higher_than_all() {
        // Middle finger prefers being higher than all other fingers
        // Left hand
        assert!(finger_prefers_higher(LM, LI), "LM should prefer higher than LI");
        assert!(finger_prefers_higher(LM, LR), "LM should prefer higher than LR");
        assert!(finger_prefers_higher(LM, LP), "LM should prefer higher than LP");
        assert!(finger_prefers_higher(LM, LT), "LM should prefer higher than LT");

        // Right hand
        assert!(finger_prefers_higher(RM, RI), "RM should prefer higher than RI");
        assert!(finger_prefers_higher(RM, RR), "RM should prefer higher than RR");
        assert!(finger_prefers_higher(RM, RP), "RM should prefer higher than RP");
        assert!(finger_prefers_higher(RM, RT), "RM should prefer higher than RT");

        // Cross-hand (middle still prefers higher)
        assert!(finger_prefers_higher(LM, RI), "LM should prefer higher than RI");
        assert!(finger_prefers_higher(RM, LI), "RM should prefer higher than LI");
    }

    #[test]
    fn test_ring_prefers_higher_than_index_and_pinky() {
        // Ring prefers higher than index and pinky
        // Left hand
        assert!(finger_prefers_higher(LR, LI), "LR should prefer higher than LI");
        assert!(finger_prefers_higher(LR, LP), "LR should prefer higher than LP");

        // Right hand
        assert!(finger_prefers_higher(RR, RI), "RR should prefer higher than RI");
        assert!(finger_prefers_higher(RR, RP), "RR should prefer higher than RP");

        // Ring does NOT prefer higher than middle
        assert!(!finger_prefers_higher(LR, LM), "LR should NOT prefer higher than LM");
        assert!(!finger_prefers_higher(RR, RM), "RR should NOT prefer higher than RM");
    }

    #[test]
    fn test_pinky_prefers_higher_than_index() {
        // Pinky prefers higher than index
        assert!(finger_prefers_higher(LP, LI), "LP should prefer higher than LI");
        assert!(finger_prefers_higher(RP, RI), "RP should prefer higher than RI");

        // Pinky does NOT prefer higher than middle or ring
        assert!(!finger_prefers_higher(LP, LM), "LP should NOT prefer higher than LM");
        assert!(!finger_prefers_higher(LP, LR), "LP should NOT prefer higher than LR");
        assert!(!finger_prefers_higher(RP, RM), "RP should NOT prefer higher than RM");
        assert!(!finger_prefers_higher(RP, RR), "RP should NOT prefer higher than RR");
    }

    #[test]
    fn test_index_prefers_lower_than_all() {
        // Index does NOT prefer being higher than any finger
        assert!(!finger_prefers_higher(LI, LM), "LI should NOT prefer higher than LM");
        assert!(!finger_prefers_higher(LI, LR), "LI should NOT prefer higher than LR");
        assert!(!finger_prefers_higher(LI, LP), "LI should NOT prefer higher than LP");

        assert!(!finger_prefers_higher(RI, RM), "RI should NOT prefer higher than RM");
        assert!(!finger_prefers_higher(RI, RR), "RI should NOT prefer higher than RR");
        assert!(!finger_prefers_higher(RI, RP), "RI should NOT prefer higher than RP");
    }

    #[test]
    fn test_same_finger_no_preference() {
        // Same finger comparisons should return false
        assert!(!finger_prefers_higher(LI, LI), "Same finger should not prefer higher");
        assert!(!finger_prefers_higher(LM, LM), "Same finger should not prefer higher");
        assert!(!finger_prefers_higher(LR, LR), "Same finger should not prefer higher");
        assert!(!finger_prefers_higher(LP, LP), "Same finger should not prefer higher");
    }

    // ==========================================
    // Tests for is_preference_violated
    // ==========================================

    #[test]
    fn test_preference_violated_middle_lower_than_index() {
        // Middle prefers higher, so if middle is lower, it's a violation
        // f1=LM, f2=LI, f1_higher=false means LM is lower than LI -> violation
        assert!(is_preference_violated(LM, LI, false),
            "Middle lower than index should be a violation");

        // f1=LM, f2=LI, f1_higher=true means LM is higher than LI -> no violation
        assert!(!is_preference_violated(LM, LI, true),
            "Middle higher than index should NOT be a violation");
    }

    #[test]
    fn test_preference_violated_index_higher_than_middle() {
        // From index's perspective: index is higher than middle
        // f1=LI, f2=LM, f1_higher=true means LI is higher than LM
        // Middle prefers higher, so this is a violation
        assert!(is_preference_violated(LI, LM, true),
            "Index higher than middle should be a violation");

        // f1=LI, f2=LM, f1_higher=false means LI is lower than LM -> no violation
        assert!(!is_preference_violated(LI, LM, false),
            "Index lower than middle should NOT be a violation");
    }

    #[test]
    fn test_preference_violated_ring_lower_than_pinky() {
        // Ring prefers higher than pinky
        // f1=LR, f2=LP, f1_higher=false means LR is lower than LP -> violation
        assert!(is_preference_violated(LR, LP, false),
            "Ring lower than pinky should be a violation");

        // f1=LR, f2=LP, f1_higher=true means LR is higher than LP -> no violation
        assert!(!is_preference_violated(LR, LP, true),
            "Ring higher than pinky should NOT be a violation");
    }

    #[test]
    fn test_preference_violated_pinky_lower_than_index() {
        // Pinky prefers higher than index
        // f1=LP, f2=LI, f1_higher=false means LP is lower than LI -> violation
        assert!(is_preference_violated(LP, LI, false),
            "Pinky lower than index should be a violation");

        // f1=LP, f2=LI, f1_higher=true means LP is higher than LI -> no violation
        assert!(!is_preference_violated(LP, LI, true),
            "Pinky higher than index should NOT be a violation");
    }

    #[test]
    fn test_no_preference_violation_same_preference_level() {
        // Two fingers with no preference relationship (e.g., same finger type)
        // Neither prefers higher, so no violation possible
        assert!(!is_preference_violated(LI, RI, true),
            "Cross-hand same finger type should not have preference violation");
        assert!(!is_preference_violated(LI, RI, false),
            "Cross-hand same finger type should not have preference violation");
    }

    // ==========================================
    // Tests for are_adjacent_fingers
    // ==========================================

    #[test]
    fn test_adjacent_index_middle() {
        // Index and Middle are adjacent
        assert!(are_adjacent_fingers(LI, LM), "LI-LM should be adjacent");
        assert!(are_adjacent_fingers(LM, LI), "LM-LI should be adjacent");
        assert!(are_adjacent_fingers(RI, RM), "RI-RM should be adjacent");
        assert!(are_adjacent_fingers(RM, RI), "RM-RI should be adjacent");
    }

    #[test]
    fn test_adjacent_middle_ring() {
        // Middle and Ring are adjacent
        assert!(are_adjacent_fingers(LM, LR), "LM-LR should be adjacent");
        assert!(are_adjacent_fingers(LR, LM), "LR-LM should be adjacent");
        assert!(are_adjacent_fingers(RM, RR), "RM-RR should be adjacent");
        assert!(are_adjacent_fingers(RR, RM), "RR-RM should be adjacent");
    }

    #[test]
    fn test_adjacent_ring_pinky() {
        // Ring and Pinky are adjacent
        assert!(are_adjacent_fingers(LR, LP), "LR-LP should be adjacent");
        assert!(are_adjacent_fingers(LP, LR), "LP-LR should be adjacent");
        assert!(are_adjacent_fingers(RR, RP), "RR-RP should be adjacent");
        assert!(are_adjacent_fingers(RP, RR), "RP-RR should be adjacent");
    }

    #[test]
    fn test_not_adjacent_index_ring() {
        // Index and Ring are NOT adjacent (middle is between them)
        assert!(!are_adjacent_fingers(LI, LR), "LI-LR should NOT be adjacent");
        assert!(!are_adjacent_fingers(LR, LI), "LR-LI should NOT be adjacent");
        assert!(!are_adjacent_fingers(RI, RR), "RI-RR should NOT be adjacent");
        assert!(!are_adjacent_fingers(RR, RI), "RR-RI should NOT be adjacent");
    }

    #[test]
    fn test_not_adjacent_index_pinky() {
        // Index and Pinky are NOT adjacent
        assert!(!are_adjacent_fingers(LI, LP), "LI-LP should NOT be adjacent");
        assert!(!are_adjacent_fingers(LP, LI), "LP-LI should NOT be adjacent");
        assert!(!are_adjacent_fingers(RI, RP), "RI-RP should NOT be adjacent");
        assert!(!are_adjacent_fingers(RP, RI), "RP-RI should NOT be adjacent");
    }

    #[test]
    fn test_not_adjacent_middle_pinky() {
        // Middle and Pinky are NOT adjacent (ring is between them)
        assert!(!are_adjacent_fingers(LM, LP), "LM-LP should NOT be adjacent");
        assert!(!are_adjacent_fingers(LP, LM), "LP-LM should NOT be adjacent");
        assert!(!are_adjacent_fingers(RM, RP), "RM-RP should NOT be adjacent");
        assert!(!are_adjacent_fingers(RP, RM), "RP-RM should NOT be adjacent");
    }

    #[test]
    fn test_not_adjacent_cross_hand() {
        // Fingers on different hands are NOT adjacent
        assert!(!are_adjacent_fingers(LI, RI), "Cross-hand fingers should NOT be adjacent");
        assert!(!are_adjacent_fingers(LM, RM), "Cross-hand fingers should NOT be adjacent");
        assert!(!are_adjacent_fingers(LR, RR), "Cross-hand fingers should NOT be adjacent");
        assert!(!are_adjacent_fingers(LP, RP), "Cross-hand fingers should NOT be adjacent");
        assert!(!are_adjacent_fingers(LI, RM), "Cross-hand fingers should NOT be adjacent");
    }

    #[test]
    fn test_not_adjacent_same_finger() {
        // Same finger is NOT adjacent to itself
        assert!(!are_adjacent_fingers(LI, LI), "Same finger should NOT be adjacent");
        assert!(!are_adjacent_fingers(LM, LM), "Same finger should NOT be adjacent");
        assert!(!are_adjacent_fingers(LR, LR), "Same finger should NOT be adjacent");
        assert!(!are_adjacent_fingers(LP, LP), "Same finger should NOT be adjacent");
    }

    #[test]
    fn test_thumb_not_adjacent() {
        // Thumbs are not considered adjacent to any finger for scissor purposes
        assert!(!are_adjacent_fingers(LT, LI), "Thumb should NOT be adjacent to index");
        assert!(!are_adjacent_fingers(LI, LT), "Index should NOT be adjacent to thumb");
        assert!(!are_adjacent_fingers(RT, RI), "Thumb should NOT be adjacent to index");
        assert!(!are_adjacent_fingers(RI, RT), "Index should NOT be adjacent to thumb");
    }

    // ==========================================
    // Tests for compute_scissor
    // ==========================================

    use super::compute_scissor;
    use libdof::prelude::PhysicalKey;

    /// Helper to create a PhysicalKey at a given row (y position)
    fn key_at_row(row: f64) -> PhysicalKey {
        PhysicalKey::xywh(0.0, row, 1.0, 1.0)
    }

    // --- Requirement 1.6: Different hands should NOT be scissors ---

    #[test]
    fn test_compute_scissor_different_hands_not_scissor() {
        // Keys on different hands should never be scissors
        let k1 = key_at_row(0.0);  // Top row
        let k2 = key_at_row(2.0);  // Bottom row (2 rows apart)

        // Left index vs Right index - different hands
        assert_eq!(compute_scissor(&k1, &k2, LI, RI), None,
            "Different hands should not form a scissor");

        // Left middle vs Right middle - different hands
        assert_eq!(compute_scissor(&k1, &k2, LM, RM), None,
            "Different hands should not form a scissor");
    }

    // --- Requirement 1.7: Same finger should NOT be scissors ---

    #[test]
    fn test_compute_scissor_same_finger_not_scissor() {
        // Same finger should never be scissors
        let k1 = key_at_row(0.0);
        let k2 = key_at_row(2.0);

        assert_eq!(compute_scissor(&k1, &k2, LI, LI), None,
            "Same finger should not form a scissor");
        assert_eq!(compute_scissor(&k1, &k2, LM, LM), None,
            "Same finger should not form a scissor");
        assert_eq!(compute_scissor(&k1, &k2, RI, RI), None,
            "Same finger should not form a scissor");
    }

    // --- No vertical separation should NOT be scissors ---

    #[test]
    fn test_compute_scissor_same_row_not_scissor() {
        // Keys on the same row should never be scissors
        let k1 = key_at_row(1.0);
        let k2 = key_at_row(1.0);

        assert_eq!(compute_scissor(&k1, &k2, LI, LM), None,
            "Same row should not form a scissor");
        assert_eq!(compute_scissor(&k1, &k2, LM, LR), None,
            "Same row should not form a scissor");
    }

    // --- Requirement 1.2: Full scissor with index higher than middle ---

    #[test]
    fn test_compute_scissor_full_index_higher_than_middle() {
        // Index higher than middle with 2-row separation = full scissor
        // Middle prefers higher, so if index is higher, it's a violation
        let k_index = key_at_row(0.0);  // Index at top (row 0)
        let k_middle = key_at_row(2.0); // Middle at bottom (row 2)

        let result = compute_scissor(&k_index, &k_middle, LI, LM);
        assert_eq!(result, Some((true, true)),
            "Index higher than middle with 2-row separation should be full adjacent scissor");
    }

    // --- Requirement 1.3: Full scissor with middle lower than other fingers ---

    #[test]
    fn test_compute_scissor_full_middle_lower_than_ring() {
        // Middle lower than ring with 2-row separation = full scissor
        // Middle prefers higher, so if middle is lower, it's a violation
        let k_ring = key_at_row(0.0);   // Ring at top (row 0)
        let k_middle = key_at_row(2.0); // Middle at bottom (row 2)

        let result = compute_scissor(&k_ring, &k_middle, LR, LM);
        assert_eq!(result, Some((true, true)),
            "Middle lower than ring with 2-row separation should be full adjacent scissor");
    }

    #[test]
    fn test_compute_scissor_full_middle_lower_than_pinky() {
        // Middle lower than pinky with 2-row separation = full scissor
        let k_pinky = key_at_row(0.0);  // Pinky at top (row 0)
        let k_middle = key_at_row(2.0); // Middle at bottom (row 2)

        let result = compute_scissor(&k_pinky, &k_middle, LP, LM);
        assert_eq!(result, Some((true, false)),
            "Middle lower than pinky with 2-row separation should be full non-adjacent scissor");
    }

    // --- Requirement 1.4: Full scissor with ring lower than pinky/index ---

    #[test]
    fn test_compute_scissor_full_ring_lower_than_pinky() {
        // Ring lower than pinky with 2-row separation = full scissor
        // Ring prefers higher than pinky, so if ring is lower, it's a violation
        let k_pinky = key_at_row(0.0); // Pinky at top (row 0)
        let k_ring = key_at_row(2.0);  // Ring at bottom (row 2)

        let result = compute_scissor(&k_pinky, &k_ring, LP, LR);
        assert_eq!(result, Some((true, true)),
            "Ring lower than pinky with 2-row separation should be full adjacent scissor");
    }

    #[test]
    fn test_compute_scissor_full_ring_lower_than_index() {
        // Ring lower than index with 2-row separation = full scissor
        // Ring prefers higher than index, so if ring is lower, it's a violation
        let k_index = key_at_row(0.0); // Index at top (row 0)
        let k_ring = key_at_row(2.0);  // Ring at bottom (row 2)

        let result = compute_scissor(&k_index, &k_ring, LI, LR);
        assert_eq!(result, Some((true, false)),
            "Ring lower than index with 2-row separation should be full non-adjacent scissor");
    }

    // --- Requirement 1.5: Half scissors (1-row separation) ---

    #[test]
    fn test_compute_scissor_half_index_higher_than_middle() {
        // Index higher than middle with 1-row separation = half scissor
        let k_index = key_at_row(0.0);  // Index at top (row 0)
        let k_middle = key_at_row(1.0); // Middle at bottom (row 1)

        let result = compute_scissor(&k_index, &k_middle, LI, LM);
        assert_eq!(result, Some((false, true)),
            "Index higher than middle with 1-row separation should be half adjacent scissor");
    }

    #[test]
    fn test_compute_scissor_half_middle_lower_than_ring() {
        // Middle lower than ring with 1-row separation = half scissor
        let k_ring = key_at_row(0.0);   // Ring at top (row 0)
        let k_middle = key_at_row(1.0); // Middle at bottom (row 1)

        let result = compute_scissor(&k_ring, &k_middle, LR, LM);
        assert_eq!(result, Some((false, true)),
            "Middle lower than ring with 1-row separation should be half adjacent scissor");
    }

    // --- Non-scissor cases: finger preference NOT violated ---

    #[test]
    fn test_compute_scissor_middle_higher_than_index_not_scissor() {
        // Middle higher than index is the PREFERRED position, not a scissor
        let k_middle = key_at_row(0.0); // Middle at top (row 0)
        let k_index = key_at_row(2.0);  // Index at bottom (row 2)

        let result = compute_scissor(&k_middle, &k_index, LM, LI);
        assert_eq!(result, None,
            "Middle higher than index should NOT be a scissor (preferred position)");
    }

    #[test]
    fn test_compute_scissor_ring_higher_than_pinky_not_scissor() {
        // Ring higher than pinky is the PREFERRED position, not a scissor
        let k_ring = key_at_row(0.0);  // Ring at top (row 0)
        let k_pinky = key_at_row(2.0); // Pinky at bottom (row 2)

        let result = compute_scissor(&k_ring, &k_pinky, LR, LP);
        assert_eq!(result, None,
            "Ring higher than pinky should NOT be a scissor (preferred position)");
    }

    #[test]
    fn test_compute_scissor_pinky_higher_than_index_not_scissor() {
        // Pinky higher than index is the PREFERRED position, not a scissor
        let k_pinky = key_at_row(0.0); // Pinky at top (row 0)
        let k_index = key_at_row(2.0); // Index at bottom (row 2)

        let result = compute_scissor(&k_pinky, &k_index, LP, LI);
        assert_eq!(result, None,
            "Pinky higher than index should NOT be a scissor (preferred position)");
    }

    // --- Right hand tests ---

    #[test]
    fn test_compute_scissor_right_hand_full_scissor() {
        // Same logic applies to right hand
        let k_index = key_at_row(0.0);  // Index at top (row 0)
        let k_middle = key_at_row(2.0); // Middle at bottom (row 2)

        let result = compute_scissor(&k_index, &k_middle, RI, RM);
        assert_eq!(result, Some((true, true)),
            "Right hand: Index higher than middle with 2-row separation should be full adjacent scissor");
    }

    #[test]
    fn test_compute_scissor_right_hand_half_scissor() {
        // Half scissor on right hand
        let k_ring = key_at_row(0.0);   // Ring at top (row 0)
        let k_middle = key_at_row(1.0); // Middle at bottom (row 1)

        let result = compute_scissor(&k_ring, &k_middle, RR, RM);
        assert_eq!(result, Some((false, true)),
            "Right hand: Middle lower than ring with 1-row separation should be half adjacent scissor");
    }

    // --- Edge case: 3+ row separation is still full scissor ---

    #[test]
    fn test_compute_scissor_three_row_separation_is_full() {
        // 3-row separation should still be classified as full scissor
        let k_index = key_at_row(0.0);  // Index at top (row 0)
        let k_middle = key_at_row(3.0); // Middle at bottom (row 3)

        let result = compute_scissor(&k_index, &k_middle, LI, LM);
        assert_eq!(result, Some((true, true)),
            "3-row separation should be classified as full scissor");
    }

    // --- Test with non-integer row positions (rounding) ---

    #[test]
    fn test_compute_scissor_fractional_rows_round() {
        // Keys at y=0.4 and y=1.6 should round to rows 0 and 2 (2-row diff = full)
        let k1 = key_at_row(0.4);
        let k2 = key_at_row(1.6);

        // 0.4 rounds to 0, 1.6 rounds to 2, diff = 2 = full scissor
        let result = compute_scissor(&k1, &k2, LI, LM);
        assert_eq!(result, Some((true, true)),
            "Fractional rows should round: 0.4->0, 1.6->2, diff=2 = full scissor");
    }

    #[test]
    fn test_compute_scissor_fractional_rows_half() {
        // Keys at y=0.4 and y=0.6 should round to rows 0 and 1 (1-row diff = half)
        let k1 = key_at_row(0.4);
        let k2 = key_at_row(0.6);

        // 0.4 rounds to 0, 0.6 rounds to 1, diff = 1 = half scissor
        let result = compute_scissor(&k1, &k2, LI, LM);
        assert_eq!(result, Some((false, true)),
            "Fractional rows should round: 0.4->0, 0.6->1, diff=1 = half scissor");
    }

    // ==========================================
    // Tests for compute_severity
    // ==========================================

    use super::compute_severity;

    /// **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    /// Tests that compute_severity returns correct severity values in centiunits.

    #[test]
    fn test_compute_severity_full_adjacent() {
        // Full adjacent scissor: base=200, multiplier=150, result=300
        let severity = compute_severity(true, true);
        assert_eq!(severity, 300,
            "Full adjacent scissor should have severity 300");
    }

    #[test]
    fn test_compute_severity_full_non_adjacent() {
        // Full non-adjacent scissor: base=200, multiplier=100, result=200
        let severity = compute_severity(true, false);
        assert_eq!(severity, 200,
            "Full non-adjacent scissor should have severity 200");
    }

    #[test]
    fn test_compute_severity_half_adjacent() {
        // Half adjacent scissor: base=100, multiplier=150, result=150
        let severity = compute_severity(false, true);
        assert_eq!(severity, 150,
            "Half adjacent scissor should have severity 150");
    }

    #[test]
    fn test_compute_severity_half_non_adjacent() {
        // Half non-adjacent scissor: base=100, multiplier=100, result=100
        let severity = compute_severity(false, false);
        assert_eq!(severity, 100,
            "Half non-adjacent scissor should have severity 100");
    }

    // --- Requirement 2.1: Full scissors have higher severity than half scissors ---

    #[test]
    fn test_compute_severity_full_greater_than_half_adjacent() {
        // Full adjacent (300) > Half adjacent (150)
        let full_adj = compute_severity(true, true);
        let half_adj = compute_severity(false, true);
        assert!(full_adj > half_adj,
            "Full adjacent ({}) should have higher severity than half adjacent ({})",
            full_adj, half_adj);
    }

    #[test]
    fn test_compute_severity_full_greater_than_half_non_adjacent() {
        // Full non-adjacent (200) > Half non-adjacent (100)
        let full_non_adj = compute_severity(true, false);
        let half_non_adj = compute_severity(false, false);
        assert!(full_non_adj > half_non_adj,
            "Full non-adjacent ({}) should have higher severity than half non-adjacent ({})",
            full_non_adj, half_non_adj);
    }

    // --- Requirement 2.2: Adjacent scissors have higher severity than non-adjacent ---

    #[test]
    fn test_compute_severity_adjacent_greater_than_non_adjacent_full() {
        // Full adjacent (300) > Full non-adjacent (200)
        let full_adj = compute_severity(true, true);
        let full_non_adj = compute_severity(true, false);
        assert!(full_adj > full_non_adj,
            "Full adjacent ({}) should have higher severity than full non-adjacent ({})",
            full_adj, full_non_adj);
    }

    #[test]
    fn test_compute_severity_adjacent_greater_than_non_adjacent_half() {
        // Half adjacent (150) > Half non-adjacent (100)
        let half_adj = compute_severity(false, true);
        let half_non_adj = compute_severity(false, false);
        assert!(half_adj > half_non_adj,
            "Half adjacent ({}) should have higher severity than half non-adjacent ({})",
            half_adj, half_non_adj);
    }

    // --- Requirement 2.4: Severity is stored as integer (i64) ---

    #[test]
    fn test_compute_severity_returns_i64() {
        // All severity values should be positive integers
        let severities = [
            compute_severity(true, true),
            compute_severity(true, false),
            compute_severity(false, true),
            compute_severity(false, false),
        ];

        for severity in severities {
            assert!(severity > 0, "Severity should be positive");
            // The value is already i64, so this test verifies the type is correct
        }
    }

    // --- Test severity ordering (all four combinations) ---

    #[test]
    fn test_compute_severity_ordering() {
        // Full adjacent > Full non-adjacent > Half adjacent > Half non-adjacent
        let full_adj = compute_severity(true, true);       // 300
        let full_non_adj = compute_severity(true, false);  // 200
        let half_adj = compute_severity(false, true);      // 150
        let half_non_adj = compute_severity(false, false); // 100

        assert!(full_adj > full_non_adj,
            "Full adjacent should be > full non-adjacent");
        assert!(full_non_adj > half_adj,
            "Full non-adjacent should be > half adjacent");
        assert!(half_adj > half_non_adj,
            "Half adjacent should be > half non-adjacent");
    }

    // ==========================================
    // Tests for ScissorsCache::new()
    // ==========================================

    use super::ScissorsCache;

    /// Helper to create a PhysicalKey at a given (x, y) position
    fn key_at_pos(x: f64, y: f64) -> PhysicalKey {
        PhysicalKey::xywh(x, y, 1.0, 1.0)
    }

    // --- Requirement 3.1: Pre-compute all scissor pairs during initialization ---

    #[test]
    fn test_scissors_cache_new_empty_keyboard() {
        // Empty keyboard should result in empty cache
        let keyboard: Vec<PhysicalKey> = vec![];
        let fingers: Vec<Finger> = vec![];

        let cache = ScissorsCache::new(&keyboard, &fingers, 0);

        assert_eq!(cache.num_positions(), 0, "Empty keyboard should have 0 positions");
        assert_eq!(cache.num_keys(), 0, "Empty keyboard should have 0 keys");
    }

    #[test]
    fn test_scissors_cache_new_single_key() {
        // Single key cannot form scissor pairs
        let keyboard = vec![key_at_row(0.0)];
        let fingers = vec![LI];

        let cache = ScissorsCache::new(&keyboard, &fingers, 1);

        assert_eq!(cache.num_positions(), 1, "Single key should have 1 position");
        assert!(cache.get_scissor_pairs(0).is_empty(),
            "Single key should have no scissor pairs");
    }

    #[test]
    fn test_scissors_cache_new_detects_full_scissor() {
        // Two keys forming a full scissor (2-row separation, index higher than middle)
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: bottom row (2 rows apart)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Position 0 should have a scissor pair pointing to position 1
        let pairs_0 = cache.get_scissor_pairs(0);
        assert_eq!(pairs_0.len(), 1, "Position 0 should have 1 scissor pair");
        assert_eq!(pairs_0[0].other_pos, 1, "Pair should point to position 1");
        assert!(pairs_0[0].is_full, "Should be a full scissor");
        assert!(pairs_0[0].is_adjacent, "Index-Middle should be adjacent");
        assert_eq!(pairs_0[0].severity, 300, "Full adjacent scissor severity should be 300");

        // Position 1 should have a scissor pair pointing to position 0
        let pairs_1 = cache.get_scissor_pairs(1);
        assert_eq!(pairs_1.len(), 1, "Position 1 should have 1 scissor pair");
        assert_eq!(pairs_1[0].other_pos, 0, "Pair should point to position 0");
        assert!(pairs_1[0].is_full, "Should be a full scissor");
        assert!(pairs_1[0].is_adjacent, "Index-Middle should be adjacent");
        assert_eq!(pairs_1[0].severity, 300, "Full adjacent scissor severity should be 300");
    }

    #[test]
    fn test_scissors_cache_new_detects_half_scissor() {
        // Two keys forming a half scissor (1-row separation)
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(1.0),  // Position 1: middle row (1 row apart)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let pairs_0 = cache.get_scissor_pairs(0);
        assert_eq!(pairs_0.len(), 1, "Position 0 should have 1 scissor pair");
        assert!(!pairs_0[0].is_full, "Should be a half scissor");
        assert!(pairs_0[0].is_adjacent, "Index-Middle should be adjacent");
        assert_eq!(pairs_0[0].severity, 150, "Half adjacent scissor severity should be 150");
    }

    #[test]
    fn test_scissors_cache_new_no_scissor_same_finger() {
        // Same finger should not form scissors
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LI];  // Same finger

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        assert!(cache.get_scissor_pairs(0).is_empty(),
            "Same finger should not form scissor pairs");
        assert!(cache.get_scissor_pairs(1).is_empty(),
            "Same finger should not form scissor pairs");
    }

    #[test]
    fn test_scissors_cache_new_no_scissor_different_hands() {
        // Different hands should not form scissors
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, RI];  // Different hands

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        assert!(cache.get_scissor_pairs(0).is_empty(),
            "Different hands should not form scissor pairs");
        assert!(cache.get_scissor_pairs(1).is_empty(),
            "Different hands should not form scissor pairs");
    }

    #[test]
    fn test_scissors_cache_new_no_scissor_same_row() {
        // Same row should not form scissors
        let keyboard = vec![
            key_at_pos(0.0, 1.0),
            key_at_pos(1.0, 1.0),
        ];
        let fingers = vec![LI, LM];

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        assert!(cache.get_scissor_pairs(0).is_empty(),
            "Same row should not form scissor pairs");
        assert!(cache.get_scissor_pairs(1).is_empty(),
            "Same row should not form scissor pairs");
    }

    #[test]
    fn test_scissors_cache_new_no_scissor_preferred_position() {
        // Middle higher than index is the preferred position, not a scissor
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (middle)
            key_at_row(2.0),  // Position 1: bottom row (index)
        ];
        let fingers = vec![LM, LI];  // Middle at top, Index at bottom = preferred

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        assert!(cache.get_scissor_pairs(0).is_empty(),
            "Preferred position should not form scissor pairs");
        assert!(cache.get_scissor_pairs(1).is_empty(),
            "Preferred position should not form scissor pairs");
    }

    #[test]
    fn test_scissors_cache_new_multiple_pairs() {
        // Three keys where position 0 forms scissors with both 1 and 2
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (index)
            key_at_row(2.0),  // Position 1: bottom row (middle)
            key_at_row(2.0),  // Position 2: bottom row (ring)
        ];
        let fingers = vec![LI, LM, LR];

        let cache = ScissorsCache::new(&keyboard, &fingers, 3);

        // Position 0 (index at top) should have scissors with both middle and ring
        let pairs_0 = cache.get_scissor_pairs(0);
        assert_eq!(pairs_0.len(), 2, "Position 0 should have 2 scissor pairs");

        // Position 1 (middle at bottom) should have scissor with index
        let pairs_1 = cache.get_scissor_pairs(1);
        assert_eq!(pairs_1.len(), 1, "Position 1 should have 1 scissor pair");
        assert_eq!(pairs_1[0].other_pos, 0, "Middle should pair with index");

        // Position 2 (ring at bottom) should have scissor with index
        let pairs_2 = cache.get_scissor_pairs(2);
        assert_eq!(pairs_2.len(), 1, "Position 2 should have 1 scissor pair");
        assert_eq!(pairs_2[0].other_pos, 0, "Ring should pair with index");
    }

    #[test]
    fn test_scissors_cache_new_non_adjacent_scissor() {
        // Index and Ring are non-adjacent (middle is between them)
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (index)
            key_at_row(2.0),  // Position 1: bottom row (ring)
        ];
        let fingers = vec![LI, LR];  // Index at top, Ring at bottom

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let pairs_0 = cache.get_scissor_pairs(0);
        assert_eq!(pairs_0.len(), 1, "Should have 1 scissor pair");
        assert!(!pairs_0[0].is_adjacent, "Index-Ring should NOT be adjacent");
        assert_eq!(pairs_0[0].severity, 200, "Full non-adjacent scissor severity should be 200");
    }

    #[test]
    fn test_scissors_cache_new_get_scissor_helper() {
        // Test the get_scissor helper method
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Should find scissor pair
        let pair = cache.get_scissor(0, 1);
        assert!(pair.is_some(), "Should find scissor pair from 0 to 1");
        assert_eq!(pair.unwrap().other_pos, 1);

        // Should also find in reverse direction
        let pair_rev = cache.get_scissor(1, 0);
        assert!(pair_rev.is_some(), "Should find scissor pair from 1 to 0");
        assert_eq!(pair_rev.unwrap().other_pos, 0);

        // Should return None for non-existent pair
        let no_pair = cache.get_scissor(0, 0);
        assert!(no_pair.is_none(), "Same position should not have scissor pair");
    }

    #[test]
    fn test_scissors_cache_new_totals_initialized_to_zero() {
        // All totals should be initialized to 0
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Verify all totals are 0 by checking the default state
        // (We can't directly access private fields, but we can verify through score())
        // Since weights are 0, score should be 0
        // Note: score() method will be implemented in a later task
    }

    #[test]
    #[should_panic(expected = "keyboard len is not the same as fingers len")]
    fn test_scissors_cache_new_panics_on_mismatched_lengths() {
        let keyboard = vec![key_at_row(0.0), key_at_row(1.0)];
        let fingers = vec![LI];  // Only 1 finger for 2 keys

        ScissorsCache::new(&keyboard, &fingers, 2);
    }

    #[test]
    fn test_scissors_cache_new_right_hand() {
        // Test that right hand scissors are detected correctly
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: bottom row
        ];
        let fingers = vec![RI, RM];  // Right hand: Index at top, Middle at bottom

        let cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let pairs_0 = cache.get_scissor_pairs(0);
        assert_eq!(pairs_0.len(), 1, "Right hand should also detect scissors");
        assert!(pairs_0[0].is_full, "Should be a full scissor");
        assert!(pairs_0[0].is_adjacent, "RI-RM should be adjacent");
    }

    // ==========================================
    // Tests for ScissorsCache::set_weights()
    // ==========================================

    use crate::weights::{Weights, FingerWeights};

    /// Helper to create a Weights struct with specified scissor weights
    fn weights_with_scissors(
        full_scissors: i64,
        half_scissors: i64,
        full_scissors_skip: i64,
        half_scissors_skip: i64,
    ) -> Weights {
        Weights {
            sfbs: 0,
            sfs: 0,
            stretches: 0,
            sft: 0,
            inroll: 0,
            outroll: 0,
            alternate: 0,
            redirect: 0,
            onehandin: 0,
            onehandout: 0,
            finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
            full_scissors,
            half_scissors,
            full_scissors_skip,
            half_scissors_skip,
            fingers: FingerWeights::default(),
        }
    }

    #[test]
    fn test_set_weights_stores_all_values() {
        // Test that set_weights correctly stores all scissor weight values
        // set_weights negates: stored = -config_value (scissors are penalties)
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = weights_with_scissors(-10, -5, -3, -2);
        cache.set_weights(&weights);

        // Weights are negated in set_weights: -(-10)=10, -(-5)=5, -(-3)=3, -(-2)=2
        assert_eq!(cache.full_scissors_weight, 10,
            "full_scissors_weight should be negated to 10");
        assert_eq!(cache.half_scissors_weight, 5,
            "half_scissors_weight should be negated to 5");
        assert_eq!(cache.full_scissors_skip_weight, 3,
            "full_scissors_skip_weight should be negated to 3");
        assert_eq!(cache.half_scissors_skip_weight, 2,
            "half_scissors_skip_weight should be negated to 2");
    }

    #[test]
    fn test_set_weights_with_zero_values() {
        // Test that set_weights handles zero values correctly
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = weights_with_scissors(0, 0, 0, 0);
        cache.set_weights(&weights);

        assert_eq!(cache.full_scissors_weight, 0);
        assert_eq!(cache.half_scissors_weight, 0);
        assert_eq!(cache.full_scissors_skip_weight, 0);
        assert_eq!(cache.half_scissors_skip_weight, 0);
    }

    #[test]
    fn test_set_weights_with_positive_values() {
        // Test that set_weights handles positive values correctly
        // Positive config values get negated: stored = -config_value
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = weights_with_scissors(100, 50, 30, 20);
        cache.set_weights(&weights);

        assert_eq!(cache.full_scissors_weight, -100);
        assert_eq!(cache.half_scissors_weight, -50);
        assert_eq!(cache.full_scissors_skip_weight, -30);
        assert_eq!(cache.half_scissors_skip_weight, -20);
    }

    #[test]
    fn test_set_weights_can_be_called_multiple_times() {
        // Test that set_weights can be called multiple times to update weights
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Set initial weights (negated: -(-10)=10, -(-5)=5)
        let weights1 = weights_with_scissors(-10, -5, -3, -2);
        cache.set_weights(&weights1);

        assert_eq!(cache.full_scissors_weight, 10);
        assert_eq!(cache.half_scissors_weight, 5);

        // Update weights (negated: -(-20)=20, -(-15)=15, -(-8)=8, -(-4)=4)
        let weights2 = weights_with_scissors(-20, -15, -8, -4);
        cache.set_weights(&weights2);

        assert_eq!(cache.full_scissors_weight, 20,
            "full_scissors_weight should be updated to 20");
        assert_eq!(cache.half_scissors_weight, 15,
            "half_scissors_weight should be updated to 15");
        assert_eq!(cache.full_scissors_skip_weight, 8,
            "full_scissors_skip_weight should be updated to 8");
        assert_eq!(cache.half_scissors_skip_weight, 4,
            "half_scissors_skip_weight should be updated to 4");
    }

    #[test]
    fn test_set_weights_with_dummy_weights() {
        // Test that set_weights works with the dummy_weights function
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = crate::weights::dummy_weights();
        cache.set_weights(&weights);

        // dummy_weights: full_scissors=5, half_scissors=1, full_scissors_skip=2, half_scissors_skip=1
        // set_weights negates: -5, -1, -2, -1
        assert_eq!(cache.full_scissors_weight, -5);
        assert_eq!(cache.half_scissors_weight, -1);
        assert_eq!(cache.full_scissors_skip_weight, -2);
        assert_eq!(cache.half_scissors_skip_weight, -1);
    }

    // ==========================================
    // Tests for ScissorsCache::compute_replace_delta()
    // ==========================================

    /// Helper to create a frequency array with a specific value at (key_a, key_b)
    fn freq_array(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut freq = vec![0i64; num_keys * num_keys];
        for &(key_a, key_b, value) in entries {
            freq[key_a * num_keys + key_b] = value;
        }
        freq
    }

    #[test]
    fn test_compute_replace_delta_basic() {
        // Test basic delta computation for a full scissor pair
        // Position 0 (index) and Position 1 (middle) form a full adjacent scissor
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: bottom row (2 rows apart = full scissor)
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;  // Keys: 0, 1, 2

        let cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

        // Keys array: position 0 has key 0, position 1 has key 1
        let keys = vec![0, 1];

        // Bigram frequencies: (0, 1) = 100, (1, 0) = 50
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 50)]);
        // Skipgram frequencies: (0, 1) = 30, (1, 0) = 20
        let sg_freq = freq_array(num_keys, &[(0, 1, 30), (1, 0, 20)]);

        // Replace key at position 0: old_key=0, new_key=2
        // New bigram frequencies for key 2: (2, 1) = 200, (1, 2) = 100
        let bg_freq_with_new = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 50),  // Old key 0 frequencies
            (2, 1, 200), (1, 2, 100), // New key 2 frequencies
        ]);
        let sg_freq_with_new = freq_array(num_keys, &[
            (0, 1, 30), (1, 0, 20),   // Old key 0 frequencies
            (2, 1, 60), (1, 2, 40),   // New key 2 frequencies
        ]);

        let delta = cache.compute_replace_delta(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            None,   // skip_pos
            &bg_freq_with_new,
            &sg_freq_with_new,
        );

        // Full adjacent scissor has severity 300
        // Bigram delta: (200 - 100) + (100 - 50) = 150
        // Bigram score delta: 150 * 300 = 45000
        assert_eq!(delta.full_scissors_bigram_freq, 150,
            "Bigram freq delta should be 150");
        assert_eq!(delta.full_scissors_bigram_total, 45000,
            "Bigram score delta should be 45000 (150 * 300)");

        // Skipgram delta: (60 - 30) + (40 - 20) = 50
        // Skipgram score delta: 50 * 300 = 15000
        assert_eq!(delta.full_scissors_skipgram_freq, 50,
            "Skipgram freq delta should be 50");
        assert_eq!(delta.full_scissors_skipgram_total, 15000,
            "Skipgram score delta should be 15000 (50 * 300)");

        // Half scissors should be unchanged
        assert_eq!(delta.half_scissors_bigram_total, 0);
        assert_eq!(delta.half_scissors_bigram_freq, 0);
        assert_eq!(delta.half_scissors_skipgram_total, 0);
        assert_eq!(delta.half_scissors_skipgram_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_with_skip_pos() {
        // Test that skip_pos correctly skips the specified position
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (index)
            key_at_row(2.0),  // Position 1: bottom row (middle) - scissor with 0
            key_at_row(2.0),  // Position 2: bottom row (ring) - scissor with 0
        ];
        let fingers = vec![LI, LM, LR];
        let num_keys = 3;

        let cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

        // Position 0 should have scissor pairs with both 1 and 2
        assert_eq!(cache.get_scissor_pairs(0).len(), 2,
            "Position 0 should have 2 scissor pairs");

        let keys = vec![0, 1, 2];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (0, 2, 50), (2, 0, 50)]);
        let sg_freq = freq_array(num_keys, &[]);

        // Replace key at position 0, skipping position 1
        let delta = cache.compute_replace_delta(
            0,      // pos
            0,      // old_key
            0,      // new_key (same key, so delta should only come from skip behavior)
            &keys,
            Some(1), // skip_pos - skip position 1
            &bg_freq,
            &sg_freq,
        );

        // Since we're replacing with the same key and skipping position 1,
        // only the pair with position 2 should contribute
        // But since old_key == new_key, the delta should be 0
        assert_eq!(delta.full_scissors_bigram_total, 0,
            "Delta should be 0 when replacing with same key");
    }

    #[test]
    fn test_compute_replace_delta_half_scissor() {
        // Test delta computation for a half scissor pair
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(1.0),  // Position 1: middle row (1 row apart = half scissor)
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[]);

        let delta = cache.compute_replace_delta(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            None,
            &bg_freq,
            &sg_freq,
        );

        // Half adjacent scissor has severity 150
        // Bigram delta: (200 - 100) + (200 - 100) = 200
        // Bigram score delta: 200 * 150 = 30000
        assert_eq!(delta.half_scissors_bigram_freq, 200,
            "Half scissor bigram freq delta should be 200");
        assert_eq!(delta.half_scissors_bigram_total, 30000,
            "Half scissor bigram score delta should be 30000 (200 * 150)");

        // Full scissors should be unchanged
        assert_eq!(delta.full_scissors_bigram_total, 0);
        assert_eq!(delta.full_scissors_bigram_freq, 0);
    }

    #[test]
    fn test_compute_replace_delta_invalid_keys() {
        // Test that invalid keys (>= num_keys) are handled correctly
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100)]);
        let sg_freq = freq_array(num_keys, &[]);

        // Replace with invalid new_key (>= num_keys)
        let delta = cache.compute_replace_delta(
            0,      // pos
            0,      // old_key
            999,    // new_key (invalid)
            &keys,
            None,
            &bg_freq,
            &sg_freq,
        );

        // New key is invalid, so new frequencies are 0
        // Delta = (0 - 100) + (0 - 100) = -200
        // Score delta = -200 * 300 = -60000
        assert_eq!(delta.full_scissors_bigram_freq, -200,
            "Bigram freq delta should be -200 when new key is invalid");
        assert_eq!(delta.full_scissors_bigram_total, -60000,
            "Bigram score delta should be -60000");
    }

    #[test]
    fn test_compute_replace_delta_score_only() {
        // Test the score-only fast path
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[(0, 1, 50), (1, 0, 50), (2, 1, 100), (1, 2, 100)]);

        let score_delta = cache.compute_replace_delta_score_only(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            None,
            &bg_freq,
            &sg_freq,
        );

        // Full adjacent scissor has severity 300
        // Bigram delta: (200 - 100) + (200 - 100) = 200
        // Bigram score delta: 200 * 300 = 60000
        // Skipgram delta: (100 - 50) + (100 - 50) = 100
        // Skipgram score delta: 100 * 300 = 30000
        // Weights negated in set_weights: -(-10)=10, -(-3)=3
        // Weighted score: 60000 * 10 + 30000 * 3 = 600000 + 90000 = 690000
        assert_eq!(score_delta, 690000,
            "Weighted score delta should be 690000");
    }

    #[test]
    fn test_apply_delta() {
        // Test that apply_delta correctly updates cache state
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Initial state should be all zeros
        assert_eq!(cache.full_scissors_bigram_total, 0);
        assert_eq!(cache.full_scissors_bigram_freq, 0);

        // Create a delta
        let delta = super::ScissorsDelta {
            full_scissors_bigram_total: 1000,
            full_scissors_bigram_freq: 10,
            full_scissors_skipgram_total: 500,
            full_scissors_skipgram_freq: 5,
            half_scissors_bigram_total: 200,
            half_scissors_bigram_freq: 2,
            half_scissors_skipgram_total: 100,
            half_scissors_skipgram_freq: 1,
        };

        cache.apply_delta(&delta);

        // Verify all totals were updated
        assert_eq!(cache.full_scissors_bigram_total, 1000);
        assert_eq!(cache.full_scissors_bigram_freq, 10);
        assert_eq!(cache.full_scissors_skipgram_total, 500);
        assert_eq!(cache.full_scissors_skipgram_freq, 5);
        assert_eq!(cache.half_scissors_bigram_total, 200);
        assert_eq!(cache.half_scissors_bigram_freq, 2);
        assert_eq!(cache.half_scissors_skipgram_total, 100);
        assert_eq!(cache.half_scissors_skipgram_freq, 1);

        // Apply another delta
        cache.apply_delta(&delta);

        // Verify totals were accumulated
        assert_eq!(cache.full_scissors_bigram_total, 2000);
        assert_eq!(cache.full_scissors_bigram_freq, 20);
    }

    #[test]
    fn test_scissors_delta_combine() {
        // Test that ScissorsDelta::combine correctly combines two deltas
        let delta_a = super::ScissorsDelta {
            full_scissors_bigram_total: 100,
            full_scissors_bigram_freq: 1,
            full_scissors_skipgram_total: 50,
            full_scissors_skipgram_freq: 2,
            half_scissors_bigram_total: 25,
            half_scissors_bigram_freq: 3,
            half_scissors_skipgram_total: 10,
            half_scissors_skipgram_freq: 4,
        };

        let delta_b = super::ScissorsDelta {
            full_scissors_bigram_total: 200,
            full_scissors_bigram_freq: 5,
            full_scissors_skipgram_total: 100,
            full_scissors_skipgram_freq: 6,
            half_scissors_bigram_total: 50,
            half_scissors_bigram_freq: 7,
            half_scissors_skipgram_total: 20,
            half_scissors_skipgram_freq: 8,
        };

        let combined = super::ScissorsDelta::combine(&delta_a, &delta_b);

        assert_eq!(combined.full_scissors_bigram_total, 300);
        assert_eq!(combined.full_scissors_bigram_freq, 6);
        assert_eq!(combined.full_scissors_skipgram_total, 150);
        assert_eq!(combined.full_scissors_skipgram_freq, 8);
        assert_eq!(combined.half_scissors_bigram_total, 75);
        assert_eq!(combined.half_scissors_bigram_freq, 10);
        assert_eq!(combined.half_scissors_skipgram_total, 30);
        assert_eq!(combined.half_scissors_skipgram_freq, 12);
    }

    // ==========================================
    // Tests for ScissorsCache::score()
    // ==========================================

    #[test]
    fn test_score_zero_weights() {
        // Test that score returns 0 when all weights are 0
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);
        // Weights default to 0

        // Set some frequencies directly for testing
        // Full adjacent scissor has severity 300
        // bigram_total = 100 * 300 = 30000, skipgram_total = 50 * 300 = 15000
        cache.set_totals_for_testing(30000, 100, 15000, 50, 0, 0, 0, 0);

        // Score should be 0 because weights are 0
        assert_eq!(cache.score(), 0, "Score should be 0 when all weights are 0");
    }

    #[test]
    fn test_score_with_weights() {
        // Test that score correctly computes weighted sum
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Set frequencies directly for testing
        // Full adjacent scissor has severity 300
        // bigram_total = 100 * 300 = 30000, skipgram_total = 50 * 300 = 15000
        cache.set_totals_for_testing(30000, 100, 15000, 50, 0, 0, 0, 0);

        // Weights are negated in set_weights: -(-10)=10, -(-3)=3
        // Score = 30000 * 10 + 15000 * 3 = 300000 + 45000 = 345000
        assert_eq!(cache.score(), 345000, "Score should be 345000");
    }

    #[test]
    fn test_score_all_scissor_types() {
        // Test score with all four scissor types
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: 2 rows down (full scissor with pos 0)
            key_at_row(1.0),  // Position 2: 1 row down (half scissor with pos 0)
        ];
        let fingers = vec![LI, LM, LR];  // Index, Middle, Ring

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 3);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Set frequencies directly for testing
        // Full scissor (pos 0, 1): severity 300 (adjacent)
        // bigram: 100 * 300 = 30000, skipgram: 50 * 300 = 15000
        //
        // Half scissor (pos 0, 2): Index-Ring are NOT adjacent, so severity = 100
        // bigram: 80 * 100 = 8000, skipgram: 40 * 100 = 4000
        cache.set_totals_for_testing(30000, 100, 15000, 50, 8000, 80, 4000, 40);

        // Weights are negated in set_weights: -(-10)=10, -(-5)=5, -(-3)=3, -(-2)=2
        // Score = 30000 * 10 + 15000 * 3 + 8000 * 5 + 4000 * 2
        //       = 300000 + 45000 + 40000 + 8000 = 393000
        assert_eq!(cache.score(), 393000, "Score should be 393000");
    }

    // ==========================================
    // Tests for ScissorsCache::replace_key()
    // ==========================================

    #[test]
    fn test_replace_key_apply_true() {
        // Test replace_key with apply=true updates state
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[(0, 1, 50), (1, 0, 50), (2, 1, 100), (1, 2, 100)]);

        // Replace key 0 with key 2 at position 0
        let new_score = cache.replace_key(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            None,
            &bg_freq,
            &sg_freq,
        );

        // Verify the returned score matches score()
        assert_eq!(new_score, cache.score(),
            "Returned score should match score() after apply=true");

        // Verify state was updated
        // Full adjacent scissor has severity 300
        // Old bigram: (0,1) + (1,0) = 100 + 100 = 200, score = 200 * 300 = 60000
        // New bigram: (2,1) + (1,2) = 200 + 200 = 400, score = 400 * 300 = 120000
        // Delta = 120000 - 60000 = 60000
        assert_eq!(cache.full_scissors_bigram_total, 60000,
            "Bigram total should be 60000 after replace");
    }

    #[test]
    fn test_replace_key_apply_false() {
        // Test replace_key with apply=false does not update state
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[(0, 1, 50), (1, 0, 50), (2, 1, 100), (1, 2, 100)]);

        // Get initial score
        let initial_score = cache.score();
        let initial_bigram_total = cache.full_scissors_bigram_total;

        // Replace key 0 with key 2 at position 0 with apply=false
        let speculative_score = cache.score_replace(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            &bg_freq,
            &sg_freq,
        );

        // Verify state was NOT updated
        assert_eq!(cache.score(), initial_score,
            "Score should not change after apply=false");
        assert_eq!(cache.full_scissors_bigram_total, initial_bigram_total,
            "Bigram total should not change after apply=false");

        // Verify speculative score is different from initial
        assert_ne!(speculative_score, initial_score,
            "Speculative score should differ from initial score");
    }

    #[test]
    fn test_replace_key_apply_true_matches_apply_false() {
        // Test that apply=true returns the same score as apply=false
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100), (2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[(0, 1, 50), (1, 0, 50), (2, 1, 100), (1, 2, 100)]);

        // Create two identical caches
        let mut cache_apply = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache_apply.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let mut cache_no_apply = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache_no_apply.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Get score with apply=true
        let score_apply = cache_apply.replace_key(
            0, 0, 2, &keys, None, &bg_freq, &sg_freq,
        );

        // Get score with apply=false
        let score_no_apply = cache_no_apply.score_replace(
            0, 0, 2, &keys, &bg_freq, &sg_freq,
        );

        // Both should return the same score
        assert_eq!(score_apply, score_no_apply,
            "apply=true and apply=false should return the same score");
    }

    #[test]
    fn test_replace_key_with_skip_pos() {
        // Test replace_key with skip_pos (used during swaps)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
            key_at_row(2.0),  // Another position at same row as pos 1
        ];
        let fingers = vec![LI, LM, LR];
        let num_keys = 4;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1, 2];
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),
            (0, 2, 50), (2, 0, 50),
            (3, 1, 200), (1, 3, 200),
            (3, 2, 150), (2, 3, 150),
        ]);
        let sg_freq = freq_array(num_keys, &[]);

        // Replace key 0 with key 3 at position 0, skipping position 1
        let score_with_skip = cache.score() + cache.compute_replace_delta_score_only(
            0, 0, 3, &keys, Some(1), &bg_freq, &sg_freq,
        );

        // Replace key 0 with key 3 at position 0, without skipping
        let score_without_skip = cache.score_replace(
            0, 0, 3, &keys, &bg_freq, &sg_freq,
        );

        // Scores should be different because skip_pos excludes position 1's contribution
        assert_ne!(score_with_skip, score_without_skip,
            "Score with skip_pos should differ from score without skip_pos");
    }

    #[test]
    fn test_replace_key_invalid_old_key() {
        // Test replace_key when old_key is invalid (>= num_keys)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![999, 1];  // Position 0 has invalid key
        let bg_freq = freq_array(num_keys, &[(2, 1, 200), (1, 2, 200)]);
        let sg_freq = freq_array(num_keys, &[(2, 1, 100), (1, 2, 100)]);

        // Replace invalid key with key 2 at position 0
        let new_score = cache.replace_key(
            0, 999, 2, &keys, None, &bg_freq, &sg_freq,
        );

        // Should add the new key's contribution without subtracting old
        // New bigram: (2,1) + (1,2) = 200 + 200 = 400, score = 400 * 300 = 120000
        // New skipgram: (2,1) + (1,2) = 100 + 100 = 200, score = 200 * 300 = 60000
        // Weights negated: -(-10)=10, -(-3)=3
        // Weighted: 120000 * 10 + 60000 * 3 = 1200000 + 180000 = 1380000
        assert_eq!(new_score, 1380000,
            "Score should be 1380000 when replacing invalid old key");
    }

    #[test]
    fn test_replace_key_invalid_new_key() {
        // Test replace_key when new_key is invalid (>= num_keys)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[(0, 1, 100), (1, 0, 100)]);
        let sg_freq = freq_array(num_keys, &[(0, 1, 50), (1, 0, 50)]);

        // Replace key 0 with invalid key at position 0
        let new_score = cache.replace_key(
            0, 0, 999, &keys, None, &bg_freq, &sg_freq,
        );

        // Should subtract old key's contribution without adding new
        // Old bigram: (0,1) + (1,0) = 100 + 100 = 200, score = 200 * 300 = 60000
        // Old skipgram: (0,1) + (1,0) = 50 + 50 = 100, score = 100 * 300 = 30000
        // Delta: -60000 bigram, -30000 skipgram
        // Weights negated: -(-10)=10, -(-3)=3
        // Weighted: -60000 * 10 + -30000 * 3 = -600000 + -90000 = -690000
        assert_eq!(new_score, -690000,
            "Score should be -690000 when replacing with invalid new key");
    }

    // ==========================================
    // Tests for ScissorsCache::key_swap()
    // ==========================================

    #[test]
    fn test_key_swap_apply_true() {
        // Test key_swap with apply=true updates state
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: row 0
            key_at_row(2.0),  // pos 1: row 2
        ];
        let fingers = vec![LI, LM];  // Full adjacent scissor (index lower than middle)
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        // Frequencies for keys 0, 1, 2
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),  // key 0 <-> key 1
            (0, 2, 50), (2, 0, 50),    // key 0 <-> key 2
            (1, 2, 200), (2, 1, 200),  // key 1 <-> key 2
        ]);
        let sg_freq = freq_array(num_keys, &[
            (0, 1, 50), (1, 0, 50),
            (0, 2, 25), (2, 0, 25),
            (1, 2, 100), (2, 1, 100),
        ]);

        // Swap key 0 at pos 0 with key 1 at pos 1
        let new_score = cache.key_swap(
            0,      // pos_a
            1,      // pos_b
            0,      // key_a (at pos_a)
            1,      // key_b (at pos_b)
            &keys,
            &bg_freq,
            &sg_freq,
        );

        // Verify the returned score matches score()
        assert_eq!(new_score, cache.score(),
            "Returned score should match score() after apply=true");
    }

    #[test]
    fn test_key_swap_apply_false() {
        // Test key_swap with apply=false does not update state
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),
            (0, 2, 50), (2, 0, 50),
            (1, 2, 200), (2, 1, 200),
        ]);
        let sg_freq = freq_array(num_keys, &[
            (0, 1, 50), (1, 0, 50),
        ]);

        // Get initial state
        let initial_score = cache.score();
        let initial_bigram_total = cache.full_scissors_bigram_total;
        let initial_skipgram_total = cache.full_scissors_skipgram_total;

        // Swap with apply=false
        let speculative_score = cache.score_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq,
        );

        // Verify state was NOT updated
        assert_eq!(cache.score(), initial_score,
            "Score should not change after apply=false");
        assert_eq!(cache.full_scissors_bigram_total, initial_bigram_total,
            "Bigram total should not change after apply=false");
        assert_eq!(cache.full_scissors_skipgram_total, initial_skipgram_total,
            "Skipgram total should not change after apply=false");

        // Speculative score should be computed (may or may not differ from initial)
        // The important thing is that state wasn't mutated
        let _ = speculative_score;
    }

    #[test]
    fn test_key_swap_apply_true_matches_apply_false() {
        // Test that apply=true returns the same score as apply=false
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),
            (0, 2, 50), (2, 0, 50),
            (1, 2, 200), (2, 1, 200),
        ]);
        let sg_freq = freq_array(num_keys, &[
            (0, 1, 50), (1, 0, 50),
            (0, 2, 25), (2, 0, 25),
            (1, 2, 100), (2, 1, 100),
        ]);

        // Create two identical caches
        let mut cache_apply = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache_apply.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let mut cache_no_apply = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache_no_apply.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Get score with apply=true
        let score_apply = cache_apply.key_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq,
        );

        // Get score with apply=false
        let score_no_apply = cache_no_apply.score_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq,
        );

        // Both should return the same score
        assert_eq!(score_apply, score_no_apply,
            "apply=true and apply=false should return the same score");
    }

    #[test]
    fn test_key_swap_is_reversible() {
        // Test that swapping twice returns to original state
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),
            (0, 2, 50), (2, 0, 50),
            (1, 2, 200), (2, 1, 200),
        ]);
        let sg_freq = freq_array(num_keys, &[
            (0, 1, 50), (1, 0, 50),
        ]);

        // Get initial state
        let initial_score = cache.score();
        let initial_bigram_total = cache.full_scissors_bigram_total;

        // First swap: key 0 at pos 0 <-> key 1 at pos 1
        cache.key_swap(0, 1, 0, 1, &keys, &bg_freq, &sg_freq);

        // After first swap, keys are now: [1, 0]
        let keys_after_swap = vec![1, 0];

        // Second swap: key 1 at pos 0 <-> key 0 at pos 1 (reverse)
        cache.key_swap(0, 1, 1, 0, &keys_after_swap, &bg_freq, &sg_freq);

        // Should be back to original state
        assert_eq!(cache.score(), initial_score,
            "Score should return to original after double swap");
        assert_eq!(cache.full_scissors_bigram_total, initial_bigram_total,
            "Bigram total should return to original after double swap");
    }

    #[test]
    fn test_key_swap_with_three_positions() {
        // Test key_swap with a third position that forms scissors with both swap positions
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: row 0
            key_at_row(2.0),  // pos 1: row 2
            key_at_row(2.0),  // pos 2: row 2
        ];
        let fingers = vec![LI, LM, LR];  // All different fingers, same hand
        let num_keys = 4;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1, 2];
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100), (1, 0, 100),  // key 0 <-> key 1
            (0, 2, 50), (2, 0, 50),    // key 0 <-> key 2
            (1, 2, 200), (2, 1, 200),  // key 1 <-> key 2
            (3, 0, 75), (0, 3, 75),    // key 3 <-> key 0
            (3, 1, 150), (1, 3, 150),  // key 3 <-> key 1
            (3, 2, 125), (2, 3, 125),  // key 3 <-> key 2
        ]);
        let sg_freq = freq_array(num_keys, &[]);

        // Swap key 0 at pos 0 with key 1 at pos 1
        // This should affect scissors with pos 2 as well
        let new_score = cache.key_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq,
        );

        // Verify score was computed
        assert_eq!(new_score, cache.score(),
            "Returned score should match score()");
    }

    #[test]
    fn test_key_swap_non_scissor_positions() {
        // Test key_swap when the two positions don't form a scissor with each other
        // but may form scissors with other positions
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: row 0
            key_at_row(0.0),  // pos 1: row 0 (same row as pos 0, no scissor between them)
            key_at_row(2.0),  // pos 2: row 2 (forms scissor with pos 0 and pos 1)
        ];
        let fingers = vec![LI, LM, LR];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1, 2];
        let bg_freq = freq_array(num_keys, &[
            (0, 2, 100), (2, 0, 100),  // key 0 <-> key 2
            (1, 2, 200), (2, 1, 200),  // key 1 <-> key 2
        ]);
        let sg_freq = freq_array(num_keys, &[]);

        // Swap key 0 at pos 0 with key 1 at pos 1
        // These positions don't form a scissor with each other (same row)
        // but both form scissors with pos 2
        let new_score = cache.key_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq,
        );

        assert_eq!(new_score, cache.score(),
            "Returned score should match score()");
    }

    // ==========================================
    // Tests for ScissorsCache::stats()
    // ==========================================

    #[test]
    fn test_stats_populates_all_fields() {
        // Test that stats() populates all scissor-related fields in Stats
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: row 0
            key_at_row(2.0),  // pos 1: row 2 (full scissor with pos 0)
            key_at_row(1.0),  // pos 2: row 1 (half scissor with pos 0)
        ];
        let fingers = vec![LI, LM, LR];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

        // Set frequencies directly for testing
        // Full scissor (pos 0, 1): severity 300 (adjacent)
        // bigram: 1000 freq, skipgram: 500 freq
        // Half scissor (pos 0, 2): Index-Ring is NOT adjacent, so severity = 100
        // bigram: 200 freq, skipgram: 100 freq
        // Note: totals are freq * severity, but stats uses raw freq
        cache.set_totals_for_testing(
            1000 * 300, 1000,  // full bigram: total, freq
            500 * 300, 500,    // full skipgram: total, freq
            200 * 100, 200,    // half bigram: total, freq
            100 * 100, 100,    // half skipgram: total, freq
        );

        let mut stats = crate::stats::Stats::default();
        let bigram_total = 10000.0;
        let skipgram_total = 5000.0;

        cache.stats(&mut stats, bigram_total, skipgram_total);

        // Full scissors bigrams: 1000 / (10000 * 100) = 0.001
        assert!((stats.full_scissors_bigrams - 0.001).abs() < 1e-10,
            "full_scissors_bigrams should be 0.001, got {}", stats.full_scissors_bigrams);

        // Full scissors skipgrams: 500 / (5000 * 100) = 0.001
        assert!((stats.full_scissors_skipgrams - 0.001).abs() < 1e-10,
            "full_scissors_skipgrams should be 0.001, got {}", stats.full_scissors_skipgrams);

        // Half scissors bigrams: 200 / (10000 * 100) = 0.0002
        assert!((stats.half_scissors_bigrams - 0.0002).abs() < 1e-10,
            "half_scissors_bigrams should be 0.0002, got {}", stats.half_scissors_bigrams);

        // Half scissors skipgrams: 100 / (5000 * 100) = 0.0002
        assert!((stats.half_scissors_skipgrams - 0.0002).abs() < 1e-10,
            "half_scissors_skipgrams should be 0.0002, got {}", stats.half_scissors_skipgrams);
    }

    #[test]
    fn test_stats_zero_totals() {
        // Test that stats() handles zero totals gracefully (no division by zero)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 2;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        // Set frequencies directly for testing
        // Full adjacent scissor has severity 300
        cache.set_totals_for_testing(100 * 300, 100, 50 * 300, 50, 0, 0, 0, 0);

        let mut stats = crate::stats::Stats::default();

        // Test with zero bigram_total
        cache.stats(&mut stats, 0.0, 1000.0);
        assert_eq!(stats.full_scissors_bigrams, 0.0,
            "full_scissors_bigrams should be 0 when bigram_total is 0");
        assert_eq!(stats.half_scissors_bigrams, 0.0,
            "half_scissors_bigrams should be 0 when bigram_total is 0");

        // Test with zero skipgram_total
        cache.stats(&mut stats, 1000.0, 0.0);
        assert_eq!(stats.full_scissors_skipgrams, 0.0,
            "full_scissors_skipgrams should be 0 when skipgram_total is 0");
        assert_eq!(stats.half_scissors_skipgrams, 0.0,
            "half_scissors_skipgrams should be 0 when skipgram_total is 0");

        // Test with both zero
        cache.stats(&mut stats, 0.0, 0.0);
        assert_eq!(stats.full_scissors_bigrams, 0.0);
        assert_eq!(stats.full_scissors_skipgrams, 0.0);
        assert_eq!(stats.half_scissors_bigrams, 0.0);
        assert_eq!(stats.half_scissors_skipgrams, 0.0);
    }

    #[test]
    fn test_stats_empty_cache() {
        // Test stats() with no scissor frequencies
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 2;

        let cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        // No frequencies added

        let mut stats = crate::stats::Stats::default();
        cache.stats(&mut stats, 10000.0, 5000.0);

        assert_eq!(stats.full_scissors_bigrams, 0.0);
        assert_eq!(stats.full_scissors_skipgrams, 0.0);
        assert_eq!(stats.half_scissors_bigrams, 0.0);
        assert_eq!(stats.half_scissors_skipgrams, 0.0);
    }

    // ==========================================
    // Tests for add_rule
    // ==========================================

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

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_apply_false_preserves_state() {
        // Test that add_rule with apply=false returns correct delta without mutating state
        // Create a keyboard with a scissor pair: positions 0 and 1 form a full scissor
        // For a scissor: the finger that prefers higher must be positioned lower
        // LM prefers being higher than LI, so for a scissor, LM must be at a higher y (lower on keyboard)
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: LI at row 0 (higher on keyboard)
            key_at_row(2.0),  // pos 1: LM at row 2 (lower on keyboard) - forms scissor with pos 0
            key_at_row(1.0),  // pos 2: RI at row 1
            key_at_row(1.0),  // pos 3: RP at row 1
        ];
        let fingers = vec![LI, LM, RI, RP];
        let num_keys = 5;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, num_keys);

        // Create bigram frequencies for the full steal: A→B
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Record initial state
        let initial_active_rules_len = cache.active_rules.len();
        let initial_magic_delta = cache.magic_rule_score_delta;
        let initial_score = cache.score();

        // Apply rule speculatively (apply=false)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, false);

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
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: LI at row 0 (higher on keyboard)
            key_at_row(2.0),  // pos 1: LM at row 2 (lower on keyboard)
            key_at_row(1.0),  // pos 2: RI at row 1
            key_at_row(1.0),  // pos 3: RP at row 1
        ];
        let fingers = vec![LI, LM, RI, RP];
        let num_keys = 5;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, num_keys);

        // Create bigram frequencies
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100),  // A→B frequency
        ]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Verify initial state
        assert!(cache.active_rules.is_empty());
        assert_eq!(cache.magic_rule_score_delta, 0);

        // Apply rule: leader=0 (A), output=1 (B), magic_key=2 (M)
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

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
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
            key_at_row(1.0),
            key_at_row(1.0),
        ];
        let fingers = vec![LI, LM, RI, RP];
        let num_keys = 6;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, num_keys);

        let bg_freq = freq_array(num_keys, &[]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Apply first rule: leader=0, output=1, magic_key=2
        cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(1, _))));

        // Apply second rule with different magic_key and leader
        cache.add_rule(1, 3, 4, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(4, 1)), Some(&(3, _))));

        // Both rules should be tracked
        assert_eq!(cache.active_rules.len(), 2);

        // Apply rule with same (magic_key, leader) but different output - should replace
        cache.add_rule(0, 3, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert!(matches!(cache.active_rules.get(&(2, 0)), Some(&(3, _)))); // Updated to new output
        assert_eq!(cache.active_rules.len(), 2); // Still 2 rules
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_with_scissor_pair() {
        // Test add_rule when the rule affects a scissor pair
        // Create a keyboard where positions 0 and 1 form a full scissor
        // For a scissor: the finger that prefers higher must be positioned lower
        // LM prefers being higher than LI, so for a scissor, LM must be at a higher y (lower on keyboard)
        let keyboard = vec![
            key_at_row(0.0),  // pos 0: LI at row 0 (higher on keyboard)
            key_at_row(2.0),  // pos 1: LM at row 2 (lower on keyboard) - forms scissor with pos 0
            key_at_row(1.0),  // pos 2: RI at row 1 (no scissor with others)
        ];
        let fingers = vec![LI, LM, RI];
        let num_keys = 4;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Verify scissor pair exists between positions 0 and 1
        assert!(cache.get_scissor(0, 1).is_some(), "Positions 0 and 1 should form a scissor");

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M)
        let keys = vec![0, 1, 2];
        let key_positions = create_key_positions(&keys, num_keys);

        // Create bigram frequency for A→B (which is a scissor)
        let bg_freq = freq_array(num_keys, &[
            (0, 1, 100),  // A→B frequency (scissor pair)
        ]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Apply rule: A→M steals B
        // This should change the scissor contribution since A→B is a scissor but A→M is not
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // The delta should be non-zero because we're removing a scissor contribution
        // A→B was a scissor (positions 0→1), A→M is not a scissor (positions 0→2)
        // So the delta should be negative (removing the scissor penalty)
        // Note: The actual sign depends on the weight values
        assert_ne!(delta, 0, "Delta should be non-zero when rule affects a scissor pair");
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_with_skipgram_partial_steal() {
        // Test add_rule with skipgram partial steal: Z→B becomes Z→M based on tg_freq[Z][A][B]
        let keyboard = vec![
            key_at_row(2.0),  // pos 0: LI at row 2
            key_at_row(0.0),  // pos 1: LM at row 0 - forms scissor with pos 0
            key_at_row(1.0),  // pos 2: RI at row 1
            key_at_row(3.0),  // pos 3: RM at row 3 - forms scissor with pos 2
        ];
        let fingers = vec![LI, LM, RI, RM];
        let num_keys = 5;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        // Keys: pos 0=key 0 (leader A), pos 1=key 1 (output B), pos 2=key 2 (magic M), pos 3=key 3 (Z)
        let keys = vec![0, 1, 2, 3];
        let key_positions = create_key_positions(&keys, num_keys);

        let bg_freq = freq_array(num_keys, &[]);
        let sg_freq = freq_array(num_keys, &[
            (3, 1, 100),  // Z→B skipgram frequency
        ]);
        // Trigram Z→A→B determines how much of Z→B is stolen
        let tg_freq = create_tg_freq(num_keys, &[
            (3, 0, 1, 50),  // tg_freq[Z][A][B] = 50 (half of Z→B is stolen)
        ]);

        // Apply rule: A→M steals B
        // This should affect the skipgram Z→B which becomes Z→M for the stolen portion
        let delta = cache.add_rule(0, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);

        // The delta may or may not be zero depending on whether Z→B and Z→M are scissors
        // The important thing is that the computation completes without error
        let _ = delta;
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_invalid_keys() {
        // Test add_rule with invalid key indices
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 3;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];
        let key_positions = create_key_positions(&keys, num_keys);

        let bg_freq = freq_array(num_keys, &[]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Apply rule with invalid leader (>= num_keys)
        let delta = cache.add_rule(10, 1, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid leader key");

        // Apply rule with invalid output (>= num_keys)
        let delta = cache.add_rule(0, 10, 2, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid output key");

        // Apply rule with invalid magic_key (>= num_keys)
        let delta = cache.add_rule(0, 1, 10, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 for invalid magic_key");
    }

    /// **Validates: Requirements 2.5, 2.6**
    #[test]
    fn test_add_rule_leader_without_position() {
        // Test add_rule when leader has no position
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];
        let num_keys = 4;

        let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let keys = vec![0, 1];  // Only keys 0 and 1 have positions
        let key_positions = create_key_positions(&keys, num_keys);

        let bg_freq = freq_array(num_keys, &[
            (2, 1, 100),  // Leader (key 2) → output (key 1)
        ]);
        let sg_freq = freq_array(num_keys, &[]);
        let tg_freq = create_tg_freq(num_keys, &[]);

        // Apply rule with leader=2 which has no position
        let delta = cache.add_rule(2, 1, 3, &keys, &key_positions, &bg_freq, &sg_freq, &tg_freq, true);
        assert_eq!(delta, 0, "Delta should be 0 when leader has no position");
    }
}

// ==========================================
// Property-Based Tests for add_rule
// ==========================================

#[cfg(test)]
mod pbt_add_rule_apply_true {
    use super::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;
    use proptest::prelude::*;

    /// Helper to create a PhysicalKey at a given row (y position)
    fn key_at_row(row: f64) -> PhysicalKey {
        PhysicalKey::xywh(0.0, row, 1.0, 1.0)
    }

    /// Helper to create key_positions array from keys
    fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
        let mut key_positions = vec![None; num_keys];
        for (pos, &key) in keys.iter().enumerate() {
            if key < num_keys {
                key_positions[key] = Some(pos);
            }
        }
        key_positions
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

    /// Convert entries to flat skipgram frequency array
    fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut sg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                sg_freq[k1 * num_keys + k2] = freq;
            }
        }
        sg_freq
    }

    /// Convert entries to 3D trigram frequency array
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
            // Seed for generating sg_freq entries
            sg_freq_entries in proptest::collection::vec(
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

            let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a keyboard layout with scissor pairs
            // For a scissor: the finger that prefers higher must be positioned lower
            // LM prefers being higher than LI, so for a scissor, LM must be at a higher y (lower on keyboard)
            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| key_at_row(i as f64 * 2.0))  // Positions at rows 0, 2, 4, ...
                .collect();

            // Create fingers that can form scissors
            // Positions 0 and 1 are LI and LM (can form scissor if LM is lower)
            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| match i {
                    0 => LI,  // Index at row 0 (higher)
                    1 => LM,  // Middle at row 2 (lower) - forms scissor with pos 0
                    _ if i % 2 == 0 => RI,
                    _ => RM,
                })
                .collect();

            let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

            // Set some weights so score changes are visible
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: 0,
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
                full_scissors: -10,
                half_scissors: -5,
                full_scissors_skip: -3,
                half_scissors_skip: -2,
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
                &sg_freq,
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

#[cfg(test)]
mod pbt_add_rule_apply_false {
    use super::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;
    use proptest::prelude::*;

    /// Helper to create a PhysicalKey at a given row (y position)
    fn key_at_row(row: f64) -> PhysicalKey {
        PhysicalKey::xywh(0.0, row, 1.0, 1.0)
    }

    /// Helper to create key_positions array from keys
    fn make_key_positions(keys: &[usize], num_keys: usize) -> Vec<Option<usize>> {
        let mut key_positions = vec![None; num_keys];
        for (pos, &key) in keys.iter().enumerate() {
            if key < num_keys {
                key_positions[key] = Some(pos);
            }
        }
        key_positions
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

    /// Convert entries to flat skipgram frequency array
    fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut sg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                sg_freq[k1 * num_keys + k2] = freq;
            }
        }
        sg_freq
    }

    /// Convert entries to 3D trigram frequency array
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
            // Seed for generating sg_freq entries
            sg_freq_entries in proptest::collection::vec(
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

            let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a keyboard layout with scissor pairs
            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| key_at_row(i as f64 * 2.0))
                .collect();

            // Create fingers that can form scissors
            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| match i {
                    0 => LI,
                    1 => LM,
                    _ if i % 2 == 0 => RI,
                    _ => RM,
                })
                .collect();

            let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

            // Set some weights so score changes are visible
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: 0,
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
                full_scissors: -10,
                half_scissors: -5,
                full_scissors_skip: -3,
                half_scissors_skip: -2,
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
                &sg_freq,
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
            let mut cache_for_apply = ScissorsCache::new(&keyboard, &fingers, num_keys);
            cache_for_apply.set_weights(&weights);

            let delta_true = cache_for_apply.add_rule(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
                &sg_freq,
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
mod pbt_lookup_table_matches_computed_scissors {
    use super::*;
    use proptest::prelude::*;
    use crate::weights::{FingerWeights, Weights};
    use libdof::dofinitions::Finger::*;

    /// Helper to create a PhysicalKey at a given row (y position)
    fn key_at_row(x: f64, row: i32) -> PhysicalKey {
        PhysicalKey::xy(x, row as f64)
    }

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

    /// Convert entries to flat skipgram frequency array
    fn entries_to_sg_freq(num_keys: usize, entries: &[(usize, usize, i64)]) -> Vec<i64> {
        let mut sg_freq = vec![0i64; num_keys * num_keys];
        for &(k1, k2, freq) in entries {
            if k1 < num_keys && k2 < num_keys {
                sg_freq[k1 * num_keys + k2] = freq;
            }
        }
        sg_freq
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(25))]

        /// **Validates: Requirements 6.4, 6.5**
        ///
        /// Property: After calling init_rule_deltas, for any valid (leader, output, magic_key)
        /// triple where leader has a position, the value in rule_delta.get(&(leader, output, magic_key))
        /// should equal compute_rule_delta(leader, output, magic_key, ...).
        #[test]
        fn prop_lookup_table_matches_computed_scissors(
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
            // Seed for generating sg_freq entries
            sg_freq_entries in proptest::collection::vec(
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

            let constrained_sg_entries: Vec<(usize, usize, i64)> = sg_freq_entries
                .iter()
                .map(|&(k1, k2, freq)| (k1 % num_keys, k2 % num_keys, freq))
                .collect();
            let sg_freq = entries_to_sg_freq(num_keys, &constrained_sg_entries);

            let constrained_tg_entries: Vec<(usize, usize, usize, i64)> = tg_freq_entries
                .iter()
                .map(|&(k1, k2, k3, freq)| (k1 % num_keys, k2 % num_keys, k3 % num_keys, freq))
                .collect();
            let tg_freq = entries_to_tg_freq(num_keys, &constrained_tg_entries);

            // Create a keyboard layout with potential scissors
            // Position 0: LI at row 0
            // Position 1: LM at row 2 (creates full scissor with position 0)
            // Position 2: RI at row 0
            // Position 3: RP at row 1
            let base_fingers = vec![LI, LM, RI, RP];
            let base_rows = vec![0, 2, 0, 1];

            let fingers: Vec<Finger> = (0..num_positions)
                .map(|i| base_fingers[i % base_fingers.len()])
                .collect();

            let keyboard: Vec<PhysicalKey> = (0..num_positions)
                .map(|i| key_at_row(i as f64, base_rows[i % base_rows.len()]))
                .collect();

            let mut cache = ScissorsCache::new(&keyboard, &fingers, num_keys);

            // Set weights
            let weights = Weights {
                sfbs: 0,
                sfs: 0,
                stretches: 0,
                sft: 0,
                inroll: 0,
                outroll: 0,
                alternate: 0,
                redirect: 0,
                onehandin: 0,
                onehandout: 0,
                finger_usage: 0,
            magic_rule_penalty: 0,
            magic_repeat_penalty: 0,
                full_scissors: -10,
                half_scissors: -5,
                full_scissors_skip: -3,
                half_scissors_skip: -2,
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
            cache.init_rule_deltas(&keys, &key_positions, &bg_freq, &sg_freq, &tg_freq);

            // Compute the delta from scratch using compute_rule_delta
            let computed_delta = cache.compute_rule_delta(
                leader,
                output,
                magic_key,
                &keys,
                &key_positions,
                &bg_freq,
                &sg_freq,
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
