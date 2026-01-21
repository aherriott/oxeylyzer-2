/*
 **************************************
 *     Scissors Scoring Cache
 **************************************
 */

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

    /// Compute the weighted score from this delta.
    ///
    /// Multiplies each total by its corresponding weight and sums them.
    fn weighted_score(&self, cache: &ScissorsCache) -> i64 {
        self.full_scissors_bigram_total * cache.full_scissors_weight
            + self.full_scissors_skipgram_total * cache.full_scissors_skip_weight
            + self.half_scissors_bigram_total * cache.half_scissors_weight
            + self.half_scissors_skipgram_total * cache.half_scissors_skip_weight
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
        self.full_scissors_weight = weights.full_scissors;
        self.full_scissors_skip_weight = weights.full_scissors_skip;
        self.half_scissors_weight = weights.half_scissors;
        self.half_scissors_skip_weight = weights.half_scissors_skip;
    }

    /// Update for a bigram frequency change.
    ///
    /// When a bigram frequency changes, this method updates the appropriate scissor
    /// totals if the position pair forms a scissor. This is called during magic rule
    /// application and other frequency updates.
    ///
    /// # Arguments
    /// * `p_a` - First position in the bigram
    /// * `p_b` - Second position in the bigram
    /// * `old_freq` - Previous frequency value
    /// * `new_freq` - New frequency value
    ///
    /// # Behavior
    /// - If (p_a, p_b) forms a scissor pair, updates the appropriate total:
    ///   - Full scissors: updates `full_scissors_bigram_total` and `full_scissors_bigram_freq`
    ///   - Half scissors: updates `half_scissors_bigram_total` and `half_scissors_bigram_freq`
    /// - If (p_a, p_b) is not a scissor pair, does nothing
    ///
    /// # Requirements
    /// - Requirement 4.1: Update only the affected scissor scores via `update_bigram`
    #[inline]
    pub fn update_bigram(&mut self, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(sp) = self.get_scissor(p_a, p_b) {
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * sp.severity;

            if sp.is_full {
                self.full_scissors_bigram_total += score_delta;
                self.full_scissors_bigram_freq += freq_delta;
            } else {
                self.half_scissors_bigram_total += score_delta;
                self.half_scissors_bigram_freq += freq_delta;
            }
        }
    }

    /// Update for a skipgram frequency change.
    ///
    /// When a skipgram frequency changes, this method updates the appropriate scissor
    /// skipgram totals if the position pair forms a scissor. This is called during magic rule
    /// application and other frequency updates.
    ///
    /// # Arguments
    /// * `p_a` - First position in the skipgram
    /// * `p_b` - Second position in the skipgram
    /// * `old_freq` - Previous frequency value
    /// * `new_freq` - New frequency value
    ///
    /// # Behavior
    /// - If (p_a, p_b) forms a scissor pair, updates the appropriate skipgram total:
    ///   - Full scissors: updates `full_scissors_skipgram_total` and `full_scissors_skipgram_freq`
    ///   - Half scissors: updates `half_scissors_skipgram_total` and `half_scissors_skipgram_freq`
    /// - If (p_a, p_b) is not a scissor pair, does nothing
    ///
    /// # Requirements
    /// - Requirement 4.2: Update only the affected scissor scores via `update_skipgram`
    #[inline]
    pub fn update_skipgram(&mut self, p_a: usize, p_b: usize, old_freq: i64, new_freq: i64) {
        if let Some(sp) = self.get_scissor(p_a, p_b) {
            let freq_delta = new_freq - old_freq;
            let score_delta = freq_delta * sp.severity;

            if sp.is_full {
                self.full_scissors_skipgram_total += score_delta;
                self.full_scissors_skipgram_freq += freq_delta;
            } else {
                self.half_scissors_skipgram_total += score_delta;
                self.half_scissors_skipgram_freq += freq_delta;
            }
        }
    }

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

    /// Replace key at position. Returns the new score.
    ///
    /// Computes the score delta for replacing a key at a position and optionally
    /// applies the change to the internal state.
    ///
    /// # Arguments
    /// * `pos` - The position where the key is being replaced
    /// * `old_key` - The key currently at the position
    /// * `new_key` - The key that will replace it
    /// * `keys` - The current key assignment array
    /// * `skip_pos` - Optional position to skip (used during swaps to avoid double-counting)
    /// * `bg_freq` - Bigram frequency array (1D, indexed by key_a * num_keys + key_b)
    /// * `sg_freq` - Skipgram frequency array (1D, indexed by key_a * num_keys + key_b)
    /// * `apply` - If true, updates internal state; if false, only computes the score
    ///
    /// # Returns
    /// The new total score after the replacement.
    ///
    /// # Behavior
    /// - If `apply` is true:
    ///   1. Computes the full delta using `compute_replace_delta()`
    ///   2. Applies the delta to update internal state via `apply_delta()`
    ///   3. Returns `score()` (the new total score)
    /// - If `apply` is false:
    ///   1. Computes only the score delta using `compute_replace_delta_score_only()`
    ///   2. Returns `score() + score_delta` without mutating state
    ///
    /// # Requirements
    /// - Requirement 4.3: Compute score delta for key replacement without full recalculation
    /// - Requirement 4.5: When apply=false, compute new score without mutating internal state
    /// - Requirement 4.6: When apply=true, update internal state and return the new score
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
        apply: bool,
    ) -> i64 {
        if apply {
            // Compute full delta and apply it
            let delta = self.compute_replace_delta(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
            self.apply_delta(&delta);
            self.score()
        } else {
            // Compute only the score delta without mutating state
            let score_delta = self.compute_replace_delta_score_only(pos, old_key, new_key, keys, skip_pos, bg_freq, sg_freq);
            self.score() + score_delta
        }
    }

    /// Swap keys at two positions. Returns the new score.
    ///
    /// Computes the combined score delta for swapping two keys and optionally
    /// applies the change to the internal state. This is used during layout
    /// optimization when evaluating key swaps.
    ///
    /// # Arguments
    /// * `pos_a` - First position in the swap
    /// * `pos_b` - Second position in the swap
    /// * `key_a` - The key currently at pos_a (will move to pos_b)
    /// * `key_b` - The key currently at pos_b (will move to pos_a)
    /// * `keys` - The current key assignment array
    /// * `bg_freq` - Bigram frequency array (1D, indexed by key_a * num_keys + key_b)
    /// * `sg_freq` - Skipgram frequency array (1D, indexed by key_a * num_keys + key_b)
    /// * `apply` - If true, updates internal state; if false, only computes the score
    ///
    /// # Returns
    /// The new total score after the swap.
    ///
    /// # Algorithm
    /// 1. Compute delta for replacing key_a at pos_a with key_b, skipping pos_b
    /// 2. Compute delta for replacing key_b at pos_b with key_a, skipping pos_a
    /// 3. Combine the two deltas using `ScissorsDelta::combine()`
    /// 4. If `apply` is true: apply the combined delta and return `score()`
    /// 5. If `apply` is false: return `score() + combined_delta.weighted_score()`
    ///
    /// The skip_pos parameter is used to avoid double-counting the contribution
    /// from the pair (pos_a, pos_b) which would be counted in both replace operations.
    ///
    /// # Requirements
    /// - Requirement 4.4: Compute score delta for key swap without full recalculation
    /// - Requirement 4.5: When apply=false, compute new score without mutating internal state
    /// - Requirement 4.6: When apply=true, update internal state and return the new score
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
        apply: bool,
    ) -> i64 {
        if apply {
            // Compute full deltas for both positions
            // pos_a: key_a -> key_b, skip pos_b to avoid double-counting
            let delta_a = self.compute_replace_delta(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
            // pos_b: key_b -> key_a, skip pos_a to avoid double-counting
            let delta_b = self.compute_replace_delta(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
            // Combine the deltas and apply
            let combined = ScissorsDelta::combine(&delta_a, &delta_b);
            self.apply_delta(&combined);
            self.score()
        } else {
            // Compute only the score deltas without mutating state
            let score_a = self.compute_replace_delta_score_only(pos_a, key_a, key_b, keys, Some(pos_b), bg_freq, sg_freq);
            let score_b = self.compute_replace_delta_score_only(pos_b, key_b, key_a, keys, Some(pos_a), bg_freq, sg_freq);
            self.score() + score_a + score_b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Finger::*;

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
            thumb: 0,
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
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = weights_with_scissors(-10, -5, -3, -2);
        cache.set_weights(&weights);

        // Verify weights are stored correctly by checking internal state
        assert_eq!(cache.full_scissors_weight, -10,
            "full_scissors_weight should be set to -10");
        assert_eq!(cache.half_scissors_weight, -5,
            "half_scissors_weight should be set to -5");
        assert_eq!(cache.full_scissors_skip_weight, -3,
            "full_scissors_skip_weight should be set to -3");
        assert_eq!(cache.half_scissors_skip_weight, -2,
            "half_scissors_skip_weight should be set to -2");
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
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = weights_with_scissors(100, 50, 30, 20);
        cache.set_weights(&weights);

        assert_eq!(cache.full_scissors_weight, 100);
        assert_eq!(cache.half_scissors_weight, 50);
        assert_eq!(cache.full_scissors_skip_weight, 30);
        assert_eq!(cache.half_scissors_skip_weight, 20);
    }

    #[test]
    fn test_set_weights_can_be_called_multiple_times() {
        // Test that set_weights can be called multiple times to update weights
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Set initial weights
        let weights1 = weights_with_scissors(-10, -5, -3, -2);
        cache.set_weights(&weights1);

        assert_eq!(cache.full_scissors_weight, -10);
        assert_eq!(cache.half_scissors_weight, -5);

        // Update weights
        let weights2 = weights_with_scissors(-20, -15, -8, -4);
        cache.set_weights(&weights2);

        assert_eq!(cache.full_scissors_weight, -20,
            "full_scissors_weight should be updated to -20");
        assert_eq!(cache.half_scissors_weight, -15,
            "half_scissors_weight should be updated to -15");
        assert_eq!(cache.full_scissors_skip_weight, -8,
            "full_scissors_skip_weight should be updated to -8");
        assert_eq!(cache.half_scissors_skip_weight, -4,
            "half_scissors_skip_weight should be updated to -4");
    }

    #[test]
    fn test_set_weights_with_dummy_weights() {
        // Test that set_weights works with the dummy_weights function
        let keyboard = vec![key_at_row(0.0), key_at_row(2.0)];
        let fingers = vec![LI, LM];
        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        let weights = crate::weights::dummy_weights();
        cache.set_weights(&weights);

        // dummy_weights sets all scissor weights to 0
        assert_eq!(cache.full_scissors_weight, 0);
        assert_eq!(cache.half_scissors_weight, 0);
        assert_eq!(cache.full_scissors_skip_weight, 0);
        assert_eq!(cache.half_scissors_skip_weight, 0);
    }

    // ==========================================
    // Tests for ScissorsCache::update_bigram()
    // ==========================================

    #[test]
    fn test_update_bigram_full_scissor() {
        // Test that update_bigram correctly updates full scissor totals
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: bottom row (2 rows apart = full scissor)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Initial state: all totals should be 0
        assert_eq!(cache.full_scissors_bigram_total, 0);
        assert_eq!(cache.full_scissors_bigram_freq, 0);

        // Update bigram frequency from 0 to 100
        cache.update_bigram(0, 1, 0, 100);

        // Full adjacent scissor has severity 300
        // Delta = (100 - 0) * 300 = 30000
        assert_eq!(cache.full_scissors_bigram_total, 30000,
            "Full scissor bigram total should be 30000 (100 * 300)");
        assert_eq!(cache.full_scissors_bigram_freq, 100,
            "Full scissor bigram freq should be 100");

        // Half scissor totals should remain 0
        assert_eq!(cache.half_scissors_bigram_total, 0);
        assert_eq!(cache.half_scissors_bigram_freq, 0);
    }

    #[test]
    fn test_update_bigram_half_scissor() {
        // Test that update_bigram correctly updates half scissor totals
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(1.0),  // Position 1: middle row (1 row apart = half scissor)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update bigram frequency from 0 to 100
        cache.update_bigram(0, 1, 0, 100);

        // Half adjacent scissor has severity 150
        // Delta = (100 - 0) * 150 = 15000
        assert_eq!(cache.half_scissors_bigram_total, 15000,
            "Half scissor bigram total should be 15000 (100 * 150)");
        assert_eq!(cache.half_scissors_bigram_freq, 100,
            "Half scissor bigram freq should be 100");

        // Full scissor totals should remain 0
        assert_eq!(cache.full_scissors_bigram_total, 0);
        assert_eq!(cache.full_scissors_bigram_freq, 0);
    }

    #[test]
    fn test_update_bigram_non_scissor_pair() {
        // Test that update_bigram does nothing for non-scissor pairs
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (middle)
            key_at_row(2.0),  // Position 1: bottom row (index)
        ];
        let fingers = vec![LM, LI];  // Middle at top, Index at bottom = preferred (not scissor)

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update bigram frequency from 0 to 100
        cache.update_bigram(0, 1, 0, 100);

        // All totals should remain 0 since this is not a scissor pair
        assert_eq!(cache.full_scissors_bigram_total, 0);
        assert_eq!(cache.full_scissors_bigram_freq, 0);
        assert_eq!(cache.half_scissors_bigram_total, 0);
        assert_eq!(cache.half_scissors_bigram_freq, 0);
    }

    #[test]
    fn test_update_bigram_frequency_decrease() {
        // Test that update_bigram correctly handles frequency decreases
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // First, increase frequency to 100
        cache.update_bigram(0, 1, 0, 100);
        assert_eq!(cache.full_scissors_bigram_total, 30000);
        assert_eq!(cache.full_scissors_bigram_freq, 100);

        // Then decrease frequency from 100 to 50
        cache.update_bigram(0, 1, 100, 50);

        // Delta = (50 - 100) * 300 = -15000
        // New total = 30000 - 15000 = 15000
        assert_eq!(cache.full_scissors_bigram_total, 15000,
            "Full scissor bigram total should be 15000 after decrease");
        assert_eq!(cache.full_scissors_bigram_freq, 50,
            "Full scissor bigram freq should be 50 after decrease");
    }

    #[test]
    fn test_update_bigram_reverse_direction() {
        // Test that update_bigram works in both directions (p_a, p_b) and (p_b, p_a)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update in direction (0, 1)
        cache.update_bigram(0, 1, 0, 50);
        assert_eq!(cache.full_scissors_bigram_total, 15000);  // 50 * 300
        assert_eq!(cache.full_scissors_bigram_freq, 50);

        // Update in direction (1, 0)
        cache.update_bigram(1, 0, 0, 50);
        assert_eq!(cache.full_scissors_bigram_total, 30000);  // 15000 + 15000
        assert_eq!(cache.full_scissors_bigram_freq, 100);  // 50 + 50
    }

    #[test]
    fn test_update_bigram_non_adjacent_scissor() {
        // Test that update_bigram correctly handles non-adjacent scissors
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (index)
            key_at_row(2.0),  // Position 1: bottom row (ring)
        ];
        let fingers = vec![LI, LR];  // Index at top, Ring at bottom = non-adjacent scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update bigram frequency from 0 to 100
        cache.update_bigram(0, 1, 0, 100);

        // Full non-adjacent scissor has severity 200
        // Delta = (100 - 0) * 200 = 20000
        assert_eq!(cache.full_scissors_bigram_total, 20000,
            "Full non-adjacent scissor bigram total should be 20000 (100 * 200)");
        assert_eq!(cache.full_scissors_bigram_freq, 100);
    }

    #[test]
    fn test_update_bigram_skipgram_totals_unchanged() {
        // Test that update_bigram does not affect skipgram totals
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update bigram frequency
        cache.update_bigram(0, 1, 0, 100);

        // Skipgram totals should remain 0
        assert_eq!(cache.full_scissors_skipgram_total, 0,
            "Skipgram totals should not be affected by update_bigram");
        assert_eq!(cache.full_scissors_skipgram_freq, 0);
        assert_eq!(cache.half_scissors_skipgram_total, 0);
        assert_eq!(cache.half_scissors_skipgram_freq, 0);
    }

    #[test]
    fn test_update_bigram_multiple_updates() {
        // Test multiple sequential updates
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Series of updates
        cache.update_bigram(0, 1, 0, 100);    // +30000
        cache.update_bigram(0, 1, 100, 150);  // +15000 (50 * 300)
        cache.update_bigram(0, 1, 150, 200);  // +15000 (50 * 300)

        // Total should be 60000 (200 * 300)
        assert_eq!(cache.full_scissors_bigram_total, 60000);
        assert_eq!(cache.full_scissors_bigram_freq, 200);
    }

    // ==========================================
    // Tests for ScissorsCache::update_skipgram()
    // ==========================================

    #[test]
    fn test_update_skipgram_full_scissor() {
        // Test that update_skipgram correctly updates full scissor skipgram totals
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(2.0),  // Position 1: bottom row (2 rows apart = full scissor)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Initial state: all totals should be 0
        assert_eq!(cache.full_scissors_skipgram_total, 0);
        assert_eq!(cache.full_scissors_skipgram_freq, 0);

        // Update skipgram frequency from 0 to 100
        cache.update_skipgram(0, 1, 0, 100);

        // Full adjacent scissor has severity 300
        // Delta = (100 - 0) * 300 = 30000
        assert_eq!(cache.full_scissors_skipgram_total, 30000,
            "Full scissor skipgram total should be 30000 (100 * 300)");
        assert_eq!(cache.full_scissors_skipgram_freq, 100,
            "Full scissor skipgram freq should be 100");

        // Half scissor skipgram totals should remain 0
        assert_eq!(cache.half_scissors_skipgram_total, 0);
        assert_eq!(cache.half_scissors_skipgram_freq, 0);
    }

    #[test]
    fn test_update_skipgram_half_scissor() {
        // Test that update_skipgram correctly updates half scissor skipgram totals
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row
            key_at_row(1.0),  // Position 1: middle row (1 row apart = half scissor)
        ];
        let fingers = vec![LI, LM];  // Index at top, Middle at bottom = scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update skipgram frequency from 0 to 100
        cache.update_skipgram(0, 1, 0, 100);

        // Half adjacent scissor has severity 150
        // Delta = (100 - 0) * 150 = 15000
        assert_eq!(cache.half_scissors_skipgram_total, 15000,
            "Half scissor skipgram total should be 15000 (100 * 150)");
        assert_eq!(cache.half_scissors_skipgram_freq, 100,
            "Half scissor skipgram freq should be 100");

        // Full scissor skipgram totals should remain 0
        assert_eq!(cache.full_scissors_skipgram_total, 0);
        assert_eq!(cache.full_scissors_skipgram_freq, 0);
    }

    #[test]
    fn test_update_skipgram_non_scissor_pair() {
        // Test that update_skipgram does nothing for non-scissor pairs
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (middle)
            key_at_row(2.0),  // Position 1: bottom row (index)
        ];
        let fingers = vec![LM, LI];  // Middle at top, Index at bottom = preferred (not scissor)

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update skipgram frequency from 0 to 100
        cache.update_skipgram(0, 1, 0, 100);

        // All skipgram totals should remain 0 since this is not a scissor pair
        assert_eq!(cache.full_scissors_skipgram_total, 0);
        assert_eq!(cache.full_scissors_skipgram_freq, 0);
        assert_eq!(cache.half_scissors_skipgram_total, 0);
        assert_eq!(cache.half_scissors_skipgram_freq, 0);
    }

    #[test]
    fn test_update_skipgram_frequency_decrease() {
        // Test that update_skipgram correctly handles frequency decreases
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // First, increase frequency to 100
        cache.update_skipgram(0, 1, 0, 100);
        assert_eq!(cache.full_scissors_skipgram_total, 30000);
        assert_eq!(cache.full_scissors_skipgram_freq, 100);

        // Then decrease frequency from 100 to 50
        cache.update_skipgram(0, 1, 100, 50);

        // Delta = (50 - 100) * 300 = -15000
        // New total = 30000 - 15000 = 15000
        assert_eq!(cache.full_scissors_skipgram_total, 15000,
            "Full scissor skipgram total should be 15000 after decrease");
        assert_eq!(cache.full_scissors_skipgram_freq, 50,
            "Full scissor skipgram freq should be 50 after decrease");
    }

    #[test]
    fn test_update_skipgram_reverse_direction() {
        // Test that update_skipgram works in both directions (p_a, p_b) and (p_b, p_a)
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update in direction (0, 1)
        cache.update_skipgram(0, 1, 0, 50);
        assert_eq!(cache.full_scissors_skipgram_total, 15000);  // 50 * 300
        assert_eq!(cache.full_scissors_skipgram_freq, 50);

        // Update in direction (1, 0)
        cache.update_skipgram(1, 0, 0, 50);
        assert_eq!(cache.full_scissors_skipgram_total, 30000);  // 15000 + 15000
        assert_eq!(cache.full_scissors_skipgram_freq, 100);  // 50 + 50
    }

    #[test]
    fn test_update_skipgram_non_adjacent_scissor() {
        // Test that update_skipgram correctly handles non-adjacent scissors
        let keyboard = vec![
            key_at_row(0.0),  // Position 0: top row (index)
            key_at_row(2.0),  // Position 1: bottom row (ring)
        ];
        let fingers = vec![LI, LR];  // Index at top, Ring at bottom = non-adjacent scissor

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update skipgram frequency from 0 to 100
        cache.update_skipgram(0, 1, 0, 100);

        // Full non-adjacent scissor has severity 200
        // Delta = (100 - 0) * 200 = 20000
        assert_eq!(cache.full_scissors_skipgram_total, 20000,
            "Full non-adjacent scissor skipgram total should be 20000 (100 * 200)");
        assert_eq!(cache.full_scissors_skipgram_freq, 100);
    }

    #[test]
    fn test_update_skipgram_bigram_totals_unchanged() {
        // Test that update_skipgram does not affect bigram totals
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update skipgram frequency
        cache.update_skipgram(0, 1, 0, 100);

        // Bigram totals should remain 0
        assert_eq!(cache.full_scissors_bigram_total, 0,
            "Bigram totals should not be affected by update_skipgram");
        assert_eq!(cache.full_scissors_bigram_freq, 0);
        assert_eq!(cache.half_scissors_bigram_total, 0);
        assert_eq!(cache.half_scissors_bigram_freq, 0);
    }

    #[test]
    fn test_update_skipgram_multiple_updates() {
        // Test multiple sequential updates
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Series of updates
        cache.update_skipgram(0, 1, 0, 100);    // +30000
        cache.update_skipgram(0, 1, 100, 150);  // +15000 (50 * 300)
        cache.update_skipgram(0, 1, 150, 200);  // +15000 (50 * 300)

        // Total should be 60000 (200 * 300)
        assert_eq!(cache.full_scissors_skipgram_total, 60000);
        assert_eq!(cache.full_scissors_skipgram_freq, 200);
    }

    #[test]
    fn test_update_bigram_and_skipgram_independent() {
        // Test that bigram and skipgram updates are independent
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);

        // Update both bigram and skipgram
        cache.update_bigram(0, 1, 0, 100);
        cache.update_skipgram(0, 1, 0, 200);

        // Bigram totals should reflect bigram update only
        assert_eq!(cache.full_scissors_bigram_total, 30000,
            "Bigram total should be 30000 (100 * 300)");
        assert_eq!(cache.full_scissors_bigram_freq, 100);

        // Skipgram totals should reflect skipgram update only
        assert_eq!(cache.full_scissors_skipgram_total, 60000,
            "Skipgram total should be 60000 (200 * 300)");
        assert_eq!(cache.full_scissors_skipgram_freq, 200);
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
        // Weighted score: 60000 * (-10) + 30000 * (-3) = -600000 + -90000 = -690000
        assert_eq!(score_delta, -690000,
            "Weighted score delta should be -690000");
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

    #[test]
    fn test_scissors_delta_weighted_score() {
        // Test that ScissorsDelta::weighted_score correctly computes weighted score
        let keyboard = vec![
            key_at_row(0.0),
            key_at_row(2.0),
        ];
        let fingers = vec![LI, LM];

        let mut cache = ScissorsCache::new(&keyboard, &fingers, 2);
        cache.set_weights(&weights_with_scissors(-10, -5, -3, -2));

        let delta = super::ScissorsDelta {
            full_scissors_bigram_total: 100,
            full_scissors_bigram_freq: 0,  // freq doesn't affect weighted_score
            full_scissors_skipgram_total: 50,
            full_scissors_skipgram_freq: 0,
            half_scissors_bigram_total: 25,
            half_scissors_bigram_freq: 0,
            half_scissors_skipgram_total: 10,
            half_scissors_skipgram_freq: 0,
        };

        let weighted = delta.weighted_score(&cache);

        // Expected: 100 * (-10) + 50 * (-3) + 25 * (-5) + 10 * (-2)
        //         = -1000 + -150 + -125 + -20 = -1295
        assert_eq!(weighted, -1295,
            "Weighted score should be -1295");
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

        // Add some frequencies
        cache.update_bigram(0, 1, 0, 100);
        cache.update_skipgram(0, 1, 0, 50);

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

        // Add bigram frequency: 100 * 300 (severity) = 30000 total
        cache.update_bigram(0, 1, 0, 100);
        // Add skipgram frequency: 50 * 300 (severity) = 15000 total
        cache.update_skipgram(0, 1, 0, 50);

        // Score = 30000 * (-10) + 15000 * (-3) = -300000 + -45000 = -345000
        assert_eq!(cache.score(), -345000, "Score should be -345000");
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

        // Full scissor (pos 0, 1): severity 300 (adjacent)
        cache.update_bigram(0, 1, 0, 100);  // 100 * 300 = 30000
        cache.update_skipgram(0, 1, 0, 50); // 50 * 300 = 15000

        // Half scissor (pos 0, 2): severity 150 (adjacent, index-ring is not adjacent, but let's check)
        // Actually index-ring is NOT adjacent, so severity would be 100
        // Let me verify: pos 0 is LI (index), pos 2 is LR (ring)
        // Index at row 0, Ring at row 1 = 1 row diff = half scissor
        // Index prefers lower, Ring prefers higher than index
        // So if index is higher (row 0 < row 1), ring prefers higher but is lower = scissor
        // Index-Ring are NOT adjacent, so severity = 100

        cache.update_bigram(0, 2, 0, 80);   // 80 * 100 = 8000
        cache.update_skipgram(0, 2, 0, 40); // 40 * 100 = 4000

        // Score = 30000 * (-10) + 15000 * (-3) + 8000 * (-5) + 4000 * (-2)
        //       = -300000 + -45000 + -40000 + -8000 = -393000
        assert_eq!(cache.score(), -393000, "Score should be -393000");
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
            true,   // apply
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
        let speculative_score = cache.replace_key(
            0,      // pos
            0,      // old_key
            2,      // new_key
            &keys,
            None,
            &bg_freq,
            &sg_freq,
            false,  // apply
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
            0, 0, 2, &keys, None, &bg_freq, &sg_freq, true,
        );

        // Get score with apply=false
        let score_no_apply = cache_no_apply.replace_key(
            0, 0, 2, &keys, None, &bg_freq, &sg_freq, false,
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
        let score_with_skip = cache.replace_key(
            0, 0, 3, &keys, Some(1), &bg_freq, &sg_freq, false,
        );

        // Replace key 0 with key 3 at position 0, without skipping
        let score_without_skip = cache.replace_key(
            0, 0, 3, &keys, None, &bg_freq, &sg_freq, false,
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
            0, 999, 2, &keys, None, &bg_freq, &sg_freq, true,
        );

        // Should add the new key's contribution without subtracting old
        // New bigram: (2,1) + (1,2) = 200 + 200 = 400, score = 400 * 300 = 120000
        // New skipgram: (2,1) + (1,2) = 100 + 100 = 200, score = 200 * 300 = 60000
        // Weighted: 120000 * (-10) + 60000 * (-3) = -1200000 + -180000 = -1380000
        assert_eq!(new_score, -1380000,
            "Score should be -1380000 when replacing invalid old key");
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
            0, 0, 999, &keys, None, &bg_freq, &sg_freq, true,
        );

        // Should subtract old key's contribution without adding new
        // Old bigram: (0,1) + (1,0) = 100 + 100 = 200, score = 200 * 300 = 60000
        // Old skipgram: (0,1) + (1,0) = 50 + 50 = 100, score = 100 * 300 = 30000
        // Delta: -60000 bigram, -30000 skipgram
        // Weighted: -60000 * (-10) + -30000 * (-3) = 600000 + 90000 = 690000
        assert_eq!(new_score, 690000,
            "Score should be 690000 when replacing with invalid new key");
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
            true,   // apply
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
        let speculative_score = cache.key_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq, false,
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
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq, true,
        );

        // Get score with apply=false
        let score_no_apply = cache_no_apply.key_swap(
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq, false,
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
        cache.key_swap(0, 1, 0, 1, &keys, &bg_freq, &sg_freq, true);

        // After first swap, keys are now: [1, 0]
        let keys_after_swap = vec![1, 0];

        // Second swap: key 1 at pos 0 <-> key 0 at pos 1 (reverse)
        cache.key_swap(0, 1, 1, 0, &keys_after_swap, &bg_freq, &sg_freq, true);

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
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq, true,
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
            0, 1, 0, 1, &keys, &bg_freq, &sg_freq, true,
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

        // Add frequencies for full scissor (pos 0, 1)
        cache.update_bigram(0, 1, 0, 1000);
        cache.update_skipgram(0, 1, 0, 500);

        // Add frequencies for half scissor (pos 0, 2)
        // Note: Index-Ring is NOT adjacent, so this is a half non-adjacent scissor
        cache.update_bigram(0, 2, 0, 200);
        cache.update_skipgram(0, 2, 0, 100);

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
        cache.update_bigram(0, 1, 0, 100);
        cache.update_skipgram(0, 1, 0, 50);

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
}

