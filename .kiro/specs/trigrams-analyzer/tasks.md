# Implementation Plan: Trigrams Analyzer

## Overview

This plan implements the `TrigramCache` for tracking trigram type frequencies in the oxeylyzer keyboard layout analyzer. The implementation follows the established cache pattern from `SFCache`, `StretchCache`, and `ScissorsCache`.

## Tasks

- [x] 1. Implement TrigramCache core structure
  - [x] 1.1 Add TrigramCombo and TrigramDelta structs to `core/src/trigrams.rs`
    - Define `TrigramCombo` with pos_b, pos_c, and trigram_type fields
    - Define `TrigramDelta` with frequency fields for each tracked type
    - Implement `TrigramDelta::combine()` for merging deltas
    - _Requirements: 9.3_

  - [x] 1.2 Add TrigramCache struct with fields and constructor
    - Add `trigram_combos_per_key`, `num_keys`, frequency totals, weights, and fingers fields
    - Implement `new()` that pre-computes all trigram combinations using TRIGRAMS lookup
    - Filter to only store tracked types (Inroll, Outroll, Alternate, Redirect, OnehandIn, OnehandOut)
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 9.1, 9.2_

  - [ ]* 1.3 Write property test for tracked type classification
    - **Property 2: Untracked Types Ignored**
    - **Validates: Requirements 1.3, 4.3**

- [x] 2. Implement weight and score methods
  - [x] 2.1 Implement `set_weights()` method
    - Copy weight values from Weights struct for each trigram type
    - _Requirements: 3.1, 3.2_

  - [x] 2.2 Implement `score()` method
    - Return weighted sum of all frequency totals
    - _Requirements: 6.1, 6.2_

  - [ ]* 2.3 Write property test for score computation
    - **Property 7: Score Equals Weighted Sum**
    - **Validates: Requirements 6.1, 6.2**

- [x] 3. Implement update methods
  - [x] 3.1 Implement `update_trigram()` method
    - Look up trigram type from finger indices using TRIGRAMS constant
    - Update corresponding frequency total if type is tracked
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 3.2 Write property test for trigram updates
    - **Property 1: Tracked Type Frequency Updates**
    - **Validates: Requirements 1.2, 4.1, 4.2**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement key operation methods
  - [x] 5.1 Implement `compute_replace_delta()` helper method
    - Iterate over all trigram combinations involving the position
    - Compute frequency deltas for old_key vs new_key
    - Return TrigramDelta with accumulated changes
    - _Requirements: 5.1_

  - [x] 5.2 Implement `compute_replace_delta_score_only()` fast path
    - Same logic as compute_replace_delta but only compute weighted score delta
    - Used for speculative scoring when apply=false
    - _Requirements: 5.3_

  - [x] 5.3 Implement `replace_key()` method
    - If apply=true: compute full delta, apply it, return score()
    - If apply=false: compute score delta only, return score() + delta
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 5.4 Implement `key_swap()` method
    - Compute deltas for both positions with skip_pos to avoid double-counting
    - Combine deltas and apply or return speculative score
    - _Requirements: 5.4, 5.5_

  - [ ]* 5.5 Write property tests for key operations
    - **Property 4: Incremental Score Consistency**
    - **Property 5: Apply True Mutates State**
    - **Property 6: Apply False Preserves State**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [x] 6. Implement stats method
  - [x] 6.1 Implement `stats()` method
    - Populate TrigramStats fields with normalized frequencies
    - Handle division by zero by returning 0
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ]* 6.2 Write property test for stats normalization
    - **Property 8: Stats Normalization**
    - **Validates: Requirements 7.1**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Integrate with CachedLayout
  - [x] 8.1 Add TrigramCache field to CachedLayout struct
    - Add `trigram: TrigramCache` field
    - _Requirements: 8.1_

  - [x] 8.2 Initialize TrigramCache in CachedLayout::new()
    - Create TrigramCache with fingers and num_keys
    - Call set_weights with the weights parameter
    - _Requirements: 8.2_

  - [x] 8.3 Update CachedLayout::score() to include trigram score
    - Add `self.trigram.score()` to the total
    - _Requirements: 8.3_

  - [x] 8.4 Update CachedLayout::stats() to populate trigram stats
    - Call `self.trigram.stats(stats, self.data.trigram_total)`
    - _Requirements: 8.4_

  - [x] 8.5 Update CachedLayout::replace_key() to call trigram cache
    - Add call to `self.trigram.replace_key()` with tg_freq from magic cache
    - _Requirements: 8.5_

  - [x] 8.6 Update CachedLayout::swap_keys() to call trigram cache
    - Add call to `self.trigram.key_swap()` with tg_freq from magic cache
    - _Requirements: 8.6_

  - [x] 8.7 Update CachedLayout::apply_magic_rule() to update trigram cache
    - Add DeltaGram::Trigram handling to update trigram frequencies
    - Call `self.trigram.update_trigram()` for each trigram delta
    - _Requirements: 8.7_

  - [ ]* 8.8 Write integration tests for CachedLayout
    - **Property 9: CachedLayout Integration**
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7**

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation follows the established pattern from SFCache, StretchCache, and ScissorsCache
