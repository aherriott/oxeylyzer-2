# Requirements Document

## Introduction

This document specifies the requirements for refactoring the analyzer architecture to support O(1) speculative scoring. The current architecture has a performance bottleneck where frequency arrays (`bg_freq`, `sg_freq`, `tg_freq`) in `MagicCache` are mutated during magic rule application, requiring analyzers to be updated via `update_bigram`, `update_skipgram`, and `update_trigram` calls. For speculative scoring (`apply=false`), changes are applied then reverted, causing `TrigramCache.key_swap` to iterate over ~5400 trigram combinations per call (~28µs).

The refactoring will make frequency arrays constant during optimization and move magic rule effect computation into each analyzer via an `add_rule` method. This enables pre-computing O(1) lookup tables for speculative scoring, targeting ~1µs per speculative score.

## Glossary

- **MagicCache**: Structure storing frequency tables (`bg_freq`, `sg_freq`, `tg_freq`) that currently get modified by magic key rules
- **Analyzer_Cache**: Generic term for `TrigramCache`, `SFCache`, `StretchCache`, `ScissorsCache`
- **Magic_Rule**: A rule that "steals" a bigram from a regular key to a magic key (leader→output becomes leader→magic_key)
- **Speculative_Scoring**: Computing a score without mutating state (`apply=false`), used to evaluate potential moves
- **Const_Freq**: The new architecture where frequency arrays remain constant during optimization
- **Rule_Delta**: The pre-computed score contribution of a magic rule for a specific analyzer
- **Lookup_Table**: Pre-computed O(n²) tables enabling O(1) speculative score lookups

## Requirements

### Requirement 1: Constant Frequency Arrays

**User Story:** As a layout optimizer, I want frequency arrays to remain constant during optimization, so that I can pre-compute lookup tables for O(1) speculative scoring.

#### Acceptance Criteria

1. THE MagicCache SHALL NOT modify `bg_freq`, `sg_freq`, or `tg_freq` after initialization
2. THE MagicCache SHALL provide read-only access to frequency arrays via getter methods
3. WHEN magic rules are applied, THE System SHALL NOT call `steal_bigram` to modify frequencies
4. THE System SHALL remove the `steal_bigram` method from MagicCache

### Requirement 2: Analyzer Rule Interface

**User Story:** As a layout optimizer, I want each analyzer to compute magic rule effects independently, so that rule effects can be pre-computed per analyzer.

#### Acceptance Criteria

1. THE TrigramCache SHALL implement an `add_rule(leader, output, magic_key, apply)` method
2. THE SFCache SHALL implement an `add_rule(leader, output, magic_key, apply)` method
3. THE StretchCache SHALL implement an `add_rule(leader, output, magic_key, apply)` method
4. THE ScissorsCache SHALL implement an `add_rule(leader, output, magic_key, apply)` method
5. WHEN `add_rule` is called with `apply=true`, THE Analyzer_Cache SHALL update its internal state
6. WHEN `add_rule` is called with `apply=false`, THE Analyzer_Cache SHALL return the score delta without mutating state

### Requirement 3: Remove Update Methods

**User Story:** As a maintainer, I want to remove the frequency update methods from analyzers, so that the codebase reflects the new constant-frequency architecture.

#### Acceptance Criteria

1. THE TrigramCache SHALL NOT have an `update_trigram` method
2. THE SFCache SHALL NOT have `update_bigram` or `update_skipgram` methods
3. THE StretchCache SHALL NOT have an `update_bigram` method
4. THE ScissorsCache SHALL NOT have `update_bigram` or `update_skipgram` methods
5. THE CachedLayout SHALL NOT call any update methods on analyzers during magic rule application

### Requirement 4: Magic Rule Score Computation

**User Story:** As a layout optimizer, I want each analyzer to correctly compute the score effect of a magic rule, so that the total score remains accurate.

#### Acceptance Criteria

1. WHEN a magic rule A→M steals output B, THE TrigramCache SHALL compute the score delta for:
   - Trigrams Z→A→B becoming Z→A→M (for all Z)
   - Trigrams A→B→C becoming A→M→C (for all C)
2. WHEN a magic rule A→M steals output B, THE SFCache SHALL compute the score delta for:
   - Bigram A→B becoming A→M
   - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
   - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
3. WHEN a magic rule A→M steals output B, THE StretchCache SHALL compute the score delta for:
   - Bigram A→B becoming A→M
   - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
4. WHEN a magic rule A→M steals output B, THE ScissorsCache SHALL compute the score delta for:
   - Bigram A→B becoming A→M
   - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
   - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate

### Requirement 5: Pre-computed Lookup Tables

**User Story:** As a layout optimizer, I want pre-computed lookup tables for speculative scoring, so that I can evaluate potential moves in O(1) time.

#### Acceptance Criteria

1. THE TrigramCache SHALL maintain pre-computed weighted scores for O(1) `key_swap` speculative scoring
2. THE SFCache SHALL maintain pre-computed weighted scores for O(1) `key_swap` speculative scoring
3. THE StretchCache SHALL maintain pre-computed weighted scores for O(1) `key_swap` speculative scoring
4. THE ScissorsCache SHALL maintain pre-computed weighted scores for O(1) `key_swap` speculative scoring
5. WHEN `key_swap` is called with `apply=false`, THE Analyzer_Cache SHALL use lookup tables instead of iterating over combinations
6. THE lookup tables SHALL be updated when `apply=true` operations change the layout state

### Requirement 6: Rule Lookup Tables

**User Story:** As a layout optimizer, I want pre-computed lookup tables for magic rule speculative scoring, so that I can evaluate potential rule changes in O(1) time.

#### Acceptance Criteria

1. THE TrigramCache SHALL maintain pre-computed rule deltas for O(1) `add_rule` speculative scoring
2. THE SFCache SHALL maintain pre-computed rule deltas for O(1) `add_rule` speculative scoring
3. THE StretchCache SHALL maintain pre-computed rule deltas for O(1) `add_rule` speculative scoring
4. THE ScissorsCache SHALL maintain pre-computed rule deltas for O(1) `add_rule` speculative scoring
5. WHEN `add_rule` is called with `apply=false`, THE Analyzer_Cache SHALL use lookup tables instead of computing from scratch

### Requirement 7: Performance Target

**User Story:** As a layout optimizer, I want speculative scoring to complete in ~1µs, so that optimization runs faster.

#### Acceptance Criteria

1. WHEN `key_swap` is called with `apply=false`, THE TrigramCache SHALL complete in under 2µs on average
2. WHEN `key_swap` is called with `apply=false`, THE SFCache SHALL complete in under 1µs on average
3. WHEN `key_swap` is called with `apply=false`, THE StretchCache SHALL complete in under 1µs on average
4. WHEN `key_swap` is called with `apply=false`, THE ScissorsCache SHALL complete in under 1µs on average
5. THE total speculative scoring time for all analyzers SHALL be under 5µs on average (down from ~28µs)

### Requirement 8: Correctness Preservation

**User Story:** As a layout optimizer, I want the refactored system to produce identical scores to the current system, so that optimization quality is preserved.

#### Acceptance Criteria

1. FOR ALL valid layouts and magic rule configurations, THE refactored system SHALL produce the same total score as the current system
2. FOR ALL valid key swap operations, THE refactored system SHALL produce the same score delta as the current system
3. FOR ALL valid magic rule operations, THE refactored system SHALL produce the same score delta as the current system
4. ALL existing tests (214 tests) SHALL pass after the refactoring

### Requirement 9: CachedLayout Integration

**User Story:** As a layout optimizer, I want CachedLayout to orchestrate the new architecture, so that the public API remains unchanged.

#### Acceptance Criteria

1. THE CachedLayout SHALL call `add_rule` on each analyzer when applying magic rules
2. THE CachedLayout SHALL NOT maintain `affected_grams` for frequency delta tracking
3. THE CachedLayout SHALL remove the `revert_affected` logic from magic rule handling
4. THE CachedLayout public API (`apply_neighbor`, `swap_keys`, `replace_key`, `score`, `stats`) SHALL remain unchanged

### Requirement 10: Space Complexity

**User Story:** As a system designer, I want the space overhead to be acceptable, so that the system can handle typical keyboard layouts.

#### Acceptance Criteria

1. THE lookup tables SHALL use O(n²) space where n is the number of positions (typically 30-40)
2. THE rule lookup tables SHALL use O(k²) space where k is the number of keys (typically 30-50)
3. THE total additional memory usage SHALL be under 10MB for typical configurations
