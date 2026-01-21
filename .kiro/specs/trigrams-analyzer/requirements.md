# Requirements Document

## Introduction

This document specifies the requirements for implementing a `TrigramCache` in the oxeylyzer keyboard layout analyzer. The cache will track trigram type frequencies (alternate, inroll, outroll, redirect, onehandin, onehandout) and integrate with the existing scoring system. This follows the established cache pattern used by `SFCache`, `StretchCache`, and `ScissorsCache`.

## Glossary

- **TrigramCache**: A cache structure that tracks trigram type frequencies and provides incremental scoring updates during layout optimization
- **Trigram**: A sequence of three consecutive key presses
- **TrigramType**: Classification of a trigram based on finger and hand patterns (Alternate, Inroll, Outroll, Redirect, OnehandIn, OnehandOut, Sft, Sfb, Thumb, Invalid)
- **Alternate**: A trigram where each key is pressed by alternating hands (hand pattern: 1, 1, 1)
- **Roll**: A trigram with two keys on one hand and one on the other (hand pattern: 2, 1 or 1, 2), where the two keys are pressed by different fingers
- **Inroll**: A roll where the two same-hand keys move inward (toward the thumb)
- **Outroll**: A roll where the two same-hand keys move outward (away from the thumb)
- **Onehand**: A trigram where all three keys are on the same hand
- **OnehandIn**: A onehand trigram where all keys move in the same inward direction
- **OnehandOut**: A onehand trigram where all keys move in the same outward direction
- **Redirect**: A onehand trigram where the direction changes mid-sequence
- **CachedLayout**: The main layout structure that integrates all scoring caches
- **Weights**: Configuration structure containing weight multipliers for each trigram type

## Requirements

### Requirement 1: Trigram Type Classification

**User Story:** As a layout analyzer, I want to classify trigrams by type, so that I can track and score different trigram patterns.

#### Acceptance Criteria

1. THE TrigramCache SHALL reuse the existing `TRIGRAMS` lookup table from `trigrams.rs` for classification
2. THE TrigramCache SHALL track frequencies for these trigram types: Alternate, Inroll, Outroll, Redirect, OnehandIn, OnehandOut
3. THE TrigramCache SHALL NOT track Sft, Sfb, Thumb, or Invalid trigram types (these are handled elsewhere or ignored)
4. WHEN classifying a trigram, THE TrigramCache SHALL use the finger indices to look up the type in the `TRIGRAMS` constant array

### Requirement 2: Cache Initialization

**User Story:** As a layout analyzer, I want to initialize the trigram cache from keyboard layout data, so that I can begin tracking trigram frequencies.

#### Acceptance Criteria

1. WHEN creating a new TrigramCache, THE System SHALL accept keyboard positions, finger assignments, and number of keys as parameters
2. THE TrigramCache SHALL store the number of keys for frequency array indexing
3. THE TrigramCache SHALL initialize all frequency totals to zero
4. THE TrigramCache SHALL initialize all weight multipliers to zero (to be set via `set_weights`)

### Requirement 3: Weight Configuration

**User Story:** As a layout analyzer, I want to configure weights for each trigram type, so that I can customize the scoring formula.

#### Acceptance Criteria

1. WHEN `set_weights` is called, THE TrigramCache SHALL store the weight values for each trigram type (inroll, outroll, alternate, redirect, onehandin, onehandout)
2. THE TrigramCache SHALL use the weights from the existing `Weights` struct fields: `inroll`, `outroll`, `alternate`, `redirect`, `onehandin`, `onehandout`

### Requirement 4: Trigram Frequency Updates

**User Story:** As a layout analyzer, I want to update trigram frequencies incrementally, so that I can efficiently track changes during optimization.

#### Acceptance Criteria

1. WHEN `update_trigram` is called with three positions and old/new frequencies, THE TrigramCache SHALL compute the trigram type from the finger assignments
2. WHEN the trigram type is one of the tracked types, THE TrigramCache SHALL update the corresponding frequency total by the delta (new_freq - old_freq)
3. WHEN the trigram type is not tracked (Sft, Sfb, Thumb, Invalid), THE TrigramCache SHALL ignore the update

### Requirement 5: Incremental Key Operations

**User Story:** As a layout analyzer, I want to compute score changes when keys are swapped or replaced, so that I can efficiently evaluate layout modifications.

#### Acceptance Criteria

1. WHEN `replace_key` is called, THE TrigramCache SHALL compute the score delta by iterating over all trigrams involving the position
2. WHEN `replace_key` is called with `apply=true`, THE TrigramCache SHALL update internal state and return the new score
3. WHEN `replace_key` is called with `apply=false`, THE TrigramCache SHALL compute and return the new score without mutating state
4. WHEN `key_swap` is called, THE TrigramCache SHALL compute the combined score delta for both positions
5. WHEN `key_swap` is called, THE TrigramCache SHALL use skip_pos to avoid double-counting trigrams that involve both swapped positions

### Requirement 6: Score Computation

**User Story:** As a layout analyzer, I want to compute a weighted trigram score, so that I can evaluate layout quality.

#### Acceptance Criteria

1. THE TrigramCache SHALL maintain running frequency totals for each tracked trigram type
2. WHEN `score()` is called, THE TrigramCache SHALL return the weighted sum: `inroll_total * inroll_weight + outroll_total * outroll_weight + alternate_total * alternate_weight + redirect_total * redirect_weight + onehandin_total * onehandin_weight + onehandout_total * onehandout_weight`
3. THE TrigramCache SHALL update the running score incrementally during key operations

### Requirement 7: Statistics Population

**User Story:** As a layout analyzer, I want to populate trigram statistics, so that I can report trigram type percentages.

#### Acceptance Criteria

1. WHEN `stats()` is called, THE TrigramCache SHALL populate the `TrigramStats` struct fields with normalized frequencies
2. THE TrigramCache SHALL normalize frequencies by dividing by the trigram total
3. THE TrigramCache SHALL populate these fields: `inroll`, `outroll`, `alternate`, `redirect`, `onehandin`, `onehandout`

### Requirement 8: CachedLayout Integration

**User Story:** As a layout analyzer, I want the trigram cache integrated with CachedLayout, so that trigram scoring is included in layout optimization.

#### Acceptance Criteria

1. THE CachedLayout SHALL contain a `TrigramCache` field
2. WHEN CachedLayout is created, THE System SHALL initialize the TrigramCache with keyboard and finger data
3. WHEN CachedLayout's `score()` is called, THE System SHALL include the TrigramCache score in the total
4. WHEN CachedLayout's `stats()` is called, THE System SHALL call TrigramCache's stats method to populate trigram statistics
5. WHEN CachedLayout's `replace_key` is called, THE System SHALL call TrigramCache's replace_key method
6. WHEN CachedLayout's `swap_keys` is called, THE System SHALL call TrigramCache's key_swap method
7. WHEN magic rules affect trigram frequencies, THE System SHALL update the TrigramCache via `update_trigram`

### Requirement 9: Trigram Pair Pre-computation

**User Story:** As a layout analyzer, I want trigram position combinations pre-computed, so that incremental updates are efficient.

#### Acceptance Criteria

1. THE TrigramCache SHALL pre-compute all valid trigram position combinations during initialization
2. FOR EACH position, THE TrigramCache SHALL store a list of position pairs that form trigrams with it
3. THE TrigramCache SHALL store the trigram type for each pre-computed combination
4. WHEN updating for a key change, THE TrigramCache SHALL iterate only over pre-computed combinations involving that position
