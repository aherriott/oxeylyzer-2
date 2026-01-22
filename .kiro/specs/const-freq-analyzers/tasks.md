# Implementation Tasks: Constant Frequency Analyzers

## Overview

This document outlines the implementation tasks for refactoring the analyzer architecture to support O(1) speculative scoring with constant frequency arrays.

## Task List

- [x] 1. Make MagicCache Read-Only
  - [x] 1.1 Remove `steal_bigram` method from MagicCache
  - [x] 1.2 Remove `revert_affected` method from MagicCache
  - [x] 1.3 Remove `DeltaGram`, `DeltaBigram`, `DeltaSkipgram`, `DeltaTrigram` types
  - [x] 1.4 Add documentation noting frequency arrays are constant after initialization

- [x] 2. Add `add_rule` to TrigramCache
  - [x] 2.1 Add `active_rules: HashMap<(CacheKey, CacheKey), CacheKey>` field to track active magic rules
  - [x] 2.2 Implement `compute_rule_delta` method to calculate score delta for a magic rule
  - [x] 2.3 Implement `add_rule(leader, output, magic_key, keys, key_positions, tg_freq, apply)` method
  - [x] 2.4 Add unit tests for TrigramCache `add_rule`
  - [x] 2.5 (PBT) Property test: add_rule with apply=true mutates state correctly
  - [x] 2.6 (PBT) Property test: add_rule with apply=false preserves state

- [x] 3. Add `add_rule` to SFCache
  - [x] 3.1 Add `active_rules` field to SFCache
  - [x] 3.2 Implement `compute_rule_delta` for SFCache (bigrams + skipgrams)
  - [x] 3.3 Implement `add_rule` method for SFCache
  - [x] 3.4 Add unit tests for SFCache `add_rule`
  - [x] 3.5 (PBT) Property test: SFCache add_rule state mutation

- [x] 4. Add `add_rule` to StretchCache
  - [x] 4.1 Add `active_rules` field to StretchCache
  - [x] 4.2 Implement `compute_rule_delta` for StretchCache (bigrams only)
  - [x] 4.3 Implement `add_rule` method for StretchCache
  - [x] 4.4 Add unit tests for StretchCache `add_rule`
  - [x] 4.5 (PBT) Property test: StretchCache add_rule state mutation

- [x] 5. Add `add_rule` to ScissorsCache
  - [x] 5.1 Add `active_rules` field to ScissorsCache
  - [x] 5.2 Implement `compute_rule_delta` for ScissorsCache (bigrams + skipgrams)
  - [x] 5.3 Implement `add_rule` method for ScissorsCache
  - [x] 5.4 Add unit tests for ScissorsCache `add_rule`
  - [x] 5.5 (PBT) Property test: ScissorsCache add_rule state mutation

- [x] 6. Refactor CachedLayout `apply_magic_rule`
  - [x] 6.1 Modify `apply_magic_rule` to call `add_rule` on each analyzer instead of using `steal_bigram` + `update_*`
  - [x] 6.2 Remove `affected_grams` field from CachedLayout
  - [x] 6.3 Remove revert logic that used `affected_grams`
  - [x] 6.4 Add integration tests verifying magic rule application produces correct scores
  - [x] 6.5 (PBT) Property test: apply_magic_rule score equals sum of analyzer scores

- [x] 7. Add Rule Lookup Tables for O(1) Speculative Scoring
  - [x] 7.1 Add `rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>` to TrigramCache
  - [x] 7.2 Implement `init_rule_deltas` for TrigramCache
  - [x] 7.3 Modify TrigramCache `add_rule` to use lookup table when `apply=false`
  - [x] 7.4 Add rule lookup tables to SFCache, StretchCache, ScissorsCache
  - [x] 7.5 (PBT) Property test: lookup table values match computed values

- [x] 8. Remove Deprecated Update Methods
  - [x] 8.1 Remove `update_trigram` from TrigramCache
  - [x] 8.2 Remove `update_bigram` and `update_skipgram` from SFCache
  - [x] 8.3 Remove `update_bigram` from StretchCache
  - [x] 8.4 Remove `update_bigram` and `update_skipgram` from ScissorsCache
  - [x] 8.5 Update any remaining callers of removed methods

- [x] 9. Correctness Verification
  - [x] 9.1 (PBT) Property test: constant frequencies - MagicCache arrays unchanged after operations
  - [x] 9.2 (PBT) Property test: total score preservation - refactored system equals original
  - [x] 9.3 Verify all 214 existing tests pass
  - [x] 9.4 Add regression test comparing old vs new implementation scores

- [x] 10. Performance Optimization and Benchmarking
  - [x] 10.1 Add benchmark for speculative key_swap (target: < 2µs per analyzer)
  - [x] 10.2 Add benchmark for speculative add_rule (target: < 2µs per analyzer)
  - [x] 10.3 Add benchmark for total speculative scoring (target: < 5µs)
  - [x] 10.4 Profile and optimize hot paths if targets not met
  - [x] 10.5 Document final performance numbers

## Task Details

### Task 1: Make MagicCache Read-Only

Remove mutation methods from MagicCache to enforce constant frequencies.

**Files to modify:**
- `core/src/cached_layout.rs`

**Acceptance criteria:**
- `steal_bigram` method removed
- `revert_affected` method removed
- Delta types removed
- Frequency arrays only modified during `init_from_data`

### Task 2: Add `add_rule` to TrigramCache

Implement magic rule handling in TrigramCache.

**Files to modify:**
- `core/src/trigrams.rs`

**Algorithm for `compute_rule_delta`:**
```
For rule A→M stealing B:
  delta = 0
  For each Z with position:
    // Z→A→B becomes Z→A→M
    old_score = tg_freq[Z][A][B] * weight(type(Z,A,B))
    new_score = tg_freq[Z][A][B] * weight(type(Z,A,M))
    delta += new_score - old_score
  For each C with position:
    // A→B→C becomes A→M→C
    old_score = tg_freq[A][B][C] * weight(type(A,B,C))
    new_score = tg_freq[A][B][C] * weight(type(A,M,C))
    delta += new_score - old_score
  return delta
```

### Task 3-5: Add `add_rule` to Other Caches

Similar pattern to TrigramCache, but handling bigrams and skipgrams.

**SFCache algorithm:**
```
For rule A→M stealing B:
  // Full steal: A→B becomes A→M
  delta += (bg_freq[A][M] - bg_freq[A][B]) * sf_weight(A,M) - sf_weight(A,B))

  // Partial steal: B→C becomes M→C based on tg_freq[A][B][C]
  For each C:
    stolen = tg_freq[A][B][C]
    delta += stolen * (sf_weight(M,C) - sf_weight(B,C))

  // Skipgram partial steal: Z→B becomes Z→M based on tg_freq[Z][A][B]
  For each Z:
    stolen = tg_freq[Z][A][B]
    delta += stolen * (sfs_weight(Z,M) - sfs_weight(Z,B))
```

### Task 6: Refactor CachedLayout

Replace the current `apply_magic_rule` implementation that uses `steal_bigram` + `update_*` with direct `add_rule` calls.

**Before:**
```rust
self.magic.steal_bigram(leader, output, magic_key, ...);
for gram in &self.affected_grams {
    match gram {
        DeltaGram::Bigram(bg) => self.sfb.update_bigram(...);
        // ...
    }
}
```

**After:**
```rust
self.sfb.add_rule(leader, output, magic_key, ...);
self.stretch.add_rule(leader, output, magic_key, ...);
self.scissors.add_rule(leader, output, magic_key, ...);
self.trigram.add_rule(leader, output, magic_key, ...);
```

### Task 7: Add Rule Lookup Tables

Pre-compute rule deltas for O(1) speculative scoring.

**Storage strategy:** Use `HashMap<(CacheKey, CacheKey, CacheKey), i32>` for sparse storage to stay under 10MB memory target.

**Initialization:** Call `init_rule_deltas` after layout is fully initialized (all keys placed).

### Task 8: Remove Deprecated Methods

Clean up old update methods that are no longer needed.

**Methods to remove:**
- `TrigramCache::update_trigram`
- `SFCache::update_bigram`, `SFCache::update_skipgram`
- `StretchCache::update_bigram`
- `ScissorsCache::update_bigram`, `ScissorsCache::update_skipgram`

### Task 9: Correctness Verification

Ensure the refactored system produces identical results to the original.

**Property tests:**
1. Frequencies remain constant
2. Scores match between old and new implementations
3. All existing tests pass

### Task 10: Performance Benchmarking

Verify performance targets are met.

**Targets:**
- Individual analyzer speculative scoring: < 2µs
- Total speculative scoring: < 5µs
- Down from ~28µs baseline

## Dependencies

```
Task 1 (MagicCache read-only)
    ↓
Tasks 2-5 (add_rule to analyzers) [parallel]
    ↓
Task 6 (refactor CachedLayout)
    ↓
Task 7 (lookup tables)
    ↓
Task 8 (remove deprecated methods)
    ↓
Tasks 9-10 (verification & benchmarking) [parallel]
```

## Notes

- Tasks 2-5 can be implemented in parallel
- Task 6 depends on Tasks 2-5 being complete
- Task 8 should only be done after Task 6 is verified working
- Performance benchmarks should be run before and after to measure improvement
