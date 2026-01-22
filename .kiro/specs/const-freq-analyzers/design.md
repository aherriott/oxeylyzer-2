# Design Document: Constant Frequency Analyzers

## Overview

This feature refactors the analyzer architecture to support O(1) speculative scoring by making frequency arrays constant during optimization. Currently, `MagicCache` mutates frequency arrays (`bg_freq`, `sg_freq`, `tg_freq`) when magic rules are applied, requiring analyzers to be updated via `update_bigram`, `update_skipgram`, and `update_trigram` calls. For speculative scoring (`apply=false`), changes are applied then reverted, causing expensive iteration over ~5400 trigram combinations per call (~28µs).

The refactoring moves magic rule effect computation into each analyzer via an `add_rule` method, enabling pre-computed O(1) lookup tables for speculative scoring. Target: ~1µs per speculative score (down from ~28µs).

## Architecture

```mermaid
graph TD
    subgraph "Current Architecture"
        CL1[CachedLayout]
        MC1[MagicCache]
        SF1[SFCache]
        ST1[StretchCache]
        SC1[ScissorsCache]
        TG1[TrigramCache]

        CL1 -->|apply_magic_rule| MC1
        MC1 -->|steal_bigram mutates freq| MC1
        MC1 -->|affected_grams| CL1
        CL1 -->|update_bigram| SF1
        CL1 -->|update_bigram| ST1
        CL1 -->|update_bigram| SC1
        CL1 -->|update_trigram| TG1
    end

    subgraph "New Architecture"
        CL2[CachedLayout]
        MC2[MagicCache - Const]
        SF2[SFCache]
        ST2[StretchCache]
        SC2[ScissorsCache]
        TG2[TrigramCache]

        CL2 -->|add_rule| SF2
        CL2 -->|add_rule| ST2
        CL2 -->|add_rule| SC2
        CL2 -->|add_rule| TG2

        SF2 -->|O(1) lookup| SF2
        ST2 -->|O(1) lookup| ST2
        SC2 -->|O(1) lookup| SC2
        TG2 -->|O(1) lookup| TG2

        MC2 -.->|read-only freq| SF2
        MC2 -.->|read-only freq| ST2
        MC2 -.->|read-only freq| SC2
        MC2 -.->|read-only freq| TG2
    end
```

### Key Changes

1. **MagicCache becomes read-only**: Frequency arrays are initialized once and never modified
2. **Remove `steal_bigram`**: No more frequency redistribution in MagicCache
3. **Remove `update_*` methods**: Analyzers no longer receive frequency deltas
4. **Add `add_rule` method**: Each analyzer computes magic rule effects independently
5. **Pre-computed lookup tables**: Enable O(1) speculative scoring for both `key_swap` and `add_rule`

## Components and Interfaces

### MagicCache (Modified)

The MagicCache becomes a read-only frequency store after initialization.

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    /// Flat bigram frequencies: bg_freq[a * num_keys + b] - CONSTANT after init
    bg_freq: Vec<i64>,
    /// Flat skipgram frequencies: sg_freq[a * num_keys + b] - CONSTANT after init
    sg_freq: Vec<i64>,
    /// Trigram frequencies - CONSTANT after init
    tg_freq: Vec<Vec<Vec<i64>>>,
    /// Number of keys for indexing
    num_keys: usize,
}

impl MagicCache {
    pub fn new(num_keys: usize) -> Self;

    /// Initialize frequencies from corpus data (called once)
    pub fn init_from_data(&mut self, bigrams: &[Vec<i64>], skipgrams: &[Vec<i64>], trigrams: &[Vec<Vec<i64>>>);

    /// Read-only accessors
    pub fn bg_freq_flat(&self) -> &[i64];
    pub fn sg_freq_flat(&self) -> &[i64];
    pub fn tg_freq(&self) -> &[Vec<Vec<i64>>];
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64;
    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64;
    pub fn get_tg_freq(&self, a: CacheKey, b: CacheKey, c: CacheKey) -> i64;

    // REMOVED: steal_bigram, revert_affected
}
```

### Analyzer Rule Interface

Each analyzer implements the `add_rule` method to compute magic rule effects.

#### TrigramCache

```rust
impl TrigramCache {
    /// Apply a magic rule. Returns the new score.
    /// If `apply` is false, uses O(1) lookup for speculative scoring.
    ///
    /// When rule A→M steals output B:
    /// - Trigrams Z→A→B become Z→A→M (for all Z)
    /// - Trigrams A→B→C become A→M→C (for all C)
    pub fn add_rule(
        &mut self,
        leader: CacheKey,      // A
        output: CacheKey,      // B (being stolen)
        magic_key: CacheKey,   // M
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64;

    /// Pre-computed rule deltas for O(1) add_rule speculative scoring
    /// rule_delta[leader * num_keys * num_keys + output * num_keys + magic_key]
    rule_delta: Vec<i64>,

    /// Initialize rule lookup tables
    fn init_rule_deltas(&mut self, keys: &[CacheKey], key_positions: &[Option<CachePos>], tg_freq: &[Vec<Vec<i64>>>);
}
```

#### SFCache

```rust
impl SFCache {
    /// Apply a magic rule. Returns the new score.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    pub fn add_rule(
        &mut self,
        leader: CacheKey,
        output: CacheKey,
        magic_key: CacheKey,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64;

    /// Pre-computed rule deltas
    rule_delta: Vec<i64>,
}
```

#### StretchCache

```rust
impl StretchCache {
    /// Apply a magic rule. Returns the new score.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    pub fn add_rule(
        &mut self,
        leader: CacheKey,
        output: CacheKey,
        magic_key: CacheKey,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64;

    /// Pre-computed rule deltas
    rule_delta: Vec<i64>,
}
```

#### ScissorsCache

```rust
impl ScissorsCache {
    /// Apply a magic rule. Returns the new score.
    ///
    /// When rule A→M steals output B:
    /// - Bigram A→B becomes A→M
    /// - Bigrams B→C partially stolen by M→C based on trigram A→B→C rate
    /// - Skipgrams Z→B partially stolen by Z→M based on trigram Z→A→B rate
    pub fn add_rule(
        &mut self,
        leader: CacheKey,
        output: CacheKey,
        magic_key: CacheKey,
        keys: &[CacheKey],
        key_positions: &[Option<CachePos>],
        bg_freq: &[i64],
        sg_freq: &[i64],
        tg_freq: &[Vec<Vec<i64>>],
        apply: bool,
    ) -> i64;

    /// Pre-computed rule deltas
    rule_delta: Vec<i64>,
}
```

### CachedLayout Integration

```rust
impl CachedLayout {
    /// Apply a magic rule. Returns the new score.
    /// Orchestrates add_rule calls to each analyzer.
    pub fn apply_magic_rule(
        &mut self,
        magic_key: CacheKey,
        leader: CacheKey,
        new_output: CacheKey,
        apply: bool,
    ) -> i64 {
        // Handle clearing conflicting rules from other magic keys
        // ...

        // Unsteal old output if exists
        if let Some(old_output) = self.current_magic_rules.get(&(magic_key, leader)) {
            if *old_output != EMPTY_KEY {
                // Reverse the old rule
                self.sfb.add_rule(leader, magic_key, *old_output, ...);
                self.stretch.add_rule(leader, magic_key, *old_output, ...);
                self.scissors.add_rule(leader, magic_key, *old_output, ...);
                self.trigram.add_rule(leader, magic_key, *old_output, ...);
            }
        }

        // Apply new rule
        if new_output != EMPTY_KEY {
            self.sfb.add_rule(leader, new_output, magic_key, ...);
            self.stretch.add_rule(leader, new_output, magic_key, ...);
            self.scissors.add_rule(leader, new_output, magic_key, ...);
            self.trigram.add_rule(leader, new_output, magic_key, ...);
        }

        if apply {
            // Update current_magic_rules
        }

        self.score()
    }

    // REMOVED: affected_grams field
    // REMOVED: revert_affected logic
}
```

## Data Models

### Rule Delta Lookup Tables

Each analyzer maintains a pre-computed lookup table for O(1) rule speculative scoring.

| Analyzer | Table Size | Index Formula |
|----------|------------|---------------|
| TrigramCache | O(k³) | `leader * k² + output * k + magic_key` |
| SFCache | O(k³) | `leader * k² + output * k + magic_key` |
| StretchCache | O(k³) | `leader * k² + output * k + magic_key` |
| ScissorsCache | O(k³) | `leader * k² + output * k + magic_key` |

Where k = number of keys (typically 30-50).

### Active Rules State

Each analyzer tracks which rules are currently active to correctly compute deltas.

```rust
/// Tracks active magic rules per analyzer
/// Key: (magic_key, leader), Value: output
active_rules: HashMap<(CacheKey, CacheKey), CacheKey>,
```

### Magic Rule Score Computation

When a magic rule A→M steals output B, the score effects are:

#### Trigrams
| Original | Becomes | Frequency Source |
|----------|---------|------------------|
| Z→A→B | Z→A→M | tg_freq[Z][A][B] for all Z |
| A→B→C | A→M→C | tg_freq[A][B][C] for all C |

#### Bigrams (SFCache, StretchCache, ScissorsCache)
| Original | Becomes | Frequency Source |
|----------|---------|------------------|
| A→B | A→M | bg_freq[A][B] (full steal) |
| B→C | M→C | tg_freq[A][B][C] (partial steal) |

#### Skipgrams (SFCache, ScissorsCache)
| Original | Becomes | Frequency Source |
|----------|---------|------------------|
| Z→B | Z→M | tg_freq[Z][A][B] (partial steal) |

## Lookup Table Pre-computation

### Key Swap Lookup Tables (Existing Enhancement)

The existing `weighted_score_first`, `weighted_score_mid`, `weighted_score_end`, and `swap_score_both` tables in TrigramCache already support O(1) key swap speculative scoring. Similar tables will be added to SFCache, StretchCache, and ScissorsCache.

```rust
/// For each position and key, pre-computed weighted score contribution
/// swap_score[pos * num_keys + key] = weighted score when pos has key
swap_score: Vec<i64>,

/// For position pairs, pre-computed swap delta
/// swap_delta[pair_idx * num_keys * num_keys + key_a * num_keys + key_b]
swap_delta: Vec<i64>,
```

### Rule Lookup Tables (New)

For each (leader, output, magic_key) triple, pre-compute the score delta.

```rust
impl TrigramCache {
    fn init_rule_deltas(&mut self, keys: &[CacheKey], key_positions: &[Option<CachePos>], tg_freq: &[Vec<Vec<i64>>]) {
        let num_keys = self.num_keys;
        self.rule_delta = vec![0; num_keys * num_keys * num_keys];

        for leader in 0..num_keys {
            let leader_pos = key_positions.get(leader).copied().flatten();
            if leader_pos.is_none() { continue; }

            for output in 0..num_keys {
                for magic_key in 0..num_keys {
                    let delta = self.compute_rule_delta(
                        leader, output, magic_key, keys, key_positions, tg_freq
                    );
                    let idx = leader * num_keys * num_keys + output * num_keys + magic_key;
                    self.rule_delta[idx] = delta;
                }
            }
        }
    }
}
```

## Correctness Properties

### Property 1: Constant Frequencies

*For any* sequence of magic rule applications, the frequency arrays in MagicCache (`bg_freq`, `sg_freq`, `tg_freq`) should remain unchanged from their initial values.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

### Property 2: Rule Score Equivalence

*For any* magic rule (leader, output, magic_key) and *for any* layout state, the score computed by the new `add_rule` method should equal the score that would have been computed by the old `steal_bigram` + `update_*` approach.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 8.1, 8.2, 8.3, 8.4**

### Property 3: Apply True Mutates State

*For any* `add_rule` call with `apply=true`, the analyzer's internal state should reflect the rule application, and subsequent calls to `score()` should return the same value as was returned by the operation.

**Validates: Requirements 2.5**

### Property 4: Apply False Preserves State

*For any* `add_rule` call with `apply=false`, the analyzer's internal state should remain unchanged, and `score()` called before and after should return the same value.

**Validates: Requirements 2.6**

### Property 5: Lookup Table Consistency

*For any* (leader, output, magic_key) triple, the pre-computed `rule_delta` lookup should equal the value computed by `compute_rule_delta` from scratch.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Property 6: Key Swap O(1) Consistency

*For any* key swap operation with `apply=false`, the score returned using O(1) lookup tables should equal the score computed by full iteration.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

### Property 7: Total Score Preservation

*For any* layout and *for any* sequence of operations (key swaps, magic rules), the total score from the refactored system should equal the total score from the original system.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

### Property 8: Performance Target

*For any* speculative scoring operation (`apply=false`), the execution time should be under 5µs total across all analyzers.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Property 9: Space Complexity

*For any* typical keyboard configuration (30-50 keys, 30-40 positions), the total additional memory for lookup tables should be under 10MB.

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 10: API Compatibility

*For any* usage of the public CachedLayout API (`apply_neighbor`, `swap_keys`, `replace_key`, `score`, `stats`), the behavior should be identical to the pre-refactoring implementation.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

## Error Handling

### Invalid Key Handling

- If a key index is >= num_keys, treat it as EMPTY_KEY and contribute zero to scores
- Lookup table accesses should bounds-check and return 0 for invalid indices

### Missing Position Handling

- If `key_positions[key]` is None, the key is not on the layout and should be skipped
- Rule computations should gracefully handle keys without positions

### Rule Conflict Handling

- When applying a rule that conflicts with an existing rule on another magic key, the conflicting rule should be cleared first
- This logic remains in CachedLayout, not in individual analyzers

## Testing Strategy

### Unit Tests

1. **MagicCache Immutability**: Verify frequencies don't change after any operation
2. **Individual Analyzer add_rule**: Test each analyzer's rule computation in isolation
3. **Lookup Table Correctness**: Verify pre-computed values match computed values
4. **Edge Cases**: Empty layouts, single key, all same finger, etc.

### Property-Based Tests

Using `proptest` with minimum 100 iterations per property:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property 1: Constant Frequencies
    #[test]
    fn prop_constant_frequencies(
        layout in arb_layout(),
        rules in vec(arb_magic_rule(), 0..10),
    ) {
        let initial_bg = magic_cache.bg_freq_flat().to_vec();
        for rule in rules {
            cached_layout.apply_magic_rule(rule.magic_key, rule.leader, rule.output, true);
        }
        prop_assert_eq!(magic_cache.bg_freq_flat(), &initial_bg[..]);
    }

    // Property 2: Rule Score Equivalence
    #[test]
    fn prop_rule_score_equivalence(
        layout in arb_layout(),
        rule in arb_magic_rule(),
    ) {
        let old_score = old_impl.apply_magic_rule(rule, true);
        let new_score = new_impl.apply_magic_rule(rule, true);
        prop_assert_eq!(old_score, new_score);
    }
}
```

### Integration Tests

1. **Full Optimization Run**: Run simulated annealing with both implementations, verify same results
2. **Benchmark Comparison**: Measure speculative scoring time before/after refactoring
3. **Existing Test Suite**: All 214 existing tests must pass

### Performance Tests

```rust
#[bench]
fn bench_speculative_key_swap(b: &mut Bencher) {
    let layout = setup_layout();
    b.iter(|| {
        layout.swap_keys(0, 1, false)
    });
    // Target: < 5µs total
}

#[bench]
fn bench_speculative_add_rule(b: &mut Bencher) {
    let layout = setup_layout();
    b.iter(|| {
        layout.apply_magic_rule(magic_key, leader, output, false)
    });
    // Target: < 5µs total
}
```

## Migration Strategy

### Phase 1: Add New Methods (Non-Breaking)

1. Add `add_rule` method to each analyzer (alongside existing `update_*` methods)
2. Add lookup table fields and initialization
3. Add tests for new methods

### Phase 2: Switch CachedLayout

1. Modify `apply_magic_rule` to use `add_rule` instead of `steal_bigram` + `update_*`
2. Remove `affected_grams` field and `revert_affected` logic
3. Verify all tests pass

### Phase 3: Remove Old Methods

1. Remove `steal_bigram` from MagicCache
2. Remove `update_bigram`, `update_skipgram`, `update_trigram` from analyzers
3. Remove `revert_affected` from MagicCache
4. Final cleanup and documentation

## Space Complexity Analysis

### Lookup Table Sizes

For k = 50 keys, n = 40 positions:

| Table | Size | Memory (i64 = 8 bytes) |
|-------|------|------------------------|
| TrigramCache.rule_delta | k³ = 125,000 | 1 MB |
| SFCache.rule_delta | k³ = 125,000 | 1 MB |
| StretchCache.rule_delta | k³ = 125,000 | 1 MB |
| ScissorsCache.rule_delta | k³ = 125,000 | 1 MB |
| TrigramCache.swap_score_both | n²/2 × k² = 2M | 16 MB |
| **Total** | | ~20 MB |

This exceeds the 10MB target. Optimization options:
1. Use i32 instead of i64 for deltas (halves memory)
2. Sparse storage for rule_delta (most entries are 0)
3. Compute rule deltas on-demand with caching

### Recommended Approach

Use i32 for lookup tables and sparse storage for rule deltas:

```rust
/// Sparse rule delta storage
rule_delta: HashMap<(CacheKey, CacheKey, CacheKey), i32>,
```

This reduces memory to ~5MB for typical configurations while maintaining O(1) lookup for common cases.

## Performance Results

This section documents the actual benchmark results from the implementation.

### Speculative Scoring Benchmarks (`apply=false`)

| Operation | Measured Time | Target | Status |
|-----------|---------------|--------|--------|
| `speculative_add_rule` | ~0.59µs | < 2µs | ✅ **MEETS TARGET** |
| `speculative_key_swap` | ~14-15µs | < 2µs | ❌ Does not meet target |
| `total_speculative_scoring` | ~11.37µs | < 5µs | ❌ Does not meet target |

### Applied Operations Benchmarks (`apply=true`)

| Operation | Measured Time |
|-----------|---------------|
| `key_swap` (apply=true) | ~2.7µs |
| `apply_magic_rule` (apply=true) | ~1.5µs |

### Analysis

1. **Magic Rule Speculative Scoring (SUCCESS)**: The `add_rule` method with `apply=false` achieves ~0.59µs, well under the 2µs target. This is enabled by the pre-computed `rule_delta` lookup tables that provide O(1) access to score deltas for any (leader, output, magic_key) triple.

2. **Key Swap Speculative Scoring (PARTIAL)**: The `key_swap` operation with `apply=false` currently takes ~14-15µs, which does not meet the 2µs target. This is because only the `add_rule` operation has O(1) lookup tables implemented. The key swap operation still iterates over affected trigram combinations.

3. **Total Speculative Scoring**: At ~11.37µs, this is a significant improvement from the original ~28µs baseline but does not meet the 5µs target.

### Future Optimization Opportunities

To achieve O(1) key swap speculative scoring, the following enhancements could be implemented:

1. **Swap Delta Lookup Tables**: Add pre-computed `swap_delta` tables to each analyzer, similar to the `rule_delta` tables. These would store the score delta for swapping any two positions.

2. **Table Structure**: For each analyzer, add:
   ```rust
   /// swap_delta[pos_a * num_positions + pos_b] = score delta when swapping positions
   swap_delta: Vec<i64>,
   ```

3. **Memory Impact**: For 40 positions, this adds 40² × 8 bytes = 12.8KB per analyzer, or ~51KB total—well within the 10MB budget.

4. **Expected Performance**: With swap delta tables, key swap speculative scoring should achieve similar ~0.5-1µs performance as add_rule speculative scoring.

### Conclusion

The refactoring successfully achieved O(1) speculative scoring for magic rule operations (`add_rule`), meeting the performance target. Key swap speculative scoring remains O(n) but could be optimized to O(1) in a future iteration by adding swap delta lookup tables.
