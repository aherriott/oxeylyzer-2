# Requirements: Hot Loop Performance Optimization

## Target
100 independent layout optimizations (SA 1K-10K iterations + depth-3) in < 3 minutes, parallelized across cores.

## Current Bottlenecks
| Bottleneck | Current | Impact |
|---|---|---|
| `init_cached_layout` | 31.37s | Blocks start of every optimization |
| `apply_neighbor(true)` | 28µs/op | 450x slower than speculative path |
| `swap_delta` HashMap init | ~30s of the 31.37s init | O(pos² × keys²) × 4 caches |
| `swap_delta` HashMap lookup | ~62ns but with hash overhead | 4 HashMap lookups per score_neighbor |
| `Data` clone per layout | ~780KB trigram table copied | Memory pressure under parallelism |
| `std::HashMap` in sub-caches | SipHash on 32-byte keys | Should be FxHashMap or eliminated |

## Functional Requirements

### R1: New API Surface
Replace `apply_neighbor(neighbor, apply: bool)` with:
- `score_neighbor(neighbor) -> i64` — O(1) speculative, no mutation
- `score() -> i64` — current score from running totals, always valid
- `apply_neighbor(neighbor)` — mutate keys + running totals only (fast)
- `apply_neighbor_and_update(neighbor)` — apply + rebuild weighted_score arrays
- `update_scores()` — rebuild weighted_score arrays from current state

### R2: Eliminate swap_delta HashMaps
Remove `swap_delta: HashMap<(CachePos, CachePos, CacheKey, CacheKey), i32>` from all 4 caches. Derive speculative swap scores from flat arrays at score_neighbor time.

### R3: Eliminate init_swap_deltas
Remove the O(pos² × keys²) precomputation entirely. The 31s init becomes near-zero.

### R4: Flat-Array Speculative Scoring for All Caches
Extend the weighted_score flat array pattern (currently only in TrigramCache) to SFCache, StretchCache, and ScissorsCache. Each cache gets `weighted_score[pos * num_keys + key]` arrays.

### R5: Share AnalyzerData via Arc
Use `Arc<AnalyzerData>` instead of cloning Data into each CachedLayout. All 100 parallel optimizations share one copy of the frequency tables.

### R6: Switch to FxHashMap
Replace `std::collections::HashMap` with `fxhash::FxHashMap` in all sub-caches for remaining maps (rule_delta, active_rules).

### R7: Update Optimization Loops
- SA: use `score_neighbor` + `apply_neighbor_and_update` on accept
- Depth-N: use `apply_neighbor` + `score()` at leaf, `apply_neighbor` to revert
- Remove `test_neighbor` and `get_revert_neighbor` from Analyzer (neighbors revert themselves for KeySwap)

## Non-Functional Requirements
- Maintain correctness: all scores must match the current implementation
- Maintain readability: no unsafe code, clear method names
- Benchmarks must pass and show improvement
- Existing tests must pass (modulo API changes)
