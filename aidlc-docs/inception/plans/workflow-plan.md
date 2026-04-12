# Workflow Plan: Hot Loop Performance Optimization

## Overview
Eliminate the 31s init_cached_layout bottleneck and optimize the hot loop for SA + depth-3 optimization. Target: 100 layouts in < 3 minutes.

## Phases Included
- [x] Inception: Requirements ✓
- [ ] Construction: Implementation (3 units)
- [ ] Build and Test

## Skipped Phases
- Reverse Engineering (developer has deep familiarity)
- User Stories (pure performance optimization, no user-facing changes)
- Application Design (architecture is sound, we're optimizing data structures)
- NFR Requirements/Design (the requirement IS the NFR)
- Infrastructure Design (no infra changes)

---

## Unit 1: Sub-Cache API Refactor + Flat Array Scoring

**Scope**: All 4 sub-caches (SFCache, StretchCache, ScissorsCache, TrigramCache)

### Steps:
- [ ] 1.1 Add `weighted_score` flat arrays to SFCache, StretchCache, ScissorsCache
  - `weighted_score[pos * num_keys + key] -> i64` for each scoring dimension
  - SFCache: one array combining bg+sg weighted by finger dist
  - StretchCache: one array for bg weighted by stretch dist
  - ScissorsCache: one array per scissor type (full_bg, full_sg, half_bg, half_sg)
- [ ] 1.2 Add `init_weighted_scores(keys, bg_freq, sg_freq)` to each bigram cache
  - Precompute weighted_score arrays from current key positions
  - Pattern matches TrigramCache.init_weighted_scores
- [ ] 1.3 Add `update_weighted_scores_for_key_change(pos, old_key, new_key, ...)` to each bigram cache
  - Incremental update: only touch positions that share pairs with `pos`
  - Pattern matches TrigramCache.update_weighted_scores_for_key_change
- [ ] 1.4 Add `score_swap(pos_a, pos_b, key_a, key_b) -> i64` to each cache
  - Pure function: compute speculative swap score from flat arrays, no mutation
  - For bigram caches: `ws[pos_a][key_b] - ws[pos_a][key_a] + ws[pos_b][key_a] - ws[pos_b][key_b]`
  - For TrigramCache: same + swap_score_both correction
- [ ] 1.5 Add `score_replace(pos, old_key, new_key) -> i64` to each cache
  - For magic rule speculative scoring
- [ ] 1.6 Split `key_swap(apply: bool)` into `key_swap()` (mutate) and remove apply=false path
  - key_swap() updates running totals + weighted_scores (full apply)
- [ ] 1.7 Add `key_swap_fast()` that only updates running totals, not weighted_scores
  - For depth-N inner loop
- [ ] 1.8 Split `replace_key(apply: bool)` similarly
- [ ] 1.9 Remove `swap_delta` HashMap from all 4 caches
- [ ] 1.10 Remove `init_swap_deltas` and `clear_swap_deltas` from all 4 caches
- [ ] 1.11 Switch `active_rules` and `rule_delta` HashMaps from std to FxHashMap

## Unit 2: CachedLayout + Analyzer API Refactor

**Scope**: cached_layout.rs, analyze.rs

### Steps:
- [ ] 2.1 Replace `apply_neighbor(neighbor, apply: bool)` with:
  - `score_neighbor(neighbor) -> i64` — delegates to sub-cache score_swap/score_replace
  - `apply_neighbor(neighbor)` — delegates to sub-cache key_swap_fast/replace_key_fast
  - `apply_neighbor_and_update(neighbor)` — delegates to sub-cache key_swap/replace_key (full)
  - `update_scores()` — calls init_weighted_scores on all sub-caches
- [ ] 2.2 Remove `init_swap_deltas()` from CachedLayout::new()
  - The 31s init is eliminated
- [ ] 2.3 Update Analyzer to match new API
  - `test_neighbor` -> `score_neighbor` (delegates to CachedLayout.score_neighbor)
  - `apply_neighbor` -> two variants matching CachedLayout
  - Remove `get_revert_neighbor` (KeySwap is self-inverse, MagicRule tracks previous)
- [ ] 2.4 Share AnalyzerData via Arc
  - Change `CachedLayout.data: AnalyzerData` to `Arc<AnalyzerData>`
  - Change `Analyzer.data: Data` to store processed `Arc<AnalyzerData>` once
  - `use_layout` reuses the shared AnalyzerData instead of cloning

## Unit 3: Optimization Loop Updates + Benchmarks

**Scope**: simulated_annealing.rs, depth_optimization.rs, benchmarks.rs

### Steps:
- [ ] 3.1 Update SA loop to use `score_neighbor` + `apply_neighbor_and_update`
- [ ] 3.2 Update depth-N to use `apply_neighbor` (fast) + `score()` at leaf
  - Remove `get_revert_neighbor` calls — KeySwap reverts by applying same swap
  - Call `update_scores()` once after depth-N search completes
- [ ] 3.3 Update `greedy_improve` and `always_better_swap` to new API
- [ ] 3.4 Update benchmarks to new API
  - Remove benchmarks for removed methods
  - Add benchmarks for score_neighbor, apply_neighbor, apply_neighbor_and_update
- [ ] 3.5 Update repl and web-components to new API
- [ ] 3.6 Run full test suite, fix any failures
- [ ] 3.7 Run benchmarks, verify improvement
- [ ] 3.8 Git commit

## Dependency Order
Unit 1 → Unit 2 → Unit 3 (strict sequential)

## Risk Assessment
- **Medium**: Correctness of flat-array scoring for bigram caches (mitigated by existing tests)
- **Low**: API migration (mechanical, compiler-guided)
- **Low**: Arc<AnalyzerData> (straightforward refactor)
