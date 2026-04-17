---
inclusion: auto
---

# Testing & Known Bug Patterns

## How to Test

```bash
# All tests (slow — some take 200s due to use_layout calls)
cargo test -p oxeylyzer-core

# Just the API correctness tests (14 tests, ~200s)
cargo test -p oxeylyzer-core api_tests

# Quick compile check
cargo check --tests --benches

# Build release binary
cargo build -p oxeylyzer-repl --release
```

## Critical Test: api_tests

Located in `core/src/api_tests.rs`. These 14 tests cover the exact bugs we encountered:

1. **score_is_negative** — Score should always be ≤ 0 for any layout (magic and non-magic)
2. **swap_key_is_reversible** — Double swap restores exact score (magic and non-magic)
3. **swap_key_no_update_plus_update_matches_swap_key** — Both paths produce identical scores
4. **score_neighbor_keyswap_matches_apply** — Speculative score matches actual apply
5. **score_neighbor_magic_matches_apply** — Same for magic rule neighbors
6. **score_neighbor_does_not_mutate_magic** — 50 speculative calls don't change state
7. **many_swaps_no_drift_magic** — Score stays negative after many swaps
8. **swap_and_revert_100_times_magic** — 100 swap-reverts cause zero drift
9. **replace_rule_and_revert** — Magic rule change + revert restores score
10. **greedy_produces_negative_score_magic** — Greedy on random magic layout gives negative score
11. **greedy_improves_random_layout** — Greedy never makes score worse
12. **apply_neighbor_keyswap_matches_swap_key** — Dispatch works correctly

## Bug Pattern: Magic Rule Score Drift

The #1 recurring bug. Magic rules add score deltas to sub-cache running totals. When keys move (swap/replace), the magic deltas become stale because they were computed with the old key positions.

### Root cause:
Sub-cache `add_rule` methods store the delta when a rule is first applied. When removing/replacing a rule, they subtract the stored delta. But if keys moved since the rule was applied, the stored delta is wrong.

### The fix:
`update()` recomputes ALL magic deltas from scratch. Every mutation that moves keys must be followed by `update()` before `score()` is read. The base API versions (swap_key, replace_key, replace_rule) call update() automatically. The _no_update versions require explicit update() calls.

### How to detect:
If you ever see positive scores, magic deltas have drifted. Add a debug assert: `debug_assert!(score < 0)` after any score() call on a fully-placed layout.

## Bug Pattern: Speculative Magic Scoring

`apply_magic_rule(apply=false)` was the old speculative path. It computed deltas relative to running totals that could be stale. This produced wrong speculative scores, causing greedy to make bad decisions (not pruning bad magic rules).

### The fix:
`score_neighbor` for MagicRule now uses apply-score-revert: actually applies the rule, reads the true score, then reverts. Slower but correct. A TODO exists for implementing a proper speculative path.

## Bug Pattern: Auto-scale After Key Placement

The trigram auto-scale used to change weights AFTER keys were placed. But `replace_key_fast` dumps deltas into `magic_rule_score_delta` (a pre-multiplied value), so changing weights after placement has no effect on the score.

### The fix:
Scale factors are now computed from corpus data BEFORE any layout is loaded, in `Weights::compute_scale_factors()`. The scaled weights are passed to `CachedLayout::new` which applies them before placing keys.

## Bug Pattern: replace_key_fast Trigram Tracking

`replace_key_fast` (now `replace_key_no_update`) uses the flat trigram array and dumps the total delta into `magic_rule_score_delta`. It does NOT update per-type frequency totals (inroll_freq, outroll_freq, etc.). This means:
- `score()` works correctly (reads magic_rule_score_delta)
- Per-type frequency breakdown is not available after using this path
- Weight changes after placement don't affect the score (the delta is pre-multiplied)

This is a known architectural limitation. The fix would be to track per-type frequencies in replace_key_fast, but it would add overhead to the hot path.
