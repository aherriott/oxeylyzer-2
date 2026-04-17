---
inclusion: auto
---

# Architecture & Design Decisions

## CachedLayout Mutation API

The central scoring engine. Three mutation operations, each with two variants:

### Public API (score() valid after):
- `replace_key(pos, old, new)` - Place or remove a key
- `swap_key(pos_a, pos_b)` - Swap two keys (uses dedicated key_swap on sub-caches for perf)
- `replace_rule(magic_key, leader, output)` - Change a magic rule
- `apply_neighbor(neighbor)` - Dispatches to swap_key or replace_rule

### No-update API (score() INVALID until update()):
- `replace_key_no_update(pos, old, new)`
- `swap_key_no_update(pos_a, pos_b)`
- `replace_rule_no_update(magic_key, leader, output)`
- `apply_neighbor_no_update(neighbor)`
- `update()` - Recomputes magic rule deltas from current key positions

### Speculative scoring (no permanent state change):
- `score_neighbor(neighbor)` - KeySwap: fast delta math (~11µs). MagicRule: apply-score-revert.

### Why this design:
Magic rule deltas in sub-caches become stale when keys move. Rather than tracking this incrementally (which caused drift bugs), `update()` recomputes all magic deltas from scratch. The base API versions call `_no_update` + `update()`. The `_no_update` versions are for batch operations where you want to defer the recompute.

## Sub-caches

CachedLayout has 4 scoring sub-caches:
- **SFCache** - Same-finger bigrams and skipgrams. Score = Σ freq × distance × finger_weight × metric_weight
- **StretchCache** - Finger stretches. Score = Σ freq × stretch_distance × weight
- **ScissorsCache** - Scissor motions (full/half, bigram/skipgram). 4 separate weights.
- **TrigramCache** - Trigram patterns (inroll, outroll, alternate, redirect, onehandin, onehandout). Uses offset scoring where all contributions ≤ 0.

Each sub-cache maintains running totals updated incrementally by replace_key and key_swap operations. Magic rules add deltas via `add_rule` methods.

## Score Normalization

Trigram scores are ~100x smaller than bigram scores in raw magnitude. Scale factors are computed from corpus data in `Weights::compute_scale_factors()` and applied to trigram weights BEFORE key placement. This ensures running totals are correct from the start.

Magic penalty and finger usage also have auto-computed scale factors so that weight=1 produces meaningful score contributions.

## Weight System

All weights in analyzer-config.toml are positive. The code negates penalties internally:
- Penalties (sfbs, sfs, stretches, scissors, redirect, magic_rule_penalty): negated in set_weights
- Rewards (inroll, outroll, alternate, onehandin, onehandout): kept positive
- Finger weights: multiply with SFB/SFS weights. Higher = more penalty for SFBs on that finger.
- finger_usage: penalizes placing high-frequency keys on weak fingers

## Magic Key Architecture

Magic keys produce different output depending on the preceding key. In the scoring system:
- Magic rules redirect bigram/trigram frequencies (e.g., t→magic becomes t→h)
- Each sub-cache has `add_rule` methods that compute the score delta from a rule change
- `current_magic_rules` HashMap on CachedLayout tracks active rules
- `update()` recomputes all magic deltas from scratch (needed after key swaps)
- Neighbor list includes all possible (leader, output) combinations including repeats and EMPTY (clear)
