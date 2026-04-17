---
inclusion: auto
---

# Optimization Strategy

## What Works: Random-Restart Greedy

The best approach for this problem is simple: randomize a layout, run greedy hill-climbing, repeat. Keep the best results. This beats all sophisticated algorithms we tried.

The `gen` command implements this: continuous random-restart greedy across all CPU threads, tracking top N layouts.

### Local search: Greedy hill-climb
- Iterates all neighbors (key swaps + magic rule changes)
- Uses `score_neighbor` for speculative scoring (no state change, ~11µs per neighbor)
- Applies the first improvement found via `apply_neighbor`
- Repeats until no single swap/rule change improves the score

### Why greedy wins:
- Each greedy run takes ~1-2s for a 32-key layout
- With 14 threads, that's ~6-10 layouts/second
- In 30 seconds, you test ~200 random starting points
- The diversity from random starts matters more than sophisticated exploration

## What Doesn't Work (and why)

### Dual Annealing (DA)
- Global perturbation (multi-swap) + local greedy
- Problem: partial perturbations from one layout are less diverse than completely random starts
- DA with 30s budget: best ~-1.4T. Random-restart greedy with 30s: best ~-800B to -1.0T
- The global SA parameters (temp, qv, qa, restart ratio, max_swaps) barely matter — all within noise

### MCTS
- Tree search over key placements with SA/greedy rollouts
- Problem: rollout cost (~11µs per score) is too high relative to the decision space
- Best MCTS result: -226B. SA+greedy: -118B. Random-restart greedy: much better.
- Tree depth 1 was best — essentially just trying all positions for the most frequent key

### Branch & Bound
- Exhaustive search with pruning
- Problem: 30 positions = 30! search space. Average prune depth ~13 with good bounds.
- Infeasible for full-depth search. Useful for understanding the problem structure.

### Low-temperature SA as local search
- SA at temp 0.001 with 1000 iterations before greedy
- Marginally better than pure greedy (~24B improvement) but 4x slower per iteration
- Not worth the throughput tradeoff for random-restart approach

## Progressive Deepening Greedy
Available via `-g 2` or `-g 3` flag. Runs depth-1 greedy until stuck, then tries depth-2 (all 2-swap combos), then depth-3. Any improvement restarts from depth-1. Too slow for the DA/gen hot loop but useful for final polish of a specific layout.

## Weight Tuning

Used `goal_seek_weights.py` (Nelder-Mead optimization) to find weights where sturdy ranks #1 among reference layouts. The optimizer found:
- sfbs=7, sfs=2, stretches=3, inroll=7, outroll=4, alternate=4, redirect=4
- Higher inroll weight produces sturdy-like (inroll-heavy) layouts
- Higher sfbs weight produces lower SFB layouts at the cost of trigram flow

## Magic Rule Optimization

Magic rules are explored as part of the greedy neighbor search. The neighbor list includes all possible (leader, output) combinations for each magic key, plus EMPTY_KEY to clear rules.

`magic_rule_penalty` controls how many rules survive: penalty=10 ≈ 2-4 rules, penalty=5 ≈ 1 rule, penalty=20+ ≈ no rules. `magic_repeat_penalty` separately controls repeat rules (leader→same key) which are easy to remember.

Random layouts start with random magic rules for every key. Greedy prunes the ones that aren't worth the penalty.
