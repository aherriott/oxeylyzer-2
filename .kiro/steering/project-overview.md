---
inclusion: auto
---

# Oxeylyzer-2: Keyboard Layout Optimizer

## What This Project Is

A Rust-based keyboard layout analyzer and optimizer. Given a corpus of text (e.g., English), a keyboard geometry (e.g., colstag), and a set of weights, it finds optimal key placements that minimize same-finger bigrams, stretches, scissors, and maximize good trigram patterns (inrolls, outrolls, alternates).

## Key Concepts

- **Layout**: A mapping of characters to physical key positions on a keyboard
- **Magic keys**: Special keys that produce different output depending on the preceding key (e.g., typing 't' then magic produces 'h'). Defined in .dof layout files.
- **Repeat rules**: Magic rules where output equals leader (e.g., t→t). Eliminates same-finger bigrams for repeated characters. Cheap to remember.
- **Score**: Always negative. Less negative = better layout. Composed of: SFB + stretch + scissors + trigram + finger_usage + magic_penalty
- **Weights**: All positive numbers in analyzer-config.toml. The code handles sign convention internally (penalties negated in set_weights).

## Project Structure

```
core/           - Core library (scoring, caching, optimization algorithms)
  src/
    cached_layout.rs  - CachedLayout: the central scoring engine with incremental updates
    optimization.rs   - RolloutPolicy, OptStep, greedy/SA implementations
    dual_annealing.rs - Dual annealing optimizer (global + local search)
    mcts.rs          - Monte Carlo Tree Search (experimental, not competitive)
    branch_bound.rs  - Branch and bound search (experimental, not competitive)
    analyze.rs       - Analyzer: high-level wrapper around CachedLayout
    weights.rs       - Weight definitions and scale factor computation
    trigrams.rs      - Trigram scoring cache (~5000 lines, most complex sub-cache)
    same_finger.rs   - SFB/SFS scoring cache
    stretches.rs     - Stretch scoring cache
    scissors.rs      - Scissors scoring cache
    layout.rs        - Layout struct, .dof file loading, Display impl
    data.rs          - Corpus data loading
repl/           - REPL binary (commands: gen, da, sa, analyze, rank, etc.)
  src/
    lib.rs           - All REPL commands
    flags.rs         - CLI flag definitions (xflags)
    config.rs        - Config file loading
layouts/        - .dof layout files
data/           - Corpus JSON files
analyzer-config.toml - Weights and corpus configuration
```

## Build & Run

```bash
cargo build -p oxeylyzer-repl --release
echo "gen my-layout -t 30" | ./target/release/main
```

## Key Commands

- `gen <layout> [-t secs] [-n top]` - Continuous random-restart greedy optimization. Runs forever or until timeout.
- `da <layout> [-t secs]` - Dual annealing (global perturbation + local greedy). Less effective than gen for this problem.
- `sa <layout> [count] [-s iters]` - Classic SA + greedy. Good for comparison.
- `a <layout>` - Analyze a layout (show score breakdown and stats)
- `r` - Reload config (pick up weight changes without restarting)
