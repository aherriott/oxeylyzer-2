#![allow(dead_code)]

mod util;

#[cfg(not(target_arch = "wasm32"))]
mod bench {
    use super::util;

    use std::hint::black_box;

    use diol::prelude::*;
    use oxeylyzer_core::{optimization::*, prelude::*};
    use rand::{distributions::Alphanumeric, Rng};

    const N: i32 = 10000;

    pub(super) fn main() -> std::io::Result<()> {
        let swaps = [
            Neighbor::KeySwap(PosPair(1, 4)),
            Neighbor::KeySwap(PosPair(5, 28)),
        ];

        let mut bench = Bench::new(BenchConfig::from_args()?);

        // === Data Loading ===
        bench.register(load_corpus, ["english"]);
        bench.register(load_layout, ["qwerty"]);

        // === Cache Initialization ===
        bench.register(init_cached_layout, ["qwerty"]);

        // === Core Operations (per-swap benchmarks) ===
        bench.register_many(
            list![
                // Scoring
                score_only,
                // Neighbor operations - granular breakdown
                apply_neighbor_only,
                test_neighbor_full,
                // Breakdown of test_neighbor
                apply_then_score,
                apply_score_revert,
                apply_score_revert_copy,
                // CachedLayout low-level operations
                replace_key,
                key_swap,
            ],
            swaps,
        );

        // === Magic Key Operations ===
        bench.register(apply_magic_rule, ["magic"]);

        // === Speculative Scoring Operations ===
        // These measure O(1) speculative scoring performance (apply=false)
        // Targets: key_swap < 2µs, add_rule < 2µs, total < 5µs
        bench.register_many(
            list![speculative_key_swap],
            swaps,
        );
        bench.register(speculative_add_rule, ["magic"]);
        bench.register(total_speculative_scoring, ["magic"]);

        // === High-level Operations ===
        bench.register(best_neighbor, ["qwerty"]);
        bench.register(stats_calculation, ["qwerty"]);

        // === Optimization (quick) ===
        bench.register(optimize, [OptimizationMethod::Greedy]);

        // === Simulated Annealing ===
        bench.register(annealing_iterations, [1000]);

        bench.run()?;
        Ok(())
    }

    // ==================== Data Loading ====================

    fn load_corpus(bencher: Bencher, corpus: &str) {
        bencher.bench(|| {
            black_box(
                oxeylyzer_core::data::Data::load(format!("./data/{corpus}.json"))
                    .expect("corpus should exist"),
            )
        });
    }

    fn load_layout(bencher: Bencher, layout: &str) {
        bencher.bench(|| {
            black_box(
                oxeylyzer_core::layout::Layout::load(format!("./layouts/{layout}.dof"))
                    .expect("layout should exist"),
            )
        });
    }

    // ==================== Cache Initialization ====================

    fn init_cached_layout(bencher: Bencher, layout_name: &str) {
        let (mut analyzer, layout) = util::analyzer_layout("english", layout_name);

        bencher.bench(|| {
            analyzer.use_layout(&layout, &[]);
            black_box(analyzer.score());
        })
    }

    // ==================== Scoring ====================

    fn score_only(bencher: Bencher, _swap: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                black_box(analyzer.score());
            }
        })
    }

    // ==================== Neighbor Operations - Granular ====================

    /// Just apply_neighbor (apply + apply to revert)
    fn apply_neighbor_only(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                analyzer.apply_neighbor(black_box(neighbor));
                analyzer.apply_neighbor(black_box(neighbor)); // revert
            }
        })
    }

    /// Full test_neighbor (apply + score + revert + copy_from)
    fn test_neighbor_full(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                black_box(analyzer.test_neighbor(black_box(neighbor)));
            }
        })
    }

    /// Apply then score (no revert)
    fn apply_then_score(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                analyzer.apply_neighbor(black_box(neighbor));
                black_box(analyzer.score());
                analyzer.apply_neighbor(black_box(neighbor)); // revert to keep state consistent
            }
        })
    }

    /// Apply, score, revert (no copy_from)
    fn apply_score_revert(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                let revert = analyzer.get_revert_neighbor(black_box(neighbor));
                analyzer.apply_neighbor(black_box(neighbor));
                black_box(analyzer.score());
                analyzer.apply_neighbor(black_box(revert));
            }
        })
    }

    /// Apply, score, revert, copy_from (full test_neighbor equivalent)
    fn apply_score_revert_copy(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);

        // Note: This is what test_neighbor does internally
        bencher.bench(|| {
            for _ in 0..N {
                black_box(analyzer.test_neighbor(black_box(neighbor)));
            }
        })
    }

    // ==================== High-level Operations ====================

    fn best_neighbor(bencher: Bencher, layout_name: &str) {
        let (mut analyzer, layout) = util::analyzer_layout("english", layout_name);
        analyzer.use_layout(&layout, &[]);
        let neighbors = analyzer.neighbors();

        bencher.bench(|| {
            black_box(analyzer.best_neighbor(&neighbors));
        })
    }

    fn stats_calculation(bencher: Bencher, layout_name: &str) {
        let (mut analyzer, layout) = util::analyzer_layout("english", layout_name);
        analyzer.use_layout(&layout, &[]);

        bencher.bench(|| {
            for _ in 0..N {
                black_box(analyzer.stats());
            }
        })
    }

    // ==================== Optimization ====================

    fn optimize(bencher: Bencher, method: OptimizationMethod) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        let layout = layout.random();

        bencher.bench(|| {
            black_box(method.optimize(&mut analyzer, layout.clone()));
        })
    }

    fn annealing_iterations(bencher: Bencher, iterations: usize) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        let pins = vec![];
        let initial_temperature = 1E-4;
        let final_temperature = 1E-7;

        bencher.bench(|| {
            analyzer.use_layout(&layout, &pins);
            black_box(analyzer.annealing_improve(
                layout.clone(),
                &pins,
                initial_temperature,
                final_temperature,
                iterations,
            ));
        })
    }

    // ==================== Utility benchmarks ====================

    fn generate_data(bencher: Bencher, length: usize) {
        let v = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect::<Vec<_>>();

        bencher.bench(|| black_box(v.iter().copied().collect::<Data>()))
    }

    // ==================== CachedLayout Low-Level Operations ====================

    /// Benchmark replace_key operation
    fn replace_key(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::{CachedLayout, EMPTY_KEY};
        use oxeylyzer_core::weights::dummy_weights;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        // Get position from the swap neighbor
        let pos = match swap {
            Neighbor::KeySwap(PosPair(a, _)) => a,
            _ => 0,
        };

        bencher.bench(|| {
            for _ in 0..N {
                // Replace key with EMPTY and back
                let key = cached.get_key(pos);
                cached.replace_key(pos, key, EMPTY_KEY);
                cached.replace_key(pos, EMPTY_KEY, key);
            }
        })
    }

    /// Benchmark key_swap operation via swap_keys
    fn key_swap(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        let (pos_a, pos_b) = match swap {
            Neighbor::KeySwap(PosPair(a, b)) => (a, b),
            _ => (0, 1),
        };

        bencher.bench(|| {
            for _ in 0..N {
                // Swap and swap back
                cached.swap_keys(pos_a, pos_b);
                cached.swap_keys(pos_a, pos_b);
            }
        })
    }

    /// Benchmark apply_magic_rule operation (magic key functionality)
    fn apply_magic_rule(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        // Use the magic test layout which has magic keys defined
        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        // Get magic key info from the layout's char_mapping
        // The magic layout has: mag2 with rules "a" -> "b" and "b" -> "c"
        let char_mapping = cached.char_mapping();
        let leader_a = char_mapping.get_u('a');
        let output_b = char_mapping.get_u('b');
        let output_c = char_mapping.get_u('c');
        let magic_key = char_mapping.get_u('µ'); // mag2 uses µ

        bencher.bench(|| {
            for _ in 0..N {
                // Alternate between setting output to 'c' and 'b'
                cached.apply_magic_rule(magic_key, leader_a, output_c, true);
                cached.apply_magic_rule(magic_key, leader_a, output_b, true);
            }
        })
    }

    // ==================== Speculative Scoring Benchmarks ====================
    // These benchmarks measure O(1) speculative scoring performance (apply=false).
    // Target: < 2µs per analyzer for key_swap, < 2µs per analyzer for add_rule,
    // < 5µs total for all speculative scoring operations.
    // See .kiro/specs/const-freq-analyzers/requirements.md Requirement 7.

    /// Benchmark speculative key_swap operation (apply=false).
    /// This measures the O(1) lookup table performance for speculative scoring.
    /// Target: < 2µs per analyzer (< 5µs total across all analyzers)
    fn speculative_key_swap(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        let (pos_a, pos_b) = match swap {
            Neighbor::KeySwap(PosPair(a, b)) => (a, b),
            _ => (0, 1),
        };

        bencher.bench(|| {
            for _ in 0..N {
                // Speculative scoring: apply=false, no state mutation
                black_box(cached.score_neighbor(Neighbor::KeySwap(PosPair(pos_a, pos_b))));
            }
        })
    }

    /// Benchmark speculative add_rule operation (apply=false).
    /// This measures the O(1) lookup table performance for magic rule speculative scoring.
    /// Target: < 2µs per analyzer (< 5µs total across all analyzers)
    fn speculative_add_rule(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        // Use the magic test layout which has magic keys defined
        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        // Get magic key info from the layout's char_mapping
        // The magic layout has: mag2 with rules "a" -> "b" and "b" -> "c"
        let char_mapping = cached.char_mapping();
        let leader_a = char_mapping.get_u('a');
        let output_c = char_mapping.get_u('c');
        let magic_key = char_mapping.get_u('µ'); // mag2 uses µ

        bencher.bench(|| {
            for _ in 0..N {
                // Speculative scoring: apply=false, no state mutation
                // Test changing the rule output speculatively
                black_box(cached.apply_magic_rule(magic_key, leader_a, output_c, false));
            }
        })
    }

    /// Benchmark total speculative scoring path.
    /// This measures the combined performance of speculative key_swap and add_rule.
    /// Target: < 5µs total for all speculative scoring operations.
    fn total_speculative_scoring(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        // Use the magic test layout which has magic keys defined
        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        // Get magic key info from the layout's char_mapping
        let char_mapping = cached.char_mapping();
        let leader_a = char_mapping.get_u('a');
        let output_c = char_mapping.get_u('c');
        let magic_key = char_mapping.get_u('µ'); // mag2 uses µ

        // Use positions 1 and 4 for key swap (typical swap positions)
        let pos_a = 1;
        let pos_b = 4;

        bencher.bench(|| {
            for _ in 0..N {
                // Full speculative scoring path: key_swap + add_rule
                black_box(cached.score_neighbor(Neighbor::KeySwap(PosPair(pos_a, pos_b))));
                black_box(cached.apply_magic_rule(magic_key, leader_a, output_c, false));
            }
        })
    }
}

#[cfg(target_arch = "wasm32")]
mod bench {
    pub(super) fn main() -> std::io::Result<()> {
        println!("Benchmarking for wasm is currently not possible");

        Ok(())
    }
}

fn main() -> std::io::Result<()> {
    bench::main()
}
