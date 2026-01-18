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
                remove_key,
                add_key,
                key_swap,
            ],
            swaps,
        );

        // === Magic Key Operations ===
        bench.register(steal_bigram, ["magic"]);

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

        bencher.bench(|| {
            black_box(analyzer.best_neighbor());
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

    /// Benchmark remove_key operation
    fn remove_key(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let mut cached = CachedLayout::new(&layout, data.clone());

        // Get position from the swap neighbor
        let pos = match swap {
            Neighbor::KeySwap(PosPair(a, _)) => a,
            _ => 0,
        };

        bencher.bench(|| {
            for _ in 0..N {
                // We need to re-add the key after removing to keep benchmarking
                let key = cached.get_key(pos);
                cached.remove_key(pos);
                cached.add_key(pos, key);
            }
        })
    }

    /// Benchmark add_key operation
    fn add_key(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let mut cached = CachedLayout::new(&layout, data.clone());

        // Get position from the swap neighbor
        let pos = match swap {
            Neighbor::KeySwap(PosPair(a, _)) => a,
            _ => 0,
        };

        bencher.bench(|| {
            for _ in 0..N {
                let key = cached.get_key(pos);
                cached.remove_key(pos);
                cached.add_key(pos, key);
            }
        })
    }

    /// Benchmark key_swap operation (remove_key x2 + add_key x2)
    fn key_swap(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let mut cached = CachedLayout::new(&layout, data.clone());

        let (pos_a, pos_b) = match swap {
            Neighbor::KeySwap(PosPair(a, b)) => (a, b),
            _ => (0, 1),
        };

        bencher.bench(|| {
            for _ in 0..N {
                // Perform key swap manually
                let key_a = cached.get_key(pos_a);
                let key_b = cached.get_key(pos_b);
                cached.remove_key(pos_a);
                cached.remove_key(pos_b);
                cached.add_key(pos_a, key_b);
                cached.add_key(pos_b, key_a);

                // Swap back to original state
                cached.remove_key(pos_a);
                cached.remove_key(pos_b);
                cached.add_key(pos_a, key_a);
                cached.add_key(pos_b, key_b);
            }
        })
    }

    /// Benchmark apply_magic_rule operation (magic key functionality)
    fn steal_bigram(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;

        // Use the magic test layout which has magic keys defined
        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let mut cached = CachedLayout::new(&layout, data.clone());

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
                cached.apply_magic_rule(magic_key, leader_a, output_c);
                cached.apply_magic_rule(magic_key, leader_a, output_b);
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
