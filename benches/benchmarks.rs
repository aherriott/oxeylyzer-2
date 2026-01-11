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
            ],
            swaps,
        );

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
                analyzer.apply_neighbor(black_box(neighbor));
                black_box(analyzer.score());
                analyzer.apply_neighbor(black_box(neighbor.revert()));
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
