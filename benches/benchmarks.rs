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

        // === Data Loading & Init ===
        bench.register(load_corpus, ["english"]);
        bench.register(load_layout, ["qwerty"]);
        bench.register(init_cached_layout, ["qwerty"]);

        // === Hot Loop Primitives (init outside bench closure) ===
        bench.register_many(
            list![
                score_only,
                score_neighbor_only,
                apply_neighbor_only,
                apply_and_score,
                apply_score_revert,
            ],
            swaps,
        );

        // === CachedLayout Low-Level ===
        bench.register_many(list![replace_key, key_swap], swaps);

        // === Magic Key Operations ===
        bench.register(apply_magic_rule, ["magic"]);
        bench.register(speculative_add_rule, ["magic"]);

        // === High-level (init outside bench closure) ===
        bench.register(best_neighbor, ["qwerty"]);
        bench.register(stats_calculation, ["qwerty"]);

        // === End-to-end Optimization (hot loop only, init excluded) ===
        bench.register(sa_hot_loop, [1000, 10000]);
        bench.register(greedy_optimize, ["qwerty"]);

        bench.run()?;
        Ok(())
    }

    // ==================== Data Loading & Init ====================

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

    fn init_cached_layout(bencher: Bencher, layout_name: &str) {
        let (mut analyzer, layout) = util::analyzer_layout("english", layout_name);
        bencher.bench(|| {
            analyzer.use_layout(&layout, &[]);
            black_box(analyzer.score());
        })
    }

    // ==================== Hot Loop Primitives ====================

    /// score() — read running totals, O(1)
    fn score_only(bencher: Bencher, _swap: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);
        bencher.bench(|| {
            for _ in 0..N { black_box(analyzer.score()); }
        })
    }

    /// score_neighbor() — speculative score, no mutation
    fn score_neighbor_only(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);
        bencher.bench(|| {
            for _ in 0..N { black_box(analyzer.score_neighbor(black_box(neighbor))); }
        })
    }

    /// apply_neighbor() — mutate + revert (depth-N hot path)
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

    /// apply + score (depth-N leaf pattern)
    fn apply_and_score(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);
        bencher.bench(|| {
            for _ in 0..N {
                analyzer.apply_neighbor(black_box(neighbor));
                black_box(analyzer.score());
                analyzer.apply_neighbor(black_box(neighbor)); // revert
            }
        })
    }

    /// apply + score + revert (full depth-N cycle)
    fn apply_score_revert(bencher: Bencher, neighbor: Neighbor) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        analyzer.use_layout(&layout, &[]);
        bencher.bench(|| {
            for _ in 0..N {
                analyzer.apply_neighbor(black_box(neighbor));
                black_box(analyzer.score());
                analyzer.apply_neighbor(black_box(neighbor)); // KeySwap is self-inverse
            }
        })
    }

    // ==================== CachedLayout Low-Level ====================

    fn replace_key(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::{CachedLayout, EMPTY_KEY};
        use oxeylyzer_core::weights::dummy_weights;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);
        let pos = match swap { Neighbor::KeySwap(PosPair(a, _)) => a, _ => 0 };

        bencher.bench(|| {
            for _ in 0..N {
                let key = cached.get_key(pos);
                cached.replace_key(pos, key, EMPTY_KEY);
                cached.replace_key(pos, EMPTY_KEY, key);
            }
        })
    }

    fn key_swap(bencher: Bencher, swap: Neighbor) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        let (_, layout) = util::analyzer_layout("english", "qwerty");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);
        let (pos_a, pos_b) = match swap { Neighbor::KeySwap(PosPair(a, b)) => (a, b), _ => (0, 1) };

        bencher.bench(|| {
            for _ in 0..N {
                cached.swap_keys(pos_a, pos_b);
                cached.swap_keys(pos_a, pos_b);
            }
        })
    }

    // ==================== Magic Key Operations ====================

    fn apply_magic_rule(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        let char_mapping = cached.char_mapping();
        let leader_a = char_mapping.get_u('a');
        let output_b = char_mapping.get_u('b');
        let output_c = char_mapping.get_u('c');
        let magic_key = char_mapping.get_u('µ');

        bencher.bench(|| {
            for _ in 0..N {
                cached.apply_magic_rule(magic_key, leader_a, output_c, true);
                cached.apply_magic_rule(magic_key, leader_a, output_b, true);
            }
        })
    }

    fn speculative_add_rule(bencher: Bencher, _layout_name: &str) {
        use oxeylyzer_core::cached_layout::CachedLayout;
        use oxeylyzer_core::weights::dummy_weights;

        let layout = oxeylyzer_core::layout::Layout::load("./layouts/test/magic.dof")
            .expect("magic layout should exist");
        let data = oxeylyzer_core::data::Data::load("./data/english.json").unwrap();
        let weights = dummy_weights();
        let mut cached = CachedLayout::new(&layout, data.clone(), &weights);

        let char_mapping = cached.char_mapping();
        let leader_a = char_mapping.get_u('a');
        let output_c = char_mapping.get_u('c');
        let magic_key = char_mapping.get_u('µ');

        bencher.bench(|| {
            for _ in 0..N {
                black_box(cached.apply_magic_rule(magic_key, leader_a, output_c, false));
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
            for _ in 0..N { black_box(analyzer.stats()); }
        })
    }

    // ==================== End-to-end Optimization (hot loop only) ====================

    /// SA hot loop only — measures pure iteration cost.
    /// Clones the initialized state to avoid re-running CachedLayout::new() each iteration.
    fn sa_hot_loop(bencher: Bencher, iterations: usize) {
        let (mut analyzer, layout) = util::analyzer_layout("english", "qwerty");
        let pins = vec![];
        let initial_temperature = 1E-4;
        let final_temperature = 1E-7;

        // Init once, then clone for each bench iteration
        analyzer.use_layout(&layout, &pins);
        let snapshot = analyzer.clone();

        bencher.bench(|| {
            analyzer = snapshot.clone();
            black_box(analyzer.annealing_improve(
                layout.clone(),
                &pins,
                initial_temperature,
                final_temperature,
                iterations,
            ));
        })
    }

    /// Greedy optimize — clone snapshot to avoid re-init.
    fn greedy_optimize(bencher: Bencher, layout_name: &str) {
        let (mut analyzer, layout) = util::analyzer_layout("english", layout_name);
        let random = layout.random();
        analyzer.use_layout(&random, &[]);
        let snapshot = analyzer.clone();

        bencher.bench(|| {
            analyzer = snapshot.clone();
            black_box(analyzer.greedy_improve(&random, &[]));
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
