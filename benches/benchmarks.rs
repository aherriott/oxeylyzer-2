#![allow(dead_code)]

mod util;

#[cfg(not(target_arch = "wasm32"))]
mod bench {
    use super::util;

    use std::hint::black_box;

    use diol::prelude::*;
    use oxeylyzer_core::{optimization::*, prelude::*};
    use rand::{distributions::Alphanumeric, Rng};

    pub(super) fn main() -> std::io::Result<()> {
        let swaps = [PosPair(1, 4), PosPair(5, 28), PosPair(3, 13), PosPair(7, 7)];

        let mut bench = Bench::new(BenchConfig::from_args()?);
        // bench.register_many(
        //     list![generate_data, collect_s],
        //     [100, 1000, 10000, 100_000, 1_000_000, 10_000_000],
        // );
        // bench.register(generate_real_data, ["monkeyracer", "english"]);
        // bench.register(find_best_swap, ["rstn-oxey"]);
        bench.register_many(
            list![analyze_swap_bigrams, analyze_swap_stretches, analyze_swap],
            swaps,
        );
        // bench.register(init_cached_layout, ["rstn-oxey", "colemak-dh", "sturdy"]);
        // bench.register(
        //     optimize,
        //     [
        //         OptimizationMethod::Greedy,
        //         OptimizationMethod::GreedyDepth2,
        //         OptimizationMethod::GreedyDepth3,
        //         // OptimizationMethod::GreedyDepth4,
        //         OptimizationMethod::GreedyAlternative,
        //         OptimizationMethod::GreedyAlternativeD3,
        //     ],
        // );

        bench.run()?;
        Ok(())
    }

    fn optimize(bencher: Bencher, method: OptimizationMethod) {
        let (analyzer, layout) = util::analyzer_layout("english", "rstn-oxey");
        let layout = layout.random();

        bencher.bench(|| {
            black_box(method.optimize(&analyzer, layout.clone()));
        })
    }

    fn init_cached_layout(bencher: Bencher, layout_name: &str) {
        let (analyzer, layout) = util::analyzer_layout("english", layout_name);

        bencher.bench(|| {
            black_box(analyzer.cached_layout(layout.clone(), &[]));
        })
    }

    fn analyze_swap(bencher: Bencher, swap: PosPair) {
        let (analyzer, layout) = util::analyzer_layout("english", "rstn-oxey");
        let mut cache = analyzer.cached_layout(layout, &[]);

        bencher.bench(|| black_box(analyzer.score_cached_swap(&mut cache, black_box(swap))))
    }

    fn analyze_swap_bigrams(bencher: Bencher, swap: PosPair) {
        let (analyzer, layout) = util::analyzer_layout("english", "rstn-oxey");
        let mut cache = analyzer.cached_layout(layout, &[]);

        bencher
            .bench(|| black_box(analyzer.score_swap_weighted_bigrams(&mut cache, black_box(swap))))
    }

    fn analyze_swap_stretches(bencher: Bencher, swap: PosPair) {
        let (analyzer, layout) = util::analyzer_layout("english", "rstn-oxey");
        let mut cache = analyzer.cached_layout(layout, &[]);

        bencher
            .bench(|| black_box(analyzer.score_swap_stretch_bigrams(&mut cache, black_box(swap))))
    }

    fn generate_data(bencher: Bencher, length: usize) {
        let v = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect::<Vec<_>>();

        bencher.bench(|| black_box(v.iter().copied().collect::<Data>()))
    }

    fn collect_s(bencher: Bencher, length: usize) {
        let v = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect::<Vec<_>>();

        bencher.bench(|| black_box(v.iter().collect::<String>()))
    }

    fn generate_real_data(bencher: Bencher, name: &str) {
        let cleaner = CorpusCleaner::builder()
            .with_chars("abcdefghijklmnopqrstuvwxyz".chars())
            .qwerty_punctuation_mappings(true)
            .with_chars([' '])
            .build();

        bencher.bench(|| {
            black_box(
                Data::from_path(format!("./corpora/{name}"), name, &cleaner)
                    .expect("this path should exist"),
            )
        });
    }

    fn find_best_swap(bencher: Bencher, layout_name: &str) {
        let (analyzer, layout) = util::analyzer_layout("english", layout_name);

        let mut cache = analyzer.cached_layout(layout, &[]);

        bencher.bench(|| {
            analyzer.best_swap(&mut cache);
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
