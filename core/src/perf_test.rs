#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use crate::weights::dummy_weights;
    use std::time::Instant;

    #[test]
    fn profile_score_neighbor() {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let mut analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");
        analyzer.use_layout(&layout, &[]);
        let neighbors = analyzer.neighbors();

        let key_swaps: Vec<_> = neighbors.iter().filter(|n| matches!(n, Neighbor::KeySwap(_))).copied().collect();
        println!("\n=== {} neighbors: {} KeySwap ===", neighbors.len(), key_swaps.len());

        let iterations = 10_000;
        let start = Instant::now();
        for i in 0..iterations {
            let n = key_swaps[i % key_swaps.len()];
            std::hint::black_box(analyzer.score_neighbor(n));
        }
        let elapsed = start.elapsed();
        println!("=== score_neighbor: {iterations} iters in {elapsed:?} ({:?}/iter) ===", elapsed / iterations as u32);
    }

    #[test]
    fn profile_apply_neighbor() {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let mut analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");
        analyzer.use_layout(&layout, &[]);
        let neighbors = analyzer.neighbors();

        let iterations = 10_000;
        let start = Instant::now();
        for i in 0..iterations {
            let n = neighbors[i % neighbors.len()];
            analyzer.apply_neighbor(n);
            analyzer.apply_neighbor(n);
        }
        let elapsed = start.elapsed();
        println!("\n=== apply_neighbor (fast): {iterations} iters in {elapsed:?} ({:?}/iter) ===", elapsed / iterations as u32);
    }

    #[test]
    fn profile_apply_and_update() {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let mut analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");
        analyzer.use_layout(&layout, &[]);
        let neighbors = analyzer.neighbors();

        let iterations = 1_000;
        let start = Instant::now();
        for i in 0..iterations {
            let n = neighbors[i % neighbors.len()];
            analyzer.apply_neighbor_and_update(n);
            analyzer.apply_neighbor_and_update(n);
        }
        let elapsed = start.elapsed();
        println!("\n=== apply_neighbor_and_update: {iterations} iters in {elapsed:?} ({:?}/iter) ===", elapsed / iterations as u32);
    }
}
