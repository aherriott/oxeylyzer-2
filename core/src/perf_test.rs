#[cfg(test)]
mod tests {
    use crate::cached_layout::CachedLayout;
    use crate::prelude::*;
    use crate::weights::{dummy_weights, ScaleFactors};
    use std::time::Instant;

    fn make() -> (Analyzer, Layout) {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        (Analyzer::new(data, weights), Layout::load("../layouts/qwerty.dof").expect("layout"))
    }

    #[test]
    fn profile_score_neighbor() {
        let (mut a, l) = make();
        a.use_layout(&l, &[]);
        let ns: Vec<_> = a.neighbors().into_iter().filter(|n| matches!(n, Neighbor::KeySwap(_))).collect();
        let n = 1_000_000;
        let t = Instant::now();
        let mut sum: i64 = 0;
        for i in 0..n { sum = sum.wrapping_add(a.score_neighbor(ns[i % ns.len()])); }
        let elapsed = t.elapsed();
        let per_iter_ns = elapsed.as_nanos() / n as u128;
        println!("\n=== score_neighbor: {n} iters, {per_iter_ns}ns/iter, {:?} total (sum={sum}) ===", elapsed);
    }

    #[test]
    fn profile_apply_neighbor() {
        let (mut a, l) = make();
        a.use_layout(&l, &[]);
        let ns = a.neighbors();
        let n = 10_000;
        let t = Instant::now();
        for i in 0..n { let nb = ns[i % ns.len()]; a.apply_neighbor(nb); a.apply_neighbor(nb); }
        println!("\n=== apply_neighbor: {n} iters, {:?}/iter ===", t.elapsed() / n as u32);
    }

    #[test]
    fn profile_apply_and_update() {
        let (mut a, l) = make();
        a.use_layout(&l, &[]);
        let ns = a.neighbors();
        let n = 1_000;

        // Profile just the swap (no neighbor recompute)
        let t = Instant::now();
        for i in 0..n { let nb = ns[i % ns.len()]; a.apply_neighbor(nb); a.apply_neighbor(nb); }
        println!("\n=== apply_neighbor (fast): {n} iters, {:?}/iter ===", t.elapsed() / n as u32);

        // Profile full apply_and_update (swap + weighted_score update + neighbor recompute)
        let t = Instant::now();
        for i in 0..n { let nb = ns[i % ns.len()]; a.apply_neighbor(nb); a.apply_neighbor(nb); }
        println!("=== apply_and_update: {n} iters, {:?}/iter ===", t.elapsed() / n as u32);
    }

    #[test]
    fn profile_init() {
        let (_, l) = make();
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let mut c = CachedLayout::new(&l, data, &weights, &ScaleFactors::default());
        let n = 3u32;
        let t = Instant::now();
        for _ in 0..n { c.update(); }
        println!("\n=== update: {n} iters, {:?}/iter ===", t.elapsed() / n);
    }

    #[test]
    fn verify_score_consistency() {
        let (mut a, l) = make();
        a.use_layout(&l, &[]);
        let score = a.score();
        // compute_score now delegates to score(), so they should always match
        let computed = a.compute_score();
        println!("\n=== score()={score}, compute_score()={computed} ===");
        assert_eq!(score, computed, "compute_score must match score()");
    }

    #[test]
    fn profile_magic_greedy_breakdown() {
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/magic-one.dof").expect("layout");
        let scale_factors = weights.compute_scale_factors(&data);

        let random = layout.random_with_pins(&[]);
        let mut cache = CachedLayout::new(&random, data, &weights, &scale_factors);

        let neighbors = cache.neighbors();
        let swaps: Vec<_> = neighbors.iter().filter(|n| matches!(n, Neighbor::KeySwap(_))).copied().collect();
        let magics: Vec<_> = neighbors.iter().filter(|n| matches!(n, Neighbor::MagicRule(_))).copied().collect();
        println!("\n=== {} swap neighbors, {} magic neighbors ===", swaps.len(), magics.len());

        // Profile score_neighbor for swaps
        let n = 100;
        let t = Instant::now();
        for _ in 0..n { for &s in &swaps { std::hint::black_box(cache.score_neighbor(s)); } }
        let swap_ns = t.elapsed().as_nanos() / (n * swaps.len()) as u128;
        println!("score_neighbor(KeySwap): {}ns/call ({} calls)", swap_ns, swaps.len());

        // Profile score_neighbor for magic rules
        let t = Instant::now();
        for _ in 0..n { for &m in &magics { std::hint::black_box(cache.score_neighbor(m)); } }
        let magic_ns = t.elapsed().as_nanos() / (n * magics.len()) as u128;
        println!("score_neighbor(MagicRule): {}ns/call ({} calls)", magic_ns, magics.len());

        // Profile update() (called once per apply_neighbor)
        let t = Instant::now();
        for _ in 0..1000 { cache.update(); }
        let update_ns = t.elapsed().as_nanos() / 1000;
        println!("update(): {}ns/call", update_ns);

        // Estimated iteration cost
        let swap_total = swap_ns * swaps.len() as u128;
        let magic_total = magic_ns * magics.len() as u128;
        let iter_total = swap_total + magic_total + update_ns;
        println!("Estimated iteration: {}µs (swaps: {}µs, magic: {}µs, update: {}µs)",
            iter_total / 1000, swap_total / 1000, magic_total / 1000, update_ns / 1000);
    }
}
