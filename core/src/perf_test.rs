#[cfg(test)]
mod tests {
    use crate::cached_layout::CachedLayout;
    use crate::prelude::*;
    use crate::weights::dummy_weights;
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
        for i in 0..n { let nb = ns[i % ns.len()]; a.apply_neighbor_and_update(nb); a.apply_neighbor_and_update(nb); }
        println!("=== apply_and_update: {n} iters, {:?}/iter ===", t.elapsed() / n as u32);
    }

    #[test]
    fn profile_init() {
        let (_, l) = make();
        let data = crate::data::Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let mut c = CachedLayout::new(&l, data, &weights);
        let n = 3u32;
        let t = Instant::now();
        for _ in 0..n { c.update_scores(); }
        println!("\n=== update_scores: {n} iters, {:?}/iter ===", t.elapsed() / n);
    }
}
