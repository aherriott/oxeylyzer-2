#[cfg(test)]
mod tests {
    use crate::branch_bound::BranchBound;
    use crate::cached_layout::EMPTY_KEY;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;

    #[test]
    fn diagnose_pruning_with_known_good_bound() {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/nrts-oxey.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let mut cache = bb.create_empty_cache();

        let num_pos = bb.num_positions();
        let all_keys: Vec<usize> = bb.chars_by_frequency().iter()
            .map(|&c| cache.char_mapping().get_u(c))
            .collect();

        // nrts-oxey actual score
        let known_good_bound: i64 = -143_714_244_130;

        println!("\n=== When does greedy placement exceed the known-good bound? ===");
        println!("Known-good bound (nrts-oxey score): {}", known_good_bound);
        println!("Positions: {}", num_pos);

        for depth in 0..num_pos {
            let score = cache.score();
            let prune = score < known_good_bound;

            println!("  depth {:2}: score={:>15} {}", depth, score,
                if prune { "<-- PRUNE" } else { "" });

            if prune {
                println!("\n  First prune at depth {depth}. This means B&B with a perfect bound");
                println!("  would prune at depth {depth} on this greedy path.");
                break;
            }

            if depth < all_keys.len() {
                cache.replace_key_fast(depth, EMPTY_KEY, all_keys[depth]);
            }
        }

        // Also check: what score does greedy completion give from depth 0?
        let mut cache2 = bb.create_empty_cache();
        let all_avail: Vec<usize> = (0..num_pos).collect();
        let greedy_score = cache2.greedy_completion_score(&all_keys[..num_pos], &all_avail);
        println!("\n  Greedy completion score from empty: {}", greedy_score);
        println!("  Known-good (nrts-oxey): {}", known_good_bound);
        println!("  Gap: greedy is {:.1}x worse", greedy_score as f64 / known_good_bound as f64);
    }
}
