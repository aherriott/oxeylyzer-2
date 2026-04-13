#[cfg(test)]
mod tests {
    use crate::branch_bound::BranchBound;
    use crate::cached_layout::EMPTY_KEY;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;

    #[test]
    fn diagnose_greedy_remaining_table() {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/nrts-oxey.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let cache = bb.create_empty_cache();

        let num_pos = bb.num_positions();
        let all_keys: Vec<usize> = bb.chars_by_frequency().iter()
            .map(|&c| cache.char_mapping().get_u(c))
            .collect();

        let bound: i64 = -143_714_244_130;

        // Build greedy remaining cost table:
        // greedy_score[d] = score after greedily placing first d keys
        // remaining_cost[d] = greedy_score[num_pos] - greedy_score[d]
        let all_avail: Vec<usize> = (0..num_pos).collect();
        let mut greedy_scores: Vec<i64> = Vec::new();

        let mut greedy_cache = cache.clone();
        greedy_scores.push(greedy_cache.score()); // depth 0

        for depth in 0..num_pos {
            let key = all_keys[depth];
            // Find best position for this key
            let nk = greedy_cache.trigram_num_keys();
            if key >= nk { continue; }

            let mut best_pos = depth; // fallback
            let mut best_score = i64::MIN;
            for pos in 0..num_pos {
                if greedy_cache.get_key_at(pos) != EMPTY_KEY { continue; }
                greedy_cache.replace_key_fast(pos, EMPTY_KEY, key);
                let s = greedy_cache.score();
                greedy_cache.replace_key_fast(pos, key, EMPTY_KEY);
                if s > best_score {
                    best_score = s;
                    best_pos = pos;
                }
            }
            greedy_cache.replace_key_fast(best_pos, EMPTY_KEY, key);
            greedy_scores.push(greedy_cache.score());
        }

        let full_greedy = *greedy_scores.last().unwrap();

        println!("\n=== Greedy remaining cost table ===");
        println!("Full greedy score: {}", full_greedy);
        println!("Bound (nrts-oxey): {}", bound);

        // Now simulate B&B with this table
        // At each depth, remaining_cost = full_greedy - greedy_score[depth]
        // This is how much MORE penalty the greedy placement adds from this depth.
        // It's an upper bound on the optimal remaining cost (greedy is suboptimal).
        // So: projected = current_score + remaining_cost is an UPPER bound.
        // For pruning we need: if projected < bound, prune.
        // But projected is an upper bound, so if even the upper bound is < bound, prune.
        // Wait - that's wrong. If the upper bound is < bound, it means even the BEST
        // completion from this point is worse than bound. That IS valid for pruning!
        //
        // Actually no. remaining_cost from greedy is the greedy's remaining penalty.
        // The optimal remaining penalty is LESS negative (better). So:
        //   optimal_remaining >= greedy_remaining (less negative)
        //   current + optimal_remaining >= current + greedy_remaining
        // If current + greedy_remaining >= bound, we can't prune.
        // If current + greedy_remaining < bound, the greedy completion is worse than bound,
        // but the optimal might still be better. So we CAN'T prune based on this.
        //
        // The greedy remaining cost is an UPPER bound on remaining penalty (less negative).
        // For pruning we need a LOWER bound (more negative). Greedy gives the wrong direction.

        // BUT: we can use the greedy table differently.
        // The greedy remaining cost tells us: "from this depth, the greedy adds X penalty."
        // The ACTUAL remaining cost is somewhere between X (greedy) and the optimal.
        // We can't use this for pruning directly.
        //
        // However, we CAN use it to estimate: if the current path is WORSE than the greedy
        // path at this depth, it's unlikely to lead to a good solution.
        // greedy_score[depth] is the score of the greedy partial layout at depth d.
        // If current_score < greedy_score[depth] - margin, prune.
        // This is heuristic, not exact.

        let mut test_cache = cache.clone();
        for depth in 0..12 {
            let score = test_cache.score();
            let greedy_at_d = greedy_scores[depth];
            let greedy_remaining = full_greedy - greedy_at_d;

            // Heuristic: if current score + greedy remaining < bound, prune
            // This is valid IF greedy remaining is a lower bound on actual remaining.
            // Greedy remaining is an UPPER bound (less negative), so this is NOT valid.
            // But let's see the numbers anyway.
            let projected = score + greedy_remaining;

            println!("  depth {:2}: score={:>15}, greedy_at_d={:>15}, greedy_remaining={:>15}, projected={:>15} {}",
                depth, score, greedy_at_d, greedy_remaining, projected,
                if projected < bound { "<-- would prune (heuristic)" } else { "" });

            if depth < all_keys.len() {
                test_cache.replace_key_fast(depth, EMPTY_KEY, all_keys[depth]);
            }
        }
    }
}
