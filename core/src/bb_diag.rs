#[cfg(test)]
mod tests {
    use crate::branch_bound::BranchBound;
    use crate::cached_layout::EMPTY_KEY;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;

    #[test]
    fn diagnose_pruning() {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let mut cache = bb.create_empty_cache();

        let num_pos = bb.num_positions();
        let keys_by_freq = bb.chars_by_frequency();
        let all_keys: Vec<usize> = keys_by_freq.iter()
            .map(|&c| cache.char_mapping().get_u(c))
            .collect();
        let mut available: Vec<usize> = (0..num_pos).collect();

        println!("\n=== Lower bound breakdown at each depth ===");

        for depth in 0..std::cmp::min(8, num_pos) {
            let remaining = &all_keys[depth..];
            let lb = cache.lower_bound_remaining(remaining, &available);
            let score = cache.score();

            println!("depth {:2}: score={:>15}, lb={:>20}, projected={:>20}, remaining_keys={}",
                depth, score, lb, score + lb, remaining.len());

            if depth < all_keys.len() {
                cache.replace_key_fast(depth, EMPTY_KEY, all_keys[depth]);
                available.retain(|&p| p != depth);
            }
        }

        // Full score
        let mut full = bb.create_empty_cache();
        for (i, &k) in all_keys.iter().enumerate().take(num_pos) {
            full.replace_key_fast(i, EMPTY_KEY, k);
        }
        println!("full score: {}", full.score());
    }
}
