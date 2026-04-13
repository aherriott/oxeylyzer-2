#[cfg(test)]
mod tests {
    use crate::branch_bound::BranchBound;
    use crate::cached_layout::{CachedLayout, EMPTY_KEY};
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;
    use std::time::Instant;

    fn setup() -> (BranchBound, CachedLayout) {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");
        let mut bb = BranchBound::new(layout, data, weights);
        let cache = bb.create_empty_cache();
        (bb, cache)
    }

    #[test]
    fn bench_replace_key_at_various_depths() {
        let (bb, mut cache) = setup();
        let keys = bb.chars_by_frequency().iter().enumerate().collect::<Vec<_>>();
        let num_positions = bb.num_positions();

        println!("\n=== replace_key cost at various depths ===");

        // Place keys one at a time, measuring replace_key cost at each depth
        for depth in 0..std::cmp::min(15, num_positions) {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            let pos = depth;

            let n = 10_000u32;
            let t = Instant::now();
            for _ in 0..n {
                cache.replace_key_fast(pos, EMPTY_KEY, key_id);
                cache.replace_key_fast(pos, key_id, EMPTY_KEY);
            }
            let per_iter = t.elapsed() / n;
            println!("  depth {depth:2}: {:?}/replace_key_fast (place+remove)", per_iter);

            // Actually place the key for the next depth
            cache.replace_key_fast(pos, EMPTY_KEY, key_id);
        }
    }

    #[test]
    fn bench_score_at_various_depths() {
        let (bb, mut cache) = setup();
        let num_positions = bb.num_positions();

        println!("\n=== score() cost at various depths ===");

        for depth in 0..std::cmp::min(15, num_positions) {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            let pos = depth;

            let n = 100_000u32;
            let t = Instant::now();
            for _ in 0..n {
                std::hint::black_box(cache.score());
            }
            let per_iter = t.elapsed() / n;
            println!("  depth {depth:2} ({} keys placed): {:?}/score()", depth, per_iter);

            cache.replace_key(pos, EMPTY_KEY, key_id);
        }
    }

    #[test]
    fn bench_bb_node_simulation() {
        let (bb, mut cache) = setup();
        let num_positions = bb.num_positions();

        // Place first 5 keys to simulate being at depth 5
        for depth in 0..5 {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key(depth, EMPTY_KEY, key_id);
        }

        let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[5]);
        let score_before = cache.score();

        println!("\n=== B&B node simulation at depth 5 ===");
        println!("  Score with 5 keys: {}", score_before);

        // Simulate trying all remaining positions for the 6th key
        let n = 1_000u32;
        let t = Instant::now();
        for _ in 0..n {
            for pos in 5..num_positions {
                cache.replace_key_fast(pos, EMPTY_KEY, key_id);
                std::hint::black_box(cache.score());
                cache.replace_key_fast(pos, key_id, EMPTY_KEY);
            }
        }
        let elapsed = t.elapsed();
        let positions_tried = (num_positions - 5) as u32;
        let per_position = elapsed / (n * positions_tried);
        let per_full_node = elapsed / n;
        println!("  {} positions × {n} iterations in {:?}", positions_tried, elapsed);
        println!("  {:?}/position (replace + score + unreplace)", per_position);
        println!("  {:?}/full node (try all {} positions)", per_full_node, positions_tried);
    }

    #[test]
    fn bench_bb_depth_limited() {
        let (bb, _) = setup();
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");

        println!("\n=== B&B search to various depths ===");

        for max_depth in [4, 5, 6, 7] {
            let mut bb = BranchBound::new(layout.clone(), data.clone(), weights.clone());
            let t = Instant::now();
            let (results, stats) = bb.search_limited(i64::MIN, 5, max_depth);
            let elapsed = t.elapsed();
            println!("  depth {max_depth}: {elapsed:?}, {} nodes, {} solutions, best={:?}",
                stats.nodes_visited, stats.solutions_found as u64,
                results.first().map(|r| r.score));
        }
    }

    #[test]
    fn test_lower_bound_no_unplaced_keys_returns_zero() {
        let (bb, mut cache) = setup();

        // Place a few keys so the cache has some state
        for depth in 0..3 {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let unplaced: Vec<usize> = vec![];
        let available: Vec<usize> = (3..bb.num_positions()).collect();

        let lb = cache.lower_bound_remaining(&unplaced, &available);
        assert_eq!(lb, 0, "Lower bound with no unplaced keys should be 0");
    }

    #[test]
    fn test_lower_bound_is_non_positive() {
        let (bb, mut cache) = setup();

        // Place first 5 keys
        for depth in 0..5 {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let unplaced: Vec<usize> = (5..bb.num_positions())
            .map(|d| cache.char_mapping().get_u(bb.chars_by_frequency()[d]))
            .collect();
        let available: Vec<usize> = (5..bb.num_positions()).collect();

        let lb = cache.lower_bound_remaining(&unplaced, &available);
        assert!(lb <= 0, "Lower bound should be <= 0 (all penalties are non-positive), got {lb}");
    }

    #[test]
    fn test_lower_bound_is_valid_bound() {
        let (bb, _) = setup();
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let mut cache = bb.create_empty_cache();

        // Place first 5 keys at positions 0..5
        let num_to_place = 5;
        for depth in 0..num_to_place {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let score_after_partial = cache.score();

        // Compute lower bound for remaining keys
        let unplaced: Vec<usize> = (num_to_place..bb.num_positions())
            .map(|d| cache.char_mapping().get_u(bb.chars_by_frequency()[d]))
            .collect();
        let available: Vec<usize> = (num_to_place..bb.num_positions()).collect();

        let lb = cache.lower_bound_remaining(&unplaced, &available);

        // Now actually place all remaining keys and get the real score
        for depth in num_to_place..bb.num_positions() {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let final_score = cache.score();
        let actual_remaining = final_score - score_after_partial;

        println!("Score after {} keys: {}", num_to_place, score_after_partial);
        println!("Final score: {}", final_score);
        println!("Actual remaining cost: {}", actual_remaining);
        println!("Lower bound estimate: {}", lb);

        assert!(
            lb <= actual_remaining,
            "Lower bound ({lb}) should be <= actual remaining cost ({actual_remaining})"
        );
    }

    #[test]
    fn bench_lower_bound_cost() {
        let (bb, mut cache) = setup();

        // Place first 5 keys
        for depth in 0..5 {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let unplaced: Vec<usize> = (5..bb.num_positions())
            .map(|d| cache.char_mapping().get_u(bb.chars_by_frequency()[d]))
            .collect();
        let available: Vec<usize> = (5..bb.num_positions()).collect();

        let n = 1_000u32;
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(cache.lower_bound_remaining(&unplaced, &available));
        }
        let elapsed = t.elapsed();
        let per_call = elapsed / n;
        let lb = cache.lower_bound_remaining(&unplaced, &available);
        println!("\n=== lower_bound_remaining cost ===");
        println!("  {} unplaced keys, {} available positions", unplaced.len(), available.len());
        println!("  {:?}/call ({n} iterations)", per_call);
        println!("  lower bound value: {}", lb);
    }

    #[test]
    fn test_lower_bound_by_lookahead() {
        let (bb, _) = setup();
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let mut cache = bb.create_empty_cache();

        // Place first 5 keys
        for depth in 0..5 {
            let key_id = cache.char_mapping().get_u(bb.chars_by_frequency()[depth]);
            cache.replace_key_fast(depth, EMPTY_KEY, key_id);
        }

        let score_after_5 = cache.score();
        let all_remaining: Vec<usize> = (5..bb.num_positions())
            .map(|d| cache.char_mapping().get_u(bb.chars_by_frequency()[d]))
            .collect();
        let available: Vec<usize> = (5..bb.num_positions()).collect();

        println!("\n=== Lower bound by lookahead depth (5 keys placed) ===");
        println!("  Current score: {}", score_after_5);

        for lookahead in [1, 2, 3, 5, 8, 10, 15, 20, all_remaining.len()] {
            let keys_to_check = &all_remaining[..lookahead.min(all_remaining.len())];
            let lb = cache.lower_bound_remaining(keys_to_check, &available);
            println!("  lookahead {:2}: bound = {:>20}, projected = {:>20}",
                lookahead, lb, score_after_5 + lb);
        }

        // Also compute the actual final score for reference
        let mut cache2 = cache.clone();
        for (i, &key_id) in all_remaining.iter().enumerate() {
            cache2.replace_key_fast(5 + i, EMPTY_KEY, key_id);
        }
        println!("  actual final score: {}", cache2.score());
    }
}
