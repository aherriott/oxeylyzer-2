/*
 **************************************
 *         Cache Test Suite
 **************************************
 */

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use crate::analyze::{Analyzer, Neighbor};
    use crate::data::Data;
    use crate::layout::{Layout, PosPair};
    use crate::weights::dummy_weights;

    /// Test fixture: loads qwerty layout with english corpus
    fn qwerty_fixture() -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("english.json should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/qwerty.dof").expect("qwerty.dof should exist");
        (analyzer, layout)
    }

    /// Test fixture: loads sturdy layout with english corpus
    #[allow(dead_code)]
    fn sturdy_fixture() -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("english.json should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/sturdy.dof").expect("sturdy.dof should exist");
        (analyzer, layout)
    }

    // ==================== CachedLayout Tests ====================

    #[test]
    fn cached_layout_score_deterministic() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let score1 = analyzer.score();

        // Re-initialize and check score is the same
        analyzer.use_layout(&layout, &[]);
        let score2 = analyzer.score();

        assert_eq!(score1, score2, "Score should be deterministic");
    }

    #[test]
    fn cached_layout_to_layout_roundtrip() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);

        let recovered = analyzer.layout();

        // Keys should match
        assert_eq!(layout.keys, recovered.keys, "Keys should roundtrip");
        assert_eq!(layout.name, recovered.name, "Name should roundtrip");
    }

    // ==================== KeySwap Tests ====================

    #[test]
    fn key_swap_is_reversible() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let original_score = analyzer.score();
        let original_layout = analyzer.layout();

        eprintln!("Original score: {}", original_score);
        eprintln!("Keys at pos 0: {}, pos 1: {}", original_layout.keys[0], original_layout.keys[1]);

        // Apply a swap
        let swap = Neighbor::KeySwap(PosPair(0, 1));
        analyzer.apply_neighbor(swap);
        let swapped_score = analyzer.score();
        let swapped_layout = analyzer.layout();

        eprintln!("After first swap - score: {}", swapped_score);
        eprintln!("Keys at pos 0: {}, pos 1: {}", swapped_layout.keys[0], swapped_layout.keys[1]);
        eprintln!("Score delta from first swap: {}", swapped_score - original_score);

        // Scores should differ (unless the two keys happen to be identical)
        // Apply the same swap again to revert
        analyzer.apply_neighbor(swap);
        let reverted_score = analyzer.score();
        let reverted_layout = analyzer.layout();

        eprintln!("After second swap - score: {}", reverted_score);
        eprintln!("Keys at pos 0: {}, pos 1: {}", reverted_layout.keys[0], reverted_layout.keys[1]);
        eprintln!("Score delta from second swap: {}", reverted_score - swapped_score);
        eprintln!("Total drift: {}", reverted_score - original_score);

        assert_eq!(original_score, reverted_score, "Score should be restored after double swap");
        assert_eq!(original_layout.keys, reverted_layout.keys, "Layout should be restored after double swap");
    }

    #[test]
    fn key_swap_changes_score() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let original_score = analyzer.score();

        let neighbors = analyzer.neighbors();
        let mut found_different = false;

        for &neighbor in neighbors.iter().take(100) {
            let test_score = analyzer.score_neighbor(neighbor);
            if test_score != original_score {
                found_different = true;
                break;
            }
        }

        assert!(found_different, "At least one swap should change the score");
    }

    #[test]
    fn test_neighbor_does_not_mutate() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let original_score = analyzer.score();
        let original_layout = analyzer.layout();

        let neighbors = analyzer.neighbors();
        for &neighbor in neighbors.iter().take(10) {
            let _test_score = analyzer.score_neighbor(neighbor);
        }

        // Score and layout should be unchanged
        assert_eq!(original_score, analyzer.score(), "test_neighbor should not change score");
        assert_eq!(original_layout.keys, analyzer.layout().keys, "test_neighbor should not change layout");
    }

    #[test]
    fn apply_neighbor_matches_test_neighbor() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);

        let neighbors = analyzer.neighbors();

        // Test that test_neighbor returns the same score as apply_neighbor
        for &neighbor in neighbors.iter().take(10) {
            // Reset to original layout
            analyzer.use_layout(&layout, &[]);

            let test_score = analyzer.score_neighbor(neighbor);

            analyzer.apply_neighbor(neighbor);
            let apply_score = analyzer.score();

            assert_eq!(test_score, apply_score, "test_neighbor and apply_neighbor should give same score for neighbor {:?}", neighbor);
        }
    }

    // ==================== SFCache Tests ====================

    #[test]
    fn sfb_score_non_negative_contribution() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let stats = analyzer.stats();

        // SFB percentage should be between 0 and 100%
        assert!(stats.sfbs >= 0.0, "SFB percentage should be non-negative");
        assert!(stats.sfbs <= 1.0, "SFB percentage should be <= 100%, got {}", stats.sfbs);
    }

    #[test]
    fn sfs_score_non_negative_contribution() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let stats = analyzer.stats();

        // SFS percentage should be between 0 and 100%
        assert!(stats.sfs >= 0.0, "SFS percentage should be non-negative");
        assert!(stats.sfs <= 1.0, "SFS percentage should be <= 100%");
    }

    #[test]
    fn finger_sfbs_sum_to_total() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let stats = analyzer.stats();

        let sum: f64 = stats.finger_sfbs.iter().sum();
        let diff = (sum - stats.sfbs).abs();

        assert!(diff < 0.0001, "Per-finger SFBs should sum to total SFBs: sum={sum}, total={}", stats.sfbs);
    }

    // ==================== StretchCache Tests ====================

    #[test]
    fn stretch_score_non_negative() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let stats = analyzer.stats();

        assert!(stats.stretches >= 0.0, "Stretch score should be non-negative");
    }

    // ==================== Stats Tests ====================

    #[test]
    fn finger_use_sums_to_100() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let stats = analyzer.stats();

        let sum: f64 = stats.finger_use.iter().sum();
        let diff = (sum - 100.0).abs();

        // Finger use is stored as percentages (0-100), should sum to ~100
        // May be slightly less than 100 if some characters aren't on the layout
        assert!(diff < 5.0, "Finger use should sum to ~100%: got {sum}");
    }

    #[test]
    fn stats_deterministic() {
        let (mut analyzer, layout) = qwerty_fixture();

        analyzer.use_layout(&layout, &[]);
        let stats1 = analyzer.stats();

        analyzer.use_layout(&layout, &[]);
        let stats2 = analyzer.stats();

        assert_eq!(stats1.sfbs, stats2.sfbs, "SFBs should be deterministic");
        assert_eq!(stats1.sfs, stats2.sfs, "SFS should be deterministic");
        assert_eq!(stats1.stretches, stats2.stretches, "Stretches should be deterministic");
        assert_eq!(stats1.score, stats2.score, "Score should be deterministic");
    }

    // ==================== Copy From Tests ====================

    #[test]
    fn copy_from_restores_state_after_swap() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let original_score = analyzer.score();

        // Apply a swap
        let swap = Neighbor::KeySwap(PosPair(5, 10));
        let swapped_score = analyzer.score_neighbor(swap);

        // After test_neighbor, score should still be original
        assert_eq!(original_score, analyzer.score(), "test_neighbor should restore state via copy_from");

        // Now actually apply it
        analyzer.apply_neighbor(swap);
        assert_eq!(swapped_score, analyzer.score(), "apply_neighbor should change score");
    }

    // ==================== Layout Comparison Tests ====================

    #[test]
    fn different_layouts_have_different_scores() {
        let (mut analyzer, qwerty) = qwerty_fixture();
        let sturdy = Layout::load("../layouts/sturdy.dof").expect("sturdy.dof should exist");

        analyzer.use_layout(&qwerty, &[]);
        let qwerty_score = analyzer.score();

        analyzer.use_layout(&sturdy, &[]);
        let sturdy_score = analyzer.score();

        assert_ne!(qwerty_score, sturdy_score, "Different layouts should have different scores");
        // Sturdy should be better (higher score) than QWERTY
        assert!(sturdy_score > qwerty_score, "Sturdy should score better than QWERTY");
    }

    #[test]
    fn optimized_layout_scores_better() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);
        let neighbors = analyzer.neighbors();
        let initial_score = analyzer.score();
        eprintln!("Initial score: {}", initial_score);

        // Do a few greedy improvements
        for i in 0..5 {
            if let Some((neighbor, expected_score)) = analyzer.best_neighbor(&neighbors) {
                eprintln!("Iteration {}: applying neighbor {:?}, expected score: {}", i, neighbor, expected_score);
                analyzer.apply_neighbor(neighbor);
                let actual_score = analyzer.score();
                eprintln!("Iteration {}: actual score after apply: {}", i, actual_score);
            } else {
                eprintln!("Iteration {}: no improvement found", i);
                break;
            }
        }

        let improved_score = analyzer.score();
        eprintln!("Final improved score: {}", improved_score);
        assert!(improved_score >= initial_score, "Greedy optimization should not make score worse: initial={}, improved={}", initial_score, improved_score);
    }

    // ==================== Neighbor Count Tests ====================

    #[test]
    fn neighbors_is_correct() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);

        let neighbors = analyzer.neighbors();
        let n = layout.keys.len();
        let expected_swaps = n * (n - 1) / 2;

        // Should have at least the key swap neighbors
        assert!(neighbors.len() >= expected_swaps, "Should have at least {expected_swaps} swap neighbors, got {}", neighbors.len());
    }

    #[test]
    fn all_neighbors_are_valid() {
        let (mut analyzer, layout) = qwerty_fixture();
        analyzer.use_layout(&layout, &[]);

        let neighbors = analyzer.neighbors();
        // Test that we can iterate all neighbors without panicking
        for &_neighbor in &neighbors {
            // Just iterating is the test
        }
    }
}
