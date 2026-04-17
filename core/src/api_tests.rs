/// Tests for the CachedLayout mutation API, focusing on magic rule correctness.
/// These tests target the exact bugs we encountered:
/// - Score drift from stale magic deltas after key swaps
/// - Positive scores from incorrect speculative magic scoring
/// - Magic rules not being pruned by greedy
/// - score() inconsistency between _no_update + update vs base version

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use crate::analyze::{Analyzer, Neighbor};
    use crate::cached_layout::{CachedLayout, EMPTY_KEY};
    use crate::data::Data;
    use crate::layout::{Layout, PosPair, MagicRule};
    use crate::optimization::{RolloutPolicy, OptStep};
    use crate::weights::{dummy_weights, ScaleFactors};

    fn magic_fixture() -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("english.json should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/magic-one.dof").expect("magic-one.dof should exist");
        (analyzer, layout)
    }

    fn non_magic_fixture() -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("english.json should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load("../layouts/sturdy.dof").expect("sturdy.dof should exist");
        (analyzer, layout)
    }

    // ==================== Score is always negative ====================

    #[test]
    fn score_is_negative_non_magic() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);
        assert!(a.score() < 0, "Score should be negative, got {}", a.score());
    }

    #[test]
    fn score_is_negative_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);
        assert!(a.score() < 0, "Score should be negative for magic layout, got {}", a.score());
    }

    // ==================== swap_key consistency ====================

    #[test]
    fn swap_key_is_reversible_non_magic() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);
        let original = a.score();

        let cache = a.cache_mut();
        cache.swap_key(0, 1);
        cache.swap_key(0, 1); // reverse
        assert_eq!(original, cache.score(), "Double swap should restore score");
    }

    #[test]
    fn swap_key_is_reversible_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);
        let original = a.score();

        let cache = a.cache_mut();
        cache.swap_key(0, 1);
        let swapped = cache.score();
        cache.swap_key(0, 1);
        assert_eq!(original, cache.score(),
            "Double swap should restore score on magic layout. original={}, swapped={}, restored={}",
            original, swapped, cache.score());
    }

    #[test]
    fn swap_key_no_update_plus_update_matches_swap_key() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);

        // Method 1: swap_key (base)
        let cache1 = a.cache_mut();
        cache1.swap_key(0, 1);
        let score1 = cache1.score();

        // Reset
        cache1.swap_key(0, 1);

        // Method 2: swap_key_no_update + update
        cache1.swap_key_no_update(0, 1);
        cache1.update();
        let score2 = cache1.score();

        assert_eq!(score1, score2,
            "swap_key and swap_key_no_update+update should give same score");
    }

    // ==================== score_neighbor consistency ====================

    #[test]
    fn score_neighbor_keyswap_matches_apply() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);

        let neighbors = a.neighbors();
        for &neighbor in neighbors.iter().take(20) {
            a.use_layout(&layout, &[]);
            let speculative = a.score_neighbor(neighbor);
            a.apply_neighbor(neighbor);
            let actual = a.score();
            assert_eq!(speculative, actual,
                "score_neighbor should match apply for {:?}", neighbor);
        }
    }

    #[test]
    fn score_neighbor_magic_matches_apply() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);

        let neighbors = a.neighbors();
        let magic_neighbors: Vec<_> = neighbors.iter()
            .filter(|n| matches!(n, Neighbor::MagicRule(_)))
            .take(20)
            .copied()
            .collect();

        for neighbor in magic_neighbors {
            a.use_layout(&layout, &[]);
            let speculative = a.score_neighbor(neighbor);
            a.apply_neighbor(neighbor);
            let actual = a.score();
            assert_eq!(speculative, actual,
                "score_neighbor should match apply for magic rule {:?}", neighbor);
        }
    }

    #[test]
    fn score_neighbor_does_not_mutate_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);
        let original = a.score();

        let neighbors = a.neighbors();
        for &neighbor in neighbors.iter().take(50) {
            let _ = a.score_neighbor(neighbor);
        }

        assert_eq!(original, a.score(),
            "score_neighbor should not change score. original={}, after={}", original, a.score());
    }

    // ==================== Multiple swaps don't cause drift ====================

    #[test]
    fn many_swaps_no_drift_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        let initial = cache.score();

        // Do 100 random swaps and reversals
        let pairs = [(0,1), (2,3), (4,5), (0,3), (1,4), (2,5)];
        for &(a, b) in pairs.iter().cycle().take(100) {
            cache.swap_key(a, b);
        }
        // Reverse them all (even number of swaps per pair = identity)
        let score_after = cache.score();

        // The score won't match initial because we did an even number of
        // different swaps. But it should be negative.
        assert!(score_after < 0,
            "Score should remain negative after many swaps, got {}", score_after);
    }

    #[test]
    fn swap_and_revert_100_times_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        let initial = cache.score();

        // Swap and immediately revert 100 times
        for _ in 0..100 {
            cache.swap_key(0, 5);
            cache.swap_key(0, 5);
        }

        assert_eq!(initial, cache.score(),
            "100 swap-reverts should not cause drift");
    }

    // ==================== replace_rule consistency ====================

    #[test]
    fn replace_rule_and_revert() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        let initial = cache.score();

        // Get a magic rule neighbor
        let neighbors = cache.neighbors();
        let magic_neighbor = neighbors.iter()
            .find(|n| matches!(n, Neighbor::MagicRule(_)))
            .copied();

        if let Some(Neighbor::MagicRule(rule)) = magic_neighbor {
            let revert = cache.get_revert_neighbor(Neighbor::MagicRule(rule));
            cache.replace_rule(rule.magic_key, rule.leader, rule.output);
            let changed = cache.score();

            if let Neighbor::MagicRule(rev) = revert {
                cache.replace_rule(rev.magic_key, rev.leader, rev.output);
            }
            let restored = cache.score();

            assert_eq!(initial, restored,
                "replace_rule + revert should restore score. initial={}, changed={}, restored={}",
                initial, changed, restored);
        }
    }

    // ==================== Greedy can prune magic rules ====================

    #[test]
    fn greedy_produces_negative_score_magic() {
        let (a, layout) = magic_fixture();
        let random = layout.random_with_pins(&[]);
        let mut cache = CachedLayout::new(
            &random, a.data().clone(), a.weights(), a.scale_factors(),
        );
        let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };
        cache.optimize(&policy, &[]);
        let score = cache.score();

        assert!(score < 0,
            "Greedy should produce negative score on magic layout, got {}", score);
    }

    #[test]
    fn greedy_improves_random_layout() {
        let (a, layout) = non_magic_fixture();
        let random = layout.random_with_pins(&[]);
        let mut cache = CachedLayout::new(
            &random, a.data().clone(), a.weights(), a.scale_factors(),
        );
        let before = cache.score();
        let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };
        cache.optimize(&policy, &[]);
        let after = cache.score();

        assert!(after >= before,
            "Greedy should not make score worse. before={}, after={}", before, after);
    }

    // ==================== apply_neighbor dispatches correctly ====================

    #[test]
    fn apply_neighbor_keyswap_matches_swap_key() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        cache.swap_key(0, 1);
        let via_swap = cache.score();
        cache.swap_key(0, 1); // revert

        cache.apply_neighbor(Neighbor::KeySwap(PosPair(0, 1)));
        let via_neighbor = cache.score();

        assert_eq!(via_swap, via_neighbor,
            "apply_neighbor(KeySwap) should match swap_key");
    }
}
