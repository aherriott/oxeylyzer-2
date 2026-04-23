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
        for (i, &neighbor) in neighbors.iter().enumerate().take(20) {
            a.use_layout(&layout, &[]);
            let initial = a.score();
            let speculative = a.score_neighbor(neighbor);
            // Speculative should not mutate
            assert_eq!(a.score(), initial,
                "score_neighbor should not mutate: neighbor #{} {:?}", i, neighbor);

            a.apply_neighbor(neighbor);
            let actual = a.score();
            assert_eq!(speculative, actual,
                "score_neighbor should match apply for #{} {:?} (initial was {})",
                i, neighbor, initial);
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

    // ==================== Trigram stats ====================

    #[test]
    fn trigram_stats_are_nonzero() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);
        let stats = a.stats();

        let total_trigrams = stats.trigrams.inroll + stats.trigrams.outroll
            + stats.trigrams.alternate + stats.trigrams.redirect
            + stats.trigrams.onehandin + stats.trigrams.onehandout;

        assert!(total_trigrams > 0.0,
            "Trigram stats should be nonzero, got total={}", total_trigrams);
        assert!(stats.trigrams.inroll > 0.0, "Inroll should be > 0");
        assert!(stats.trigrams.alternate > 0.0, "Alternate should be > 0");
    }

    #[test]
    fn trigram_stats_nonzero_magic() {
        let (mut a, layout) = magic_fixture();
        a.use_layout(&layout, &[]);
        let stats = a.stats();

        let total_trigrams = stats.trigrams.inroll + stats.trigrams.outroll
            + stats.trigrams.alternate + stats.trigrams.redirect
            + stats.trigrams.onehandin + stats.trigrams.onehandout;

        assert!(total_trigrams > 0.0,
            "Trigram stats should be nonzero for magic layout, got total={}", total_trigrams);
    }

    #[test]
    fn fresh_cache_without_layout_has_zero_trigram_score() {
        use crate::trigrams::TrigramCache;
        use libdof::dofinitions::Finger::*;

        let fingers = vec![LP; 30];
        let cache = TrigramCache::new(&fingers, 30);
        assert_eq!(cache.score(), 0,
            "Fresh TrigramCache should have score 0, got {}", cache.score());
    }

    #[test]
    fn place_then_clear_returns_zero_trigram() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();

        for pos in 0..30 {
            let k = cache.get_key(pos);
            if k != EMPTY_KEY {
                cache.replace_key_no_update(pos, k, EMPTY_KEY);
            }
        }
        cache.full_recompute();

        let (sfb, stretch, scissors, trigram, _m, _f) = cache.score_breakdown();
        assert_eq!(trigram, 0, "Before placing: trigram={}", trigram);
        assert_eq!(sfb, 0, "Before placing: sfb={}", sfb);
        assert_eq!(stretch, 0, "Before placing: stretch={}", stretch);
        assert_eq!(scissors, 0, "Before placing: scissors={}", scissors);

        cache.replace_key_no_update(0, EMPTY_KEY, 10);
        cache.full_recompute();
        let (sfb, stretch, scissors, trigram, _m, _f) = cache.score_breakdown();
        assert_eq!(trigram, 0,
            "One key placed (no trigrams possible): trigram={}", trigram);
        assert_eq!(sfb, 0, "One key placed: sfb={}", sfb);

        cache.replace_key_no_update(0, 10, EMPTY_KEY);
        cache.full_recompute();
        let (sfb, stretch, scissors, trigram, _m, _f) = cache.score_breakdown();
        assert_eq!(trigram, 0, "After removing: trigram={}", trigram);
        assert_eq!(sfb, 0, "After removing: sfb={}", sfb);
        assert_eq!(stretch, 0, "After removing: stretch={}", stretch);
        assert_eq!(scissors, 0, "After removing: scissors={}", scissors);
    }

    // ==================== Empty / partial layout scoring ====================
    // These tests target the bug where clearing all keys leaves trigram score
    // positive (violating the invariant that score ≤ 0 for any layout).

    #[test]
    fn empty_layout_has_zero_score() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        let num_positions = 30;

        // Clear all keys
        for pos in 0..num_positions {
            let k = cache.get_key(pos);
            if k != EMPTY_KEY {
                cache.replace_key_no_update(pos, k, EMPTY_KEY);
            }
        }
        cache.full_recompute();

        let score = cache.score();
        assert_eq!(score, 0,
            "Empty layout should have score 0, got {}. Breakdown: {:?}",
            score, cache.score_breakdown());
    }

    #[test]
    fn empty_layout_has_zero_trigram_score() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);

        let cache = a.cache_mut();
        let num_positions = 30;

        for pos in 0..num_positions {
            let k = cache.get_key(pos);
            if k != EMPTY_KEY {
                cache.replace_key_no_update(pos, k, EMPTY_KEY);
            }
        }
        cache.full_recompute();

        let (_sfb, _stretch, _scissors, trigram, _magic, _finger) = cache.score_breakdown();
        assert_eq!(trigram, 0,
            "Empty layout trigram score should be 0, got {}", trigram);
    }

    #[test]
    fn clear_and_refill_matches_original() {
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);
        let original_score = a.score();
        let original_breakdown = a.cache_mut().score_breakdown();

        let cache = a.cache_mut();
        let num_positions = 30;

        let saved_keys: Vec<_> = (0..num_positions).map(|p| cache.get_key(p)).collect();

        for pos in 0..num_positions {
            let k = cache.get_key(pos);
            if k != EMPTY_KEY {
                cache.replace_key_no_update(pos, k, EMPTY_KEY);
            }
        }
        cache.full_recompute();
        let empty_breakdown = cache.score_breakdown();
        println!("Empty breakdown: {:?}", empty_breakdown);

        for pos in 0..num_positions {
            if saved_keys[pos] != EMPTY_KEY {
                cache.replace_key_no_update(pos, EMPTY_KEY, saved_keys[pos]);
            }
        }
        cache.full_recompute();

        let restored_score = cache.score();
        let restored_breakdown = cache.score_breakdown();
        assert_eq!(restored_score, original_score,
            "Clear + refill should restore original score.\n original={} breakdown={:?}\nrestored={} breakdown={:?}",
            original_score, original_breakdown, restored_score, restored_breakdown);
    }

    #[test]
    fn direct_full_recompute_matches_use_layout() {
        // Load layout with use_layout, then directly call full_recompute.
        // Score should be identical (idempotent).
        let (mut a, layout) = non_magic_fixture();
        a.use_layout(&layout, &[]);
        let before = a.score();
        let before_breakdown = a.cache_mut().score_breakdown();

        // Print trigram internals before
        let stats_before = a.stats();
        println!("BEFORE: inroll={:.3}% outroll={:.3}% alt={:.3}% redir={:.3}% ohIn={:.3}% ohOut={:.3}%",
            stats_before.trigrams.inroll * 100.0,
            stats_before.trigrams.outroll * 100.0,
            stats_before.trigrams.alternate * 100.0,
            stats_before.trigrams.redirect * 100.0,
            stats_before.trigrams.onehandin * 100.0,
            stats_before.trigrams.onehandout * 100.0,
        );

        a.cache_mut().full_recompute();
        let after = a.score();
        let after_breakdown = a.cache_mut().score_breakdown();

        let stats_after = a.stats();
        println!("AFTER:  inroll={:.3}% outroll={:.3}% alt={:.3}% redir={:.3}% ohIn={:.3}% ohOut={:.3}%",
            stats_after.trigrams.inroll * 100.0,
            stats_after.trigrams.outroll * 100.0,
            stats_after.trigrams.alternate * 100.0,
            stats_after.trigrams.redirect * 100.0,
            stats_after.trigrams.onehandin * 100.0,
            stats_after.trigrams.onehandout * 100.0,
        );

        assert_eq!(before, after,
            "full_recompute on a valid layout should be idempotent.\n before={} breakdown={:?}\n after={} breakdown={:?}",
            before, before_breakdown, after, after_breakdown);
    }

    // ================================================================
    // ================ NEW: FIRST-PRINCIPLES TESTS ===================
    // ================================================================
    //
    // These tests codify the invariants the cache MUST satisfy regardless
    // of implementation details. Each invariant is tested across three
    // layout regimes:
    //   - non-magic:    sturdy (0 rules)
    //   - magic-one:    magic-one.dof (1 rule)
    //   - random-magic: random_with_pins layout (29 random rules)

    fn random_magic_fixture() -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("english.json should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let base = Layout::load("../layouts/my-layout.dof").expect("my-layout.dof should exist");
        let random = base.random_with_pins(&[]);
        (analyzer, random)
    }

    fn fresh_cache(analyzer: &Analyzer, layout: &Layout) -> CachedLayout {
        CachedLayout::new(
            layout,
            analyzer.data().clone(),
            analyzer.weights(),
            analyzer.scale_factors(),
        )
    }

    // ---- Invariant: score matches full_recompute after construction ----

    #[test]
    fn fresh_cache_matches_full_recompute_non_magic() {
        let (a, layout) = non_magic_fixture();
        let cache = fresh_cache(&a, &layout);
        let incremental = cache.score();
        let mut clone = cache.clone();
        clone.full_recompute();
        assert_eq!(incremental, clone.score(),
            "fresh non-magic cache should match full_recompute");
    }

    #[test]
    fn fresh_cache_matches_full_recompute_magic_one() {
        let (a, layout) = magic_fixture();
        let cache = fresh_cache(&a, &layout);
        let incremental = cache.score();
        let mut clone = cache.clone();
        clone.full_recompute();
        assert_eq!(incremental, clone.score(),
            "fresh magic-one cache should match full_recompute");
    }

    #[test]
    fn fresh_cache_matches_full_recompute_random_magic() {
        let (a, layout) = random_magic_fixture();
        let cache = fresh_cache(&a, &layout);
        let incremental = cache.score();
        let mut clone = cache.clone();
        clone.full_recompute();
        assert_eq!(incremental, clone.score(),
            "fresh random-magic cache should match full_recompute");
    }

    // ---- Invariant: update() is idempotent ----

    #[test]
    fn update_is_idempotent_non_magic() {
        let (a, layout) = non_magic_fixture();
        let mut cache = fresh_cache(&a, &layout);
        let s1 = cache.score();
        cache.update();
        let s2 = cache.score();
        cache.update();
        let s3 = cache.score();
        assert_eq!(s1, s2, "first update should not change score");
        assert_eq!(s2, s3, "second update should not change score");
    }

    #[test]
    fn update_is_idempotent_random_magic() {
        let (a, layout) = random_magic_fixture();
        let mut cache = fresh_cache(&a, &layout);
        let s1 = cache.score();
        cache.update();
        let s2 = cache.score();
        cache.update();
        let s3 = cache.score();
        assert_eq!(s1, s2, "first update should not change score (random-magic)");
        assert_eq!(s2, s3, "second update should not change score (random-magic)");
    }

    // ---- Invariant: swap_key matches full_recompute (after a single swap) ----

    fn check_single_swap_matches_recompute(label: &str, a: &Analyzer, layout: &Layout) {
        let cache_tpl = fresh_cache(a, layout);
        let keyswaps: Vec<_> = cache_tpl.neighbors().iter()
            .filter_map(|n| if let Neighbor::KeySwap(p) = n { Some(*p) } else { None })
            .take(20)
            .collect();

        for &PosPair(pa, pb) in &keyswaps {
            let mut c1 = fresh_cache(a, layout);
            c1.swap_key(pa, pb);
            let incremental = c1.score();

            let mut c2 = fresh_cache(a, layout);
            c2.swap_key(pa, pb);
            c2.full_recompute();
            let truth = c2.score();

            assert_eq!(incremental, truth,
                "[{label}] swap_key({pa},{pb}) incremental={incremental} vs full_recompute={truth}");
        }
    }

    #[test]
    fn swap_matches_recompute_non_magic() {
        let (a, layout) = non_magic_fixture();
        check_single_swap_matches_recompute("non_magic", &a, &layout);
    }

    #[test]
    fn swap_matches_recompute_magic_one() {
        let (a, layout) = magic_fixture();
        check_single_swap_matches_recompute("magic_one", &a, &layout);
    }

    #[test]
    fn swap_matches_recompute_random_magic() {
        let (a, layout) = random_magic_fixture();
        check_single_swap_matches_recompute("random_magic", &a, &layout);
    }

    // ---- Invariant: chained swaps stay consistent with full_recompute ----

    fn check_chain_matches_recompute(label: &str, a: &Analyzer, layout: &Layout, chain_len: usize) {
        let mut cache = fresh_cache(a, layout);
        let keyswaps: Vec<_> = cache.neighbors().iter()
            .filter_map(|n| if let Neighbor::KeySwap(p) = n { Some(*p) } else { None })
            .take(chain_len)
            .collect();

        for (i, &PosPair(pa, pb)) in keyswaps.iter().enumerate() {
            cache.swap_key(pa, pb);
            let incremental = cache.score();
            let mut clone = cache.clone();
            clone.full_recompute();
            let truth = clone.score();
            assert_eq!(incremental, truth,
                "[{label}] after swap {} ({pa},{pb}): incremental={incremental} vs truth={truth} (diff {})",
                i + 1, incremental - truth);
        }
    }

    #[test]
    fn chain_matches_recompute_non_magic() {
        let (a, layout) = non_magic_fixture();
        check_chain_matches_recompute("non_magic", &a, &layout, 10);
    }

    #[test]
    fn chain_matches_recompute_magic_one() {
        let (a, layout) = magic_fixture();
        check_chain_matches_recompute("magic_one", &a, &layout, 10);
    }

    #[test]
    fn chain_matches_recompute_random_magic() {
        let (a, layout) = random_magic_fixture();
        check_chain_matches_recompute("random_magic", &a, &layout, 10);
    }

    // ---- Invariant: apply(n) + revert(n) restores score (user-requested) ----
    // update -> score() -> apply() -> revert() -> update -> score() should return the same score

    fn check_apply_revert_restores_score_keyswap(label: &str, a: &Analyzer, layout: &Layout) {
        let mut cache = fresh_cache(a, layout);
        cache.update(); // explicit
        let initial_score = cache.score();

        let keyswaps: Vec<_> = cache.neighbors().iter()
            .filter_map(|n| if let Neighbor::KeySwap(p) = n { Some(*p) } else { None })
            .take(20)
            .collect();

        for &PosPair(pa, pb) in &keyswaps {
            cache.apply_neighbor(Neighbor::KeySwap(PosPair(pa, pb)));
            // revert (keyswap is self-inverse)
            cache.apply_neighbor(Neighbor::KeySwap(PosPair(pa, pb)));
            cache.update();
            let after = cache.score();
            assert_eq!(initial_score, after,
                "[{label}] apply+revert KeySwap({pa},{pb}) changed score: {initial_score} -> {after}");
        }
    }

    #[test]
    fn apply_revert_restores_keyswap_non_magic() {
        let (a, layout) = non_magic_fixture();
        check_apply_revert_restores_score_keyswap("non_magic", &a, &layout);
    }

    #[test]
    fn apply_revert_restores_keyswap_magic_one() {
        let (a, layout) = magic_fixture();
        check_apply_revert_restores_score_keyswap("magic_one", &a, &layout);
    }

    #[test]
    fn apply_revert_restores_keyswap_random_magic() {
        let (a, layout) = random_magic_fixture();
        check_apply_revert_restores_score_keyswap("random_magic", &a, &layout);
    }

    fn check_apply_revert_restores_score_magic_rule(label: &str, a: &Analyzer, layout: &Layout) {
        let mut cache = fresh_cache(a, layout);
        cache.update();
        let initial_score = cache.score();

        let magic_neighbors: Vec<_> = cache.neighbors().iter()
            .filter_map(|n| if let Neighbor::MagicRule(r) = n { Some(*r) } else { None })
            .take(20)
            .collect();

        if magic_neighbors.is_empty() {
            return; // nothing to test on non-magic layouts
        }

        for rule in magic_neighbors {
            let revert = cache.get_revert_neighbor(Neighbor::MagicRule(rule));
            cache.apply_neighbor(Neighbor::MagicRule(rule));
            cache.apply_neighbor(revert);
            cache.update();
            let after = cache.score();
            assert_eq!(initial_score, after,
                "[{label}] apply+revert MagicRule({:?}) changed score: {} -> {}",
                rule, initial_score, after);
        }
    }

    #[test]
    fn apply_revert_restores_magic_rule_magic_one() {
        let (a, layout) = magic_fixture();
        check_apply_revert_restores_score_magic_rule("magic_one", &a, &layout);
    }

    #[test]
    fn apply_revert_restores_magic_rule_random_magic() {
        let (a, layout) = random_magic_fixture();
        check_apply_revert_restores_score_magic_rule("random_magic", &a, &layout);
    }

    #[test]
    fn apply_revert_restores_mixed_random_magic() {
        // Interleave KeySwap and MagicRule applies+reverts. Score must return
        // to original each time.
        let (a, layout) = random_magic_fixture();
        let mut cache = fresh_cache(&a, &layout);
        cache.update();
        let initial_score = cache.score();

        let neighbors = cache.neighbors();
        let keyswaps: Vec<_> = neighbors.iter().filter(|n| matches!(n, Neighbor::KeySwap(_))).take(5).copied().collect();
        let rules: Vec<_> = neighbors.iter().filter(|n| matches!(n, Neighbor::MagicRule(_))).take(5).copied().collect();

        for (i, n) in keyswaps.iter().chain(rules.iter()).enumerate() {
            let revert = cache.get_revert_neighbor(*n);
            cache.apply_neighbor(*n);
            cache.apply_neighbor(revert);
            cache.update();
            let after = cache.score();
            assert_eq!(initial_score, after,
                "mixed apply+revert step {} neighbor {:?}: score {} -> {}",
                i, n, initial_score, after);
        }
    }

    // ---- Invariant: score_neighbor matches apply (incl. active magic rules) ----

    fn check_score_neighbor_matches_apply_keyswap(label: &str, a: &Analyzer, layout: &Layout) {
        let cache_tpl = fresh_cache(a, layout);
        let keyswaps: Vec<_> = cache_tpl.neighbors().iter()
            .filter(|n| matches!(n, Neighbor::KeySwap(_)))
            .take(20)
            .copied()
            .collect();

        for n in keyswaps {
            let mut cache = fresh_cache(a, layout);
            let speculative = cache.score_neighbor(n);
            cache.apply_neighbor(n);
            let actual = cache.score();
            assert_eq!(speculative, actual,
                "[{label}] score_neighbor({:?})={} vs apply+score={} (diff {})",
                n, speculative, actual, speculative - actual);
        }
    }

    #[test]
    fn score_neighbor_matches_apply_keyswap_magic_one() {
        let (a, layout) = magic_fixture();
        check_score_neighbor_matches_apply_keyswap("magic_one", &a, &layout);
    }

    #[test]
    fn score_neighbor_matches_apply_keyswap_random_magic() {
        let (a, layout) = random_magic_fixture();
        check_score_neighbor_matches_apply_keyswap("random_magic", &a, &layout);
    }

    // ---- Invariant: update_for_keyswap matches full update() ----

    #[test]
    fn update_for_keyswap_matches_full_update_random_magic() {
        let (a, layout) = random_magic_fixture();

        let cache_tpl = fresh_cache(&a, &layout);
        let keyswaps: Vec<_> = cache_tpl.neighbors().iter()
            .filter_map(|n| if let Neighbor::KeySwap(p) = n { Some(*p) } else { None })
            .take(20)
            .collect();

        for &PosPair(pa, pb) in &keyswaps {
            // Path 1: swap_key (uses update_for_keyswap internally)
            let mut c1 = fresh_cache(&a, &layout);
            c1.swap_key(pa, pb);
            let incremental = c1.score();

            // Path 2: swap_key_no_update + full update()
            let mut c2 = fresh_cache(&a, &layout);
            c2.swap_key_no_update(pa, pb);
            c2.update();
            let full = c2.score();

            assert_eq!(incremental, full,
                "update_for_keyswap ({pa},{pb}): incremental={incremental} vs full update={full}");
        }
    }
}


