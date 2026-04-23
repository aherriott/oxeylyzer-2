//! Tests that score components are always ≤ 0.
//!
//! Invariant: The best score possible is 0 (no penalties, best trigram types).
//! Every score component must be non-positive, regardless of how many magic
//! rules are active. When multiple rules are applied, the composition of their
//! bigram/trigram frequency shifts must be computed correctly — naively
//! summing per-rule deltas double-counts overlapping trigram side-effects.

#[cfg(test)]
mod helpers {
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::Weights;

    pub fn load_weights() -> Weights {
        let toml_str = std::fs::read_to_string("../analyzer-config.toml")
            .expect("Failed to read analyzer-config.toml");
        let config: toml::Value = toml::from_str(&toml_str).expect("Failed to parse toml");
        let weights_section = config.get("weights").expect("No [weights] section");
        weights_section.clone().try_into().expect("Failed to parse weights")
    }

    pub fn load_data() -> Data {
        let json_str = std::fs::read_to_string("../data/english.json")
            .expect("Failed to read english.json");
        serde_json::from_str(&json_str).expect("Failed to parse english.json")
    }

    pub fn load_layout(name: &str) -> Layout {
        let path = format!("../layouts/{name}.dof");
        Layout::load(&path).unwrap_or_else(|e| panic!("Failed to load {path}: {e}"))
    }

    pub fn assert_all_non_positive(label: &str, cache: &crate::cached_layout::CachedLayout) {
        let (sfb, stretch, scissors, trigram, magic_penalty, finger_usage) = cache.score_breakdown();
        let total = cache.score();

        let positives: Vec<(&str, i64)> = [
            ("sfb", sfb),
            ("stretch", stretch),
            ("scissors", scissors),
            ("trigram", trigram),
            ("magic_penalty", magic_penalty),
            ("finger_usage", finger_usage),
            ("total", total),
        ]
        .into_iter()
        .filter(|(_, v)| *v > 0)
        .collect();

        if !positives.is_empty() {
            panic!(
                "{label}: score components must be ≤ 0. Positives: {:?}\n\
                 breakdown: sfb={sfb} stretch={stretch} scissors={scissors} \
                 trigram={trigram} magic_penalty={magic_penalty} \
                 finger_usage={finger_usage} total={total}\n\
                 active rules: {}",
                positives, cache.magic_rule_count(),
            );
        }
    }
}

/// Regression test for the rule-composition bug found in `gen`.
///
/// `failing-many-rules.dof` was captured from a greedy run that produced a
/// positive sfb. It has 29 magic rules on a single magic key. Multiple rules
/// with overlapping trigram side-effects cause `compute_rule_delta` to
/// double-count the same frequency shifts, pushing `magic_rule_score_delta`
/// past `|total_score|` and flipping the sign.
#[cfg(test)]
mod regression {
    use super::helpers::*;
    use crate::cached_layout::CachedLayout;

    #[test]
    fn failing_layout_from_gen_keeps_all_components_non_positive() {
        let layout = load_layout("failing-many-rules");
        let data = load_data();
        let weights = load_weights();
        let scale_factors = weights.compute_scale_factors(&data);

        let cache = CachedLayout::new(&layout, data, &weights, &scale_factors);
        assert_all_non_positive("failing-many-rules", &cache);
    }
}

/// Lightweight sanity check that every committed magic layout has all score
/// components ≤ 0.
#[cfg(test)]
mod basic {
    use super::helpers::*;
    use crate::cached_layout::CachedLayout;

    fn check(name: &str) {
        let layout = load_layout(name);
        let data = load_data();
        let weights = load_weights();
        let scale_factors = weights.compute_scale_factors(&data);
        let cache = CachedLayout::new(&layout, data, &weights, &scale_factors);
        assert_all_non_positive(name, &cache);
    }

    #[test] fn magic_one() { check("magic-one"); }
    #[test] fn magic_two() { check("magic-two"); }
    #[test] fn my_layout() { check("my-layout"); }
}


/// Verify the pre-computed magic_swap_delta table matches per-call computation.
#[cfg(test)]
mod swap_table_validation {
    use super::helpers::*;
    use crate::cached_layout::CachedLayout;
    use crate::analyze::Neighbor;
    use crate::layout::PosPair;

    #[test]
    fn precomputed_table_matches_apply_score_revert() {
        let layout = load_layout("failing-many-rules");
        let data = load_data();
        let weights = load_weights();
        let scale_factors = weights.compute_scale_factors(&data);

        let mut cache = CachedLayout::new(&layout, data, &weights, &scale_factors);

        let neighbors = cache.neighbors();
        let swaps: Vec<_> = neighbors.iter()
            .filter(|n| matches!(n, Neighbor::KeySwap(_)))
            .copied()
            .collect();

        // Compare table against apply-score-revert
        let current_magic_total = cache.sfb_magic_delta() + cache.stretch_magic_delta()
            + cache.scissors_magic_delta() + cache.trigram_magic_delta();

        let mut table_failures = 0;
        for &swap in &swaps[..5.min(swaps.len())] {
            if let Neighbor::KeySwap(PosPair(a, b)) = swap {
                let table_change = cache.magic_swap_delta_at(a, b);

                // Apply, read new magic total, revert
                cache.swap_key(a, b);
                let new_sfb = cache.sfb_magic_delta();
                let new_stretch = cache.stretch_magic_delta();
                let new_scissors = cache.scissors_magic_delta();
                let new_trigram = cache.trigram_magic_delta();
                let new_magic_total = new_sfb + new_stretch + new_scissors + new_trigram;
                cache.swap_key(a, b);

                let expected_change = new_magic_total - current_magic_total;

                // Also compute per-analyzer to find which one is wrong
                let key_a = cache.get_key(a);
                let key_b = cache.get_key(b);

                let spec_sfb = cache.sfb_speculative_swap(key_a, key_b, a, b);
                let spec_stretch = cache.stretch_speculative_swap(key_a, key_b, a, b);
                let spec_scissors = cache.scissors_speculative_swap(key_a, key_b, a, b);
                let spec_trigram = cache.trigram_speculative_swap(key_a, key_b, a, b);

                let sfb_ok = spec_sfb == new_sfb;
                let stretch_ok = spec_stretch == new_stretch;
                let scissors_ok = spec_scissors == new_scissors;
                let trigram_ok = spec_trigram == new_trigram;

                if table_change != expected_change {
                    eprintln!("swap ({a},{b}): table={table_change} expected={expected_change} diff={}",
                        table_change - expected_change);
                    eprintln!("  sfb: spec={spec_sfb} actual={new_sfb} {}",
                        if sfb_ok { "OK" } else { "MISMATCH" });
                    eprintln!("  stretch: spec={spec_stretch} actual={new_stretch} {}",
                        if stretch_ok { "OK" } else { "MISMATCH" });
                    eprintln!("  scissors: spec={spec_scissors} actual={new_scissors} {}",
                        if scissors_ok { "OK" } else { "MISMATCH" });
                    eprintln!("  trigram: spec={spec_trigram} actual={new_trigram} {}",
                        if trigram_ok { "OK" } else { "MISMATCH" });
                    table_failures += 1;
                }
            }
        }

        assert_eq!(table_failures, 0, "{table_failures} swaps had wrong table value");
    }
}
