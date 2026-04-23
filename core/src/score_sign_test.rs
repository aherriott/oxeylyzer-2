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
