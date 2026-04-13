#[cfg(test)]
mod tests {
    use crate::branch_bound::BranchBound;
    use crate::cached_layout::EMPTY_KEY;
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;

    #[test]
    fn diagnose_precomputed_bound() {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();
        let layout = Layout::load("../layouts/nrts-oxey.dof").expect("layout");

        let mut bb = BranchBound::new(layout, data, weights);
        let mut cache = bb.create_empty_cache();

        let num_pos = bb.num_positions();
        let all_keys: Vec<usize> = bb.chars_by_frequency().iter()
            .map(|&c| cache.char_mapping().get_u(c))
            .collect();

        let bound: i64 = -143_714_244_130;
        let bg_freq = cache.magic_bg_freq().to_vec();
        let tg_flat = cache.magic_tg_freq_flat().to_vec();
        let nk = cache.trigram_num_keys();
        let nk2 = nk * nk;
        let max_w = cache.trigram_max_weight();

        println!("\n=== Trigram-based remaining cost bound ===");
        println!("Bound (nrts-oxey): {}", bound);
        println!("max_trigram_weight: {}", max_w);

        // For each depth, compute the total trigram frequency among remaining keys
        // Every trigram contributes (weight - max_w) * freq, and weight <= max_w,
        // so every contribution is <= 0. The total is a valid lower bound.
        for depth in 0..12 {
            let score = cache.score();
            let remaining = &all_keys[depth..num_pos];

            // Sum ALL trigram frequencies among remaining keys
            // For each triple (a, b, c) of remaining keys, sum tg_freq[a][b][c]
            // weighted by the MINIMUM possible (weight - max_w) for any position triple.
            // Since we don't know positions, use the most negative offset weight.
            let min_offset_weight: i64 = [-6, -1, -1, -4, -5].iter().copied().min().unwrap(); // redirect=-6 is worst

            let mut total_tg_freq: i64 = 0;
            for &ka in remaining {
                for &kb in remaining {
                    for &kc in remaining {
                        if ka < nk && kb < nk && kc < nk {
                            total_tg_freq += tg_flat[ka * nk2 + kb * nk + kc];
                        }
                    }
                }
            }

            // Every trigram gets at least min_offset_weight penalty
            // But not all position triples are tracked trigram types.
            // The fraction that are tracked is roughly 60-70%.
            // Use 100% for a valid (conservative) bound.
            let tg_bound = total_tg_freq * min_offset_weight;

            let projected = score + tg_bound;
            let prune = projected < bound;

            println!("  depth {:2}: score={:>15}, tg_freq_sum={:>15}, tg_bound={:>15}, projected={:>15} {}",
                depth, score, total_tg_freq, tg_bound, projected,
                if prune { "<-- PRUNE" } else { "" });

            if depth < all_keys.len() {
                cache.replace_key_fast(depth, EMPTY_KEY, all_keys[depth]);
            }
        }
    }
}
