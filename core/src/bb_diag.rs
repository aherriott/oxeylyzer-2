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
        let nk = cache.trigram_num_keys();

        // Count same-finger position pairs
        let mut sf_pair_count = 0;
        for pos in 0..num_pos {
            sf_pair_count += cache.sfb_pairs(pos).len();
        }
        sf_pair_count /= 2; // each pair counted twice
        println!("\n=== Precomputed bound analysis ===");
        println!("Same-finger position pairs: {}", sf_pair_count);

        // For each depth, compute:
        // 1. Current score (from placed keys)
        // 2. Minimum remaining SFB penalty (from unplaced key pairs)
        for depth in 0..12 {
            let score = cache.score();

            // Collect all bigram frequencies between remaining keys
            let remaining = &all_keys[depth..num_pos];
            let mut remaining_bigrams: Vec<i64> = Vec::new();
            for (i, &ka) in remaining.iter().enumerate() {
                for &kb in &remaining[i+1..] {
                    if ka < nk && kb < nk {
                        let freq = bg_freq[ka * nk + kb] + bg_freq[kb * nk + ka];
                        if freq > 0 {
                            remaining_bigrams.push(freq);
                        }
                    }
                }
            }
            remaining_bigrams.sort_unstable(); // ascending - smallest first

            // The minimum SFB penalty: the `sf_pair_count` highest-frequency
            // remaining bigrams MUST go on same-finger pairs (pigeonhole).
            // Wait - that's backwards. We WANT to minimize penalty, so we'd
            // put the LOWEST frequency bigrams on same-finger pairs.
            // But we can't choose - the positions are fixed.
            //
            // Actually: there are `sf_pair_count` same-finger position pairs
            // among the available positions. Each will have some bigram on it.
            // The minimum penalty is when the lowest-frequency bigrams land there.
            //
            // But we have C(remaining, 2) bigrams and sf_pair_count positions.
            // The minimum SFB is: sum of the sf_pair_count smallest bigram freqs
            // × minimum same-finger distance × minimum finger weight.

            let min_sfb_weight: i64 = (0..10)
                .map(|f| cache.sfb_weight(f))
                .filter(|&w| w != 0)
                .min()
                .unwrap_or(0);

            let min_sf_dist: i64 = (0..num_pos)
                .flat_map(|p| cache.sfb_pairs(p).iter().map(|sf| sf.dist))
                .min()
                .unwrap_or(0);

            // Sum the sf_pair_count smallest remaining bigram frequencies
            let n_pairs = sf_pair_count.min(remaining_bigrams.len());
            let min_sfb_penalty: i64 = remaining_bigrams[..n_pairs].iter().sum::<i64>()
                * min_sf_dist * min_sfb_weight;

            let projected = score + min_sfb_penalty;
            let prune = projected < bound;

            println!("  depth {:2}: score={:>15}, remaining_bigrams={:>5}, min_sfb_penalty={:>15}, projected={:>15} {}",
                depth, score, remaining_bigrams.len(), min_sfb_penalty, projected,
                if prune { "<-- PRUNE" } else { "" });

            if depth < all_keys.len() {
                cache.replace_key_fast(depth, EMPTY_KEY, all_keys[depth]);
            }
        }
    }
}
