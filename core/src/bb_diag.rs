#[cfg(test)]
mod tests {
    use crate::data::Data;
    use crate::layout::Layout;
    use crate::weights::dummy_weights;
    use crate::analyze::Analyzer;
    use std::collections::HashMap;

    #[test]
    fn decomposition_feasibility() {
        let data = Data::load("../data/english.json").expect("data");
        let weights = dummy_weights();

        // Test with multiple layouts to see different finger structures
        for name in ["nrts-oxey", "qwerty"] {
            let layout = Layout::load(format!("../layouts/{name}.dof")).expect("layout");

            println!("\n=== {} ({:?} board) ===", name, layout.shape);
            println!("Positions: {}", layout.keyboard.len());

            // Group positions by finger
            let mut finger_groups: HashMap<u8, Vec<usize>> = HashMap::new();
            for (pos, finger) in layout.fingers.iter().enumerate() {
                finger_groups.entry(*finger as u8).or_default().push(pos);
            }

            let mut groups: Vec<(u8, Vec<usize>)> = finger_groups.into_iter().collect();
            groups.sort_by_key(|(f, _)| *f);

            println!("\nFinger groups:");
            let mut total_within = 1u128;
            for (finger, positions) in &groups {
                let n = positions.len();
                let factorial: u128 = (1..=n as u128).product();
                total_within *= factorial;
                let keys: Vec<char> = positions.iter().map(|&p| layout.keys[p]).collect();
                println!("  finger {:2}: {} positions {:?} -> {:?} ({}! = {} orderings)",
                    finger, n, positions, keys, n, factorial);
            }

            let num_fingers = groups.len();
            let finger_sizes: Vec<usize> = groups.iter().map(|(_, p)| p.len()).collect();
            let total_positions: usize = finger_sizes.iter().sum();
            let num_keys = total_positions; // assume 1 key per position

            println!("\nSearch space analysis:");
            println!("  Fingers: {}", num_fingers);
            println!("  Finger sizes: {:?}", finger_sizes);
            println!("  Total positions: {}", total_positions);

            // Full search space
            let full_factorial: f64 = (1..=total_positions).map(|i| i as f64).product();
            println!("  Full search ({}!): {:.2e}", total_positions, full_factorial);

            // Within-finger permutations only
            println!("  Within-finger perms: {:.2e}", total_within as f64);

            // Key-to-finger assignment (multinomial)
            let assignment_space = full_factorial / finger_sizes.iter()
                .map(|&s| (1..=s).map(|i| i as f64).product::<f64>())
                .product::<f64>();
            println!("  Key-to-finger assignments: {:.2e}", assignment_space);

            // Phase 1: Hand assignment (left vs right)
            let left_fingers: Vec<&(u8, Vec<usize>)> = groups.iter()
                .filter(|(f, _)| (*f as usize) < 5) // LP=0..LT=4
                .collect();
            let right_fingers: Vec<&(u8, Vec<usize>)> = groups.iter()
                .filter(|(f, _)| (*f as usize) >= 5) // RT=5..RP=9
                .collect();
            let left_size: usize = left_fingers.iter().map(|(_, p)| p.len()).sum();
            let right_size: usize = right_fingers.iter().map(|(_, p)| p.len()).sum();

            println!("\n  Hand split: left={} positions, right={} positions", left_size, right_size);

            // C(n, left_size) ways to assign keys to hands
            let hand_assignments = n_choose_k(num_keys, left_size);
            println!("  Hand assignments C({},{}): {:.2e}", num_keys, left_size, hand_assignments);

            // Phase 2: Within-hand finger assignment
            // Left hand: partition left_size keys into finger groups
            let left_sizes: Vec<usize> = left_fingers.iter().map(|(_, p)| p.len()).collect();
            let right_sizes: Vec<usize> = right_fingers.iter().map(|(_, p)| p.len()).collect();

            let left_finger_assignments = multinomial_count(left_size, &left_sizes);
            let right_finger_assignments = multinomial_count(right_size, &right_sizes);

            println!("  Left finger assignments: {:.2e} (sizes {:?})", left_finger_assignments, left_sizes);
            println!("  Right finger assignments: {:.2e} (sizes {:?})", right_finger_assignments, right_sizes);

            // Phase 3: Within-finger ordering
            let left_within: f64 = left_sizes.iter()
                .map(|&s| (1..=s).map(|i| i as f64).product::<f64>())
                .product();
            let right_within: f64 = right_sizes.iter()
                .map(|&s| (1..=s).map(|i| i as f64).product::<f64>())
                .product();

            println!("  Left within-finger orderings: {:.2e}", left_within);
            println!("  Right within-finger orderings: {:.2e}", right_within);

            // Total decomposed
            let total_decomposed = hand_assignments * left_finger_assignments * right_finger_assignments
                * left_within * right_within;
            println!("\n  Total decomposed: {:.2e} (should equal {:.2e})", total_decomposed, full_factorial);

            // Feasibility of each phase
            println!("\n  FEASIBILITY:");
            println!("  Phase 1 (hand assignment): {:.2e} - {}",
                hand_assignments,
                if hand_assignments < 1e9 { "FEASIBLE with pruning" } else { "needs beam search" });
            println!("  Phase 2a (left finger assignment): {:.2e} - {}",
                left_finger_assignments,
                if left_finger_assignments < 1e9 { "FEASIBLE" } else { "needs beam search" });
            println!("  Phase 2b (right finger assignment): {:.2e} - {}",
                right_finger_assignments,
                if right_finger_assignments < 1e9 { "FEASIBLE" } else { "needs beam search" });
            println!("  Phase 3 (within-finger): {:.2e} - {}",
                left_within * right_within,
                if left_within * right_within < 1e9 { "FEASIBLE (independent per finger)" } else { "needs pruning" });

            // Alternative: beam search at each phase
            let beam_k = 10000;
            let beam_phase1 = (hand_assignments as f64).min(beam_k as f64) * left_size as f64;
            let beam_phase2 = beam_k as f64 * (left_finger_assignments + right_finger_assignments);
            let beam_phase3 = beam_k as f64 * (left_within + right_within);
            println!("\n  With beam search (K={}):", beam_k);
            println!("    Phase 1 evaluations: {:.2e}", beam_phase1);
            println!("    Phase 2 evaluations: {:.2e}", beam_phase2);
            println!("    Phase 3 evaluations: {:.2e}", beam_phase3);
            println!("    Total evaluations: {:.2e}", beam_phase1 + beam_phase2 + beam_phase3);
            println!("    At 5µs/eval: {:.1}s", (beam_phase1 + beam_phase2 + beam_phase3) * 5e-6);
        }
    }

    fn n_choose_k(n: usize, k: usize) -> f64 {
        if k > n { return 0.0; }
        let k = k.min(n - k);
        let mut result = 1.0f64;
        for i in 0..k {
            result *= (n - i) as f64 / (i + 1) as f64;
        }
        result
    }

    fn multinomial_count(n: usize, groups: &[usize]) -> f64 {
        let numerator: f64 = (1..=n).map(|i| i as f64).product();
        let denominator: f64 = groups.iter()
            .map(|&s| (1..=s).map(|i| i as f64).product::<f64>())
            .product();
        numerator / denominator
    }
}
