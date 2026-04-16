//! Dual Annealing for keyboard layout optimization.
//!
//! Combines Generalized Simulated Annealing (Tsallis visiting distribution)
//! with a local search (SA + greedy via RolloutPolicy).
//!
//! The global layer makes large perturbations (multi-swaps) controlled by
//! a visiting temperature. The local layer polishes each candidate.
//! Acceptance uses the Tsallis generalized probability.

use crate::cached_layout::CachedLayout;
use crate::data::Data;
use crate::layout::Layout;
use crate::optimization::RolloutPolicy;
use crate::weights::Weights;
use nanorand::{Rng, WyRand};

/// Configuration for dual annealing.
pub struct DualAnnealingConfig {
    /// Initial visiting temperature. Higher = wider jumps. Default: 5230.0
    pub initial_temp: f64,
    /// Restart ratio: when temp drops to initial_temp * ratio, restart. Default: 2e-5
    pub restart_temp_ratio: f64,
    /// Visiting parameter qv. Controls tail heaviness. Range (1, 3]. Default: 2.62
    pub visit: f64,
    /// Acceptance parameter qa. Lower = stricter acceptance. Range (-1e4, -5]. Default: -5.0
    pub accept: f64,
    /// Number of swaps per perturbation scales with temperature.
    /// At max temp, up to this many swaps. Default: 8
    pub max_perturb_swaps: usize,
    /// Pin the top K most frequent keys during local search (greedy).
    /// Global perturbations still move all keys. Default: 0 (no pinning).
    pub pin_top_k: usize,
}

impl Default for DualAnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temp: 5230.0,
            restart_temp_ratio: 2e-5,
            visit: 2.62,
            accept: -5.0,
            max_perturb_swaps: 8,
            pin_top_k: 0,
        }
    }
}

/// Result from dual annealing.
pub struct DualAnnealingResult {
    pub best_score: i64,
    pub best_layout: Layout,
    pub iterations: u64,
    pub restarts: u64,
}

pub struct DualAnnealing {
    data: Data,
    weights: Weights,
    scale_factors: crate::weights::ScaleFactors,
}

impl DualAnnealing {
    pub fn new(data: Data, weights: Weights, scale_factors: crate::weights::ScaleFactors) -> Self {
        Self { data, weights, scale_factors }
    }

    /// Run dual annealing on a layout.
    /// `pins`: positions that must not be swapped.
    /// `local_policy`: optimization applied after each global perturbation.
    /// `progress`: callback(iteration, restarts, current_score, best_score) -> should_stop
    pub fn search(
        &self,
        layout: &Layout,
        pins: &[usize],
        config: &DualAnnealingConfig,
        local_policy: &RolloutPolicy,
        max_iter: u64,
        mut progress: impl FnMut(u64, u64, i64, i64) -> bool,
    ) -> DualAnnealingResult {
        let mut rng = WyRand::new();

        // Initialize cache with the starting layout
        let mut cache = CachedLayout::new(layout, self.data.clone(), &self.weights, &self.scale_factors);

        // Build list of swappable positions (excluding pins)
        let pin_set: fxhash::FxHashSet<usize> = pins.iter().copied().collect();
        let swappable: Vec<usize> = (0..layout.keyboard.len())
            .filter(|p| !pin_set.contains(p))
            .collect();

        if swappable.len() < 2 {
            return DualAnnealingResult {
                best_score: cache.score(),
                best_layout: cache.to_layout(),
                iterations: 0,
                restarts: 0,
            };
        }

        // Run local search on initial layout
        cache.optimize(local_policy, pins);
        let mut current_score = cache.score();
        let mut best_score = current_score;
        let mut best_layout = cache.to_layout();

        // Save current state for reverting rejected perturbations
        let mut current_keys: Vec<usize> = (0..layout.keyboard.len())
            .map(|p| cache.get_key(p))
            .collect();

        let qv = config.visit;
        let qa = config.accept;
        let t1 = config.initial_temp;
        let restart_temp = t1 * config.restart_temp_ratio;
        let mut restarts = 0u64;
        let mut t_offset = 0u64; // artificial time offset for restarts

        for iter in 0..max_iter {
            // Visiting temperature: T_qv(t) = T1 * (2^(qv-1) - 1) / ((1+t)^(qv-1) - 1)
            let t = (iter - t_offset) + 1;
            let temp_v = visiting_temp(t1, qv, t as f64);

            // Check for restart
            if temp_v < restart_temp {
                t_offset = iter;
                restarts += 1;
                // Don't reset current state — keep exploring from where we are
            }

            // Perturbation: number of swaps proportional to temperature
            let temp_ratio = (temp_v / t1).min(1.0);
            let n_swaps = 1 + (temp_ratio * (config.max_perturb_swaps - 1) as f64) as usize;
            let n_swaps = n_swaps.min(swappable.len() / 2);

            // Apply random swaps
            let mut swapped_pairs: Vec<(usize, usize)> = Vec::with_capacity(n_swaps);
            let mut used = vec![false; swappable.len()];
            for _ in 0..n_swaps {
                // Pick two unused swappable positions
                let mut a_idx = rng.generate_range(0..swappable.len());
                let mut attempts = 0;
                while used[a_idx] && attempts < swappable.len() {
                    a_idx = (a_idx + 1) % swappable.len();
                    attempts += 1;
                }
                if used[a_idx] { break; }
                used[a_idx] = true;

                let mut b_idx = rng.generate_range(0..swappable.len());
                attempts = 0;
                while used[b_idx] && attempts < swappable.len() {
                    b_idx = (b_idx + 1) % swappable.len();
                    attempts += 1;
                }
                if used[b_idx] { break; }
                used[b_idx] = true;

                let pos_a = swappable[a_idx];
                let pos_b = swappable[b_idx];
                cache.swap_key(pos_a, pos_b);
                swapped_pairs.push((pos_a, pos_b));
            }

            // Local search: if pin_top_k > 0, pin the top K keys' current positions,
            // randomize the rest, then greedy optimize only unpinned keys.
            if config.pin_top_k > 0 {
                // Find positions of the top K most frequent keys
                let mut key_freqs: Vec<(usize, i64)> = swappable.iter()
                    .map(|&pos| {
                        let key = cache.get_key(pos);
                        let freq = if key < cache.data().chars.len() { cache.data().chars[key] } else { 0 };
                        (pos, freq)
                    })
                    .collect();
                key_freqs.sort_by(|a, b| b.1.cmp(&a.1));
                let top_k_pins: Vec<usize> = key_freqs.iter()
                    .take(config.pin_top_k)
                    .map(|&(pos, _)| pos)
                    .chain(pins.iter().copied())
                    .collect();

                // Randomize unpinned keys: 100 random swaps among non-pinned positions
                let pin_set_local: fxhash::FxHashSet<usize> = top_k_pins.iter().copied().collect();
                let unpinned: Vec<usize> = swappable.iter()
                    .filter(|p| !pin_set_local.contains(p))
                    .copied()
                    .collect();
                if unpinned.len() >= 2 {
                    for _ in 0..100 {
                        let a = rng.generate_range(0..unpinned.len());
                        let mut b = rng.generate_range(0..unpinned.len() - 1);
                        if b >= a { b += 1; }
                        cache.swap_key(unpinned[a], unpinned[b]);
                    }
                }

                // Greedy with top K pinned
                cache.optimize(local_policy, &top_k_pins);
            } else {
                cache.optimize(local_policy, pins);
            }
            let new_score = cache.score();

            // Tsallis acceptance probability
            let delta_e = (new_score - current_score) as f64;
            let accepted = if new_score > current_score {
                true
            } else {
                // Acceptance temperature (same schedule, different parameter)
                let temp_a = visiting_temp(t1, qa.abs(), (iter - t_offset + 1) as f64);
                let ap = tsallis_acceptance(qa, delta_e, temp_a, best_score);
                let r: f64 = rng.generate_range(0u64..u64::MAX) as f64 / u64::MAX as f64;
                ap > r
            };

            if accepted {
                current_score = new_score;
                // Save current keys
                for p in 0..current_keys.len() {
                    current_keys[p] = cache.get_key(p);
                }

                if new_score > best_score {
                    best_score = new_score;
                    best_layout = cache.to_layout();
                }
            } else {
                // Revert to current state by restoring keys
                restore_keys(&mut cache, &current_keys);
            }

            // Progress callback
            if (iter + 1) % 1 == 0 {
                if progress(iter + 1, restarts, current_score, best_score) {
                    break;
                }
            }
        }

        DualAnnealingResult {
            best_score,
            best_layout,
            iterations: max_iter,
            restarts,
        }
    }
}

/// Tsallis visiting temperature schedule.
/// T_qv(t) = T1 * (2^(qv-1) - 1) / ((1+t)^(qv-1) - 1)
fn visiting_temp(t1: f64, qv: f64, t: f64) -> f64 {
    let num = (2.0_f64).powf(qv - 1.0) - 1.0;
    let den = (1.0 + t).powf(qv - 1.0) - 1.0;
    if den <= 0.0 { return t1; }
    t1 * num / den
}

/// Tsallis generalized acceptance probability.
/// Uses the score normalization approach from our SA.
fn tsallis_acceptance(qa: f64, delta_e: f64, temp_a: f64, best_score: i64) -> f64 {
    if delta_e >= 0.0 { return 1.0; }
    // Normalize by best score magnitude (same as our SA)
    let norm = if best_score != 0 { best_score.abs() as f64 } else { 1.0 };
    let delta_norm = delta_e / norm;
    let beta = 1.0 / temp_a.max(1e-100);
    let arg = 1.0 - (1.0 - qa) * beta * delta_norm;
    if arg <= 0.0 { return 0.0; }
    arg.powf(1.0 / (1.0 - qa)).min(1.0).max(0.0)
}

/// Restore cache keys to a saved state.
fn restore_keys(cache: &mut CachedLayout, saved_keys: &[usize]) {
    use crate::cached_layout::EMPTY_KEY;
    let n = saved_keys.len();

    // First clear all positions, then place saved keys
    for pos in 0..n {
        let k = cache.get_key(pos);
        if k != EMPTY_KEY {
            cache.replace_key_no_update(pos, k, EMPTY_KEY);
        }
    }
    for pos in 0..n {
        let k = saved_keys[pos];
        if k != EMPTY_KEY {
            cache.replace_key_no_update(pos, EMPTY_KEY, k);
        }
    }
    cache.update();
}
