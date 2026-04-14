use crate::analyze::Neighbor;
use crate::cached_layout::CachedLayout;
use crate::layout::PosPair;
use nanorand::{RandomGen, Rng, WyRand};

/// Describes a sequence of optimization steps to apply to a layout.
/// Each step runs in order, with pins respected throughout.
#[derive(Debug, Clone)]
pub struct RolloutPolicy {
    pub steps: Vec<OptStep>,
}

/// A single optimization step.
#[derive(Debug, Clone)]
pub enum OptStep {
    /// Simulated annealing with given parameters.
    SA {
        initial_temp: f64,
        final_temp: f64,
        iterations: usize,
    },
    /// Greedy hill-climb (swap best neighbor until no improvement).
    Greedy,
    /// Greedy depth-N search (try all N-swap combos).
    GreedyDepthN(usize),
}

impl RolloutPolicy {
    /// SA only.
    pub fn sa(iterations: usize) -> Self {
        Self {
            steps: vec![OptStep::SA {
                initial_temp: 10.0,
                final_temp: 1E-5,
                iterations,
            }],
        }
    }

    /// SA followed by greedy hill-climb.
    pub fn sa_greedy(sa_iterations: usize) -> Self {
        Self {
            steps: vec![
                OptStep::SA {
                    initial_temp: 10.0,
                    final_temp: 1E-5,
                    iterations: sa_iterations,
                },
                OptStep::Greedy,
            ],
        }
    }

    /// SA followed by greedy depth-N.
    pub fn sa_greedy_depth(sa_iterations: usize, depth: usize) -> Self {
        Self {
            steps: vec![
                OptStep::SA {
                    initial_temp: 10.0,
                    final_temp: 1E-5,
                    iterations: sa_iterations,
                },
                OptStep::GreedyDepthN(depth),
            ],
        }
    }

    /// Greedy only.
    pub fn greedy() -> Self {
        Self {
            steps: vec![OptStep::Greedy],
        }
    }

    /// Greedy depth-N only.
    pub fn greedy_depth(depth: usize) -> Self {
        Self {
            steps: vec![OptStep::GreedyDepthN(depth)],
        }
    }
}

impl CachedLayout {
    /// Run a rollout policy in-place on this cache. `pins` are positions that
    /// must not be swapped. Running totals are valid after this returns.
    /// Returns the final score.
    pub fn optimize(&mut self, policy: &RolloutPolicy, pins: &[usize]) -> i64 {
        for step in &policy.steps {
            match step {
                OptStep::SA { initial_temp, final_temp, iterations } => {
                    self.run_sa(pins, *initial_temp, *final_temp, *iterations);
                }
                OptStep::Greedy => {
                    self.run_greedy(pins);
                }
                OptStep::GreedyDepthN(depth) => {
                    self.greedy_improve_depth_n(pins, *depth);
                }
            }
        }
        self.score()
    }

    /// Simulated annealing directly on the cache. Pins are excluded from swaps.
    fn run_sa(&mut self, pins: &[usize], initial_temp: f64, final_temp: f64, iterations: usize) {
        if iterations == 0 { return; }

        let neighbors = self.build_unpinned_neighbors(pins);
        if neighbors.is_empty() { return; }

        let mut rng = WyRand::new();
        let cooling_rate = (final_temp / initial_temp).powf(1.0 / iterations as f64);
        let mut temperature = initial_temp;
        let mut current_score = self.score();
        let mut worst_score = current_score;

        for _ in 0..iterations {
            let idx = rng.generate_range(0..neighbors.len());
            let neighbor = neighbors[idx];
            let new_score = self.score_neighbor(neighbor);

            if new_score < worst_score {
                worst_score = new_score;
            }

            let ap = if new_score > current_score {
                1.0
            } else {
                ((new_score - current_score) as f64 / worst_score.abs() as f64 / temperature).exp()
            };

            if ap > f64::random(&mut rng) {
                current_score = new_score;
                self.apply_neighbor_and_update(neighbor);
            }

            temperature *= cooling_rate;
        }
    }

    /// Greedy hill-climb: keep swapping the best neighbor until no improvement.
    /// Pins are excluded from swaps.
    fn run_greedy(&mut self, pins: &[usize]) {
        let neighbors = self.build_unpinned_neighbors(pins);
        if neighbors.is_empty() { return; }

        let mut best_score = self.score();
        loop {
            let mut improved = false;
            for &neighbor in &neighbors {
                let score = self.score_neighbor(neighbor);
                if score > best_score {
                    best_score = score;
                    self.apply_neighbor_and_update(neighbor);
                    improved = true;
                    break;
                }
            }
            if !improved { break; }
        }
    }

    /// Build the list of KeySwap neighbors excluding pinned positions.
    fn build_unpinned_neighbors(&self, pins: &[usize]) -> Vec<Neighbor> {
        if pins.is_empty() {
            return self.neighbors();
        }
        let pin_set: fxhash::FxHashSet<usize> = pins.iter().copied().collect();
        self.neighbors()
            .into_iter()
            .filter(|n| match n {
                Neighbor::KeySwap(PosPair(a, b)) => !pin_set.contains(a) && !pin_set.contains(b),
                Neighbor::MagicRule(_) => true,
            })
            .collect()
    }
}
