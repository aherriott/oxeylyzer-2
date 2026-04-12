use fxhash::FxHashSet as HashSet;
use libdof::dofinitions::Finger;
use nanorand::{Rng, WyRand};

use crate::{
    cached_layout::CachedLayout,
    data::Data,
    layout::*,
    stats::Stats,
    weights::Weights,
};

// The difference between two neighboring layouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Neighbor {
    KeySwap(PosPair),
    MagicRule(MagicRule),
}

impl Neighbor {
    pub fn default() -> Self {
        Neighbor::KeySwap(PosPair(0, 0))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Analyzer {
    data: Data,
    weights: Weights,
    analyze_bigrams: bool,
    analyze_stretches: bool,
    analyze_trigrams: bool,
    cache: Option<CachedLayout>,
}

impl Analyzer {
    pub fn new(data: Data, weights: Weights) -> Self {
        let analyze_bigrams = weights.has_bigram_weights();
        let analyze_stretches = weights.has_stretch_weights();
        let analyze_trigrams = weights.has_trigram_weights();

        Self {
            data,
            weights,
            analyze_bigrams,
            analyze_stretches,
            analyze_trigrams,
            cache: None,
        }
    }

    pub fn use_layout(&mut self, layout: &Layout, _pins: &[usize]) {
        // TODO: use pins
        self.cache = Some(CachedLayout::new(layout, self.data.clone(), &self.weights));
    }

    pub fn layout(&self) -> Layout {
        self.cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .to_layout()
    }

    pub fn score(&self) -> i64 {
        self.cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .score()
    }

    /*
     **************************************
     *         Neighbor actions
     **************************************
     */

    /// Get all possible neighbors for the current layout
    pub fn neighbors(&self) -> Vec<Neighbor> {
        self.cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .neighbors()
    }

    /// Get the neighbor that would revert the given neighbor.
    /// Must be called BEFORE apply_neighbor since it needs the current state.
    pub fn get_revert_neighbor(&self, neighbor: Neighbor) -> Neighbor {
        self.cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .get_revert_neighbor(neighbor)
    }

    pub fn random_neighbor(&self, rng: &mut WyRand, neighbors: &[Neighbor]) -> Neighbor {
        neighbors[rng.generate_range(0..neighbors.len())]
    }

    pub fn best_neighbor(&mut self, neighbors: &[Neighbor]) -> Option<(Neighbor, i64)> {
        let mut best_score = self.cache
            .as_ref()
            .expect("Analyzer has no Layout set")
            .score();
        let mut best = None;

        for &neighbor in neighbors {
            let score = self.score_neighbor(neighbor);
            if score > best_score {
                best_score = score;
                best = Some((neighbor, score));
            }
        }
        best
    }

    /// Speculative score for a neighbor. No mutation for KeySwap.
    pub fn score_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        self.cache
            .as_mut()
            .expect("Analyzer has no Layout set")
            .score_neighbor_mut(neighbor)
    }

    /// Test a neighbor without applying it. Returns the score.
    /// Alias for score_neighbor for backwards compatibility.
    pub fn test_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        self.score_neighbor(neighbor)
    }

    /// Apply a neighbor (fast path, no weighted_score update).
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        self.cache
            .as_mut()
            .expect("Analyzer has no Layout set")
            .apply_neighbor(neighbor)
    }

    /// Apply a neighbor and update weighted_score arrays.
    pub fn apply_neighbor_and_update(&mut self, neighbor: Neighbor) {
        self.cache
            .as_mut()
            .expect("Analyzer has no Layout set")
            .apply_neighbor_and_update(neighbor)
    }

    /// Rebuild weighted_score arrays from current state.
    pub fn update_scores(&mut self) {
        self.cache
            .as_mut()
            .expect("Analyzer has no Layout set")
            .update_scores()
    }

    /*
     **************************************
     *              Stats
     **************************************
     */

    pub fn stats(&self) -> Stats {
        let cache = self.cache
            .as_ref()
            .expect("Analyzer has no Layout set");

        let mut stats = Stats::default();
        stats.score = cache.score();
        cache.stats(&mut stats);
        stats
    }

    pub fn similarity(&self, layout1: &Layout, layout2: &Layout) -> i64 {
        let cache = self.cache
            .as_ref()
            .expect("Analyzer has no Layout set");
        let data = cache.data();

        // TODO: Magic
        let per_column = Finger::FINGERS
            .into_iter()
            .map(|f| {
                let col1 = layout1
                    .keys
                    .iter()
                    .zip(&layout1.fingers)
                    .filter_map(|(c1, f1)| (f == *f1).then_some(*c1))
                    .collect::<HashSet<_>>();

                layout2
                    .keys
                    .iter()
                    .zip(&layout1.fingers)
                    .filter(|&(c2, f2)| f == *f2 && col1.contains(c2))
                    .map(|(c2, _)| data.get_char(*c2))
                    .sum::<i64>()
            })
            .sum::<i64>();

        per_column
    }
}

impl std::fmt::Display for Analyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(cache) = &self.cache {
            write!(f, "{}", cache.to_layout())
        } else {
            write!(f, "<no layout>")
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use crate::weights::dummy_weights;

    use super::*;

    fn _analyzer_layout(layout_name: &str) -> (Analyzer, Layout) {
        let data = Data::load("../data/english.json").expect("this should exist");
        let weights = dummy_weights();
        let analyzer = Analyzer::new(data, weights);
        let layout = Layout::load(format!("../layouts/{layout_name}.dof"))
            .expect("this layout is valid and exists, soooo");
        (analyzer, layout)
    }
}
