use crate::{
    analyze::{Analyzer, Neighbor},
    layout::Layout,
};

impl Analyzer {
    pub fn always_better_swap(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        self.use_layout(&layout, pins);
        let mut best_score = self.score();

        loop {
            let mut best_loop_score = i64::MIN;

            let count = self.neighbor_count();
            for i in 0..count {
                let neighbor = self.get_neighbor(i);
                let score = self.test_neighbor(neighbor);

                if score > best_score {
                    best_loop_score = score;
                    self.apply_neighbor(neighbor);
                    break;
                }
            }

            if best_loop_score <= best_score {
                break;
            }

            best_score = best_loop_score;
        }

        (self.layout(), best_score)
    }

    pub fn alternative_d3(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        let (layout, _) = self.always_better_swap(layout, pins);
        self.greedy_depth3_improve(layout, pins)
    }

    pub fn optimize_depth3(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        let (layout, _) = self.greedy_improve(&layout, pins);
        let (layout, _) = self.greedy_depth2_improve(layout, pins);
        self.greedy_depth3_improve(layout, pins)
    }

    pub fn optimize_depth4(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        let (layout, _) = self.greedy_improve(&layout, pins);
        let (layout, _) = self.greedy_depth2_improve(layout, pins);
        let (layout, _) = self.greedy_depth3_improve(layout, pins);
        self.greedy_depth4_improve(layout, pins)
    }

    pub fn greedy_improve(&mut self, layout: &Layout, pins: &[usize]) -> (Layout, i64) {
        self.use_layout(&layout, pins);
        let mut best_score = self.score();

        while let Some((neighbor, score)) = self.best_neighbor() {
            if score <= best_score {
                break;
            }

            best_score = score;
            self.apply_neighbor(neighbor);
        }

        (self.layout(), best_score)
    }

    pub fn greedy_depth2_improve(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        self.greedy_improve_depth_n(layout, pins, 2)
    }

    pub fn greedy_depth4_improve(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        self.greedy_improve_depth_n(layout, pins, 4)
    }

    pub fn greedy_depth3_improve(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64) {
        self.greedy_improve_depth_n(layout, pins, 3)
    }

    fn greedy_improve_depth_n(
        &mut self,
        layout: Layout,
        pins: &[usize],
        depth: usize,
    ) -> (Layout, i64) {
        self.use_layout(&layout, pins);
        let mut diffs = vec![Neighbor::default(); depth];
        let mut cur_best = self.score();

        while self.best_neighbor_recursive(depth, &mut diffs, &mut cur_best) {
            for neighbor in diffs.iter() {
                self.apply_neighbor(*neighbor);
            }
        }

        (self.layout(), cur_best)
    }

    fn best_neighbor_recursive(
        &mut self,
        depth: usize,
        diffs: &mut Vec<Neighbor>,
        cur_best: &mut i64,
    ) -> bool {
        if depth > 0 {
            let mut return_best = false;
            let count = self.neighbor_count();
            for i in 0..count {
                let neighbor = self.get_neighbor(i);
                // Apply the neighbor
                self.apply_neighbor(neighbor);

                // Recurse
                let best = self.best_neighbor_recursive(depth - 1, diffs, cur_best);

                // Revert the neighbor
                self.apply_neighbor(neighbor.revert());

                // This chain is the current known best. Update diffs
                if best {
                    diffs[depth - 1] = neighbor;
                    return_best = true;
                }
            }
            return_best
        } else {
            let score = self.score();
            if score > *cur_best {
                // This chain is the current known best. Update cur_best
                *cur_best = score;
                true
            } else {
                false
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer_layout(name: &str) -> (Analyzer, Layout) {
        let data = crate::prelude::Data::load("../data/english.json").expect("this should exist");

        let weights = crate::weights::dummy_weights();

        let analyzer = Analyzer::new(data, weights);

        let layout = Layout::load(format!("../layouts/{name}.dof"))
            .expect("this layout is valid and exists, soooo");

        (analyzer, layout)
    }

    #[test]
    fn cache_intact() {
        let (analyzer, layout) = analyzer_layout("rstn-oxey");
        analyzer.use_layout(&layout, &[]);
        let reference = cache.clone();
        let mut diffs = vec![Neighbor::default(); 4];
        let mut cur_best = i64::MIN;

        analyzer.best_neighbor_recursive(1, &mut diffs, &mut cur_best);
        assert_eq!(cache, reference);

        analyzer.best_neighbor_recursive(&mut cache, 2, &mut diffs, &mut cur_best, &neighbors);
        assert_eq!(cache, reference);

        // TODO: This test is too slow
        // analyzer.best_neighbor_recursive(&mut cache, 3, &mut diffs, &mut cur_best);
        //assert_eq!(cache, reference);
    }

    #[test]
    fn strectch_cache_integrity() {
        let (analyzer, layout) = analyzer_layout("rstn-oxey");
        let mut cache = analyzer.cached_layout(layout, &[]);

        println!("stretches before swap: {}", analyzer.stretches(&cache));

        match analyzer.best_neighbor(&mut cache) {
            Some((pair, score)) => {
                println!("pair: {:?}, score: {}", pair, score);
                analyzer.apply_neighbor(&mut cache, pair);
                println!("stretches after swap: {}", analyzer.stretches(&cache));
            }
            None => println!("No improvement found"),
        }
    }
}
