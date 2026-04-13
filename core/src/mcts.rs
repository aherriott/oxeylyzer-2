//! Monte Carlo Tree Search for keyboard layout optimization.
//!
//! Places keys in frequency order (most frequent first). At each node,
//! children represent placing the next key at a specific position.
//! Rollouts use greedy completion to estimate node values.

use crate::cached_layout::{CachedLayout, EMPTY_KEY};
use crate::data::Data;
use crate::layout::Layout;
use crate::weights::Weights;
use crate::types::CacheKey;
use nanorand::Rng;

/// A node in the MCTS tree.
struct MctsNode {
    /// Position chosen at this node (EMPTY for root)
    position: usize,
    /// Number of times this node has been visited
    visits: u32,
    /// Sum of scores from all rollouts through this node
    score_sum: f64,
    /// Best score seen through this node
    best_score: i64,
    /// Children: one per available position
    children: Vec<MctsNode>,
    /// Whether children have been expanded
    expanded: bool,
}

impl MctsNode {
    fn new(position: usize) -> Self {
        Self {
            position,
            visits: 0,
            score_sum: 0.0,
            best_score: i64::MIN,
            children: Vec::new(),
            expanded: false,
        }
    }

    fn avg_score(&self) -> f64 {
        if self.visits == 0 { f64::NEG_INFINITY }
        else { self.score_sum / self.visits as f64 }
    }

    /// UCB1 selection: balance exploitation (high avg score) with exploration (low visits)
    fn ucb1(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY; // always explore unvisited nodes
        }
        self.avg_score() + exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt()
    }
}

/// Result from MCTS search
pub struct MctsResult {
    pub score: i64,
    pub positions: Vec<(char, usize)>,
}

/// MCTS search state
pub struct MctsSearch {
    layout: Layout,
    data: Data,
    weights: Weights,
    /// Keys sorted by frequency (CacheKey)
    keys_by_freq: Vec<CacheKey>,
    /// Characters sorted by frequency
    chars_by_freq: Vec<char>,
    num_positions: usize,
    /// Top-K best complete layouts found
    best_layouts: Vec<(i64, Vec<usize>)>,
    top_k: usize,
    /// Total rollouts performed
    total_rollouts: u64,
}

impl MctsSearch {
    pub fn new(layout: Layout, data: Data, weights: Weights, top_k: usize) -> Self {
        let num_positions = layout.keyboard.len();

        let mut char_freqs: Vec<(char, f64)> = data.chars.iter()
            .map(|(&c, &f)| (c, f))
            .collect();
        char_freqs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let chars_by_freq: Vec<char> = char_freqs.into_iter()
            .take(num_positions)
            .map(|(c, _)| c)
            .collect();

        Self {
            layout,
            data,
            weights,
            keys_by_freq: Vec::new(), // populated on first use
            chars_by_freq,
            num_positions,
            best_layouts: Vec::new(),
            top_k,
            total_rollouts: 0,
        }
    }

    /// Run MCTS for a given number of iterations with SA rollouts.
    pub fn search(
        &mut self,
        iterations: u64,
        exploration_constant: f64,
        sa_iterations: usize,
        mut progress: impl FnMut(u64, u64, i64, f64),
    ) {
        let mut cache = CachedLayout::new(&self.layout, self.data.clone(), &self.weights);

        // Initialize keys_by_freq
        if self.keys_by_freq.is_empty() {
            self.keys_by_freq = self.chars_by_freq.iter()
                .map(|&c| cache.char_mapping().get_u(c))
                .collect();
        }

        // Clear cache to empty
        for pos in 0..self.num_positions {
            let k = cache.get_key(pos);
            if k != EMPTY_KEY {
                cache.replace_key_fast(pos, k, EMPTY_KEY);
            }
        }

        let mut root = MctsNode::new(usize::MAX); // root has no position
        let num_keys = self.keys_by_freq.len().min(self.num_positions);
        let mut rng = nanorand::WyRand::new();

        for iter in 0..iterations {
            // Path from root to the selected leaf
            let mut path: Vec<usize> = Vec::with_capacity(num_keys);

            // SELECT: walk down tree using UCB1
            let mut node = &mut root;
            let mut depth = 0;

            // Place keys along the selected path
            while node.expanded && !node.children.is_empty() && depth < num_keys {
                // Pick child with best UCB1
                let parent_visits = node.visits;
                let best_child_idx = node.children.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.ucb1(parent_visits, exploration_constant)
                            .partial_cmp(&b.ucb1(parent_visits, exploration_constant))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap();

                let child = &node.children[best_child_idx];
                let pos = child.position;
                let key = self.keys_by_freq[depth];
                cache.replace_key_fast(pos, EMPTY_KEY, key);
                path.push(pos);

                node = &mut node.children[best_child_idx];
                depth += 1;
            }

            // EXPAND: if this node hasn't been expanded, create children
            if !node.expanded && depth < num_keys {
                let mut occ = vec![false; self.num_positions];
                for &p in &path { occ[p] = true; }

                for pos in 0..self.num_positions {
                    if !occ[pos] {
                        node.children.push(MctsNode::new(pos));
                    }
                }
                node.expanded = true;

                // Select first unvisited child
                if !node.children.is_empty() {
                    let child = &node.children[0];
                    let pos = child.position;
                    let key = self.keys_by_freq[depth];
                    cache.replace_key_fast(pos, EMPTY_KEY, key);
                    path.push(pos);
                    depth += 1;
                }
            }

            // ROLLOUT: random placement + SA polish directly on cache
            let mut rollout_positions: Vec<usize> = Vec::new();
            let mut occupied = vec![false; self.num_positions];
            for &p in &path { occupied[p] = true; }

            // Randomly place remaining keys
            let mut avail: Vec<usize> = (0..self.num_positions).filter(|&p| !occupied[p]).collect();

            for d in depth..num_keys {
                let key = self.keys_by_freq[d];
                let nk = cache.trigram_num_keys();
                if key >= nk || avail.is_empty() { continue; }

                let idx = rng.generate_range(0..avail.len());
                let pos = avail.swap_remove(idx);
                cache.replace_key_fast(pos, EMPTY_KEY, key);
                rollout_positions.push(pos);
            }

            // SA polish: swap random unpinned key pairs using swap_keys_only + compute_score
            // Only swap among rollout positions (MCTS path positions are pinned)
            if rollout_positions.len() >= 2 && sa_iterations > 0 {
                let mut current_score = cache.compute_score();
                for _ in 0..sa_iterations {
                    let a = rng.generate_range(0..rollout_positions.len());
                    let mut b = rng.generate_range(0..rollout_positions.len() - 1);
                    if b >= a { b += 1; }

                    let pos_a = rollout_positions[a];
                    let pos_b = rollout_positions[b];

                    cache.swap_keys_only(pos_a, pos_b);
                    let new_score = cache.compute_score();

                    if new_score > current_score {
                        current_score = new_score;
                    } else {
                        cache.swap_keys_only(pos_a, pos_b); // revert
                    }
                }
            }

            let final_score = cache.compute_score();
            self.total_rollouts += 1;

            let mut full_positions = path.clone();
            full_positions.extend_from_slice(&rollout_positions);
            self.record_layout(final_score, &full_positions);

            // Undo rollout
            for (i, &pos) in rollout_positions.iter().enumerate().rev() {
                let key = self.keys_by_freq[depth + i];
                cache.replace_key_fast(pos, key, EMPTY_KEY);
            }

            // BACKPROPAGATE: update scores along the path
            let score_f64 = final_score as f64;
            // Walk back up — we need to update the nodes along the path
            // Since we can't easily walk back up the tree with mutable refs,
            // we'll do it by re-traversing from root
            Self::backpropagate(&mut root, &path, score_f64, final_score);

            // Undo the path placements
            for (d, &pos) in path.iter().enumerate().rev() {
                let key = self.keys_by_freq[d];
                cache.replace_key_fast(pos, key, EMPTY_KEY);
            }

            // Progress callback
            if (iter + 1) % 10 == 0 {
                let best = self.best_layouts.first().map(|(s, _)| *s).unwrap_or(i64::MIN);
                let avg = root.avg_score();
                progress(iter + 1, self.total_rollouts, best, avg);
            }
        }
    }

    fn backpropagate(node: &mut MctsNode, path: &[usize], score: f64, raw_score: i64) {
        node.visits += 1;
        node.score_sum += score;
        if raw_score > node.best_score {
            node.best_score = raw_score;
        }

        if !path.is_empty() {
            let target_pos = path[0];
            if let Some(child) = node.children.iter_mut().find(|c| c.position == target_pos) {
                Self::backpropagate(child, &path[1..], score, raw_score);
            }
        }
    }

    fn record_layout(&mut self, score: i64, positions: &[usize]) {
        if self.best_layouts.len() < self.top_k {
            self.best_layouts.push((score, positions.to_vec()));
            self.best_layouts.sort_by(|a, b| b.0.cmp(&a.0));
        } else if score > self.best_layouts.last().unwrap().0 {
            self.best_layouts.pop();
            self.best_layouts.push((score, positions.to_vec()));
            self.best_layouts.sort_by(|a, b| b.0.cmp(&a.0));
        }
    }

    pub fn best_score(&self) -> Option<i64> {
        self.best_layouts.first().map(|(s, _)| *s)
    }

    pub fn total_rollouts(&self) -> u64 {
        self.total_rollouts
    }

    pub fn results(&self) -> Vec<MctsResult> {
        self.best_layouts.iter().map(|(score, positions)| {
            let assignment: Vec<(char, usize)> = positions.iter().enumerate()
                .map(|(d, &pos)| (self.chars_by_freq[d], pos))
                .collect();
            MctsResult { score: *score, positions: assignment }
        }).collect()
    }
}
