//! Branch and Bound Layout Optimizer
//!
//! Explores the layout search space as a tree where each level corresponds to
//! placing the next most frequent character. Uses pruning based on a known
//! "good" score (e.g., from simulated annealing) to cut branches early.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::{
    cached_layout::CachedLayout,
    data::Data,
    layout::Layout,
    weights::Weights,
    types::CacheKey,
};

/// A complete layout solution with its score
#[derive(Debug, Clone)]
pub struct ScoredLayout {
    pub score: i64,
    pub key_positions: Vec<(char, usize)>, // (character, position)
}

/// Progress information for branch and bound search
#[derive(Debug, Clone)]
pub struct BranchBoundProgress {
    /// Nodes (tree positions) we've visited
    pub nodes_visited: u64,
    /// Nodes we've pruned (skipped subtrees) - f64 to handle large numbers
    pub nodes_pruned: f64,
    /// Complete layouts (leaves) we've evaluated
    pub layouts_evaluated: f64,
    /// Complete layouts (leaves) we've pruned
    pub layouts_pruned: f64,
    /// Estimated total layouts (leaves) in search space
    pub estimated_total_layouts: f64,
    /// Estimated total nodes in search space
    pub estimated_total_nodes: f64,
    /// Number of prune events
    pub prune_count: u64,
    /// Sum of depths at which we pruned (for average)
    pub prune_depth_sum: u64,
    /// Weighted sum: depth * layouts_pruned_at_that_depth
    pub weighted_prune_depth_sum: f64,
    /// Current depth in tree
    pub current_depth: usize,
    /// Maximum search depth
    pub max_depth: usize,
    /// Best score found so far
    pub best_score: Option<i64>,
    /// Solutions found
    pub solutions_found: f64,
}

impl Eq for ScoredLayout {}

impl PartialEq for ScoredLayout {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl PartialOrd for ScoredLayout {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredLayout {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score is better, so reverse ordering for max-heap behavior
        self.score.cmp(&other.score)
    }
}

/// Statistics from the branch and bound search
#[derive(Debug, Clone, Default)]
pub struct BranchBoundStats {
    /// Nodes (tree positions) we've visited
    pub nodes_visited: u64,
    /// Nodes we've pruned (skipped subtrees) - f64 to handle large numbers
    pub nodes_pruned: f64,
    /// Complete layouts (leaves) we've evaluated
    pub layouts_evaluated: f64,
    /// Complete layouts (leaves) we've pruned
    pub layouts_pruned: f64,
    /// Solutions found
    pub solutions_found: f64,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Number of prune events
    pub prune_count: u64,
    /// Sum of depths at which we pruned (for average)
    pub prune_depth_sum: u64,
    /// Weighted sum: depth * layouts_pruned_at_that_depth
    pub weighted_prune_depth_sum: f64,
}

/// Maintains the top K best layouts found
pub struct TopKHeap {
    heap: BinaryHeap<std::cmp::Reverse<ScoredLayout>>,
    capacity: usize,
}


impl TopKHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity + 1),
            capacity,
        }
    }

    /// Try to insert a layout. Returns true if inserted.
    pub fn try_insert(&mut self, layout: ScoredLayout) -> bool {
        if self.heap.len() < self.capacity {
            self.heap.push(std::cmp::Reverse(layout));
            true
        } else if let Some(worst) = self.heap.peek() {
            if layout.score > worst.0.score {
                self.heap.pop();
                self.heap.push(std::cmp::Reverse(layout));
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get the worst score in the heap (used for pruning bound)
    pub fn worst_score(&self) -> Option<i64> {
        self.heap.peek().map(|r| r.0.score)
    }

    /// Get the best score in the heap
    pub fn best_score(&self) -> Option<i64> {
        self.heap.iter().map(|r| r.0.score).max()
    }

    /// Convert to sorted vec (best first)
    pub fn into_sorted_vec(self) -> Vec<ScoredLayout> {
        let mut v: Vec<_> = self.heap.into_iter().map(|r| r.0).collect();
        v.sort_by(|a, b| b.score.cmp(&a.score));
        v
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Branch and Bound optimizer
pub struct BranchBound {
    base_layout: Layout,
    data: Data,
    weights: Weights,
    /// Characters sorted by frequency (highest first)
    chars_by_freq: Vec<char>,
    /// CacheKeys for chars_by_freq
    keys_by_freq: Vec<CacheKey>,
    num_positions: usize,
    /// Positions ordered by finger (fill one finger before the next, adjacent fingers next)
    positions_by_finger: Vec<usize>,
    /// All available keys sorted by frequency (for position-first search)
    all_keys_by_freq: Vec<CacheKey>,
    /// All available chars sorted by frequency
    all_chars_by_freq: Vec<char>,
}


impl BranchBound {
    pub fn new(base_layout: Layout, data: Data, weights: Weights) -> Self {
        let num_positions = base_layout.keyboard.len();

        // Sort characters by frequency (descending)
        let mut char_freqs: Vec<(char, f64)> = data.chars.iter()
            .map(|(&c, &f)| (c, f))
            .collect();
        char_freqs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Only keep as many chars as we have positions
        let chars_by_freq: Vec<char> = char_freqs.into_iter()
            .take(num_positions)
            .map(|(c, _)| c)
            .collect();

        // Compute positions ordered by finger.
        // Strategy: start from the center (index fingers), work outward.
        // Within each finger, order by row (home row first).
        // Adjacent fingers are next to maximize early interactions.
        //
        // Finger order (center-out): LI, RI, LM, RM, LR, RR, LP, RP, LT, RT
        use libdof::dofinitions::Finger::*;
        let finger_order = [LI, RI, LM, RM, LR, RR, LP, RP, LT, RT];

        let mut positions_by_finger = Vec::with_capacity(num_positions);
        for &finger in &finger_order {
            let mut finger_positions: Vec<(usize, f64)> = base_layout.fingers.iter()
                .enumerate()
                .filter(|(_, &f)| f == finger)
                .map(|(pos, _)| {
                    // Sort within finger: home row (row 1) first, then row 0, then row 2
                    let row = base_layout.keyboard[pos].y();
                    let row_priority = if (row - 1.0).abs() < 0.5 { 0.0 } // home row
                        else if row < 1.0 { 1.0 } // top row
                        else { 2.0 }; // bottom row
                    (pos, row_priority)
                })
                .collect();
            finger_positions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            positions_by_finger.extend(finger_positions.iter().map(|(pos, _)| *pos));
        }

        Self {
            base_layout,
            data,
            weights,
            chars_by_freq: chars_by_freq.clone(),
            keys_by_freq: Vec::new(),
            num_positions,
            positions_by_finger,
            all_keys_by_freq: Vec::new(),
            all_chars_by_freq: chars_by_freq,
        }
    }

    /// Create an empty CachedLayout and initialize keys_by_freq
    pub fn create_empty_cache(&mut self) -> CachedLayout {
        use crate::cached_layout::EMPTY_KEY;

        let mut cache = CachedLayout::new(&self.base_layout, self.data.clone(), &self.weights);

        // Initialize keys_by_freq from the cache's char mapping
        if self.keys_by_freq.is_empty() {
            self.keys_by_freq = self.chars_by_freq.iter()
                .map(|&c| cache.char_mapping().get_u(c))
                .collect();
        }

        // Initialize all_keys_by_freq (same keys, for position-first search)
        if self.all_keys_by_freq.is_empty() {
            self.all_keys_by_freq = self.all_chars_by_freq.iter()
                .map(|&c| cache.char_mapping().get_u(c))
                .collect();
        }

        // Remove all keys to get empty positions
        for pos in 0..self.num_positions {
            let key = cache.get_key(pos);
            cache.replace_key_no_update(pos, key, EMPTY_KEY);
        }
        cache
    }

    /// Get characters sorted by frequency
    pub fn chars_by_frequency(&self) -> &[char] {
        &self.chars_by_freq
    }


    /// Run branch and bound search
    ///
    /// # Arguments
    /// * `bound` - Initial bound (e.g., from simulated annealing). Branches with
    ///             scores worse than this are pruned.
    /// * `top_k` - Number of best layouts to keep
    ///
    /// # Returns
    /// The top K layouts found and search statistics
    pub fn search(&mut self, bound: i64, top_k: usize) -> (Vec<ScoredLayout>, BranchBoundStats) {
        let mut cache = self.create_empty_cache();
        let mut top_layouts = TopKHeap::new(top_k);
        let mut stats = BranchBoundStats::default();
        let mut available: Vec<usize> = (0..self.num_positions).collect();
        let mut assignment: Vec<(char, usize)> = Vec::with_capacity(self.num_positions);

        self.search_recursive(
            &mut cache, 0, &mut available, &mut assignment,
            bound, &mut top_layouts, &mut stats,
        );

        (top_layouts.into_sorted_vec(), stats)
    }

    fn search_recursive(
        &self,
        cache: &mut CachedLayout,
        depth: usize,
        available_positions: &mut Vec<usize>,
        assignment: &mut Vec<(char, usize)>,
        bound: i64,
        top_layouts: &mut TopKHeap,
        stats: &mut BranchBoundStats,
    ) {
        self.search_recursive_limited(
            cache, depth, available_positions, assignment,
            bound, top_layouts, stats, usize::MAX,
        );
    }

    fn search_recursive_limited(
        &self,
        cache: &mut CachedLayout,
        depth: usize,
        available_positions: &mut Vec<usize>,
        assignment: &mut Vec<(char, usize)>,
        bound: i64,
        top_layouts: &mut TopKHeap,
        stats: &mut BranchBoundStats,
        max_depth: usize,
    ) {
        use crate::cached_layout::EMPTY_KEY;

        stats.max_depth_reached = stats.max_depth_reached.max(depth);
        stats.nodes_visited += 1;

        let current_score = cache.score();

        // Pruning: if current score is already worse than bound, prune
        if current_score < bound {
            let remaining_levels = max_depth.saturating_sub(depth);
            let leaves_pruned = Self::estimate_leaf_nodes_f64(available_positions.len(), remaining_levels);
            let nodes_pruned = Self::count_subtree_nodes_f64(available_positions.len(), remaining_levels);
            stats.layouts_pruned += leaves_pruned;
            stats.nodes_pruned += nodes_pruned;
            stats.prune_depth_sum += depth as u64;
            stats.prune_count += 1;
            stats.weighted_prune_depth_sum += depth as f64 * leaves_pruned;
            return;
        }

        // Bound tightening: do greedy completion at early depths to find good
        // feasible solutions that tighten the bound for deeper pruning.
        // Throttle: only at depths 1-5, and only every Nth node to control cost.
        if depth >= 1 && depth <= 5 {
            let remaining_keys = &self.keys_by_freq[depth..max_depth.min(self.keys_by_freq.len())];
            if !remaining_keys.is_empty() && !available_positions.is_empty() {
                let greedy_score = cache.greedy_completion_score(remaining_keys, available_positions);
                if greedy_score > bound {
                    if top_layouts.try_insert(crate::branch_bound::ScoredLayout {
                        score: greedy_score,
                        key_positions: assignment.clone(),
                    }) {
                        stats.solutions_found += 1.0;
                        stats.layouts_evaluated += 1.0;
                    }
                }
            }
        }

        // Base case: all characters placed OR depth limit reached
        if depth >= self.keys_by_freq.len() || available_positions.is_empty() || depth >= max_depth {
            stats.layouts_evaluated += 1.0;
            stats.solutions_found += 1.0;
            let solution = ScoredLayout {
                score: current_score,
                key_positions: assignment.clone(),
            };
            top_layouts.try_insert(solution);
            return;
        }

        let key = self.keys_by_freq[depth];
        let c = self.chars_by_freq[depth];
        let num_available = available_positions.len();

        for i in 0..num_available {
            let pos = available_positions[i];

            cache.replace_key_no_update(pos, EMPTY_KEY, key);
            assignment.push((c, pos));
            available_positions.swap_remove(i);

            let effective_bound = if top_layouts.len() >= top_layouts.capacity {
                top_layouts.worst_score().unwrap_or(bound).max(bound)
            } else {
                bound
            };

            self.search_recursive_limited(
                cache, depth + 1, available_positions, assignment,
                effective_bound, top_layouts, stats, max_depth,
            );

            available_positions.push(pos);
            let last = available_positions.len() - 1;
            available_positions.swap(i, last);

            assignment.pop();
            cache.replace_key_no_update(pos, key, EMPTY_KEY);
        }
    }

    /// Run depth-limited branch and bound search
    ///
    /// This is useful for testing and for exploring partial solutions.
    ///
    /// # Arguments
    /// * `bound` - Initial bound for pruning
    /// * `top_k` - Number of best layouts to keep
    /// * `max_depth` - Maximum number of characters to place
    pub fn search_limited(&mut self, bound: i64, top_k: usize, max_depth: usize) -> (Vec<ScoredLayout>, BranchBoundStats) {
        self.search_limited_with_progress(bound, top_k, max_depth, |_| {})
    }

    /// Run depth-limited search with progress callback
    ///
    /// The callback is called periodically with progress information.
    pub fn search_limited_with_progress<F>(
        &mut self,
        bound: i64,
        top_k: usize,
        max_depth: usize,
        mut progress_callback: F,
    ) -> (Vec<ScoredLayout>, BranchBoundStats)
    where
        F: FnMut(&BranchBoundProgress),
    {
        let mut cache = self.create_empty_cache();
        let mut top_layouts = TopKHeap::new(top_k);
        let mut stats = BranchBoundStats::default();

        let mut available: Vec<usize> = (0..self.num_positions).collect();
        let mut assignment: Vec<(char, usize)> = Vec::with_capacity(max_depth);

        let estimated_total_layouts = Self::estimate_leaf_nodes_f64(self.num_positions, max_depth);
        let estimated_total_nodes = Self::estimate_total_nodes_f64(self.num_positions, max_depth);

        self.search_recursive_with_progress(
            &mut cache, 0, &mut available, &mut assignment,
            bound, &mut top_layouts, &mut stats, max_depth,
            estimated_total_layouts, estimated_total_nodes, &mut progress_callback,
        );

        (top_layouts.into_sorted_vec(), stats)
    }

    /// Estimate leaf nodes at given depth: n * (n-1) * ... * (n-depth+1)
    /// This is P(n, depth) = n! / (n-depth)!
    fn estimate_leaf_nodes_f64(n: usize, depth: usize) -> f64 {
        let depth = depth.min(n);
        (0..depth).fold(1.0f64, |acc, i| acc * (n - i) as f64)
    }

    /// Estimate total nodes in search tree (all levels): 1 + n + n*(n-1) + ... + P(n,depth)
    fn estimate_total_nodes_f64(n: usize, depth: usize) -> f64 {
        let depth = depth.min(n);
        let mut total = 0.0f64;
        let mut level_nodes = 1.0f64;
        for i in 0..=depth {
            total += level_nodes;
            if i < depth {
                level_nodes *= (n - i) as f64;
            }
        }
        total
    }

    /// Count nodes in subtree rooted at current position (excluding current node)
    /// available = number of available positions, remaining_levels = levels below current
    fn count_subtree_nodes_f64(available: usize, remaining_levels: usize) -> f64 {
        if remaining_levels == 0 || available == 0 {
            return 0.0;
        }
        // Sum of nodes at each level below: available + available*(available-1) + ...
        let mut total = 0.0f64;
        let mut level_nodes = available as f64;
        for i in 0..remaining_levels {
            total += level_nodes;
            if i + 1 < remaining_levels && available > i + 1 {
                level_nodes *= (available - i - 1) as f64;
            }
        }
        total
    }

    fn search_recursive_with_progress<F>(
        &self,
        cache: &mut CachedLayout,
        depth: usize,
        available_positions: &mut Vec<usize>,
        assignment: &mut Vec<(char, usize)>,
        bound: i64,
        top_layouts: &mut TopKHeap,
        stats: &mut BranchBoundStats,
        max_depth: usize,
        estimated_total_layouts: f64,
        estimated_total_nodes: f64,
        progress_callback: &mut F,
    )
    where
        F: FnMut(&BranchBoundProgress),
    {
        use crate::cached_layout::EMPTY_KEY;

        stats.max_depth_reached = stats.max_depth_reached.max(depth);
        stats.nodes_visited += 1;

        let current_score = cache.score();

        // Pruning: if current score is already worse than bound, prune
        if current_score < bound {
            let remaining_levels = max_depth.saturating_sub(depth);
            let leaves_pruned = Self::estimate_leaf_nodes_f64(available_positions.len(), remaining_levels);
            let nodes_pruned = Self::count_subtree_nodes_f64(available_positions.len(), remaining_levels);
            stats.layouts_pruned += leaves_pruned;
            stats.nodes_pruned += nodes_pruned;
            stats.prune_depth_sum += depth as u64;
            stats.prune_count += 1;
            stats.weighted_prune_depth_sum += depth as f64 * leaves_pruned;

            let progress = BranchBoundProgress {
                nodes_visited: stats.nodes_visited,
                nodes_pruned: stats.nodes_pruned,
                layouts_evaluated: stats.layouts_evaluated,
                layouts_pruned: stats.layouts_pruned,
                estimated_total_layouts,
                estimated_total_nodes,
                prune_count: stats.prune_count,
                prune_depth_sum: stats.prune_depth_sum,
                weighted_prune_depth_sum: stats.weighted_prune_depth_sum,
                current_depth: depth,
                max_depth,
                best_score: top_layouts.best_score(),
                solutions_found: stats.solutions_found,
            };
            progress_callback(&progress);
            return;
        }

        // Bound tightening via greedy completion at early depths
        if depth >= 1 && depth <= 5 {
            let remaining_keys = &self.keys_by_freq[depth..max_depth.min(self.keys_by_freq.len())];
            if !remaining_keys.is_empty() && !available_positions.is_empty() {
                let greedy_score = cache.greedy_completion_score(remaining_keys, available_positions);
                if greedy_score > bound {
                    if top_layouts.try_insert(crate::branch_bound::ScoredLayout {
                        score: greedy_score,
                        key_positions: assignment.clone(),
                    }) {
                        stats.solutions_found += 1.0;
                        stats.layouts_evaluated += 1.0;
                    }
                }
            }
        }

        if depth >= self.keys_by_freq.len() || available_positions.is_empty() || depth >= max_depth {
            stats.layouts_evaluated += 1.0;
            stats.solutions_found += 1.0;
            let solution = ScoredLayout {
                score: current_score,
                key_positions: assignment.clone(),
            };
            top_layouts.try_insert(solution);
            return;
        }

        let key = self.keys_by_freq[depth];
        let c = self.chars_by_freq[depth];
        let num_available = available_positions.len();

        for i in 0..num_available {
            let pos = available_positions[i];

            cache.replace_key_no_update(pos, EMPTY_KEY, key);
            assignment.push((c, pos));
            available_positions.swap_remove(i);

            let effective_bound = if top_layouts.len() >= top_layouts.capacity {
                top_layouts.worst_score().unwrap_or(bound).max(bound)
            } else {
                bound
            };

            self.search_recursive_with_progress(
                cache, depth + 1, available_positions, assignment,
                effective_bound, top_layouts, stats, max_depth,
                estimated_total_layouts, estimated_total_nodes, progress_callback,
            );

            available_positions.push(pos);
            let last = available_positions.len() - 1;
            available_positions.swap(i, last);

            assignment.pop();
            cache.replace_key_no_update(pos, key, EMPTY_KEY);
        }
    }

    /// Get the number of positions in the layout
    pub fn num_positions(&self) -> usize {
        self.num_positions
    }

    // ==================== Beam Search ====================

    /// Beam search: at each depth, keep the top-K partial layouts.
    /// Places keys in frequency order (most frequent first).
    /// Uses a single CachedLayout, reconstructing state incrementally.
    pub fn beam_search(&mut self, beam_width: usize) -> Vec<ScoredLayout> {
        self.beam_search_with_interval(beam_width, 1)
    }

    /// Beam search with configurable prune interval.
    /// Prunes to beam_width every `prune_interval` depths.
    /// Between prunes, the beam grows freely (more diversity).
    pub fn beam_search_with_interval(&mut self, beam_width: usize, prune_interval: usize) -> Vec<ScoredLayout> {
        use crate::cached_layout::EMPTY_KEY;

        let mut cache = self.create_empty_cache();
        let num_keys = self.keys_by_freq.len().min(self.num_positions);
        let prune_interval = prune_interval.max(1);

        let mut beam: Vec<Vec<usize>> = vec![vec![]];

        for depth in 0..num_keys {
            let key = self.keys_by_freq[depth];

            let mut candidates: Vec<(i64, Vec<usize>)> = Vec::with_capacity(beam.len() * (self.num_positions - depth));

            beam.sort();
            let mut prev_positions: Vec<usize> = Vec::new();

            for positions in &beam {
                let shared = prev_positions.iter().zip(positions.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                for d in (shared..prev_positions.len()).rev() {
                    cache.replace_key_no_update(prev_positions[d], self.keys_by_freq[d], EMPTY_KEY);
                }
                for d in shared..positions.len() {
                    cache.replace_key_no_update(positions[d], EMPTY_KEY, self.keys_by_freq[d]);
                }
                prev_positions = positions.clone();

                let mut occupied = vec![false; self.num_positions];
                for &p in positions { occupied[p] = true; }

                for pos in 0..self.num_positions {
                    if occupied[pos] { continue; }

                    cache.replace_key_no_update(pos, EMPTY_KEY, key);
                    let score = cache.score();
                    cache.replace_key_no_update(pos, key, EMPTY_KEY);

                    let mut new_positions = positions.clone();
                    new_positions.push(pos);
                    candidates.push((score, new_positions));
                }
            }

            for d in (0..prev_positions.len()).rev() {
                cache.replace_key_no_update(prev_positions[d], self.keys_by_freq[d], EMPTY_KEY);
            }

            // Prune at interval or at last depth
            if (depth + 1) % prune_interval == 0 || depth == num_keys - 1 {
                candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                candidates.truncate(beam_width);
            }

            beam = candidates.into_iter().map(|(_, positions)| positions).collect();
            if beam.is_empty() { break; }
        }

        beam.into_iter().map(|positions| {
            for (d, &pos) in positions.iter().enumerate() {
                cache.replace_key_no_update(pos, EMPTY_KEY, self.keys_by_freq[d]);
            }
            let score = cache.score();
            for d in (0..positions.len()).rev() {
                cache.replace_key_no_update(positions[d], self.keys_by_freq[d], EMPTY_KEY);
            }

            let assignment: Vec<(char, usize)> = positions.iter().enumerate()
                .map(|(d, &pos)| (self.chars_by_freq[d], pos))
                .collect();

            ScoredLayout { score, key_positions: assignment }
        }).collect()
    }

    // ==================== BB3: Hybrid key-freq + finger-fill ====================

    /// Search by placing the most frequent key first (at any position),
    /// then filling the rest of that finger, then the next most frequent key, etc.
    pub fn search_hybrid(
        &mut self,
        bound: i64,
        top_k: usize,
        mut progress_callback: impl FnMut(&BranchBoundProgress),
    ) -> (Vec<ScoredLayout>, BranchBoundStats) {
        let mut cache = self.create_empty_cache();
        let mut top_layouts = TopKHeap::new(top_k);
        let mut stats = BranchBoundStats::default();
        let mut assignment: Vec<(char, usize)> = Vec::with_capacity(self.num_positions);

        // Build the finger map: for each position, which other positions share its finger
        let mut finger_map: Vec<Vec<usize>> = vec![Vec::new(); self.num_positions];
        for i in 0..self.num_positions {
            for j in 0..self.num_positions {
                if i != j && self.base_layout.fingers[i] == self.base_layout.fingers[j] {
                    finger_map[i].push(j);
                }
            }
        }

        // Available keys (sorted by frequency, highest first)
        let mut available_keys: Vec<(CacheKey, char)> = self.all_keys_by_freq.iter()
            .zip(self.all_chars_by_freq.iter())
            .map(|(&k, &c)| (k, c))
            .collect();

        // Available positions
        let mut available_positions: Vec<usize> = (0..self.num_positions).collect();

        // Pending finger positions to fill (positions on the same finger as the last placed key)
        let mut pending_finger_positions: Vec<usize> = Vec::new();

        let estimated_total = Self::estimate_total_nodes_f64(self.num_positions, self.num_positions);

        self.search_hybrid_recursive(
            &mut cache,
            0,
            &mut available_keys,
            &mut available_positions,
            &mut pending_finger_positions,
            &finger_map,
            &mut assignment,
            bound,
            &mut top_layouts,
            &mut stats,
            estimated_total,
            &mut progress_callback,
        );

        (top_layouts.into_sorted_vec(), stats)
    }

    fn search_hybrid_recursive(
        &self,
        cache: &mut CachedLayout,
        depth: usize,
        available_keys: &mut Vec<(CacheKey, char)>,
        available_positions: &mut Vec<usize>,
        pending_finger_positions: &mut Vec<usize>,
        finger_map: &[Vec<usize>],
        assignment: &mut Vec<(char, usize)>,
        bound: i64,
        top_layouts: &mut TopKHeap,
        stats: &mut BranchBoundStats,
        estimated_total: f64,
        progress_callback: &mut impl FnMut(&BranchBoundProgress),
    ) {
        use crate::cached_layout::EMPTY_KEY;

        stats.max_depth_reached = stats.max_depth_reached.max(depth);
        stats.nodes_visited += 1;

        let current_score = cache.score();

        // Pruning
        if current_score < bound {
            let remaining = available_keys.len();
            let leaves_pruned = Self::estimate_leaf_nodes_f64(remaining, remaining);
            let nodes_pruned = Self::count_subtree_nodes_f64(remaining, remaining);
            stats.layouts_pruned += leaves_pruned;
            stats.nodes_pruned += nodes_pruned;
            stats.prune_depth_sum += depth as u64;
            stats.prune_count += 1;
            stats.weighted_prune_depth_sum += depth as f64 * leaves_pruned;

            if stats.nodes_visited % 100_000 == 0 {
                let progress = BranchBoundProgress {
                    nodes_visited: stats.nodes_visited,
                    nodes_pruned: stats.nodes_pruned,
                    layouts_evaluated: stats.layouts_evaluated,
                    layouts_pruned: stats.layouts_pruned,
                    estimated_total_layouts: estimated_total,
                    estimated_total_nodes: estimated_total,
                    prune_count: stats.prune_count,
                    prune_depth_sum: stats.prune_depth_sum,
                    weighted_prune_depth_sum: stats.weighted_prune_depth_sum,
                    current_depth: depth,
                    max_depth: self.num_positions,
                    best_score: top_layouts.best_score(),
                    solutions_found: stats.solutions_found,
                };
                progress_callback(&progress);
            }
            return;
        }

        // Base case
        if available_keys.is_empty() || available_positions.is_empty() {
            stats.layouts_evaluated += 1.0;
            stats.solutions_found += 1.0;
            top_layouts.try_insert(ScoredLayout {
                score: current_score,
                key_positions: assignment.clone(),
            });
            return;
        }

        if !pending_finger_positions.is_empty() {
            // MODE: Fill the current finger — pick a position from pending, try all keys
            let pos = pending_finger_positions.pop().unwrap();

            let num_keys = available_keys.len();
            for i in 0..num_keys {
                let (key, c) = available_keys[i];

                cache.replace_key_no_update(pos, EMPTY_KEY, key);
                assignment.push((c, pos));
                available_keys.swap_remove(i);

                let effective_bound = if top_layouts.len() >= top_layouts.capacity {
                    top_layouts.worst_score().unwrap_or(bound).max(bound)
                } else { bound };

                self.search_hybrid_recursive(
                    cache, depth + 1, available_keys, available_positions,
                    pending_finger_positions, finger_map, assignment,
                    effective_bound, top_layouts, stats, estimated_total, progress_callback,
                );

                available_keys.push((key, c));
                let last = available_keys.len() - 1;
                available_keys.swap(i, last);
                assignment.pop();
                cache.replace_key_no_update(pos, key, EMPTY_KEY);
            }

            // Restore the pending position
            pending_finger_positions.push(pos);
        } else {
            // MODE: Place the next most frequent key — try all available positions
            // The first key in available_keys is the most frequent remaining
            let (key, c) = available_keys[0];
            available_keys.swap_remove(0);

            let num_positions = available_positions.len();
            for i in 0..num_positions {
                let pos = available_positions[i];

                cache.replace_key_no_update(pos, EMPTY_KEY, key);
                assignment.push((c, pos));
                available_positions.swap_remove(i);

                // Queue up the other positions on this finger for filling
                let same_finger: Vec<usize> = finger_map[pos].iter()
                    .filter(|&&p| available_positions.contains(&p))
                    .copied()
                    .collect();
                // Remove them from available_positions and add to pending
                for &fp in &same_finger {
                    if let Some(idx) = available_positions.iter().position(|&p| p == fp) {
                        available_positions.swap_remove(idx);
                        pending_finger_positions.push(fp);
                    }
                }

                let effective_bound = if top_layouts.len() >= top_layouts.capacity {
                    top_layouts.worst_score().unwrap_or(bound).max(bound)
                } else { bound };

                self.search_hybrid_recursive(
                    cache, depth + 1, available_keys, available_positions,
                    pending_finger_positions, finger_map, assignment,
                    effective_bound, top_layouts, stats, estimated_total, progress_callback,
                );

                // Restore: move pending finger positions back to available
                for &fp in &same_finger {
                    if let Some(idx) = pending_finger_positions.iter().position(|&p| p == fp) {
                        pending_finger_positions.swap_remove(idx);
                        available_positions.push(fp);
                    }
                }

                available_positions.push(pos);
                let last = available_positions.len() - 1;
                available_positions.swap(i, last);
                assignment.pop();
                cache.replace_key_no_update(pos, key, EMPTY_KEY);
            }

            // Restore the key
            available_keys.push((key, c));
            let last = available_keys.len() - 1;
            available_keys.swap(0, last);
        }
    }

    /// Get the finger-ordered positions
    pub fn positions_by_finger(&self) -> &[usize] {
        &self.positions_by_finger
    }

    // ==================== Position-First Search ====================

    /// Search by filling positions in finger order, trying all available keys at each position.
    /// This realizes SFB/stretch/scissor penalties earlier, enabling deeper pruning.
    pub fn search_position_first(
        &mut self,
        bound: i64,
        top_k: usize,
        progress_callback: impl FnMut(&BranchBoundProgress),
    ) -> (Vec<ScoredLayout>, BranchBoundStats) {
        let mut cache = self.create_empty_cache();
        let mut top_layouts = TopKHeap::new(top_k);
        let mut stats = BranchBoundStats::default();
        let mut assignment: Vec<(char, usize)> = Vec::with_capacity(self.num_positions);

        // Available keys (all keys initially)
        let mut available_keys: Vec<(CacheKey, char)> = self.all_keys_by_freq.iter()
            .zip(self.all_chars_by_freq.iter())
            .map(|(&k, &c)| (k, c))
            .collect();

        let max_depth = self.positions_by_finger.len();
        let estimated_total_layouts = Self::estimate_leaf_nodes_f64(self.num_positions, max_depth);
        let estimated_total_nodes = Self::estimate_total_nodes_f64(self.num_positions, max_depth);

        let mut progress_callback = progress_callback;

        self.search_position_first_recursive(
            &mut cache,
            0,
            &mut available_keys,
            &mut assignment,
            bound,
            &mut top_layouts,
            &mut stats,
            estimated_total_layouts,
            estimated_total_nodes,
            &mut progress_callback,
        );

        (top_layouts.into_sorted_vec(), stats)
    }

    fn search_position_first_recursive(
        &self,
        cache: &mut CachedLayout,
        depth: usize,
        available_keys: &mut Vec<(CacheKey, char)>,
        assignment: &mut Vec<(char, usize)>,
        bound: i64,
        top_layouts: &mut TopKHeap,
        stats: &mut BranchBoundStats,
        estimated_total_layouts: f64,
        estimated_total_nodes: f64,
        progress_callback: &mut impl FnMut(&BranchBoundProgress),
    ) {
        use crate::cached_layout::EMPTY_KEY;

        stats.max_depth_reached = stats.max_depth_reached.max(depth);
        stats.nodes_visited += 1;

        let current_score = cache.score();

        // Pruning: if current score is already worse than bound, prune
        if current_score < bound {
            let remaining_levels = self.positions_by_finger.len().saturating_sub(depth);
            let leaves_pruned = Self::estimate_leaf_nodes_f64(available_keys.len(), remaining_levels);
            let nodes_pruned = Self::count_subtree_nodes_f64(available_keys.len(), remaining_levels);
            stats.layouts_pruned += leaves_pruned;
            stats.nodes_pruned += nodes_pruned;
            stats.prune_depth_sum += depth as u64;
            stats.prune_count += 1;
            stats.weighted_prune_depth_sum += depth as f64 * leaves_pruned;

            if stats.nodes_visited % 100_000 == 0 {
                let progress = BranchBoundProgress {
                    nodes_visited: stats.nodes_visited,
                    nodes_pruned: stats.nodes_pruned,
                    layouts_evaluated: stats.layouts_evaluated,
                    layouts_pruned: stats.layouts_pruned,
                    estimated_total_layouts,
                    estimated_total_nodes,
                    prune_count: stats.prune_count,
                    prune_depth_sum: stats.prune_depth_sum,
                    weighted_prune_depth_sum: stats.weighted_prune_depth_sum,
                    current_depth: depth,
                    max_depth: self.positions_by_finger.len(),
                    best_score: top_layouts.best_score(),
                    solutions_found: stats.solutions_found,
                };
                progress_callback(&progress);
            }
            return;
        }

        // Base case: all positions filled
        if depth >= self.positions_by_finger.len() || available_keys.is_empty() {
            stats.layouts_evaluated += 1.0;
            stats.solutions_found += 1.0;
            let solution = ScoredLayout {
                score: current_score,
                key_positions: assignment.clone(),
            };
            top_layouts.try_insert(solution);
            return;
        }

        // The position to fill at this depth
        let pos = self.positions_by_finger[depth];
        let num_available = available_keys.len();

        // Try each available key at this position
        for i in 0..num_available {
            let (key, c) = available_keys[i];

            cache.replace_key_no_update(pos, EMPTY_KEY, key);
            assignment.push((c, pos));
            available_keys.swap_remove(i);

            let effective_bound = if top_layouts.len() >= top_layouts.capacity {
                top_layouts.worst_score().unwrap_or(bound).max(bound)
            } else {
                bound
            };

            self.search_position_first_recursive(
                cache, depth + 1, available_keys, assignment,
                effective_bound, top_layouts, stats,
                estimated_total_layouts, estimated_total_nodes, progress_callback,
            );

            // Backtrack
            available_keys.push((key, c));
            let last = available_keys.len() - 1;
            available_keys.swap(i, last);

            assignment.pop();
            cache.replace_key_no_update(pos, key, EMPTY_KEY);
        }
    }
}


impl std::fmt::Display for BranchBoundStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Branch & Bound Statistics")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Nodes visited:     {}", self.nodes_visited)?;
        writeln!(f, "Nodes pruned:      {:.2e}", self.nodes_pruned)?;
        writeln!(f, "Layouts evaluated: {:.0}", self.layouts_evaluated)?;
        writeln!(f, "Layouts pruned:    {:.2e}", self.layouts_pruned)?;
        writeln!(f, "Solutions found:   {:.0}", self.solutions_found)?;
        writeln!(f, "Max depth:         {}", self.max_depth_reached)?;
        if self.prune_count > 0 {
            let avg_prune_depth = self.prune_depth_sum as f64 / self.prune_count as f64;
            writeln!(f, "Avg prune depth:   {:.2} (unweighted)", avg_prune_depth)?;
        }
        if self.layouts_pruned > 0.0 {
            let weighted_avg = self.weighted_prune_depth_sum / self.layouts_pruned;
            writeln!(f, "Effective prune:   {:.2} (weighted by layouts)", weighted_avg)?;
        }
        let total = self.layouts_evaluated + self.layouts_pruned;
        if total > 0.0 {
            let prune_rate = self.layouts_pruned / total * 100.0;
            writeln!(f, "Prune rate:        {:.6}%", prune_rate)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::dummy_weights;

    #[test]
    fn test_top_k_heap() {
        let mut heap = TopKHeap::new(3);

        heap.try_insert(ScoredLayout { score: -100, key_positions: vec![] });
        heap.try_insert(ScoredLayout { score: -50, key_positions: vec![] });
        heap.try_insert(ScoredLayout { score: -200, key_positions: vec![] });

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.worst_score(), Some(-200));
        assert_eq!(heap.best_score(), Some(-50));

        // This should replace -200
        heap.try_insert(ScoredLayout { score: -75, key_positions: vec![] });
        assert_eq!(heap.worst_score(), Some(-100));

        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted[0].score, -50);
        assert_eq!(sorted[1].score, -75);
        assert_eq!(sorted[2].score, -100);
    }

    #[test]
    fn test_branch_bound_depth_limited() {
        let data = Data::load("../data/english.json").expect("data should exist");
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout should exist");
        let weights = dummy_weights();

        let mut bb = BranchBound::new(layout, data, weights);

        println!("Characters by frequency: {:?}", &bb.chars_by_frequency()[..10]);
        println!("Num positions: {}", bb.num_positions);

        // Depth 3: 30 * 29 * 28 = 24,360 nodes max
        let max_depth = 3;
        let bound = i64::MIN; // no pruning for this test
        let top_k = 5;

        let (results, stats) = bb.search_limited(bound, top_k, max_depth);

        println!("\n{}", stats);
        println!("\nTop {} partial layouts (depth {}):", top_k, max_depth);
        for (i, sol) in results.iter().enumerate() {
            println!("  {}: score={}, keys={:?}", i + 1, sol.score, sol.key_positions);
        }

        assert!(stats.nodes_visited > 0);
        assert!(stats.solutions_found > 0.0);
        assert_eq!(stats.max_depth_reached, max_depth);
        assert!(results.len() <= top_k);
    }

    #[test]
    fn test_branch_bound_with_pruning() {
        let data = Data::load("../data/english.json").expect("data should exist");
        let layout = Layout::load("../layouts/qwerty.dof").expect("layout should exist");
        let weights = dummy_weights();

        let mut bb = BranchBound::new(layout, data, weights);

        // Use a moderately tight bound to test pruning
        // Score of 0 should prune most branches since scores are negative
        let bound = -1_000_000_000_000; // -1 trillion, loose enough to find some solutions
        let max_depth = 4;
        let top_k = 10;

        let (results, stats) = bb.search_limited(bound, top_k, max_depth);

        println!("\n{}", stats);
        println!("\nFound {} solutions", results.len());
        if !results.is_empty() {
            println!("Best score: {}", results[0].score);
            println!("Worst score: {}", results.last().unwrap().score);
        }

        // With pruning, we should have pruned some nodes
        // (unless bound is too loose)
        println!("Prune rate: {:.2}%",
            stats.nodes_pruned as f64 / stats.nodes_visited as f64 * 100.0);
    }
}
