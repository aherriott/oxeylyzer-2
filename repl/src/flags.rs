xflags::xflags! {
    cmd oxeylyzer {
        /// Analyze a layout.
        cmd analyze a view layout {
            /// The name of the layout to analyze.
            required name: String
        }
        /// Rank all layouts for the currently specified language. A higher score is better.
        cmd rank {}
        /// Generate layouts by randomizing a base layout.
        cmd gen g generate {
            /// Name of the layout to use as a basis.
            required name: String
            /// Name of iterations to do. 10 by default.
            optional count: usize
            /// Characters to pin. Setting pins speeds up generation.
            optional -p, --pins pins: String
        }
        /// Shows the top n sfbs on a layout.
        cmd sfbs {
            /// Name of the layout to show sfbs of.
            required name: String
            /// Amount of sfbs to show. 10 by default.
            optional -c, --count count: usize
        }
        /// Shows the top n stretches on a layout.
        cmd stretches {
            /// Name of the layout to show stretches of.
            required name: String
            /// Amount of stretches to show. 10 by default.
            optional -c, --count count: usize
        }
        /// Shows every trigram stat of a layout.
        cmd trigrams t {
            required name: String
        }
        /// Shows layouts most similar to a reference layout.
        cmd similarity similar sim {
            required name: String
        }
        /// Reload the analyzer config file
        cmd r reload refresh {}
        /// Branch and bound search from a base layout
        cmd bb branchbound {
            /// Name of the layout to use as a basis.
            required name: String
            /// Max depth to search. Defaults to all positions.
            optional -d, --depth depth: usize
            /// Number of top layouts to keep. Defaults to 5.
            optional -k, --top top_k: usize
        }
        /// Branch and bound (position-first, finger order)
        cmd bb2 {
            /// Name of the layout to use as a basis.
            required name: String
            /// Number of top layouts to keep. Defaults to 5.
            optional -k, --top top_k: usize
        }
        /// Branch and bound (hybrid: key-freq + finger-fill)
        cmd bb3 {
            /// Name of the layout to use as a basis.
            required name: String
            /// Number of top layouts to keep. Defaults to 5.
            optional -k, --top top_k: usize
        }
        /// Beam search
        cmd beam {
            /// Name of the layout to use as a basis.
            required name: String
            /// Beam width (number of candidates kept per prune). Defaults to 1000.
            optional -w, --width width: usize
            /// Prune interval (prune every N depths). Defaults to 1.
            optional -n, --interval interval: usize
        }
        /// Monte Carlo Tree Search
        cmd mcts {
            /// Name of the layout to use as a basis.
            required name: String
            /// Number of rollouts. Defaults to unlimited.
            optional -i, --iterations iterations: usize
            /// Exploration constant. Defaults to 1.41.
            optional -c, --explore explore: f64
            /// SA iterations per rollout. Defaults to 1000.
            optional -s, --sa sa_iters: usize
            /// Greedy polish depth after SA. 0=off, 1=hill-climb, 2+=depth-N. Defaults to 0.
            optional -g, --greedy greedy_depth: usize
            /// Tree depth limit (keys decided by MCTS). 0 = all keys. Defaults to 0.
            optional -d, --tree-depth tree_depth: usize
            /// Time limit in seconds. Overrides -i.
            optional -t, --time time_secs: usize
        }
        /// Quit the analyzer
        cmd q quit {}
    }
}

//You can also specify a number to analyze a previously generated layout.
