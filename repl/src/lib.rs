mod config;
mod flags;

use config::Config;
use oxeylyzer_core::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{stdout, Write as _},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReplError {
    #[error("Layout '{0}' not found. It might exist, but it's not currently loaded.")]
    UnknownLayout(String),
    #[error("Path '{0}' either doesn't exist or is not a directory")]
    NotADirectory(PathBuf),
    #[error("Invalid quotation marks")]
    ShlexError,
    #[error("{0}")]
    XflagsError(#[from] xflags::Error),
    #[error("{0}")]
    IoError(#[from] std::io::Error),
    #[error("{0}")]
    OxeylyzerDataError(#[from] OxeylyzerError),
    #[error("{0}")]
    DofError(#[from] libdof::DofError),
    #[error("{0}")]
    TomlSerializeError(#[from] toml::ser::Error),
    #[error("{0}")]
    TomlDeserializeError(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, ReplError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplStatus {
    Continue,
    Quit,
}

pub struct Repl {
    a: Analyzer,
    layouts: HashMap<String, Layout>,
    config_path: PathBuf,
    stop: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Repl {
    pub fn with_config<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        let config = Config::load(&config_path)?;

        let data = Data::load(&config.corpus)?;

        let a = Analyzer::new(data, config.weights);

        let layouts = config
            .layouts
            .iter()
            .flat_map(|p| {
                load_layouts(p)
                    .inspect_err(|e| println!("Error loading layout at '{}': {e}", p.display()))
            })
            .flat_map(|h| h.into_iter())
            .collect();

        let stop = Arc::new(AtomicBool::new(false));
        let stop_handler = stop.clone();
        ctrlc::set_handler(move || {
            stop_handler.store(true, Ordering::SeqCst);
        }).ok();

        Ok(Self {
            a,
            layouts,
            config_path,
            stop,
        })
    }

    pub fn layout(&self, name: &str) -> Result<&Layout> {
        self.layouts
            .get(&name.to_lowercase())
            .ok_or(ReplError::UnknownLayout(name.into()))
    }

    fn analyze(&mut self, name: &str) -> Result<()> {
        let layout = self.layout(name)?.clone();
        self.a.use_layout(&layout, &[]);
        let stats = self.a.stats();

        // Print layout from cache (has reconstructed magic rules)
        let display_layout = self.a.layout();

        let finger_use: String = stats.finger_use
            .iter()
            .map(|f| format!("{:.2}", f * 100.0))
            .collect::<Vec<_>>()
            .join(", ");
        let finger_sfbs: String = stats.finger_sfbs
            .iter()
            .map(|f| format!("{:.3}", f * 100.0))
            .collect::<Vec<_>>()
            .join(", ");

        print!("{}", display_layout);

        let (sfb_s, stretch_s, scissors_s, trigram_s, magic_p, fu_s) = self.a.score_breakdown();
        let mut extras = String::new();
        if magic_p != 0 { extras += &format!(", magic: {}", fmt_num(magic_p as f64)); }
        if fu_s != 0 { extras += &format!(", finger: {}", fmt_num(fu_s as f64)); }
        println!(
            concat!(
                "score:     {} (sfb: {}, stretch: {}, scissors: {}, trigram: {}{})\n\n",
                "sfbs:      {:.3}%\n",
                "sfs:       {:.3}%\n",
                "stretches: {:.3}\n",
                "finger usage:\n  {}\n",
                "finger sfbs:\n  {}\n"
            ),
            stats.score,
            fmt_num(sfb_s as f64), fmt_num(stretch_s as f64),
            fmt_num(scissors_s as f64), fmt_num(trigram_s as f64),
            extras,
            stats.sfbs * 100.0,
            stats.sfs * 100.0,
            stats.stretches,
            finger_use,
            finger_sfbs,
        );

        self.trigrams(name)
    }

    fn rank(&mut self) {
        let mut ranked: Vec<_> = self.layouts
            .iter()
            .map(|(name, layout)| {
                self.a.use_layout(layout, &[]);
                let score = self.a.score();
                (name.clone(), score)
            })
            .collect();

        ranked.sort_by(|(_, a), (_, b)| b.cmp(a)); // Higher score = better

        for (name, score) in ranked {
            println!("{:<15} {}", name, score);
        }
    }

    fn generate(&mut self, name: &str, count: Option<usize>, pin_chars: Option<String>, time_secs: Option<usize>) -> Result<()> {
        use oxeylyzer_core::dual_annealing::{DualAnnealing, DualAnnealingConfig};
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};
        use rayon::prelude::*;
        use std::sync::atomic::AtomicUsize;

        let layout = self.layout(name)?.clone();
        let count = count.unwrap_or(10);
        let time_per = std::time::Duration::from_secs(time_secs.unwrap_or(30) as u64);
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };

        let policy = RolloutPolicy {
            steps: vec![
                OptStep::SA {
                    initial_temp: 0.001,
                    final_temp: 1E-7,
                    iterations: 1000,
                },
                OptStep::Greedy,
            ],
        };
        let config = DualAnnealingConfig::default();

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();
        let completed = Arc::new(AtomicUsize::new(0));

        let ncpus = rayon::current_num_threads();
        println!("Generating {} variants, {}s each, {} threads, from {}",
            count, time_per.as_secs(), ncpus, name);

        let start = std::time::Instant::now();
        let data = self.a.data().clone();
        let weights = self.a.weights().clone();

        let mut results: Vec<(Layout, i64)> = (0..count)
            .into_par_iter()
            .map(|_| {
                if stop.load(Ordering::Relaxed) {
                    return (layout.clone(), i64::MIN);
                }
                let variant_start = std::time::Instant::now();
                let da = DualAnnealing::new(data.clone(), weights.clone());
                let result = da.search(&layout, &pins, &config, &policy, u64::MAX, |_iter, _restarts, _current, _best| {
                    stop.load(Ordering::Relaxed) || variant_start.elapsed() >= time_per
                });

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprint!("\r  {}/{} completed   ", done, count);

                (result.best_layout, result.best_score)
            })
            .collect();

        println!();

        // Filter out any skipped results
        results.retain(|(_, s)| *s != i64::MIN);

        // Sort by score (higher = better)
        results.sort_by(|(_, s1), (_, s2)| s2.cmp(s1));

        for (i, (mut layout, score)) in results.into_iter().enumerate().take(10) {
            layout.name = "".into();
            println!("#{}, score: {}{}", i + 1, fmt_num(score as f64), layout);
        }

        println!(
            "generated {} variants in {:.1}s",
            count,
            start.elapsed().as_secs_f64()
        );

        Ok(())
    }

    fn sfbs(&mut self, name: &str, _count: Option<usize>) -> Result<()> {
        let layout = self.layout(name)?.clone();
        self.a.use_layout(&layout, &[]);
        let stats = self.a.stats();

        println!("Total SFBs: {:.3}%", stats.sfbs * 100.0);
        println!("Per-finger SFBs:");
        for (i, &sfb) in stats.finger_sfbs.iter().enumerate() {
            if sfb > 0.0 {
                println!("  Finger {}: {:.3}%", i, sfb * 100.0);
            }
        }

        Ok(())
    }

    fn stretches(&mut self, name: &str, _count: Option<usize>) -> Result<()> {
        let layout = self.layout(name)?.clone();
        self.a.use_layout(&layout, &[]);
        let stats = self.a.stats();

        println!("Total stretches: {:.3}", stats.stretches);

        Ok(())
    }

    pub fn trigrams(&mut self, name: &str) -> Result<()> {
        let layout = self.layout(name)?.clone();
        self.a.use_layout(&layout, &[]);
        let stats = self.a.stats();
        let trigrams = &stats.trigrams;

        if trigrams.sft != 0.0 {
            println!("Sft:          {:.3}%", trigrams.sft * 100.0);
        }
        if trigrams.sfb != 0.0 {
            println!("Sfb:          {:.3}%", trigrams.sfb * 100.0);
        }
        if trigrams.inroll != 0.0 {
            println!("Inroll:       {:.3}%", trigrams.inroll * 100.0);
        }
        if trigrams.outroll != 0.0 {
            println!("Outroll:      {:.3}%", trigrams.outroll * 100.0);
        }
        if trigrams.alternate != 0.0 {
            println!("Alternate:    {:.3}%", trigrams.alternate * 100.0);
        }
        if trigrams.redirect != 0.0 {
            println!("Redirect:     {:.3}%", trigrams.redirect * 100.0);
        }
        if trigrams.onehandin != 0.0 {
            println!("Onehand In:   {:.3}%", trigrams.onehandin * 100.0);
        }
        if trigrams.onehandout != 0.0 {
            println!("Onehand Out:  {:.3}%", trigrams.onehandout * 100.0);
        }
        if trigrams.thumb != 0.0 {
            println!("Thumb:        {:.3}%", trigrams.thumb * 100.0);
        }
        if trigrams.invalid != 0.0 {
            println!("Invalid:      {:.3}%", trigrams.invalid * 100.0);
        }

        Ok(())
    }

    pub fn similarity(&mut self, name: &str) -> Result<()> {
        let layout = self.layout(name)?.clone();

        let mut similarities: Vec<_> = self.layouts
            .values()
            .filter(|cmp| cmp.name.to_lowercase() != name.to_lowercase())
            .map(|cmp| {
                let sim = self.a.similarity(&layout, cmp);
                (cmp.name.as_str(), sim)
            })
            .collect();

        similarities.sort_by(|(_, s1), (_, s2)| s2.cmp(s1));

        for (n, s) in similarities {
            println!("{:<15} {}", n, s);
        }

        Ok(())
    }

    fn branch_bound(&mut self, name: &str, max_depth: Option<usize>, top_k: Option<usize>) -> Result<()> {
        use oxeylyzer_core::branch_bound::BranchBound;

        let layout = self.layout(name)?.clone();
        let top_k = top_k.unwrap_or(5);
        let max_depth = max_depth.unwrap_or(layout.keyboard.len());

        println!("Branch & bound: {} positions, depth limit {}, top-{}", layout.keyboard.len(), max_depth, top_k);

        // Get a tight initial bound by running multiple SA + greedy passes
        let bound_start = std::time::Instant::now();
        let mut best_bound = i64::MIN;
        let num_passes = 10;
        for i in 0..num_passes {
            let random = layout.random();
            self.a.use_layout(&random, &[]);
            let (sa_layout, _) = self.a.annealing_improve(random.clone(), &[], 1.0, 1E-4, 1_000_000);
            let (_, greedy_score) = self.a.greedy_improve(&sa_layout, &[]);
            if greedy_score > best_bound {
                best_bound = greedy_score;
            }
            print!("\r  bound pass {}/{}: {}", i + 1, num_passes, fmt_num(best_bound as f64));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!("\nInitial bound: {} (found in {:.1}s)", fmt_num(best_bound as f64), bound_start.elapsed().as_secs_f64());

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone());
        let (results, stats) = bb.search_limited_with_progress(
            best_bound,
            top_k,
            max_depth,
            &mut |progress: &oxeylyzer_core::branch_bound::BranchBoundProgress| {
                if progress.nodes_visited % 100_000 == 0 && last_print.elapsed().as_millis() > 500 {
                    last_print = std::time::Instant::now();
                    let elapsed = start.elapsed().as_secs_f64();
                    let nodes_per_sec = progress.nodes_visited as f64 / elapsed;
                    let total_nodes = progress.nodes_visited as f64 + progress.nodes_pruned;
                    let avg_prune_depth = if progress.prune_count > 0 {
                        progress.prune_depth_sum as f64 / progress.prune_count as f64
                    } else {
                        0.0
                    };
                    // Estimate remaining: ratio of pruned to visited gives a rough completion %
                    let pct = if progress.estimated_total_nodes > 0.0 {
                        total_nodes / progress.estimated_total_nodes * 100.0
                    } else {
                        0.0
                    };
                    let est_remaining = if pct > 0.01 {
                        elapsed / (pct / 100.0) - elapsed
                    } else {
                        f64::INFINITY
                    };

                    print!("\r  {} visited | {} pruned | {} solutions | best: {} | avg prune depth: {:.1} | {} nodes/s | {:.1}% done | ~{} left   ",
                        fmt_num(progress.nodes_visited as f64),
                        fmt_num(progress.nodes_pruned),
                        fmt_num(progress.solutions_found),
                        progress.best_score.map_or("none".to_string(), |s| fmt_num(s as f64)),
                        avg_prune_depth,
                        fmt_num(nodes_per_sec),
                        pct,
                        fmt_duration(est_remaining),
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            },
        );
        println!();

        println!("{}", stats);
        println!("Search took {:.2}s", start.elapsed().as_secs_f64());

        for (i, result) in results.iter().enumerate() {
            println!("#{}: score {}", i + 1, fmt_num(result.score as f64));
            for (c, pos) in &result.key_positions {
                print!("{}@{} ", c, pos);
            }
            println!();
        }

        Ok(())
    }

    fn branch_bound_position_first(&mut self, name: &str, top_k: Option<usize>) -> Result<()> {
        use oxeylyzer_core::branch_bound::BranchBound;

        let layout = self.layout(name)?.clone();
        let top_k = top_k.unwrap_or(5);

        println!("B&B position-first: {} positions, top-{}", layout.keyboard.len(), top_k);

        // Get initial bound from SA + greedy
        let bound_start = std::time::Instant::now();
        let mut best_bound = i64::MIN;
        let num_passes = 10;
        for i in 0..num_passes {
            let random = layout.random();
            self.a.use_layout(&random, &[]);
            let (sa_layout, _) = self.a.annealing_improve(random.clone(), &[], 1.0, 1E-4, 1_000_000);
            let (_, greedy_score) = self.a.greedy_improve(&sa_layout, &[]);
            if greedy_score > best_bound {
                best_bound = greedy_score;
            }
            print!("\r  bound pass {}/{}: {}", i + 1, num_passes, fmt_num(best_bound as f64));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!("\nInitial bound: {} (found in {:.1}s)", fmt_num(best_bound as f64), bound_start.elapsed().as_secs_f64());

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone());

        // Print finger order
        println!("Position order (finger-first): {:?}", bb.positions_by_finger());

        let (results, stats) = bb.search_position_first(
            best_bound,
            top_k,
            |progress: &oxeylyzer_core::branch_bound::BranchBoundProgress| {
                if progress.nodes_visited % 100_000 == 0 && last_print.elapsed().as_millis() > 500 {
                    last_print = std::time::Instant::now();
                    let elapsed = start.elapsed().as_secs_f64();
                    let nodes_per_sec = progress.nodes_visited as f64 / elapsed;
                    let total_nodes = progress.nodes_visited as f64 + progress.nodes_pruned;
                    let avg_prune_depth = if progress.prune_count > 0 {
                        progress.prune_depth_sum as f64 / progress.prune_count as f64
                    } else { 0.0 };
                    let pct = if progress.estimated_total_nodes > 0.0 {
                        total_nodes / progress.estimated_total_nodes * 100.0
                    } else { 0.0 };
                    let est_remaining = if pct > 0.01 {
                        elapsed / (pct / 100.0) - elapsed
                    } else { f64::INFINITY };

                    print!("\r  {} visited | {} pruned | {} solutions | best: {} | avg prune depth: {:.1} | {} nodes/s | {:.1}% done | ~{} left   ",
                        fmt_num(progress.nodes_visited as f64),
                        fmt_num(progress.nodes_pruned),
                        fmt_num(progress.solutions_found),
                        progress.best_score.map_or("none".to_string(), |s| fmt_num(s as f64)),
                        avg_prune_depth,
                        fmt_num(nodes_per_sec),
                        pct,
                        fmt_duration(est_remaining),
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            },
        );
        println!();

        println!("{}", stats);
        println!("Search took {:.2}s", start.elapsed().as_secs_f64());

        for (i, result) in results.iter().enumerate() {
            println!("#{}: score {}", i + 1, fmt_num(result.score as f64));
            for (c, pos) in &result.key_positions {
                print!("{}@{} ", c, pos);
            }
            println!();
        }

        Ok(())
    }

    fn branch_bound_hybrid(&mut self, name: &str, top_k: Option<usize>) -> Result<()> {
        use oxeylyzer_core::branch_bound::BranchBound;

        let layout = self.layout(name)?.clone();
        let top_k = top_k.unwrap_or(5);

        println!("B&B hybrid (key-freq + finger-fill): {} positions, top-{}", layout.keyboard.len(), top_k);

        let bound_start = std::time::Instant::now();
        let mut best_bound = i64::MIN;
        let num_passes = 10;
        for i in 0..num_passes {
            let random = layout.random();
            self.a.use_layout(&random, &[]);
            let (sa_layout, _) = self.a.annealing_improve(random.clone(), &[], 1.0, 1E-4, 1_000_000);
            let (_, greedy_score) = self.a.greedy_improve(&sa_layout, &[]);
            if greedy_score > best_bound { best_bound = greedy_score; }
            print!("\r  bound pass {}/{}: {}", i + 1, num_passes, fmt_num(best_bound as f64));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!("\nInitial bound: {} (found in {:.1}s)", fmt_num(best_bound as f64), bound_start.elapsed().as_secs_f64());

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone());

        let (results, stats) = bb.search_hybrid(
            best_bound,
            top_k,
            |progress: &oxeylyzer_core::branch_bound::BranchBoundProgress| {
                if progress.nodes_visited % 100_000 == 0 && last_print.elapsed().as_millis() > 500 {
                    last_print = std::time::Instant::now();
                    let elapsed = start.elapsed().as_secs_f64();
                    let nodes_per_sec = progress.nodes_visited as f64 / elapsed;
                    let total_nodes = progress.nodes_visited as f64 + progress.nodes_pruned;
                    let avg_prune_depth = if progress.prune_count > 0 {
                        progress.prune_depth_sum as f64 / progress.prune_count as f64
                    } else { 0.0 };
                    let pct = if progress.estimated_total_nodes > 0.0 {
                        total_nodes / progress.estimated_total_nodes * 100.0
                    } else { 0.0 };
                    let est_remaining = if pct > 0.01 {
                        elapsed / (pct / 100.0) - elapsed
                    } else { f64::INFINITY };

                    print!("\r  {} visited | {} pruned | {} solutions | best: {} | avg prune depth: {:.1} | {} nodes/s | {:.1}% done | ~{} left   ",
                        fmt_num(progress.nodes_visited as f64),
                        fmt_num(progress.nodes_pruned),
                        fmt_num(progress.solutions_found),
                        progress.best_score.map_or("none".to_string(), |s| fmt_num(s as f64)),
                        avg_prune_depth,
                        fmt_num(nodes_per_sec),
                        pct,
                        fmt_duration(est_remaining),
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            },
        );
        println!();

        println!("{}", stats);
        println!("Search took {:.2}s", start.elapsed().as_secs_f64());

        for (i, result) in results.iter().enumerate() {
            println!("#{}: score {}", i + 1, fmt_num(result.score as f64));
            for (c, pos) in &result.key_positions {
                print!("{}@{} ", c, pos);
            }
            println!();
        }

        Ok(())
    }

    fn beam_search_cmd(&mut self, name: &str, width: Option<usize>, interval: Option<usize>) -> Result<()> {
        use oxeylyzer_core::branch_bound::BranchBound;

        let layout = self.layout(name)?.clone();
        let width = width.unwrap_or(1000);
        let interval = interval.unwrap_or(1);

        println!("Beam search: {} positions, beam width {}, prune every {} depths", layout.keyboard.len(), width, interval);

        let start = std::time::Instant::now();
        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone());
        let results = bb.beam_search_with_interval(width, interval);
        let elapsed = start.elapsed();

        println!("Beam search completed in {:.2}s", elapsed.as_secs_f64());
        println!("Found {} layouts", results.len());

        for (i, result) in results.iter().enumerate().take(10) {
            println!("#{}: score {}", i + 1, fmt_num(result.score as f64));
        }

        Ok(())
    }

    fn mcts_cmd(&mut self, name: &str, iterations: Option<usize>, explore: Option<f64>, sa_iters: Option<usize>, greedy_depth: Option<usize>, tree_depth: Option<usize>, time_secs: Option<usize>) -> Result<()> {
        use oxeylyzer_core::mcts::MctsSearch;
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        let layout = self.layout(name)?.clone();
        let iterations = iterations.map(|i| i as u64).unwrap_or(u64::MAX);
        let explore = explore.unwrap_or(1.41);
        let sa_iters = sa_iters.unwrap_or(1_000);
        let greedy_depth = greedy_depth.unwrap_or(0);
        let tree_depth = tree_depth.unwrap_or(0);
        let time_limit = time_secs.map(|s| std::time::Duration::from_secs(s as u64));
        let num_pos = layout.keyboard.len();

        // Build rollout policy from flags
        let mut steps = Vec::new();
        if sa_iters > 0 {
            steps.push(OptStep::SA {
                initial_temp: 10.0,
                final_temp: 1E-5,
                iterations: sa_iters,
            });
        }
        match greedy_depth {
            0 => {} // no greedy
            1 => steps.push(OptStep::Greedy), // fast hill-climb
            n => steps.push(OptStep::GreedyDepthN(n)), // depth-N search
        }
        if steps.is_empty() {
            steps.push(OptStep::Greedy);
        }
        let policy = RolloutPolicy { steps };

        let td_display = if tree_depth == 0 { format!("all {}", num_pos) } else { format!("{}", tree_depth) };
        let iter_display = if time_limit.is_some() {
            format!("{}s", time_secs.unwrap())
        } else if iterations == u64::MAX {
            "∞".to_string()
        } else {
            format!("{}", iterations)
        };
        println!("MCTS: {} positions, {} rollouts, exploration={:.2}, tree depth={}, policy={:?}",
            num_pos, iter_display, explore, td_display, policy.steps);
        if iterations == u64::MAX && time_limit.is_none() {
            println!("  (Ctrl+C to stop)");
        }

        // Reset stop flag before starting
        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let mut search = MctsSearch::new(layout, self.a.data().clone(), self.a.weights().clone(), 10);

        search.search(iterations, explore, &policy, tree_depth, |_iter, total, best, avg| {
            if last_print.elapsed().as_millis() > 500 {
                last_print = std::time::Instant::now();
                let elapsed = start.elapsed().as_secs_f64();
                let rate = total as f64 / elapsed;
                print!("\r  {} rollouts | best: {} | avg: {} | {:.0} rollouts/s | {:.1}s elapsed   ",
                    fmt_num(total as f64),
                    fmt_num(best as f64),
                    fmt_num(avg),
                    rate,
                    elapsed,
                );
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            if stop.load(Ordering::SeqCst) { return true; }
            if let Some(limit) = time_limit {
                if start.elapsed() >= limit { return true; }
            }
            false
        });
        println!();

        let elapsed = start.elapsed();
        println!("MCTS completed in {:.2}s ({} rollouts)", elapsed.as_secs_f64(),
            fmt_num(search.total_rollouts() as f64));

        let results = search.results();
        for (i, result) in results.iter().enumerate().take(10) {
            println!("#{}: score {}", i + 1, fmt_num(result.score as f64));
        }

        Ok(())
    }

    fn sa_cmd(&mut self, name: &str, count: Option<usize>, sa_iters: Option<usize>, greedy_depth: Option<usize>, pin_chars: Option<String>) -> Result<()> {
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        let layout = self.layout(name)?.clone();
        let count = count.unwrap_or(1);
        let sa_iters = sa_iters.unwrap_or(10_000_000);
        let greedy_depth = greedy_depth.unwrap_or(1);
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };

        let mut steps = vec![
            OptStep::SA {
                initial_temp: 10.0,
                final_temp: 1E-5,
                iterations: sa_iters,
            },
        ];
        match greedy_depth {
            0 => {}
            1 => steps.push(OptStep::Greedy),
            n => steps.push(OptStep::ProgressiveGreedy { max_depth: n }),
        }
        let policy = RolloutPolicy { steps };

        println!("SA: {} variants, {} iters, greedy={}, from {}", count, fmt_num(sa_iters as f64), greedy_depth, name);

        let start = std::time::Instant::now();
        let mut results: Vec<(Layout, i64)> = Vec::with_capacity(count);

        for i in 0..count {
            let random_layout = layout.random_with_pins(&pins);
            let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                &random_layout, self.a.data().clone(), self.a.weights(),
            );
            let final_score = cache.optimize(&policy, &pins);
            let final_layout = cache.to_layout();
            results.push((final_layout, final_score));

            print!("\r  {}/{} | score: {}   ", i + 1, count, fmt_num(final_score as f64));
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();

        results.sort_by(|(_, s1), (_, s2)| s2.cmp(s1));

        for (i, (mut layout, score)) in results.into_iter().enumerate().take(10) {
            layout.name = "".into();
            println!("#{}, score: {}{}", i + 1, fmt_num(score as f64), layout);
        }

        println!("SA completed in {:.1}s", start.elapsed().as_secs_f64());
        Ok(())
    }

    fn dual_annealing_cmd(
        &mut self,
        name: &str,
        sa_iters: Option<usize>,
        sa_temp: Option<f64>,
        sa_final: Option<f64>,
        greedy_depth: Option<usize>,
        iterations: Option<usize>,
        time_secs: Option<usize>,
        temp: Option<f64>,
        qv: Option<f64>,
        qa: Option<f64>,
        restart_ratio: Option<f64>,
        max_swaps: Option<usize>,
        pin_chars: Option<String>,
    ) -> Result<()> {
        use oxeylyzer_core::dual_annealing::{DualAnnealing, DualAnnealingConfig};
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        let layout = self.layout(name)?.clone();
        let sa_iters = sa_iters.unwrap_or(1000);
        let greedy_depth = greedy_depth.unwrap_or(1);
        let max_iter = iterations.map(|i| i as u64).unwrap_or(u64::MAX);
        let time_limit = time_secs.map(|s| std::time::Duration::from_secs(s as u64));
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };

        // Build local search policy
        let mut steps = Vec::new();
        if sa_iters > 0 {
            steps.push(OptStep::SA {
                initial_temp: sa_temp.unwrap_or(0.001),
                final_temp: sa_final.unwrap_or(1E-7),
                iterations: sa_iters,
            });
        }
        match greedy_depth {
            0 => {}
            1 => steps.push(OptStep::Greedy),
            n => steps.push(OptStep::ProgressiveGreedy { max_depth: n }),
        }
        if steps.is_empty() {
            steps.push(OptStep::ProgressiveGreedy { max_depth: 2 });
        }
        let policy = RolloutPolicy { steps };

        let mut config = DualAnnealingConfig::default();
        if let Some(t) = temp { config.initial_temp = t; }
        if let Some(v) = qv { config.visit = v; }
        if let Some(a) = qa { config.accept = a; }
        if let Some(r) = restart_ratio { config.restart_temp_ratio = r; }
        if let Some(s) = max_swaps { config.max_perturb_swaps = s; }

        let iter_display = if time_limit.is_some() {
            format!("{}s", time_secs.unwrap())
        } else if max_iter == u64::MAX {
            "∞".to_string()
        } else {
            format!("{}", max_iter)
        };
        println!("Dual annealing: {} positions, {} iters, qv={:.2}, max_swaps={}, local={:?}",
            layout.keyboard.len(), iter_display, config.visit, config.max_perturb_swaps, policy.steps);
        if max_iter == u64::MAX && time_limit.is_none() {
            println!("  (Ctrl+C to stop)");
        }

        // Reset stop flag before starting
        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let da = DualAnnealing::new(self.a.data().clone(), self.a.weights().clone());
        let result = da.search(&layout, &pins, &config, &policy, max_iter, |iter, restarts, current, best| {
            if last_print.elapsed().as_millis() > 500 {
                last_print = std::time::Instant::now();
                let elapsed = start.elapsed().as_secs_f64();
                let rate = iter as f64 / elapsed;
                print!("\r  iter {} | restarts: {} | current: {} | best: {} | {:.1}/s | {:.1}s   ",
                    fmt_num(iter as f64),
                    restarts,
                    fmt_num(current as f64),
                    fmt_num(best as f64),
                    rate,
                    elapsed,
                );
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            if stop.load(Ordering::SeqCst) { return true; }
            if let Some(limit) = time_limit {
                if start.elapsed() >= limit { return true; }
            }
            false
        });
        println!();

        let elapsed = start.elapsed();
        println!("Dual annealing completed in {:.2}s ({} restarts)", elapsed.as_secs_f64(), result.restarts);
        println!("Best score: {}", fmt_num(result.best_score as f64));
        println!("{}", result.best_layout);

        Ok(())
    }

    pub fn reload(&mut self) -> Result<()> {
        let new = Self::with_config(&self.config_path)?;

        self.a = new.a;
        self.layouts = new.layouts;

        Ok(())
    }

    pub fn respond(&mut self, line: &str) -> Result<ReplStatus> {
        use crate::flags::*;

        let args = shlex::split(line)
            .ok_or(ReplError::ShlexError)?
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>();

        let flags = Oxeylyzer::from_vec(args)?;

        match flags.subcommand {
            OxeylyzerCmd::Analyze(a) => self.analyze(&a.name)?,
            OxeylyzerCmd::Rank(_) => self.rank(),
            OxeylyzerCmd::Gen(g) => self.generate(&g.name, g.count, g.pins, g.time)?,
            OxeylyzerCmd::Sfbs(s) => self.sfbs(&s.name, s.count)?,
            OxeylyzerCmd::Stretches(s) => self.stretches(&s.name, s.count)?,
            OxeylyzerCmd::Trigrams(t) => self.trigrams(&t.name)?,
            OxeylyzerCmd::Similarity(s) => self.similarity(&s.name)?,
            OxeylyzerCmd::R(_) => self.reload()?,
            OxeylyzerCmd::Bb(b) => self.branch_bound(&b.name, b.depth, b.top)?,
            OxeylyzerCmd::Bb2(b) => self.branch_bound_position_first(&b.name, b.top)?,
            OxeylyzerCmd::Bb3(b) => self.branch_bound_hybrid(&b.name, b.top)?,
            OxeylyzerCmd::Beam(b) => self.beam_search_cmd(&b.name, b.width, b.interval)?,
            OxeylyzerCmd::Mcts(m) => self.mcts_cmd(&m.name, m.iterations, m.explore, m.sa, m.greedy, m.tree_depth, m.time)?,
            OxeylyzerCmd::Da(d) => self.dual_annealing_cmd(&d.name, d.sa, d.sa_temp, d.sa_final, d.greedy, d.iterations, d.time, d.temp, d.qv, d.qa, d.restart, d.swaps, d.pins)?,
            OxeylyzerCmd::Sa(s) => self.sa_cmd(&s.name, s.count, s.sa, s.greedy, s.pins)?,
            OxeylyzerCmd::Q(_) => return Ok(ReplStatus::Quit),
        }

        Ok(ReplStatus::Continue)
    }

    pub fn run(&mut self) -> Result<()> {
        use ReplStatus::*;

        loop {
            let line = readline()?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            match self.respond(line) {
                Ok(Continue) => continue,
                Ok(Quit) => break,
                Err(err) => {
                    println!("Invalid line: {err}");
                }
            }
        }

        Ok(())
    }
}

fn readline() -> std::io::Result<String> {
    write!(std::io::stdout(), "> ")?;
    std::io::stdout().flush()?;

    let mut buf = String::new();

    std::io::stdin().read_line(&mut buf)?;
    Ok(buf)
}

fn fmt_num(n: f64) -> String {
    let abs = n.abs();
    let suffixes = [
        (1e63, "Vg"), (1e60, "Nv"), (1e57, "Ov"), (1e54, "Sv"), (1e51, "Sxd"),
        (1e48, "Qid"), (1e45, "Qad"), (1e42, "Td"), (1e39, "Dd"),
        (1e36, "Ud"), (1e33, "D"), (1e30, "N"), (1e27, "Oc"), (1e24, "Sp"),
        (1e21, "Sx"), (1e18, "Qi"), (1e15, "Qa"), (1e12, "T"),
        (1e9, "B"), (1e6, "M"), (1e3, "K"),
    ];
    for &(threshold, suffix) in &suffixes {
        if abs >= threshold {
            let scaled = n / threshold;
            // Use 3 significant figures
            let decimals = if scaled.abs() >= 100.0 { 0 }
                else if scaled.abs() >= 10.0 { 1 }
                else { 2 };
            return format!("{:.*}{}", decimals, scaled, suffix);
        }
    }
    format!("{:.0}", n)
}

fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() { return "?".to_string(); }
    let s = secs.abs();
    if s < 60.0 { format!("{:.0}s", s) }
    else if s < 3600.0 { format!("{:.1}m", s / 60.0) }
    else if s < 86400.0 { format!("{:.1}h", s / 3600.0) }
    else if s < 604800.0 { format!("{:.1}d", s / 86400.0) }
    else if s < 31536000.0 { format!("{:.1}w", s / 604800.0) }
    else { format!("{:.1}y", s / 31536000.0) }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_layouts<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Layout>> {
    if let Ok(readdir) = fs::read_dir(&path) {
        let map = readdir
            .flatten()
            .flat_map(|p| {
                Layout::load(p.path()).inspect_err(|e| {
                    println!("Error loading layout from '{}': {e}", p.path().display())
                })
            })
            .map(|l| (l.name.to_lowercase(), l))
            .collect();

        Ok(map)
    } else {
        Err(ReplError::NotADirectory(path.as_ref().into()))
    }
}

pub fn pin_positions(layout: &Layout, pin_chars: String) -> Vec<usize> {
    match pin_chars.len() {
        0 => vec![],
        1 => {
            let find = &pin_chars.chars().next().unwrap();

            match layout.keys.iter().position(|c| find == c) {
                Some(i) => Vec::from([i]),
                None => vec![],
            }
        }
        _ => {
            let m = HashSet::<char>::from_iter(pin_chars.chars());

            layout
                .keys
                .iter()
                .enumerate()
                .filter_map(|(i, k)| m.contains(k).then_some(i))
                .collect()
        }
    }
}
