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

    fn generate(&mut self, name: &str, pin_chars: Option<String>, time_secs: Option<usize>, top_n: Option<usize>, save_dir: Option<String>) -> Result<()> {
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};
        use rayon::prelude::*;
        use std::sync::Mutex;
        use std::sync::atomic::AtomicU64;

        let layout = self.layout(name)?.clone();
        let top_n = top_n.unwrap_or(10);
        let time_limit = time_secs.map(|s| std::time::Duration::from_secs(s as u64));
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };

        // Auto-pin positions with REPLACEMENT_CHAR (unused ~ positions) so the
        // optimizer never swaps them into main rows
        let mut all_pins = pins.clone();
        for (i, &c) in layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR && !all_pins.contains(&i) {
                all_pins.push(i);
            }
        }

        let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let ncpus = rayon::current_num_threads();
        let time_str = time_limit.map_or("∞".to_string(), |d| format!("{}s", d.as_secs()));
        println!("gen: random restart greedy, {} threads, top {}, from {} ({})",
            ncpus, top_n, name, time_str);
        if time_limit.is_none() {
            println!("  (Ctrl+C to stop)");
        }

        let start = std::time::Instant::now();
        let data = self.a.data().clone();
        let weights = self.a.weights().clone();
        let scale_factors = self.a.scale_factors().clone();
        let total = AtomicU64::new(0);

        // Shared top-N list protected by mutex
        let best_list: Arc<Mutex<Vec<(i64, Layout)>>> = Arc::new(Mutex::new(Vec::new()));
        let last_print = Arc::new(Mutex::new(std::time::Instant::now()));

        // Run greedy on random layouts continuously across all threads
        (0..ncpus).into_par_iter().for_each(|_| {
            loop {
                if stop.load(Ordering::Relaxed) { break; }
                if let Some(limit) = time_limit {
                    if start.elapsed() >= limit { break; }
                }

                let random_layout = layout.random_with_pins(&all_pins);
                let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                    &random_layout, data.clone(), &weights, &scale_factors,
                );
                cache.optimize(&policy, &all_pins);
                let score = cache.score();
                let count = total.fetch_add(1, Ordering::Relaxed) + 1;

                // Update top-N
                {
                    let mut list = best_list.lock().unwrap();
                    if list.len() < top_n || score > list.last().unwrap().0 {
                        let result_layout = cache.to_layout();
                        list.push((score, result_layout));
                        list.sort_by(|a, b| b.0.cmp(&a.0));
                        list.truncate(top_n);
                    }
                }

                // Progress update
                {
                    let mut lp = last_print.lock().unwrap();
                    if lp.elapsed().as_millis() > 500 {
                        *lp = std::time::Instant::now();
                        let elapsed = start.elapsed().as_secs_f64();
                        let rate = count as f64 / elapsed;
                        let best = best_list.lock().unwrap().first().map(|(s, _)| *s).unwrap_or(i64::MIN);
                        eprint!("\r  {} layouts | best: {} | {:.0}/s | {:.1}s   ",
                            fmt_num(count as f64),
                            fmt_num(best as f64),
                            rate,
                            elapsed,
                        );
                    }
                }
            }
        });
        eprintln!();

        let elapsed = start.elapsed();
        let count = total.load(Ordering::Relaxed);
        println!("gen completed: {} layouts in {:.1}s ({:.0}/s)",
            fmt_num(count as f64), elapsed.as_secs_f64(), count as f64 / elapsed.as_secs_f64());

        let list = best_list.lock().unwrap();
        for (i, (score, layout)) in list.iter().enumerate() {
            let mut l = layout.clone();
            l.name = "".into();
            println!("#{}, score: {}{}", i + 1, fmt_num(*score as f64), l);
        }

        // Save top layouts as .dof files if --save was specified
        if let Some(ref dir) = save_dir {
            std::fs::create_dir_all(dir).ok();
            for (i, (_score, layout)) in list.iter().enumerate() {
                let dof_name = format!("gen-{}", i + 1);
                let path = format!("{}/{}.dof", dir, dof_name);
                match layout.save_dof(&path, &dof_name) {
                    Ok(()) => println!("Saved: {}", path),
                    Err(e) => eprintln!("Failed to save {}: {}", path, e),
                }
            }
        }

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

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone());
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

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone());

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

        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone());

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
        let mut bb = BranchBound::new(layout, self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone());
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

        let mut search = MctsSearch::new(layout, self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone(), 10);

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

    fn sa_cmd(&mut self, name: &str, count: Option<usize>, sa_iters: Option<usize>, sa_temp: Option<f64>, sa_final: Option<f64>, greedy_depth: Option<usize>, time_secs: Option<usize>, pin_chars: Option<String>) -> Result<()> {
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        let layout = self.layout(name)?.clone();
        let sa_iters = sa_iters.unwrap_or(1_000_000);
        let greedy_depth = greedy_depth.unwrap_or(1);
        let time_limit = time_secs.map(|s| std::time::Duration::from_secs(s as u64));
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };
        // Auto-pin REPLACEMENT_CHAR positions
        let mut all_pins = pins.clone();
        for (i, &c) in layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR && !all_pins.contains(&i) {
                all_pins.push(i);
            }
        }

        let mut steps = Vec::new();
        if sa_iters > 0 {
            steps.push(OptStep::SA {
                initial_temp: sa_temp.unwrap_or(10.0),
                final_temp: sa_final.unwrap_or(1E-5),
                iterations: sa_iters,
            });
        }
        match greedy_depth {
            0 => {}
            1 => steps.push(OptStep::Greedy),
            n => steps.push(OptStep::ProgressiveGreedy { max_depth: n }),
        }
        if steps.is_empty() {
            steps.push(OptStep::Greedy);
        }
        let policy = RolloutPolicy { steps };

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let time_str = time_limit.map_or(
            format!("{} variants", count.unwrap_or(1)),
            |d| format!("{}s", d.as_secs())
        );
        println!("SA: {} iters, greedy={}, from {} ({})", fmt_num(sa_iters as f64), greedy_depth, name, time_str);
        if time_limit.is_some() {
            println!("  (Ctrl+C to stop)");
        }

        let start = std::time::Instant::now();
        let max_count = count.unwrap_or(usize::MAX);
        let mut best_score: i64 = i64::MIN;
        let mut best_layout: Option<Layout> = None;
        let mut variants_run = 0usize;
        let mut last_print = std::time::Instant::now();

        loop {
            if stop.load(Ordering::Relaxed) { break; }
            if let Some(limit) = time_limit {
                if start.elapsed() >= limit { break; }
            }
            if time_limit.is_none() && variants_run >= max_count { break; }

            let random_layout = layout.random_with_pins(&all_pins);
            let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                &random_layout, self.a.data().clone(), self.a.weights(), self.a.scale_factors(),
            );
            let final_score = cache.optimize(&policy, &all_pins);
            variants_run += 1;

            if final_score > best_score {
                best_score = final_score;
                best_layout = Some(cache.to_layout());
            }

            if last_print.elapsed().as_millis() > 500 {
                last_print = std::time::Instant::now();
                let elapsed = start.elapsed().as_secs_f64();
                let rate = variants_run as f64 / elapsed;
                eprint!("\r  {} variants | best: {} | {:.2}/s | {:.1}s   ",
                    fmt_num(variants_run as f64),
                    fmt_num(best_score as f64),
                    rate,
                    elapsed,
                );
            }
        }
        eprintln!();

        let elapsed = start.elapsed().as_secs_f64();
        println!("SA completed: {} variants in {:.1}s ({:.2}/s)",
            fmt_num(variants_run as f64), elapsed, variants_run as f64 / elapsed);

        if let Some(mut l) = best_layout {
            l.name = "".into();
            println!("#1, score: {}{}", fmt_num(best_score as f64), l);
        }

        Ok(())
    }

    /// Measure landscape ruggedness: Pearson correlation between layout score and
    /// single-swap neighbor scores. Parallelized across threads.
    /// Low correlation (<0.3) means swap-neighbors are nearly uncorrelated with the
    /// parent layout — a rugged landscape where hill-climbing is weak.
    fn basin_hopping_cmd(
        &mut self,
        name: &str,
        perturbation_swaps: Option<usize>,
        initial_temp: Option<f64>,
        restart_temp: Option<f64>,
        cooling_rate: Option<f64>,
        restart_after_stale: Option<usize>,
        time_secs: Option<usize>,
        iterations: Option<usize>,
        pin_chars: Option<String>,
    ) -> Result<()> {
        use oxeylyzer_core::basin_hopping::{BasinHopping, BasinHoppingConfig};
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        let layout = self.layout(name)?.clone();
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };
        // Auto-pin REPLACEMENT_CHAR positions
        let mut all_pins = pins.clone();
        for (i, &c) in layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR && !all_pins.contains(&i) {
                all_pins.push(i);
            }
        }

        let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };

        let mut config = BasinHoppingConfig::default();
        if let Some(s) = perturbation_swaps { config.perturbation_swaps = s; }
        if let Some(t) = initial_temp { config.initial_temp = t; }
        if let Some(r) = restart_temp { config.restart_temp = r; }
        if let Some(c) = cooling_rate { config.cooling_rate = c; }
        if let Some(s) = restart_after_stale { config.restart_after_stale = s; }

        let max_iter = iterations.map(|i| i as u64).unwrap_or(u64::MAX);
        let time_limit = time_secs.map(|s| std::time::Duration::from_secs(s as u64));

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let time_str = time_limit.map_or(
            format!("max {} iters", max_iter),
            |d| format!("{}s", d.as_secs())
        );
        println!("Basin hopping: swaps={}, initial_temp={:.0e}, cool={}, stale={}, from {} ({})",
            config.perturbation_swaps, config.initial_temp, config.cooling_rate,
            config.restart_after_stale, name, time_str);
        if time_limit.is_some() {
            println!("  (Ctrl+C to stop)");
        }

        let start = std::time::Instant::now();
        let mut last_print = std::time::Instant::now();

        let bh = BasinHopping::new(
            self.a.data().clone(),
            self.a.weights().clone(),
            self.a.scale_factors().clone(),
        );
        let result = bh.search(&layout, &all_pins, &config, &policy, max_iter,
            |iter, restarts, current, best| {
                if last_print.elapsed().as_millis() > 500 {
                    last_print = std::time::Instant::now();
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = iter as f64 / elapsed;
                    eprint!("\r  iter {} | restarts: {} | current: {} | best: {} | {:.1}/s | {:.1}s   ",
                        fmt_num(iter as f64),
                        restarts,
                        fmt_num(current as f64),
                        fmt_num(best as f64),
                        rate,
                        elapsed,
                    );
                }
                if stop.load(Ordering::SeqCst) { return true; }
                if let Some(limit) = time_limit {
                    if start.elapsed() >= limit { return true; }
                }
                false
            });
        eprintln!();

        let elapsed = start.elapsed().as_secs_f64();
        println!("Basin hopping completed in {:.1}s", elapsed);
        println!("  {} restarts, {} accepts, {} improvements", result.restarts, result.accepts, result.improves);
        println!("Best score: {}", fmt_num(result.best_score as f64));

        let mut best = result.best_layout;
        best.name = "".into();
        println!("#1, score: {}{}", fmt_num(result.best_score as f64), best);

        Ok(())
    }

    /// Dump feature vectors for PCA/FAMD/UMAP analysis.
    /// Each row: layout name, score, and ~90 features describing the layout.
    fn dump_features_cmd(
        &mut self,
        output: &str,
        random_greedy: Option<usize>,
        random_only: Option<usize>,
        template: Option<&str>,
    ) -> Result<()> {
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};
        use rayon::prelude::*;
        use std::sync::Mutex;

        let layout_names: Vec<String> = self.layout_names().into_iter().collect();
        println!("Found {} loaded layouts", layout_names.len());

        // Collect feature rows from loaded layouts
        let mut rows: Vec<(String, i64, Vec<(String, String)>)> = Vec::new();

        // Get template shape to filter out incompatible layouts (different boards)
        let template_name = template.unwrap_or("my-layout");
        let template_layout = self.layout(template_name)?.clone();
        let template_shape = template_layout.shape.inner().to_vec();
        let template_main_keys: usize = template_shape.iter().take(3).sum();
        println!("Template '{}' shape: {:?} (main keys: {})",
            template_name, template_shape, template_main_keys);

        let mut skipped = 0;
        for name in &layout_names {
            let layout = match self.layout(name) {
                Ok(l) => l.clone(),
                Err(_) => continue,
            };
            // Skip layouts whose shape is incompatible
            let this_shape = layout.shape.inner();
            let this_main_keys: usize = this_shape.iter().take(3).sum();
            if this_main_keys != template_main_keys || this_shape.len() != template_shape.len() {
                skipped += 1;
                continue;
            }
            self.a.use_layout(&layout, &[]);
            let features = self.compute_features();
            let score = self.a.score();
            rows.push((name.clone(), score, features));
        }
        if skipped > 0 {
            println!("Skipped {} layouts with incompatible shape", skipped);
        }
        println!("Extracted features for {} loaded layouts", rows.len());

        // Optionally add random/greedy layouts for broader coverage
        // (template_layout already loaded above)

        // Auto-pin REPLACEMENT_CHAR
        let mut all_pins: Vec<usize> = Vec::new();
        for (i, &c) in template_layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR {
                all_pins.push(i);
            }
        }

        if let Some(n) = random_only {
            println!("Generating {} random layouts (no greedy)...", n);
            let data = self.a.data().clone();
            let weights = self.a.weights().clone();
            let scale_factors = self.a.scale_factors().clone();

            let random_results: Vec<_> = (0..n).into_par_iter().map(|i| {
                let random_layout = template_layout.random_with_pins(&all_pins);
                let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                    &random_layout, data.clone(), &weights, &scale_factors,
                );
                let score = cache.score();
                let features = compute_features_from_cache(&mut cache, &[], &0.0);
                (format!("random-{}", i), score, features)
            }).collect();
            rows.extend(random_results);
        }

        if let Some(n) = random_greedy {
            println!("Generating {} random+greedy layouts...", n);
            let data = self.a.data().clone();
            let weights = self.a.weights().clone();
            let scale_factors = self.a.scale_factors().clone();
            let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };

            let done = Arc::new(Mutex::new(0usize));
            let start = std::time::Instant::now();

            let greedy_results: Vec<_> = (0..n).into_par_iter().map(|i| {
                let random_layout = template_layout.random_with_pins(&all_pins);
                let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                    &random_layout, data.clone(), &weights, &scale_factors,
                );
                cache.optimize(&policy, &all_pins);
                let score = cache.score();
                let features = compute_features_from_cache(&mut cache, &[], &0.0);

                if let Ok(mut d) = done.lock() {
                    *d += 1;
                    if *d % 10 == 0 {
                        let elapsed = start.elapsed().as_secs_f64();
                        eprint!("\r  {}/{} greedy | {:.1}/s | {:.1}s   ", *d, n, *d as f64 / elapsed, elapsed);
                    }
                }

                (format!("greedy-{}", i), score, features)
            }).collect();
            eprintln!();
            rows.extend(greedy_results);
        }

        // Write CSV
        if rows.is_empty() {
            println!("No rows to write.");
            return Ok(());
        }

        // Build header from first row
        let first_features = &rows[0].2;
        let mut header = vec!["layout_name".to_string(), "score".to_string()];
        for (name, _) in first_features {
            header.push(name.clone());
        }

        let mut csv_out = String::new();
        csv_out.push_str(&header.join(","));
        csv_out.push('\n');
        for (name, score, features) in &rows {
            csv_out.push_str(name);
            csv_out.push(',');
            csv_out.push_str(&score.to_string());
            for (_, v) in features {
                csv_out.push(',');
                csv_out.push_str(v);
            }
            csv_out.push('\n');
        }

        std::fs::write(output, csv_out)?;
        println!("Wrote {} rows × {} features to {}", rows.len(), header.len() - 2, output);

        Ok(())
    }

    fn layout_names(&self) -> Vec<String> {
        self.layouts.keys().cloned().collect()
    }

    fn compute_features(&mut self) -> Vec<(String, String)> {
        let cache = self.a.cache_mut();
        compute_features_from_cache(cache, &[], &0.0)
    }

    fn ruggedness_cmd(&mut self, name: &str, num_layouts: Option<usize>, num_neighbors: Option<usize>, use_greedy: bool) -> Result<()> {
        use rayon::prelude::*;
        use std::sync::Mutex;
        use std::sync::atomic::AtomicU64;
        use oxeylyzer_core::analyze::Neighbor;
        use oxeylyzer_core::layout::PosPair;
        use oxeylyzer_core::optimization::{RolloutPolicy, OptStep};

        // Simple xorshift RNG (no external dep needed)
        struct Xorshift64 { state: u64 }
        impl Xorshift64 {
            fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }
            fn next(&mut self) -> u64 {
                let mut x = self.state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.state = x;
                x
            }
            fn range(&mut self, n: usize) -> usize {
                (self.next() as usize) % n
            }
        }

        let layout = self.layout(name)?.clone();
        let num_layouts = num_layouts.unwrap_or(200);
        let num_neighbors = num_neighbors.unwrap_or(100);

        let policy = RolloutPolicy { steps: vec![OptStep::Greedy] };

        // Auto-pin REPLACEMENT_CHAR positions so we don't "swap" into unused slots
        let mut all_pins: Vec<usize> = Vec::new();
        for (i, &c) in layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR {
                all_pins.push(i);
            }
        }

        let total_positions = layout.keys.len();
        let valid_positions: Vec<usize> = (0..total_positions)
            .filter(|p| !all_pins.contains(p))
            .collect();

        let start_type = if use_greedy { "greedy-optimized" } else { "random" };
        println!("Ruggedness experiment: {} {} layouts × {} neighbors each",
            num_layouts, start_type, num_neighbors);
        println!("  valid positions: {}", valid_positions.len());
        println!("  pinned (~) positions: {}", all_pins.len());

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();

        let data = self.a.data().clone();
        let weights = self.a.weights().clone();
        let scale_factors = self.a.scale_factors().clone();

        let start = std::time::Instant::now();
        let done = AtomicU64::new(0);
        let last_print = Arc::new(Mutex::new(std::time::Instant::now()));

        // For each random layout: collect (parent_score, neighbor_scores[])
        // We parallelize across layouts.
        let results: Vec<(i64, Vec<i64>)> = (0..num_layouts)
            .into_par_iter()
            .filter_map(|seed_i| {
                if stop.load(Ordering::Relaxed) { return None; }

                let random_layout = layout.random_with_pins(&all_pins);
                let mut cache = oxeylyzer_core::cached_layout::CachedLayout::new(
                    &random_layout, data.clone(), &weights, &scale_factors,
                );

                // Optionally run greedy to local optimum before sampling
                if use_greedy {
                    cache.optimize(&policy, &all_pins);
                }

                let parent_score = cache.score();

                // Sample random single-swap neighbors
                let mut rng = Xorshift64::new(seed_i as u64 + 0xA1B2C3D4);
                let mut neighbor_scores = Vec::with_capacity(num_neighbors);

                for _ in 0..num_neighbors {
                    if valid_positions.len() < 2 { break; }
                    let i = rng.range(valid_positions.len());
                    let mut j = rng.range(valid_positions.len());
                    while j == i { j = rng.range(valid_positions.len()); }
                    let pos_a = valid_positions[i];
                    let pos_b = valid_positions[j];

                    let score = cache.score_neighbor(Neighbor::KeySwap(PosPair(pos_a, pos_b)));
                    neighbor_scores.push(score);
                }

                // Progress
                let count = done.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut lp) = last_print.try_lock() {
                    if lp.elapsed().as_millis() > 500 {
                        *lp = std::time::Instant::now();
                        let elapsed = start.elapsed().as_secs_f64();
                        eprint!("\r  {}/{} layouts | {:.0}/s | {:.1}s   ",
                            count, num_layouts,
                            count as f64 / elapsed,
                            elapsed,
                        );
                    }
                }

                Some((parent_score, neighbor_scores))
            })
            .collect();
        eprintln!();

        if results.is_empty() {
            println!("No results collected.");
            return Ok(());
        }

        // Compute per-layout Pearson correlation: correlation between parent_score
        // (constant for a given layout) and neighbor_scores. But that's degenerate —
        // a constant has 0 variance.
        //
        // Better metric: for each layout, compute correlation between (parent_score, neighbor_score)
        // across the pairs (one parent repeated, each neighbor distinct). This is just
        // measuring if high-parent-score layouts have high-neighbor-score on average.
        //
        // The useful ruggedness metric is different: for each random layout, measure
        // the normalized stdev of neighbor scores (how much does a single swap change things)
        // and the correlation between parent and mean neighbor score.

        // Flatten into parent-neighbor pairs for global Pearson
        let mut all_parent: Vec<f64> = Vec::new();
        let mut all_neighbor: Vec<f64> = Vec::new();
        let mut per_layout_mean_neighbor: Vec<f64> = Vec::new();
        let mut per_layout_std_neighbor: Vec<f64> = Vec::new();
        let mut per_layout_parent: Vec<f64> = Vec::new();

        for (parent, neighbors) in &results {
            if neighbors.is_empty() { continue; }
            let parent_f = *parent as f64;
            per_layout_parent.push(parent_f);
            let mean = neighbors.iter().map(|&s| s as f64).sum::<f64>() / neighbors.len() as f64;
            per_layout_mean_neighbor.push(mean);

            let var = neighbors.iter()
                .map(|&s| (s as f64 - mean).powi(2))
                .sum::<f64>() / neighbors.len() as f64;
            per_layout_std_neighbor.push(var.sqrt());

            for &n in neighbors {
                all_parent.push(parent_f);
                all_neighbor.push(n as f64);
            }
        }

        fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
            if xs.len() != ys.len() || xs.is_empty() { return 0.0; }
            let n = xs.len() as f64;
            let mean_x = xs.iter().sum::<f64>() / n;
            let mean_y = ys.iter().sum::<f64>() / n;
            let mut num = 0.0;
            let mut den_x = 0.0;
            let mut den_y = 0.0;
            for i in 0..xs.len() {
                let dx = xs[i] - mean_x;
                let dy = ys[i] - mean_y;
                num += dx * dy;
                den_x += dx * dx;
                den_y += dy * dy;
            }
            if den_x == 0.0 || den_y == 0.0 { return 0.0; }
            num / (den_x * den_y).sqrt()
        }

        let global_corr = pearson(&all_parent, &all_neighbor);
        let parent_vs_mean_corr = pearson(&per_layout_parent, &per_layout_mean_neighbor);

        // Fitness-distance correlation style: how much does a 1-swap change the score?
        // Compute: for each neighbor, |parent - neighbor| / |parent|, then mean/stdev.
        let mut rel_changes: Vec<f64> = Vec::new();
        for (parent, neighbors) in &results {
            let p = *parent as f64;
            if p == 0.0 { continue; }
            for &n in neighbors {
                rel_changes.push(((n as f64 - p) / p.abs()).abs());
            }
        }
        let mean_rel_change = rel_changes.iter().sum::<f64>() / rel_changes.len().max(1) as f64;

        // Fraction of neighbors that are improvements (score > parent, since less negative is better)
        let mut improve_fracs: Vec<f64> = Vec::new();
        for (parent, neighbors) in &results {
            if neighbors.is_empty() { continue; }
            let improvements = neighbors.iter().filter(|&&s| s > *parent).count();
            improve_fracs.push(improvements as f64 / neighbors.len() as f64);
        }
        let mean_improve_frac = improve_fracs.iter().sum::<f64>() / improve_fracs.len().max(1) as f64;

        // Parent score distribution stats
        let mean_parent = per_layout_parent.iter().sum::<f64>() / per_layout_parent.len() as f64;
        let std_parent = (per_layout_parent.iter()
            .map(|&s| (s - mean_parent).powi(2))
            .sum::<f64>() / per_layout_parent.len() as f64).sqrt();

        // Per-layout mean neighbor-spread: std(neighbors) / |mean(parent)|
        let mean_layout_std = per_layout_std_neighbor.iter().sum::<f64>()
            / per_layout_std_neighbor.len().max(1) as f64;
        let rel_layout_std = mean_layout_std / mean_parent.abs().max(1.0);

        let elapsed = start.elapsed().as_secs_f64();
        println!();
        println!("Ruggedness Results (computed in {:.1}s)", elapsed);
        println!("  samples: {} layouts × up to {} neighbors", results.len(), num_neighbors);
        println!();
        println!("  Parent score distribution:");
        println!("    mean: {}", fmt_num(mean_parent));
        println!("    stdev: {} ({:.1}% of |mean|)", fmt_num(std_parent), 100.0 * std_parent / mean_parent.abs().max(1.0));
        println!();
        println!("  Neighbor variability (per layout, averaged):");
        println!("    mean stdev of neighbor scores: {}", fmt_num(mean_layout_std));
        println!("    relative to |parent|: {:.3}%", 100.0 * rel_layout_std);
        println!();
        println!("  Pearson correlations:");
        println!("    global (parent, neighbor) pairs: {:.3}", global_corr);
        println!("    parent vs mean(neighbors) across layouts: {:.3}", parent_vs_mean_corr);
        println!();
        println!("  Neighbor improvement dynamics:");
        println!("    mean |Δscore| / |parent|: {:.4} ({:.2}%)", mean_rel_change, mean_rel_change * 100.0);
        println!("    mean fraction of improving neighbors: {:.3}", mean_improve_frac);
        println!();
        println!("Interpretation:");
        if global_corr > 0.7 {
            println!("  Smooth landscape (corr > 0.7) — neighbors carry strong fitness info.");
            println!("  Hill climbing and SA/DA should work well.");
        } else if global_corr > 0.3 {
            println!("  Moderately rugged (corr 0.3-0.7) — neighbors have some predictive value.");
            println!("  Greedy + restart is reasonable; structured moves may help further.");
        } else {
            println!("  Very rugged (corr < 0.3) — neighbors are nearly uncorrelated with parent.");
            println!("  This suggests the current key-swap neighborhood is poorly aligned with the");
            println!("  fitness landscape. Structured neighborhoods (block moves, vowel shuffling,");
            println!("  column permutation) may yield a smoother landscape.");
        }

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
        pin_top: Option<usize>,
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
        // Auto-pin REPLACEMENT_CHAR positions
        let mut all_pins = pins.clone();
        for (i, &c) in layout.keys.iter().enumerate() {
            if c == oxeylyzer_core::REPLACEMENT_CHAR && !all_pins.contains(&i) {
                all_pins.push(i);
            }
        }

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
        if let Some(k) = pin_top { config.pin_top_k = k; }

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

        let da = DualAnnealing::new(self.a.data().clone(), self.a.weights().clone(), self.a.scale_factors().clone());
        let result = da.search(&layout, &all_pins, &config, &policy, max_iter, |iter, restarts, current, best| {
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
            OxeylyzerCmd::Gen(g) => self.generate(&g.name, g.pins, g.time, g.top, g.save)?,
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
            OxeylyzerCmd::Da(d) => self.dual_annealing_cmd(&d.name, d.sa, d.sa_temp, d.sa_final, d.greedy, d.iterations, d.time, d.temp, d.qv, d.qa, d.restart, d.swaps, d.pin_top, d.pins)?,
            OxeylyzerCmd::Sa(s) => self.sa_cmd(&s.name, s.count, s.sa, s.sa_temp, s.sa_final, s.greedy, s.time, s.pins)?,
            OxeylyzerCmd::Bh(b) => self.basin_hopping_cmd(&b.name, b.swaps, b.temp, b.restart, b.cool, b.stale, b.time, b.iterations, b.pins)?,
            OxeylyzerCmd::Ruggedness(r) => self.ruggedness_cmd(&r.name, r.layouts, r.neighbors, r.greedy.unwrap_or(false))?,
            OxeylyzerCmd::Dumpfeatures(d) => self.dump_features_cmd(&d.output, d.random_greedy, d.random, d.template.as_deref())?,
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

/// Extract a feature vector from a CachedLayout for PCA/FAMD/UMAP analysis.
#[cfg(not(target_arch = "wasm32"))]
fn compute_features_from_cache(
    cache: &mut oxeylyzer_core::cached_layout::CachedLayout,
    _data_chars: &[i64],
    _char_total: &f64,
) -> Vec<(String, String)> {
    let mut features: Vec<(String, String)> = Vec::new();

    // ===== Outcome features (continuous) =====
    let stats = cache.compute_stats();

    features.push(("sfbs_pct".to_string(), format!("{:.5}", stats.sfbs * 100.0)));
    features.push(("sfs_pct".to_string(), format!("{:.5}", stats.sfs * 100.0)));
    features.push(("stretches".to_string(), format!("{:.5}", stats.stretches)));

    features.push(("inroll_pct".to_string(), format!("{:.5}", stats.trigrams.inroll * 100.0)));
    features.push(("outroll_pct".to_string(), format!("{:.5}", stats.trigrams.outroll * 100.0)));
    features.push(("alternate_pct".to_string(), format!("{:.5}", stats.trigrams.alternate * 100.0)));
    features.push(("redirect_pct".to_string(), format!("{:.5}", stats.trigrams.redirect * 100.0)));
    features.push(("onehandin_pct".to_string(), format!("{:.5}", stats.trigrams.onehandin * 100.0)));
    features.push(("onehandout_pct".to_string(), format!("{:.5}", stats.trigrams.onehandout * 100.0)));

    let finger_labels = ["lp", "lr", "lm", "li", "lt", "rt", "ri", "rm", "rr", "rp"];
    for (i, &fu) in stats.finger_use.iter().enumerate() {
        features.push((format!("fuse_{}", finger_labels[i]), format!("{:.5}", fu)));
    }
    for (i, &fsfb) in stats.finger_sfbs.iter().enumerate() {
        features.push((format!("fsfb_{}", finger_labels[i]), format!("{:.5}", fsfb)));
    }

    let (sfb, stretch, scissors, trigram, magic, finger) = cache.score_breakdown();
    features.push(("score_sfb".to_string(), sfb.to_string()));
    features.push(("score_stretch".to_string(), stretch.to_string()));
    features.push(("score_scissors".to_string(), scissors.to_string()));
    features.push(("score_trigram".to_string(), trigram.to_string()));
    features.push(("score_magic".to_string(), magic.to_string()));
    features.push(("score_finger".to_string(), finger.to_string()));

    // ===== Structural features =====
    let data = cache.data();
    let data_chars = data.chars.clone();

    // Get top 15 most frequent chars
    let mut char_freqs: Vec<(usize, i64)> = data_chars.iter().enumerate().map(|(i, &f)| (i, f)).collect();
    char_freqs.sort_by(|a, b| b.1.cmp(&a.1));
    let top_chars: Vec<usize> = char_freqs.into_iter().take(15).map(|(k, _)| k).collect();

    let char_mapping = cache.char_mapping().clone();

    for (rank, &key_id) in top_chars.iter().enumerate() {
        let c = char_mapping.get_c(key_id);
        let c_safe = if c.is_ascii_alphanumeric() { c.to_string() } else { format!("u{:04x}", c as u32) };
        let pos = cache.get_pos(key_id);
        let finger_str = if let Some(p) = pos {
            let f = cache.finger_at(p);
            finger_labels[f as usize].to_string()
        } else {
            "none".to_string()
        };
        features.push((format!("c{}_{}_finger", rank, c_safe), finger_str));

        if let Some(p) = pos {
            let (row, col) = cache.pos_row_col(p);
            features.push((format!("c{}_{}_row", rank, c_safe), row.to_string()));
            features.push((format!("c{}_{}_col", rank, c_safe), col.to_string()));
        } else {
            features.push((format!("c{}_{}_row", rank, c_safe), "-1".to_string()));
            features.push((format!("c{}_{}_col", rank, c_safe), "-1".to_string()));
        }
    }

    // Aggregates
    let (left_freq, right_freq) = cache.hand_frequencies(&data_chars);
    let total_hand = (left_freq + right_freq).max(1);
    features.push(("left_hand_pct".to_string(), format!("{:.5}", left_freq as f64 / total_hand as f64 * 100.0)));
    features.push(("hand_imbalance".to_string(), format!("{:.5}", (left_freq - right_freq).abs() as f64 / total_hand as f64 * 100.0)));

    let char_total_raw = data.char_total * 100.0;
    let home_row_pct = cache.row_usage(&data_chars, 1) as f64 / char_total_raw * 100.0;
    let top_row_pct = cache.row_usage(&data_chars, 0) as f64 / char_total_raw * 100.0;
    let bot_row_pct = cache.row_usage(&data_chars, 2) as f64 / char_total_raw * 100.0;
    features.push(("home_row_pct".to_string(), format!("{:.5}", home_row_pct)));
    features.push(("top_row_pct".to_string(), format!("{:.5}", top_row_pct)));
    features.push(("bot_row_pct".to_string(), format!("{:.5}", bot_row_pct)));

    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let (left_vowel, right_vowel) = cache.hand_freq_for_chars(&data_chars, &vowels);
    let total_vowel = (left_vowel + right_vowel).max(1);
    features.push(("vowels_left_pct".to_string(), format!("{:.5}", left_vowel as f64 / total_vowel as f64 * 100.0)));

    features.push(("magic_rule_count".to_string(), cache.magic_rule_count().to_string()));

    features
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
