mod config;
mod flags;

use config::Config;
use oxeylyzer_core::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{stdout, Write as _},
    path::{Path, PathBuf},
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

        Ok(Self {
            a,
            layouts,
            config_path,
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

        print!("{}", layout);

        println!(
            concat!(
                "score:     {}\n\n",
                "sfbs:      {:.3}%\n",
                "sfs:       {:.3}%\n",
                "stretches: {:.3}\n",
                "finger usage:\n  {}\n",
                "finger sfbs:\n  {}\n"
            ),
            stats.score,
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

    fn generate(&mut self, name: &str, count: Option<usize>, pin_chars: Option<String>) -> Result<()> {
        let layout = self.layout(name)?.clone();
        let count = count.unwrap_or(10);
        let pins = match pin_chars {
            Some(chars) => pin_positions(&layout, chars),
            None => vec![],
        };

        let start = std::time::Instant::now();

        let mut results: Vec<(Layout, i64)> = Vec::with_capacity(count);

        for i in 0..count {
            let random_layout = layout.random_with_pins(&pins);

            // Simulated annealing followed by greedy optimization
            self.a.use_layout(&random_layout, &pins);
            let (annealed_layout, _) =
                self.a.annealing_improve(random_layout, &pins, 10.0, 1E-5, 1_000_000);
            let (final_layout, final_score) = self.a.greedy_improve(&annealed_layout, &pins);

            results.push((final_layout, final_score));

            print!("\rgenerated {}/{}", i + 1, count);
            stdout().flush().unwrap();
        }
        println!();

        // Sort by score (higher = better)
        results.sort_by(|(_, s1), (_, s2)| s2.cmp(s1));

        for (i, (mut layout, score)) in results.into_iter().enumerate().take(10) {
            layout.name = "".into();
            println!("#{}, score: {}{}", i + 1, score, layout);
        }

        println!(
            "generating {} variants took {:.2} seconds.",
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
        for i in 0..5 {
            let random = layout.random();
            self.a.use_layout(&random, &[]);
            let (sa_layout, _) = self.a.annealing_improve(random.clone(), &[], 1.0, 1E-4, 100_000);
            let (_, greedy_score) = self.a.greedy_improve(&sa_layout, &[]);
            if greedy_score > best_bound {
                best_bound = greedy_score;
            }
            print!("\r  bound pass {}/5: {}", i + 1, best_bound);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!("\nInitial bound: {} (found in {:.1}s)", best_bound, bound_start.elapsed().as_secs_f64());

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
                        progress.best_score.map_or("none".to_string(), |s| s.to_string()),
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
            println!("#{}: score {}", i + 1, result.score);
            for (c, pos) in &result.key_positions {
                print!("{}@{} ", c, pos);
            }
            println!();
        }

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
            OxeylyzerCmd::Gen(g) => self.generate(&g.name, g.count, g.pins)?,
            OxeylyzerCmd::Sfbs(s) => self.sfbs(&s.name, s.count)?,
            OxeylyzerCmd::Stretches(s) => self.stretches(&s.name, s.count)?,
            OxeylyzerCmd::Trigrams(t) => self.trigrams(&t.name)?,
            OxeylyzerCmd::Similarity(s) => self.similarity(&s.name)?,
            OxeylyzerCmd::R(_) => self.reload()?,
            OxeylyzerCmd::Bb(b) => self.branch_bound(&b.name, b.depth, b.top)?,
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
            return format!("{:.1}{}", n / threshold, suffix);
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
