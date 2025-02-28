mod config;
mod flags;

use config::Config;
use itertools::Itertools;
use oxeylyzer_core::{cached_layout::BigramPair, prelude::*};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{stdout, Write as _},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
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

    fn analyze(&self, name: &str) -> Result<()> {
        let layout = self.layout(name)?;
        let stats = self.a.stats(layout);

        let finger_use = stats.finger_use.map(|f| format!("{f:.2}")).join(", ");
        let finger_sfbs = stats.finger_sfbs.map(|f| format!("{f:.2}")).join(", ");

        print!("{}", layout);

        println!(
            concat!(
                "score:     {}\n\n",
                "sfbs:      {:.3}%\n",
                "sfs:       {:.3}%\n",
                "stretches: {:.3}\n",
                "finger usage:\n{}\n",
                "finger sfbs:\n{}\n"
            ),
            stats.score, stats.sfbs, stats.sfs, stats.stretches, finger_use, finger_sfbs,
        );

        self.trigrams(name)
    }

    fn rank(&self) {
        self.layouts
            .iter()
            .map(|(n, l)| {
                let s = self.a.score(l);
                (n, s)
            })
            .sorted_by(|(_, a), (_, b)| a.cmp(b))
            .for_each(|(n, s)| println!("{n:<15} {s}"));
    }

    fn generate(&self, name: &str, count: Option<usize>, pin_chars: Option<String>) -> Result<()> {
        let layout = self.layout(name)?;
        let count = count.unwrap_or(10);
        let pins = match pin_chars {
            Some(chars) => pin_positions(layout, chars),
            None => vec![],
        };

        let start = std::time::Instant::now();
        let iteration = AtomicUsize::new(0);

        let mut layouts = Vec::with_capacity(count);
        (0..count)
            .into_par_iter()
            .map(|_| {
                let l = layout.random_with_pins(&pins);
                // let res = self.a.alternative_d3(l, &pins);
                // self.a.greedy_depth2_improve(l)
                // .annealing_improve(starting_layout, 20_500_000_000_000.0, 0.987, 5000)
                let (annealed_layout, _) =
                    self.a.annealing_improve(l, &pins, 10.0, 1E-5, 10_000_000);
                let res = self.a.alternative_d3(annealed_layout, &pins);

                iteration.fetch_add(1, Ordering::Relaxed);

                print!(
                    "\rgenerated {}/{}",
                    iteration.load(Ordering::Relaxed),
                    count
                );
                stdout().flush().unwrap();

                res
            })
            .collect_into_vec(&mut layouts);

        layouts.sort_by(|(_, s1), (_, s2)| s2.cmp(s1));

        for (i, (mut layout, score)) in layouts.into_iter().enumerate().take(10) {
            layout.name = "".into();
            println!("#{}, score: {}{}", i, score, layout);
        }

        println!(
            "generating {} variants took {:.2} seconds.",
            count,
            start.elapsed().as_secs_f64()
        );

        Ok(())
    }

    fn sfbs(&self, name: &str, count: Option<usize>) -> Result<()> {
        let layout = self.layout(name)?;
        let cache = self.a.cached_layout(layout.clone(), &[]);
        let count = count.unwrap_or(10);

        cache
            .weighted_sfb_indices
            .all
            .iter()
            .flat_map(
                |BigramPair {
                     pair: PosPair(a, b),
                     ..
                 }| {
                    let u1 = cache.keys[*a as usize];
                    let u2 = cache.keys[*b as usize];

                    let c1 = self.a.mapping().get_c(u1);
                    let c2 = self.a.mapping().get_c(u2);

                    let freq = self.a.data.get_bigram_u([u1, u2]) as f64 / self.a.data.bigram_total;
                    let freq2 =
                        self.a.data.get_bigram_u([u2, u1]) as f64 / self.a.data.bigram_total;

                    [([c1, c2], freq), ([c2, c1], freq2)]
                },
            )
            .sorted_by(|(_, f1), (_, f2)| f2.total_cmp(f1))
            .take(count)
            .for_each(|([c1, c2], f)| println!("{c1}{c2}: {f:.3}%"));

        Ok(())
    }

    fn stretches(&self, name: &str, count: Option<usize>) -> Result<()> {
        let layout = self.layout(name)?;
        let cache = self.a.cached_layout(layout.clone(), &[]);
        let count = count.unwrap_or(10);

        cache
            .stretch_bigrams
            .all_pairs
            .iter()
            .flat_map(
                |BigramPair {
                     pair: PosPair(a, b),
                     dist,
                 }| {
                    let u1 = cache.keys[*a as usize];
                    let u2 = cache.keys[*b as usize];

                    let c1 = self.a.mapping().get_c(u1);
                    let c2 = self.a.mapping().get_c(u2);

                    let f = self.a.data.get_bigram_u([u1, u2]) as f64 / self.a.data.bigram_total;
                    let f2 = self.a.data.get_bigram_u([u2, u1]) as f64 / self.a.data.bigram_total;
                    let dist = *dist as f64;

                    [([c1, c2], f * dist), ([c2, c1], f2 * dist)]
                },
            )
            .sorted_by(|(_, f1), (_, f2)| f2.total_cmp(f1))
            .take(count)
            .for_each(|([c1, c2], f)| println!("{c1}{c2}: {f:.3}"));

        Ok(())
    }

    pub fn trigrams(&self, name: &str) -> Result<()> {
        let layout = self.layout(name)?;
        let trigram_stats = self.a.stats(layout).trigrams;

        if trigram_stats.sft != 0.0 {
            println!("Sft:          {:.3}%", trigram_stats.sft);
        }
        if trigram_stats.sfb != 0.0 {
            println!("Sfb:          {:.3}%", trigram_stats.sfb);
        }
        if trigram_stats.inroll != 0.0 {
            println!("Inroll:       {:.3}%", trigram_stats.inroll);
        }
        if trigram_stats.outroll != 0.0 {
            println!("Outroll:      {:.3}%", trigram_stats.outroll);
        }
        if trigram_stats.alternate != 0.0 {
            println!("Alternate:    {:.3}%", trigram_stats.alternate);
        }
        if trigram_stats.redirect != 0.0 {
            println!("Redirect:     {:.3}%", trigram_stats.redirect);
        }
        if trigram_stats.onehandin != 0.0 {
            println!("Onehand In:   {:.3}%", trigram_stats.onehandin);
        }
        if trigram_stats.onehandout != 0.0 {
            println!("Onehand Out:  {:.3}%", trigram_stats.onehandout);
        }
        if trigram_stats.thumb != 0.0 {
            println!("Thumb:        {:.3}%", trigram_stats.thumb);
        }
        if trigram_stats.invalid != 0.0 {
            println!("Invalid:      {:.3}%", trigram_stats.invalid);
        }

        Ok(())
    }

    pub fn similarity(&self, name: &str) -> Result<()> {
        let layout = self.layout(name)?;

        self.layouts
            .values()
            .filter(|cmp| cmp.name.to_lowercase() != name.to_lowercase())
            .map(|cmp| (cmp.name.as_str(), self.a.similarity(layout, cmp)))
            .sorted_by(|(_, s1), (_, s2)| s2.cmp(s1))
            .for_each(|(n, s)| println!("{n:<15} {:.3}", s as f64 / self.a.data.char_total));

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
    // } else if !path.exists() {
    //     fs::create_dir_all(path)?;
    //     Ok(HashMap::default())
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
