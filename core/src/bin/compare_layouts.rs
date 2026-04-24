use oxeylyzer_core::prelude::*;

fn main() {
    let data = Data::load("data/english.json").expect("english.json");
    let weights_str = std::fs::read_to_string("analyzer-config.toml").expect("config");
    let config: toml::Value = toml::from_str(&weights_str).expect("parse config");
    let weights: Weights = config.get("weights").unwrap().clone().try_into().expect("weights");

    let layouts = [
        "sturdy", "Canary", "gallium",
        "my-layout-r11-1", "my-layout-r11-2", "my-layout-r11-3",
    ];

    println!("{:<20} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "Layout", "Score", "SFBs%", "SFS%", "Stretch", "Inroll", "Outroll", "Alt", "Redir", "Rules");
    println!("{}", "-".repeat(110));

    for name in &layouts {
        let path = format!("layouts/{name}.dof");
        let layout = oxeylyzer_core::layout::Layout::load(&path)
            .unwrap_or_else(|e| panic!("Failed to load {path}: {e}"));

        let scale_factors = weights.compute_scale_factors(&data);
        let mut a = Analyzer::new(data.clone(), weights.clone());
        a.use_layout(&layout, &[]);

        let stats = a.stats();
        let score = a.score();
        let rules = a.cache_mut().magic_rule_count();

        println!("{:<20} {:>10} {:>7.2}% {:>7.2}% {:>8.3} {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}% {:>6}",
            name,
            format_score(score),
            stats.sfbs * 100.0,
            stats.sfs * 100.0,
            stats.stretches,
            stats.trigrams.inroll * 100.0,
            stats.trigrams.outroll * 100.0,
            stats.trigrams.alternate * 100.0,
            stats.trigrams.redirect * 100.0,
            rules,
        );
    }
}

fn format_score(s: i64) -> String {
    let abs = s.abs() as f64;
    let sign = if s < 0 { "-" } else { "" };
    if abs >= 1e12 { format!("{sign}{:.2}T", abs / 1e12) }
    else if abs >= 1e9 { format!("{sign}{:.2}B", abs / 1e9) }
    else if abs >= 1e6 { format!("{sign}{:.2}M", abs / 1e6) }
    else { format!("{sign}{:.0}", abs) }
}
