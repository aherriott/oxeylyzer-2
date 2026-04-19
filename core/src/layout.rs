use fxhash::FxHashMap as HashMap;
use libdof::magic::MagicKey;
use libdof::prelude::{Dof, Finger, Keyboard, PhysicalKey, Shape};
use nanorand::{tls_rng, Rng as _};

use crate::{
    types::CacheKey, Result, MAGIC_CHARS, REPEAT_KEY,
    REPLACEMENT_CHAR, SHIFT_CHAR, SPACE_CHAR,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PosPair(pub CacheKey, pub CacheKey);

impl<U: Into<CacheKey>> From<(U, U)> for PosPair {
    fn from((p1, p2): (U, U)) -> Self {
        Self(p1.into(), p2.into())
    }
}

/// MagicRule represents setting a magic output for a leader key.
/// (magic_key, leader, output) - when leader is typed before magic_key, output is produced.
/// output can be EMPTY_KEY to clear the rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagicRule {
    pub magic_key: CacheKey,
    pub leader: CacheKey,
    pub output: CacheKey,
}

impl MagicRule {
    pub fn new(magic_key: CacheKey, leader: CacheKey, output: CacheKey) -> Self {
        Self { magic_key, leader, output }
    }
}

impl<U: Into<CacheKey>> From<(U, U, U)> for MagicRule {
    fn from((magic_key, leader, output): (U, U, U)) -> Self {
        Self {
            magic_key: magic_key.into(),
            leader: leader.into(),
            output: output.into(),
        }
    }
}

// #[derive(Debug, Clone, Copy, PartialEq)]
// pub struct PhysicalPos(pub i64, pub i64);

// impl PhysicalPos {
//     pub fn dist(&self, other: &Self) -> i64 {
//         let x = self.0.abs_diff(other.0) as f64;
//         let y = self.1.abs_diff(other.1) as f64;

//         x.hypot(y) as i64
//     }

//     pub fn dist_squared(&self, other: &Self) -> i64 {
//         (self.0 - other.0).pow(2) + (self.1 - other.1).pow(2)
//     }
// }

// impl<F: Into<f64>> From<(F, F)> for PhysicalPos {
//     fn from((x, y): (F, F)) -> Self {
//         Self((x.into() * 100.0) as i64, (y.into() * 100.0) as i64)
//     }
// }

// impl From<PhysicalKey> for PhysicalPos {
//     fn from(pk: PhysicalKey) -> Self {
//         let x = pk.x() + (0.5 * pk.width());
//         let y = pk.y() + (0.5 * pk.height());

//         (x, y).into()
//     }
// }

#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    pub name: String,
    pub keys: Box<[char]>,
    pub fingers: Box<[Finger]>,
    pub keyboard: Box<[PhysicalKey]>,
    pub shape: Shape,
    pub magic: HashMap<char, MagicKey>,
}

#[inline]
fn shuffle_pins<T>(slice: &mut [T], pins: &[usize]) {
    let mapping = (0..slice.len())
        .filter(|x| !pins.contains(x))
        .collect::<Vec<_>>();
    let mut rng = tls_rng();

    for (m, &swap1) in mapping.iter().enumerate() {
        let swap2 = rng.generate_range(m..mapping.len());
        slice.swap(swap1, mapping[swap2]);
    }
}

impl Layout {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;

        // Parse magic rules from raw JSON (libdof doesn't expose them)
        let raw_magic = Self::parse_magic_from_json(&s);

        let mut layout: Layout = serde_json::from_str::<Dof>(&s)
            .map(Into::into)?;

        // Populate magic rules that From<Dof> couldn't access
        for (magic_char, magic_key) in &mut layout.magic {
            if let Some(rules) = raw_magic.get(&magic_key.label().to_string()) {
                for (leader, output) in rules {
                    magic_key.add_rule(leader, output);
                }
            }
        }

        Ok(layout)
    }

    /// Extract magic rules from raw JSON since libdof doesn't expose them.
    #[cfg(not(target_arch = "wasm32"))]
    fn parse_magic_from_json(json_str: &str) -> HashMap<String, Vec<(String, String)>> {
        let mut result = HashMap::default();
        let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) else { return result };
        let Some(magic_obj) = val.get("magic").and_then(|m| m.as_object()) else { return result };

        for (label, rules_val) in magic_obj {
            let Some(rules_obj) = rules_val.as_object() else { continue };
            let mut rules = Vec::new();
            for (leader, output) in rules_obj {
                if let Some(output_str) = output.as_str() {
                    rules.push((leader.clone(), output_str.to_string()));
                }
            }
            result.insert(label.clone(), rules);
        }
        result
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load(url: &str) -> Result<Self> {
        let dof = gloo_net::http::Request::get(url)
            .send()
            .await?
            .json::<Dof>()
            .await?;

        Ok(dof.into())
    }

    pub fn random(&self) -> Self {
        self.random_with_pins(&[])
    }

    pub fn random_with_pins(&self, pins: &[usize]) -> Self {
        use nanorand::{tls_rng, Rng as _};

        let shape = self.shape.clone();
        let fingers = self.fingers.clone();
        let keyboard = self.keyboard.clone();

        let mut keys = self.keys.clone();
        // Auto-pin positions with REPLACEMENT_CHAR (unused ~ positions) so they don't
        // get shuffled into main rows
        let mut all_pins: Vec<usize> = pins.to_vec();
        for (i, &c) in self.keys.iter().enumerate() {
            if c == REPLACEMENT_CHAR && !all_pins.contains(&i) {
                all_pins.push(i);
            }
        }
        shuffle_pins(&mut keys, &all_pins);

        // Randomize magic rules: for each magic key, assign random leader→output pairs
        let non_magic_keys: Vec<char> = keys.iter()
            .filter(|&&c| !MAGIC_CHARS.contains(c) && c != REPLACEMENT_CHAR && c != ' ')
            .copied()
            .collect();

        let magic = self.magic.iter()
            .map(|(&c, mk)| {
                let mut new_mk = MagicKey::new(mk.label());
                let mut rng = tls_rng();
                for &leader in &non_magic_keys {
                    let output = non_magic_keys[rng.generate_range(0..non_magic_keys.len())];
                    new_mk.add_rule(&leader.to_string(), &output.to_string());
                }
                (c, new_mk)
            })
            .collect();

        Self {
            name: keys.iter().collect(),
            keys,
            fingers,
            keyboard,
            shape,
            magic,
        }
    }

    /// Save layout as a .dof JSON file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_dof<P: AsRef<std::path::Path>>(&self, path: P, name: &str) -> Result<()> {
        use serde_json::{json, Map, Value};

        let shape = self.shape.inner();
        let mut key_iter = self.keys.iter();
        let mut rows: Vec<String> = Vec::new();

        for &row_len in shape.iter() {
            let mut row_chars: Vec<String> = Vec::new();
            for _ in 0..row_len {
                if let Some(&c) = key_iter.next() {
                    let s = if c == REPLACEMENT_CHAR {
                        "~".to_string()
                    } else if MAGIC_CHARS.contains(c) {
                        "&mag".to_string()
                    } else if c == SPACE_CHAR || c == ' ' {
                        "spc".to_string()
                    } else if c == SHIFT_CHAR {
                        "shift".to_string()
                    } else {
                        c.to_string()
                    };
                    row_chars.push(s);
                }
            }
            // Format row with gap in the middle (5 + 5 for 10-key rows)
            if row_chars.len() >= 10 {
                let left = row_chars[..5].join(" ");
                let right = row_chars[5..].join(" ");
                rows.push(format!("{}  {}", left, right));
            } else if row_chars.len() > 3 {
                let mid = row_chars.len() / 2;
                let left = row_chars[..mid].join(" ");
                let right = row_chars[mid..].join(" ");
                rows.push(format!("{}  {}", left, right));
            } else {
                rows.push(row_chars.join(" "));
            }
        }

        // Build magic rules JSON
        let mut magic_obj = Map::new();
        for (_magic_char, magic_key) in &self.magic {
            let label = "mag".to_string(); // .dof files use "mag", not the internal char
            let mut rules_map = Map::new();
            for (leader, output) in magic_key.rules() {
                // Map special chars back to .dof format
                let leader_str = if *leader == SPACE_CHAR.to_string() { " ".to_string() } else { leader.to_string() };
                let output_str = if *output == SPACE_CHAR.to_string() { " ".to_string() } else { output.to_string() };
                rules_map.insert(leader_str, Value::String(output_str));
            }
            magic_obj.insert(label, Value::Object(rules_map));
        }

        let dof = json!({
            "name": name,
            "board": "colstag",
            "layers": {
                "main": rows
            },
            "fingering": "traditional",
            "magic": magic_obj
        });

        let json_str = serde_json::to_string_pretty(&dof)?;
        std::fs::write(path, json_str)?;
        Ok(())
    }
}

impl From<Dof> for Layout {
    fn from(dof: Dof) -> Self {
        use libdof::prelude::{Key, SpecialKey};

        let mut magic: HashMap<char, MagicKey> = HashMap::default();
        let mut magic_i = 0;
        let keys = dof
            .main_layer()
            .keys()
            .map(|k| match k {
                Key::Char(c) => *c,
                Key::Special(s) => match s {
                    SpecialKey::Repeat => REPEAT_KEY,
                    SpecialKey::Space => SPACE_CHAR,
                    SpecialKey::Shift => SHIFT_CHAR,
                    _ => REPLACEMENT_CHAR,
                },
                Key::Magic { label } => {
                    // Map magic chars to their keys
                    let c = MAGIC_CHARS.chars().nth(magic_i);
                    magic_i += 1;
                    // TODO: libdof doesn't expose magic keys directly on Dof
                    // For now, create an empty magic key - this needs proper fix
                    magic.insert(c.unwrap(), MagicKey::new(label));
                    c.unwrap()
                }
                _ => REPLACEMENT_CHAR,
            })
            .collect();

        let name = dof.name().to_owned();
        let fingers = dof.fingering().keys().copied().collect();
        let keyboard = dof.board().keys().cloned().map(Into::into).collect();
        let shape = dof.main_layer().shape();

        Layout {
            name,
            keys,
            fingers,
            keyboard,
            shape,
            magic,
        }
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.name)?;

        let mut iter = self.keys.iter();

        for l in self.shape.inner().iter() {
            let mut i = 0;
            for c in iter.by_ref() {
                write!(f, "{c} ")?;
                i += 1;

                if *l == i {
                    break;
                }
            }
            writeln!(f)?;
        }

        //TODO: add magic keys
        if !self.magic.is_empty() {
            let mut rules: Vec<(char, String, String)> = Vec::new();
            for (magic_char, magic_key) in &self.magic {
                for (leader, output) in magic_key.rules() {
                    rules.push((*magic_char, leader.to_string(), output.to_string()));
                }
            }
            rules.sort_by(|a, b| a.1.cmp(&b.1));
            let rule_strs: Vec<String> = rules.iter()
                .map(|(_mc, leader, output)| format!("{leader}→{output}"))
                .collect();
            writeln!(f, "magic: {}", rule_strs.join(" "))?;
        }

        Ok(())
    }
}
