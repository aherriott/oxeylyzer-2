use libdof::prelude::{Finger, PhysicalKey, Shape};
use libdof::magic::MagicKey;
use fxhash::FxHashMap as HashMap;

use crate::{
    analyze::Neighbor,
    analyzer_data::AnalyzerData,
    char_mapping::CharMapping,
    data::Data,
    dist::DistCache,
    layout::{Layout, MagicRule, PosPair},
    same_finger::SFCache,
    scissors::ScissorsCache,
    stats::Stats,
    stretches::StretchCache,
    trigrams::TrigramCache,
    types::{CacheKey, CachePos},
    weights::Weights,
};

pub const EMPTY_KEY: CacheKey = CacheKey::MAX;

/*
 **************************************
 *     Magic and Repeat Remapping
 **************************************
 */

/// MagicCache stores frequency tables that get modified by magic key rules.
/// When a magic rule "steals" a bigram, the frequencies are redistributed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    /// Flat bigram frequencies: bg_freq[a * num_keys + b]
    bg_freq: Vec<i64>,
    /// Flat skipgram frequencies: sg_freq[a * num_keys + b]
    sg_freq: Vec<i64>,
    /// Trigram frequencies
    tg_freq: Vec<Vec<Vec<i64>>>,
    /// Number of keys for indexing
    num_keys: usize,
}

impl MagicCache {
    pub fn new(num_keys: usize) -> Self {
        Self {
            bg_freq: vec![0; num_keys * num_keys],
            sg_freq: vec![0; num_keys * num_keys],
            tg_freq: vec![vec![vec![0; num_keys]; num_keys]; num_keys],
            num_keys,
        }
    }

    /// Initialize frequencies from corpus data (2D arrays)
    pub fn init_from_data(&mut self, bigrams: &[Vec<i64>], skipgrams: &[Vec<i64>], trigrams: &[Vec<Vec<i64>>]) {
        let num_keys = self.num_keys;
        for (a, row) in bigrams.iter().enumerate() {
            for (b, &freq) in row.iter().enumerate() {
                self.bg_freq[a * num_keys + b] = freq;
            }
        }
        for (a, row) in skipgrams.iter().enumerate() {
            for (b, &freq) in row.iter().enumerate() {
                self.sg_freq[a * num_keys + b] = freq;
            }
        }
        self.tg_freq = trigrams.to_vec();
    }

    #[inline]
    pub fn bg_freq_flat(&self) -> &[i64] {
        &self.bg_freq
    }

    #[inline]
    pub fn sg_freq_flat(&self) -> &[i64] {
        &self.sg_freq
    }

    #[inline]
    pub fn tg_freq(&self) -> &[Vec<Vec<i64>>] {
        &self.tg_freq
    }

    #[inline]
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        if a < self.num_keys && b < self.num_keys {
            self.bg_freq[a * self.num_keys + b]
        } else {
            0
        }
    }

    #[inline]
    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        if a < self.num_keys && b < self.num_keys {
            self.sg_freq[a * self.num_keys + b]
        } else {
            0
        }
    }

    #[inline]
    pub fn get_tg_freq(&self, a: CacheKey, b: CacheKey, c: CacheKey) -> i64 {
        self.tg_freq
            .get(a)
            .and_then(|r1| r1.get(b))
            .and_then(|r2| r2.get(c))
            .copied()
            .unwrap_or(0)
    }

    /// Steal a bigram: when typing leader->output, magic key M intercepts it.
    /// This redistributes frequencies from (leader, output) to (leader, magic_key).
    ///
    /// Populates affected_grams with position-based deltas for cache updates.
    ///
    /// The following calculations are based on the idea that magic keys "steal" bigrams from regular keys.
    /// i.e. when you add the rule A->B to magic key M, the bigram A->B is never typed, because you always
    /// type A->M instead.
    ///
    /// Assuming key order: Z->A->B->C
    ///
    /// If B is magic:
    /// 1. The bigram A->B is fully stolen by A->M
    /// 2. The bigram B->C is partially stolen by M->C, based on the rate of the A->B->C trigram (which is now typed A->M->C)
    /// 3. The skipgram Z->B is partially stolen by Z->M, based on the rate of the Z->A->B trigram
    /// 4. The trigram Z->A->B is fully stolen by Z->A->M
    /// 5. The trigram A->B->C is fully stolen by A->M->C
    pub fn steal_bigram(
        &mut self,
        a: CacheKey,      // leader
        b: CacheKey,      // output (being stolen)
        m: CacheKey,      // magic key
        key_positions: &[Option<CachePos>],
        num_keys: usize,
        affected_grams: &mut Vec<DeltaGram>,
    ) {
        let nk = self.num_keys;

        let get_pos = |k: CacheKey| -> Option<CachePos> {
            key_positions.get(k).copied().flatten()
        };

        macro_rules! set_bg {
            ($ka:expr, $kb:expr, $new:expr) => {{
                let idx = $ka * nk + $kb;
                let old = self.bg_freq[idx];
                self.bg_freq[idx] = $new;
                if let (Some(p_a), Some(p_b)) = (get_pos($ka), get_pos($kb)) {
                    affected_grams.push(DeltaGram::Bigram(DeltaBigram {
                        p_a,
                        p_b,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        macro_rules! set_sg {
            ($ka:expr, $kb:expr, $new:expr) => {{
                let idx = $ka * nk + $kb;
                let old = self.sg_freq[idx];
                self.sg_freq[idx] = $new;
                if let (Some(p_a), Some(p_b)) = (get_pos($ka), get_pos($kb)) {
                    affected_grams.push(DeltaGram::Skipgram(DeltaSkipgram {
                        p_a,
                        p_b,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        macro_rules! set_tg {
            ($ka:expr, $kb:expr, $kc:expr, $new:expr) => {{
                let old = self.tg_freq[$ka][$kb][$kc];
                self.tg_freq[$ka][$kb][$kc] = $new;
                if let (Some(p_a), Some(p_b), Some(p_c)) = (get_pos($ka), get_pos($kb), get_pos($kc)) {
                    affected_grams.push(DeltaGram::Trigram(DeltaTrigram {
                        p_a,
                        p_b,
                        p_c,
                        old_freq: old,
                        new_freq: $new,
                    }));
                }
            }};
        }

        // 1. The exact bigram A->B is fully stolen by A->M
        let new_am = self.bg_freq[a * nk + m] + self.bg_freq[a * nk + b];
        set_bg!(a, m, new_am);
        set_bg!(a, b, 0);

        // 2. For each key c: B->C is partially stolen by M->C based on trigram A->B->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.bg_freq[b * nk + c] >= tg);
            let new_mc = self.bg_freq[m * nk + c] + tg;
            let new_bc = self.bg_freq[b * nk + c] - tg;
            set_bg!(m, c, new_mc);
            set_bg!(b, c, new_bc);
        }

        // 3. For each key z: skipgram Z->B is partially stolen by Z->M based on trigram Z->A->B
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.sg_freq[z * nk + b] >= tg);
            let new_zm = self.sg_freq[z * nk + m] + tg;
            let new_zb = self.sg_freq[z * nk + b] - tg;
            set_sg!(z, m, new_zm);
            set_sg!(z, b, new_zb);
        }

        // 4. For each key z: trigram Z->A->B is fully stolen by Z->A->M
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            let new_zam = self.tg_freq[z][a][m] + tg;
            set_tg!(z, a, m, new_zam);
            set_tg!(z, a, b, 0);
        }

        // 5. For each key c: trigram A->B->C is fully stolen by A->M->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            let new_amc = self.tg_freq[a][m][c] + tg;
            set_tg!(a, m, c, new_amc);
            set_tg!(a, b, c, 0);
        }
    }

    /// Revert frequency changes from affected_grams
    pub fn revert_affected(&mut self, affected_grams: &[DeltaGram], key_at_pos: impl Fn(CachePos) -> CacheKey) {
        let nk = self.num_keys;
        for gram in affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    let a = key_at_pos(bg.p_a);
                    let b = key_at_pos(bg.p_b);
                    let idx = a * nk + b;
                    self.bg_freq[idx] = bg.old_freq;
                }
                DeltaGram::Skipgram(sg) => {
                    let a = key_at_pos(sg.p_a);
                    let b = key_at_pos(sg.p_b);
                    let idx = a * nk + b;
                    self.sg_freq[idx] = sg.old_freq;
                }
                DeltaGram::Trigram(tg) => {
                    let a = key_at_pos(tg.p_a);
                    let b = key_at_pos(tg.p_b);
                    let c = key_at_pos(tg.p_c);
                    self.tg_freq[a][b][c] = tg.old_freq;
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaBigram {
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaSkipgram {
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaTrigram {
    pub p_a: CachePos,
    pub p_b: CachePos,
    pub p_c: CachePos,
    pub old_freq: i64,
    pub new_freq: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeltaGram {
    Bigram(DeltaBigram),
    Skipgram(DeltaSkipgram),
    Trigram(DeltaTrigram),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedLayout {
    name: String,
    keyboard: Box<[PhysicalKey]>,
    shape: Shape,
    data: AnalyzerData,
    keys: Vec<CacheKey>,
    key_positions: Vec<Option<CachePos>>,
    num_positions: usize,
    neighbors: Vec<Neighbor>,
    current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey>,
    affected_grams: Vec<DeltaGram>,
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
    scissors: ScissorsCache,
    trigram: TrigramCache,
    magic: MagicCache,
    fingers: Vec<Finger>,
}

impl Default for CachedLayout {
    fn default() -> Self {
        Self {
            name: String::new(),
            keyboard: Box::new([]),
            shape: Shape::default(),
            data: AnalyzerData::default(),
            keys: Vec::new(),
            key_positions: Vec::new(),
            num_positions: 0,
            neighbors: Vec::new(),
            current_magic_rules: HashMap::default(),
            affected_grams: Vec::new(),
            dist: DistCache::default(),
            sfb: SFCache::default(),
            stretch: StretchCache::default(),
            scissors: ScissorsCache::default(),
            trigram: TrigramCache::default(),
            magic: MagicCache::default(),
            fingers: Vec::new(),
        }
    }
}


impl CachedLayout {
    pub fn new(layout: &Layout, data: Data, weights: &Weights) -> Self {
        let mut analyzer_data = AnalyzerData::new(data);

        // Ensure all layout keys are in the char_mapping for proper roundtrip
        for &key_char in layout.keys.iter() {
            analyzer_data.push_char(key_char);
        }

        let len = layout.keyboard.len();
        let keyboard = &layout.keyboard;
        let fingers = &layout.fingers;
        let num_keys = analyzer_data.len();

        let keys = vec![EMPTY_KEY; len];
        let key_positions = vec![None; num_keys];
        let affected_grams = Vec::with_capacity(len * len);

        let dist = DistCache::new(keyboard, fingers);
        let mut sfb = SFCache::new(fingers, keyboard, dist.distances(), num_keys);
        sfb.set_weights(weights);
        let mut stretch = StretchCache::new(keyboard, fingers, num_keys);
        stretch.set_weights(weights);
        let mut scissors = ScissorsCache::new(keyboard, fingers, num_keys);
        scissors.set_weights(weights);
        let mut trigram = TrigramCache::new(fingers, num_keys);
        trigram.set_weights(weights);
        let mut magic = MagicCache::new(num_keys);
        magic.init_from_data(&analyzer_data.bigrams, &analyzer_data.skipgrams, &analyzer_data.trigrams);

        // Pre-compute all neighbors
        let mut neighbors = Vec::new();

        // Key swap neighbors: all pairs (a, b) where a < b
        for a in 0..len {
            for b in (a + 1)..len {
                neighbors.push(Neighbor::KeySwap(PosPair(a, b)));
            }
        }

        // Magic rule neighbors
        let mut current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey> = HashMap::default();

        for (&magic_char, magic_key_def) in layout.magic.iter() {
            let magic_key = analyzer_data.char_mapping().get_u(magic_char);

            // Collect all possible outputs for this magic key
            let mut all_outputs: Vec<CacheKey> = Vec::new();

            for (leading_str, output_str) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                let output = analyzer_data.char_mapping().get_u(output_str.chars().next().unwrap_or(' '));
                current_magic_rules.insert((magic_key, leader), output);
                if !all_outputs.contains(&output) {
                    all_outputs.push(output);
                }
            }

            // Add EMPTY_KEY as a possible output (to clear rules)
            all_outputs.push(EMPTY_KEY);

            // For each leader that has a rule, create neighbors for all possible outputs
            for (leading_str, _) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                for &output in &all_outputs {
                    neighbors.push(Neighbor::MagicRule(MagicRule::new(magic_key, leader, output)));
                }
            }
        }

        let mut cache = CachedLayout {
            name: layout.name.clone(),
            keyboard: layout.keyboard.clone(),
            shape: layout.shape.clone(),
            data: analyzer_data,
            keys,
            key_positions,
            num_positions: len,
            neighbors,
            current_magic_rules,
            affected_grams,
            dist,
            sfb,
            stretch,
            scissors,
            trigram,
            magic,
            fingers: fingers.to_vec(),
        };

        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = cache.data.char_mapping().get_u(key_char);
            cache.replace_key(pos, EMPTY_KEY, key, true);
        }

        cache
    }

    pub fn data(&self) -> &AnalyzerData {
        &self.data
    }

    pub fn char_mapping(&self) -> &CharMapping {
        self.data.char_mapping()
    }

    pub fn to_layout(&self) -> Layout {
        // Convert keys back to chars
        let keys: Box<[char]> = self.keys.iter()
            .map(|&k| if k == EMPTY_KEY { ' ' } else { self.data.char_mapping().get_c(k) })
            .collect();

        // Reconstruct magic HashMap from current_magic_rules
        let mut magic: HashMap<char, MagicKey> = HashMap::default();

        for (&(magic_key_id, leader), &output) in &self.current_magic_rules {
            if output == EMPTY_KEY {
                continue;
            }

            let magic_char = self.data.char_mapping().get_c(magic_key_id);
            let leader_char = self.data.char_mapping().get_c(leader);
            let output_char = self.data.char_mapping().get_c(output);

            let magic_key = magic.entry(magic_char).or_insert_with(|| {
                MagicKey::new(&magic_char.to_string())
            });
            magic_key.add_rule(&leader_char.to_string(), &output_char.to_string());
        }

        Layout {
            name: self.name.clone(),
            keys,
            fingers: self.fingers.clone().into_boxed_slice(),
            keyboard: self.keyboard.clone(),
            shape: self.shape.clone(),
            magic,
        }
    }

    /// Get the key at a position
    #[inline]
    pub fn get_key(&self, pos: CachePos) -> CacheKey {
        self.keys[pos]
    }

    /// Get the position of a key
    #[inline]
    pub fn get_pos(&self, key: CacheKey) -> Option<CachePos> {
        self.key_positions.get(key).copied().flatten()
    }

    pub fn score(&self) -> i64 {
        self.sfb.score() + self.stretch.score() + self.scissors.score() + self.trigram.score()
    }

    /// Populate stats from the caches. Uses internal data for normalization.
    pub fn stats(&self, stats: &mut Stats) {
        let char_total = self.data.char_total;
        let bigram_total = self.data.bigram_total;
        let skipgram_total = self.data.skipgram_total;
        let chars = &self.data.chars;

        // Finger use: sum character frequencies per finger
        for (pos, &key) in self.keys.iter().enumerate() {
            if key != EMPTY_KEY && (key as usize) < chars.len() {
                let finger = self.fingers[pos] as usize;
                stats.finger_use[finger] += chars[key as usize] as f64 / char_total;
            }
        }

        self.sfb.stats(stats, bigram_total, skipgram_total);
        self.stretch.stats(stats, bigram_total);
        self.scissors.stats(stats, bigram_total, skipgram_total);
        self.trigram.stats(stats, self.data.trigram_total);
    }

    /// Apply a neighbor transformation. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    pub fn apply_neighbor(&mut self, neighbor: Neighbor, apply: bool) -> i64 {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                self.swap_keys(a, b, apply)
            }
            Neighbor::MagicRule(rule) => {
                self.apply_magic_rule(rule.magic_key, rule.leader, rule.output, apply)
            }
        }
    }

    /// Replace key at position. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn replace_key(&mut self, pos: CachePos, old_key: CacheKey, new_key: CacheKey, apply: bool) -> i64 {
        debug_assert!(self.keys[pos] == old_key, "Position {pos} has key {} but expected {old_key}", self.keys[pos]);

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        let sfb_score = self.sfb.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq, apply);
        let stretch_score = self.stretch.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, apply);
        let scissors_score = self.scissors.replace_key(pos, old_key, new_key, &self.keys, None, bg_freq, sg_freq, apply);
        let trigram_score = self.trigram.replace_key(pos, old_key, new_key, &self.keys, None, tg_freq, apply);

        if apply {
            self.keys[pos] = new_key;
            if old_key != EMPTY_KEY && old_key < self.key_positions.len() {
                self.key_positions[old_key] = None;
            }
            if new_key != EMPTY_KEY && new_key < self.key_positions.len() {
                self.key_positions[new_key] = Some(pos);
            }
        }

        sfb_score + stretch_score + scissors_score + trigram_score
    }

    /// Swap keys at two positions. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    #[inline]
    pub fn swap_keys(&mut self, pos_a: CachePos, pos_b: CachePos, apply: bool) -> i64 {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];

        debug_assert!(key_a != EMPTY_KEY, "Position {pos_a} is empty");
        debug_assert!(key_b != EMPTY_KEY, "Position {pos_b} is empty");

        let bg_freq = self.magic.bg_freq_flat();
        let sg_freq = self.magic.sg_freq_flat();
        let tg_freq = self.magic.tg_freq();

        let sfb_score = self.sfb.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq, apply);
        let stretch_score = self.stretch.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, apply);
        let scissors_score = self.scissors.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, bg_freq, sg_freq, apply);
        let trigram_score = self.trigram.key_swap(pos_a, pos_b, key_a, key_b, &self.keys, tg_freq, apply);

        if apply {
            self.keys[pos_a] = key_b;
            self.keys[pos_b] = key_a;
            if key_a < self.key_positions.len() {
                self.key_positions[key_a] = Some(pos_b);
            }
            if key_b < self.key_positions.len() {
                self.key_positions[key_b] = Some(pos_a);
            }
        }

        sfb_score + stretch_score + scissors_score + trigram_score
    }


    /// Apply a magic rule. Returns the new score.
    /// If `apply` is false, computes the score without mutating state.
    pub fn apply_magic_rule(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey, apply: bool) -> i64 {
        self.affected_grams.clear();

        let key = (magic_key, leader);
        let old_output = self.current_magic_rules.get(&key).copied();

        if old_output == Some(new_output) || (old_output.is_none() && new_output == EMPTY_KEY) {
            return self.score();
        }

        // Check if another magic key has this (leader, output) and clear it
        if new_output != EMPTY_KEY {
            let mut key_to_clear = None;
            for (&(other_magic, other_leader), &other_output) in &self.current_magic_rules {
                if other_leader == leader && other_output == new_output && other_magic != magic_key {
                    key_to_clear = Some((other_magic, other_leader));
                    break;
                }
            }
            if let Some(clear_key) = key_to_clear {
                let other_magic = clear_key.0;
                self.magic.steal_bigram(
                    leader,
                    other_magic,
                    new_output,
                    &self.key_positions,
                    self.keys.len(),
                    &mut self.affected_grams,
                );
                if apply {
                    self.current_magic_rules.remove(&clear_key);
                }
            }
        }

        // Unsteal the old output if there was one
        if let Some(old_out) = old_output {
            if old_out != EMPTY_KEY {
                self.magic.steal_bigram(
                    leader,
                    magic_key,
                    old_out,
                    &self.key_positions,
                    self.keys.len(),
                    &mut self.affected_grams,
                );
            }
        }

        // Steal the new output
        if new_output != EMPTY_KEY {
            self.magic.steal_bigram(
                leader,
                new_output,
                magic_key,
                &self.key_positions,
                self.keys.len(),
                &mut self.affected_grams,
            );
        }

        // Update caches based on affected grams
        for gram in &self.affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    self.sfb.update_bigram(bg.p_a, bg.p_b, bg.old_freq, bg.new_freq);
                    self.stretch.update_bigram(bg.p_a, bg.p_b, bg.old_freq, bg.new_freq);
                    self.scissors.update_bigram(bg.p_a, bg.p_b, bg.old_freq, bg.new_freq);
                }
                DeltaGram::Skipgram(sg) => {
                    self.sfb.update_skipgram(sg.p_a, sg.p_b, sg.old_freq, sg.new_freq);
                    self.scissors.update_skipgram(sg.p_a, sg.p_b, sg.old_freq, sg.new_freq);
                }
                DeltaGram::Trigram(tg) => {
                    self.trigram.update_trigram(tg.p_a, tg.p_b, tg.p_c, tg.old_freq, tg.new_freq);
                }
            }
        }

        let score = self.score();

        if !apply {
            // Revert the changes
            for gram in self.affected_grams.iter().rev() {
                match gram {
                    DeltaGram::Bigram(bg) => {
                        self.sfb.update_bigram(bg.p_a, bg.p_b, bg.new_freq, bg.old_freq);
                        self.stretch.update_bigram(bg.p_a, bg.p_b, bg.new_freq, bg.old_freq);
                        self.scissors.update_bigram(bg.p_a, bg.p_b, bg.new_freq, bg.old_freq);
                    }
                    DeltaGram::Skipgram(sg) => {
                        self.sfb.update_skipgram(sg.p_a, sg.p_b, sg.new_freq, sg.old_freq);
                        self.scissors.update_skipgram(sg.p_a, sg.p_b, sg.new_freq, sg.old_freq);
                    }
                    DeltaGram::Trigram(tg) => {
                        self.trigram.update_trigram(tg.p_a, tg.p_b, tg.p_c, tg.new_freq, tg.old_freq);
                    }
                }
            }
            // Revert magic cache
            let keys = &self.keys;
            self.magic.revert_affected(&self.affected_grams, |pos| keys[pos]);
        } else {
            // Apply the rule change
            if new_output != EMPTY_KEY {
                self.current_magic_rules.insert(key, new_output);
            } else {
                self.current_magic_rules.remove(&key);
            }
        }

        score
    }

    /// Returns a copy of the pre-computed list of all neighbors (key swaps + magic rules)
    #[inline]
    pub fn neighbors(&self) -> Vec<Neighbor> {
        self.neighbors.clone()
    }

    pub fn get_revert_neighbor(&self, neighbor: Neighbor) -> Neighbor {
        match neighbor {
            Neighbor::KeySwap(pair) => Neighbor::KeySwap(pair),
            Neighbor::MagicRule(rule) => {
                let key = (rule.magic_key, rule.leader);
                let current_output = self.current_magic_rules.get(&key).copied().unwrap_or(EMPTY_KEY);
                Neighbor::MagicRule(MagicRule::new(rule.magic_key, rule.leader, current_output))
            }
        }
    }

    pub fn affected_grams(&self) -> &Vec<DeltaGram> {
        &self.affected_grams
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BigramPair {
    pub pair: PosPair,
    pub dist: i64,
}

#[cfg(test)]
mod magic_tests {
    use super::*;

    #[test]
    fn magic_cache_new() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.bg_freq.len(), 100);
        assert_eq!(cache.sg_freq.len(), 100);
        assert_eq!(cache.tg_freq.len(), 10);
    }

    #[test]
    fn magic_cache_get_freq_empty() {
        let cache = MagicCache::new(10);
        assert_eq!(cache.get_bg_freq(0, 1), 0);
        assert_eq!(cache.get_sg_freq(0, 1), 0);
        assert_eq!(cache.get_tg_freq(0, 1, 2), 0);
    }

    #[test]
    fn magic_cache_init_from_data() {
        let mut cache = MagicCache::new(3);
        let bigrams = vec![
            vec![0, 100, 200],
            vec![300, 0, 400],
            vec![500, 600, 0],
        ];
        let skipgrams = vec![
            vec![0, 10, 20],
            vec![30, 0, 40],
            vec![50, 60, 0],
        ];
        let trigrams = vec![
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]],
            vec![vec![9, 10, 11], vec![12, 13, 14], vec![15, 16, 17]],
            vec![vec![18, 19, 20], vec![21, 22, 23], vec![24, 25, 26]],
        ];

        cache.init_from_data(&bigrams, &skipgrams, &trigrams);

        assert_eq!(cache.get_bg_freq(0, 1), 100);
        assert_eq!(cache.get_bg_freq(1, 2), 400);
        assert_eq!(cache.get_sg_freq(2, 1), 60);
        assert_eq!(cache.get_tg_freq(1, 2, 0), 15);
    }

    #[test]
    fn magic_cache_steal_bigram_basic() {
        let mut cache = MagicCache::new(4);
        let nk = 4;
        cache.bg_freq[0 * nk + 1] = 100;
        cache.bg_freq[0 * nk + 2] = 0;

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        assert_eq!(cache.get_bg_freq(0, 1), 0, "a->b should be stolen");
        assert_eq!(cache.get_bg_freq(0, 2), 100, "a->m should have stolen frequency");
    }

    #[test]
    fn magic_cache_steal_records_affected_grams() {
        let mut cache = MagicCache::new(4);
        let nk = 4;
        cache.bg_freq[0 * nk + 1] = 100;

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        let bigram_count = affected.iter().filter(|g| matches!(g, DeltaGram::Bigram(_))).count();
        assert!(bigram_count >= 2, "Should record at least 2 bigram changes, got {bigram_count}");
    }
}
