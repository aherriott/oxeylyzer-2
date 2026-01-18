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
    stats::Stats,
    stretches::StretchCache,
    types::{CacheKey, CachePos},
    weights::Weights,
};

/// Represents a possible magic rule neighbor: (magic_key, leader, output)
/// These are pre-computed for all valid combinations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagicRuleNeighbor {
    pub magic_key: CacheKey,
    pub leader: CacheKey,
    pub output: CacheKey,  // Can be EMPTY_KEY to clear the rule
}

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
    bg_freq: Vec<Vec<i64>>,
    sg_freq: Vec<Vec<i64>>,
    tg_freq: Vec<Vec<Vec<i64>>>,
}

impl MagicCache {
    pub fn new(num_keys: usize) -> Self {
        Self {
            bg_freq: vec![vec![0; num_keys]; num_keys],
            sg_freq: vec![vec![0; num_keys]; num_keys],
            tg_freq: vec![vec![vec![0; num_keys]; num_keys]; num_keys],
        }
    }

    /// Initialize frequencies from corpus data
    pub fn init_from_data(&mut self, bigrams: &[Vec<i64>], skipgrams: &[Vec<i64>], trigrams: &[Vec<Vec<i64>>]) {
        self.bg_freq = bigrams.to_vec();
        self.sg_freq = skipgrams.to_vec();
        self.tg_freq = trigrams.to_vec();
    }

    #[inline]
    pub fn get_bg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.bg_freq.get(a).and_then(|row| row.get(b)).copied().unwrap_or(0)
    }

    #[inline]
    pub fn get_sg_freq(&self, a: CacheKey, b: CacheKey) -> i64 {
        self.sg_freq.get(a).and_then(|row| row.get(b)).copied().unwrap_or(0)
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
        // Helper to get position
        let get_pos = |k: CacheKey| -> Option<CachePos> {
            key_positions.get(k).copied().flatten()
        };

        // Inline helper for setting bigram freq and recording delta
        macro_rules! set_bg {
            ($ka:expr, $kb:expr, $new:expr) => {{
                let old = self.bg_freq[$ka][$kb];
                self.bg_freq[$ka][$kb] = $new;
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
                let old = self.sg_freq[$ka][$kb];
                self.sg_freq[$ka][$kb] = $new;
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
        let new_am = self.bg_freq[a][m] + self.bg_freq[a][b];
        set_bg!(a, m, new_am);
        set_bg!(a, b, 0);

        // 2. For each key c: B->C is partially stolen by M->C based on trigram A->B->C
        for c in 0..num_keys {
            let tg = self.tg_freq[a][b][c];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.bg_freq[b][c] >= tg);
            let new_mc = self.bg_freq[m][c] + tg;
            let new_bc = self.bg_freq[b][c] - tg;
            set_bg!(m, c, new_mc);
            set_bg!(b, c, new_bc);
        }

        // 3. For each key z: skipgram Z->B is partially stolen by Z->M based on trigram Z->A->B
        for z in 0..num_keys {
            let tg = self.tg_freq[z][a][b];
            if tg == 0 {
                continue;
            }
            debug_assert!(self.sg_freq[z][b] >= tg);
            let new_zm = self.sg_freq[z][m] + tg;
            let new_zb = self.sg_freq[z][b] - tg;
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

    /// Copy only the frequency entries that were affected by a steal operation.
    /// `affected_grams` should contain the deltas from the steal that was applied to `other`.
    pub fn copy_from(&mut self, other: &MagicCache, affected_grams: &[DeltaGram], key_at_pos: impl Fn(CachePos) -> CacheKey) {
        for gram in affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    let a = key_at_pos(bg.p_a);
                    let b = key_at_pos(bg.p_b);
                    self.bg_freq[a][b] = other.bg_freq[a][b];
                }
                DeltaGram::Skipgram(sg) => {
                    let a = key_at_pos(sg.p_a);
                    let b = key_at_pos(sg.p_b);
                    self.sg_freq[a][b] = other.sg_freq[a][b];
                }
                DeltaGram::Trigram(tg) => {
                    let a = key_at_pos(tg.p_a);
                    let b = key_at_pos(tg.p_b);
                    let c = key_at_pos(tg.p_c);
                    self.tg_freq[a][b][c] = other.tg_freq[a][b][c];
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

// CachedLayout contains the minimum mutable data used to define a layout and store scoring.
// Designed to copy quickly and without allocation. Wrapped by Analyzer.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedLayout {
    // Layout identity
    name: String,
    keyboard: Box<[PhysicalKey]>,
    shape: Shape,

    /// Corpus data including char_mapping, character frequencies, bigrams, skipgrams, trigrams
    data: AnalyzerData,

    /// Key at each position (EMPTY_KEY if unassigned)
    keys: Vec<CacheKey>,
    /// Position of each key (None if not placed)
    key_positions: Vec<Option<CachePos>>,
    /// Number of positions (for KeySwap neighbor calculation)
    num_positions: usize,
    /// Pre-computed magic rule neighbors (constant for the layout)
    /// Each entry is a (magic_key, leader, output) that can be set
    magic_rule_neighbors: Vec<MagicRuleNeighbor>,
    /// Current magic rules: maps (magic_key, leader) -> output
    /// If not present, no rule is active for that pair
    current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey>,
    affected_grams: Vec<DeltaGram>,
    dist: DistCache,
    sfb: SFCache,
    stretch: StretchCache,
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
            magic_rule_neighbors: Vec::new(),
            current_magic_rules: HashMap::default(),
            affected_grams: Vec::new(),
            dist: DistCache::default(),
            sfb: SFCache::default(),
            stretch: StretchCache::default(),
            magic: MagicCache::default(),
            fingers: Vec::new(),
        }
    }
}

impl CachedLayout {
    /// Create a new CachedLayout from a Layout and Data
    pub fn new(layout: &Layout, data: Data) -> Self {
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
        let sfb = SFCache::new(fingers, keyboard);
        let stretch = StretchCache::new(keyboard, fingers);
        let mut magic = MagicCache::new(num_keys);
        magic.init_from_data(&analyzer_data.bigrams, &analyzer_data.skipgrams, &analyzer_data.trigrams);

        // Build magic rule neighbors from layout.magic
        // For each (magic_key, leader), we can set any output (including EMPTY_KEY to clear)
        let mut magic_rule_neighbors = Vec::new();
        let mut current_magic_rules: HashMap<(CacheKey, CacheKey), CacheKey> = HashMap::default();

        for (&magic_char, magic_key_def) in layout.magic.iter() {
            let magic_key = analyzer_data.char_mapping().get_u(magic_char);

            // Collect all possible outputs for this magic key
            let mut all_outputs: Vec<CacheKey> = Vec::new();

            for (leading_str, output_str) in magic_key_def.rules().iter() {
                let leader = analyzer_data.char_mapping().get_u(leading_str.chars().next().unwrap_or(' '));
                let output = analyzer_data.char_mapping().get_u(output_str.chars().next().unwrap_or(' '));

                // Track the initial rule
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
                    magic_rule_neighbors.push(MagicRuleNeighbor {
                        magic_key,
                        leader,
                        output,
                    });
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
            magic_rule_neighbors,
            current_magic_rules,
            affected_grams,
            dist,
            sfb,
            stretch,
            magic,
            fingers: fingers.to_vec(),
        };

        // Initialize keys from layout
        for (pos, &key_char) in layout.keys.iter().enumerate() {
            let key = cache.data.char_mapping().get_u(key_char);
            cache.add_key(pos, key);
        }


        cache
    }

    /// Access the analyzer data (char frequencies, bigrams, etc.)
    pub fn data(&self) -> &AnalyzerData {
        &self.data
    }

    /// Access the char mapping
    pub fn char_mapping(&self) -> &CharMapping {
        self.data.char_mapping()
    }

    /// Convert back to a Layout
    pub fn to_layout(&self) -> Layout {
        // Convert keys back to chars
        let keys: Box<[char]> = self.keys.iter()
            .map(|&k| if k == EMPTY_KEY { ' ' } else { self.data.char_mapping().get_c(k) })
            .collect();

        // Reconstruct magic HashMap from current_magic_rules
        let mut magic: HashMap<char, MagicKey> = HashMap::default();

        for (&(magic_key_id, leader), &output) in &self.current_magic_rules {
            if output == EMPTY_KEY {
                continue; // Skip cleared rules
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

    pub fn score(&self, weights: &Weights) -> i64 {
        self.sfb.score(weights) + self.stretch.score(weights)
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

        // SFB/SFS stats
        self.sfb.stats(stats, bigram_total, skipgram_total);

        // Stretch stats
        self.stretch.stats(stats, bigram_total);
    }

    /// Apply a neighbor transformation
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                self.swap_keys(a, b);
            }
            Neighbor::MagicRule(rule) => {
                self.apply_magic_rule(rule.magic_key, rule.leader, rule.output);
            }
        }
    }

    /// Copy only the delta resulting from a neighbor transformation from another cache.
    /// Assumes `other` has had `neighbor` applied relative to `self`.
    /// This is more efficient than cloning the entire cache.
    pub fn copy_from(&mut self, other: &CachedLayout, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                // Copy the swapped keys
                let key_a = other.keys[a];
                let key_b = other.keys[b];
                self.keys[a] = key_a;
                self.keys[b] = key_b;

                // Copy key_positions for the affected keys
                if key_a < self.key_positions.len() {
                    self.key_positions[key_a] = other.key_positions[key_a];
                }
                if key_b < self.key_positions.len() {
                    self.key_positions[key_b] = other.key_positions[key_b];
                }
            }
            Neighbor::MagicRule(rule) => {
                // Copy only affected magic frequency entries
                let keys = &other.keys;
                self.magic.copy_from(&other.magic, other.affected_grams(), |pos| keys[pos]);

                // Copy the affected magic rule state
                let key = (rule.magic_key, rule.leader);
                if let Some(&output) = other.current_magic_rules.get(&key) {
                    if output == EMPTY_KEY {
                        self.current_magic_rules.remove(&key);
                    } else {
                        self.current_magic_rules.insert(key, output);
                    }
                } else {
                    self.current_magic_rules.remove(&key);
                }
            }
        }

        // Copy scoring caches - these use fixed-size arrays, no allocation
        self.sfb.copy_from(&other.sfb);
        self.stretch.copy_from(&other.stretch);

        // Clear affected_grams to match the original state (it's a working buffer)
        self.affected_grams.clear();
    }

    /// Add a key at pos. Position should currently be empty.
    #[inline]
    pub fn add_key(&mut self, pos: CachePos, key: CacheKey) {
        debug_assert!(self.keys[pos] == EMPTY_KEY, "Position {pos} is not empty");

        self.keys[pos] = key;
        if key < self.key_positions.len() {
            self.key_positions[key] = Some(pos);
        }

        // Update SFB cache - only iterate same-finger pairs
        let sf_pair_count = self.sfb.sf_pairs(pos).len();
        for i in 0..sf_pair_count {
            let other_pos = self.sfb.sf_pairs(pos)[i].other_pos;
            let other_key = self.keys[other_pos];
            if other_key == EMPTY_KEY {
                continue;
            }

            // Bigram: pos -> other_pos
            let bg_freq = self.magic.get_bg_freq(key, other_key);
            self.sfb.update_bigram(&self.dist, pos, other_pos, 0, bg_freq);

            // Bigram: other_pos -> pos
            let bg_freq_rev = self.magic.get_bg_freq(other_key, key);
            self.sfb.update_bigram(&self.dist, other_pos, pos, 0, bg_freq_rev);

            // Skipgram: pos -> other_pos
            let sg_freq = self.magic.get_sg_freq(key, other_key);
            self.sfb.update_skipgram(&self.dist, pos, other_pos, 0, sg_freq);

            // Skipgram: other_pos -> pos
            let sg_freq_rev = self.magic.get_sg_freq(other_key, key);
            self.sfb.update_skipgram(&self.dist, other_pos, pos, 0, sg_freq_rev);
        }

        // Update stretch cache - only iterate stretch pairs
        let stretch_pair_count = self.stretch.stretch_pairs(pos).len();
        for i in 0..stretch_pair_count {
            let other_pos = self.stretch.stretch_pairs(pos)[i];
            let other_key = self.keys[other_pos];
            if other_key == EMPTY_KEY {
                continue;
            }

            // Bigram: pos -> other_pos
            let bg_freq = self.magic.get_bg_freq(key, other_key);
            self.stretch.update_bigram(pos, other_pos, 0, bg_freq);

            // Bigram: other_pos -> pos
            let bg_freq_rev = self.magic.get_bg_freq(other_key, key);
            self.stretch.update_bigram(other_pos, pos, 0, bg_freq_rev);
        }
    }

    /// Swap keys at two positions using optimized cache methods.
    /// More efficient than remove_key(a) + remove_key(b) + add_key(a, key_b) + add_key(b, key_a).
    #[inline]
    pub fn swap_keys(&mut self, pos_a: CachePos, pos_b: CachePos) {
        let key_a = self.keys[pos_a];
        let key_b = self.keys[pos_b];

        debug_assert!(key_a != EMPTY_KEY, "Position {pos_a} is empty");
        debug_assert!(key_b != EMPTY_KEY, "Position {pos_b} is empty");

        // Update SFB cache using optimized key_swap
        self.sfb.key_swap(
            &self.dist,
            pos_a,
            pos_b,
            key_a,
            key_b,
            &self.keys,
            |k1, k2| self.magic.get_bg_freq(k1, k2),
            |k1, k2| self.magic.get_sg_freq(k1, k2),
        );

        // Update stretch cache using optimized key_swap
        self.stretch.key_swap(
            pos_a,
            pos_b,
            key_a,
            key_b,
            &self.keys,
            |k1, k2| self.magic.get_bg_freq(k1, k2),
        );

        // Update key positions
        self.keys[pos_a] = key_b;
        self.keys[pos_b] = key_a;
        if key_a < self.key_positions.len() {
            self.key_positions[key_a] = Some(pos_b);
        }
        if key_b < self.key_positions.len() {
            self.key_positions[key_b] = Some(pos_a);
        }
    }

    /// Remove a key at pos. Position should currently contain a key.
    #[inline]
    pub fn remove_key(&mut self, pos: CachePos) {
        let key = self.keys[pos];
        debug_assert!(key != EMPTY_KEY, "Position {pos} is already empty");

        // Update SFB cache - only iterate same-finger pairs
        let sf_pair_count = self.sfb.sf_pairs(pos).len();
        for i in 0..sf_pair_count {
            let other_pos = self.sfb.sf_pairs(pos)[i].other_pos;
            let other_key = self.keys[other_pos];
            if other_key == EMPTY_KEY {
                continue;
            }

            // Bigram: pos -> other_pos
            let bg_freq = self.magic.get_bg_freq(key, other_key);
            self.sfb.update_bigram(&self.dist, pos, other_pos, bg_freq, 0);

            // Bigram: other_pos -> pos
            let bg_freq_rev = self.magic.get_bg_freq(other_key, key);
            self.sfb.update_bigram(&self.dist, other_pos, pos, bg_freq_rev, 0);

            // Skipgram: pos -> other_pos
            let sg_freq = self.magic.get_sg_freq(key, other_key);
            self.sfb.update_skipgram(&self.dist, pos, other_pos, sg_freq, 0);

            // Skipgram: other_pos -> pos
            let sg_freq_rev = self.magic.get_sg_freq(other_key, key);
            self.sfb.update_skipgram(&self.dist, other_pos, pos, sg_freq_rev, 0);
        }

        // Update stretch cache - only iterate stretch pairs
        let stretch_pair_count = self.stretch.stretch_pairs(pos).len();
        for i in 0..stretch_pair_count {
            let other_pos = self.stretch.stretch_pairs(pos)[i];
            let other_key = self.keys[other_pos];
            if other_key == EMPTY_KEY {
                continue;
            }

            // Bigram: pos -> other_pos
            let bg_freq = self.magic.get_bg_freq(key, other_key);
            self.stretch.update_bigram(pos, other_pos, bg_freq, 0);

            // Bigram: other_pos -> pos
            let bg_freq_rev = self.magic.get_bg_freq(other_key, key);
            self.stretch.update_bigram(other_pos, pos, bg_freq_rev, 0);
        }

        self.keys[pos] = EMPTY_KEY;
        if key < self.key_positions.len() {
            self.key_positions[key] = None;
        }
    }

    /// Apply a magic rule: set the output for (magic_key, leader).
    /// If output is EMPTY_KEY, clears the rule.
    /// If another magic key had this same (leader, output) pair, clears it from that key.
    pub fn apply_magic_rule(&mut self, magic_key: CacheKey, leader: CacheKey, new_output: CacheKey) {
        self.affected_grams.clear();

        let key = (magic_key, leader);
        let old_output = self.current_magic_rules.get(&key).copied();

        // If the rule is already set to this output, nothing to do
        if old_output == Some(new_output) || (old_output.is_none() && new_output == EMPTY_KEY) {
            return;
        }

        // If we're setting a non-empty output, check if another magic key has this (leader, output)
        // and clear it from that key
        if new_output != EMPTY_KEY {
            let mut key_to_clear = None;
            for (&(other_magic, other_leader), &other_output) in &self.current_magic_rules {
                if other_leader == leader && other_output == new_output && other_magic != magic_key {
                    key_to_clear = Some((other_magic, other_leader));
                    break;
                }
            }
            if let Some(clear_key) = key_to_clear {
                // Unsteal from the other magic key
                let other_magic = clear_key.0;
                self.magic.steal_bigram(
                    leader,
                    other_magic,  // "steal" from other magic key
                    new_output,   // back to the output
                    &self.key_positions,
                    self.keys.len(),
                    &mut self.affected_grams,
                );
                self.current_magic_rules.remove(&clear_key);
            }
        }

        // Unsteal the old output if there was one
        if let Some(old_out) = old_output {
            if old_out != EMPTY_KEY {
                self.magic.steal_bigram(
                    leader,
                    magic_key,   // "steal" from magic_key
                    old_out,     // back to old_output
                    &self.key_positions,
                    self.keys.len(),
                    &mut self.affected_grams,
                );
            }
        }

        // Steal the new output if it's not EMPTY_KEY
        if new_output != EMPTY_KEY {
            self.magic.steal_bigram(
                leader,
                new_output,
                magic_key,
                &self.key_positions,
                self.keys.len(),
                &mut self.affected_grams,
            );
            self.current_magic_rules.insert(key, new_output);
        } else {
            self.current_magic_rules.remove(&key);
        }

        // Update caches based on affected grams
        for gram in &self.affected_grams {
            match gram {
                DeltaGram::Bigram(bg) => {
                    self.sfb.update_bigram(&self.dist, bg.p_a, bg.p_b, bg.old_freq, bg.new_freq);
                    self.stretch.update_bigram(bg.p_a, bg.p_b, bg.old_freq, bg.new_freq);
                }
                DeltaGram::Skipgram(sg) => {
                    self.sfb.update_skipgram(&self.dist, sg.p_a, sg.p_b, sg.old_freq, sg.new_freq);
                }
                DeltaGram::Trigram(_) => {
                    // Trigrams don't affect SFB/stretch scores directly
                }
            }
        }
    }

    /// Total number of neighbors (KeySwaps + MagicRules)
    /// Magic rule count is constant - all possible (magic_key, leader, output) combinations are pre-computed
    #[inline]
    pub fn neighbor_count(&self) -> usize {
        self.key_swap_count() + self.magic_rule_neighbors.len()
    }

    /// Number of KeySwap neighbors: n*(n-1)/2 for n positions
    #[inline]
    fn key_swap_count(&self) -> usize {
        let n = self.num_positions;
        n * (n - 1) / 2
    }

    /// Get neighbor by index. KeySwaps come first, then MagicRules.
    #[inline]
    pub fn get_neighbor(&self, idx: usize) -> Neighbor {
        let swap_count = self.key_swap_count();
        if idx < swap_count {
            // Decode triangular index to (a, b) where a < b
            // idx = a*n - a*(a+1)/2 + (b - a - 1)
            // Solve for a: a = floor((2n-1 - sqrt((2n-1)^2 - 8*idx)) / 2)
            let n = self.num_positions;
            let a = ((2 * n - 1) as f64 - ((2 * n - 1).pow(2) as f64 - 8.0 * idx as f64).sqrt()) / 2.0;
            let a = a.floor() as usize;
            let b = idx - (a * n - a * (a + 1) / 2) + a + 1;
            Neighbor::KeySwap(PosPair(a, b))
        } else {
            // Direct lookup into pre-computed magic rule neighbors
            let magic_idx = idx - swap_count;
            let rule_neighbor = &self.magic_rule_neighbors[magic_idx];
            Neighbor::MagicRule(MagicRule::new(
                rule_neighbor.magic_key,
                rule_neighbor.leader,
                rule_neighbor.output,
            ))
        }
    }

    /// Get the neighbor that would revert the given neighbor.
    /// Must be called BEFORE apply_neighbor since it needs the current state.
    pub fn get_revert_neighbor(&self, neighbor: Neighbor) -> Neighbor {
        match neighbor {
            Neighbor::KeySwap(pair) => {
                // KeySwap is its own inverse
                Neighbor::KeySwap(pair)
            }
            Neighbor::MagicRule(rule) => {
                // To revert, we need to set the rule back to its current output
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
        assert_eq!(cache.bg_freq.len(), 10);
        assert_eq!(cache.sg_freq.len(), 10);
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
        // Setup: a=0, b=1, m=2, c=3
        // Bigram a->b = 100
        cache.bg_freq[0][1] = 100;
        cache.bg_freq[0][2] = 0; // a->m initially 0

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        // Steal bigram a->b with magic key m
        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        // a->b should now be 0
        assert_eq!(cache.get_bg_freq(0, 1), 0, "a->b should be stolen");
        // a->m should now have the stolen frequency
        assert_eq!(cache.get_bg_freq(0, 2), 100, "a->m should have stolen frequency");
    }

    #[test]
    fn magic_cache_steal_records_affected_grams() {
        let mut cache = MagicCache::new(4);
        cache.bg_freq[0][1] = 100;

        let key_positions: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3)];
        let mut affected = Vec::new();

        cache.steal_bigram(0, 1, 2, &key_positions, 4, &mut affected);

        // Should have recorded at least the two bigram changes (a->b and a->m)
        let bigram_count = affected.iter().filter(|g| matches!(g, DeltaGram::Bigram(_))).count();
        assert!(bigram_count >= 2, "Should record at least 2 bigram changes, got {bigram_count}");
    }

    #[test]
    fn magic_cache_copy_from_selective() {
        let mut cache1 = MagicCache::new(4);
        let mut cache2 = MagicCache::new(4);

        // Setup cache1 with some data
        cache1.bg_freq[0][1] = 100;
        cache1.bg_freq[1][2] = 200;
        cache1.sg_freq[0][2] = 50;

        // Setup cache2 differently
        cache2.bg_freq[0][1] = 999;
        cache2.bg_freq[1][2] = 999;
        cache2.sg_freq[0][2] = 999;

        // Create affected grams that only include [0][1] bigram
        let affected = vec![
            DeltaGram::Bigram(DeltaBigram {
                p_a: 0,
                p_b: 1,
                old_freq: 999,
                new_freq: 100,
            }),
        ];

        // Copy only affected entries from cache1 to cache2
        cache2.copy_from(&cache1, &affected, |pos| pos);

        // [0][1] should be copied
        assert_eq!(cache2.get_bg_freq(0, 1), 100, "Affected bigram should be copied");
        // [1][2] should NOT be copied (not in affected)
        assert_eq!(cache2.get_bg_freq(1, 2), 999, "Unaffected bigram should not be copied");
        // skipgram should NOT be copied
        assert_eq!(cache2.get_sg_freq(0, 2), 999, "Unaffected skipgram should not be copied");
    }
}
