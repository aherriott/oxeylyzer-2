use libdof::dofinitions::Finger;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    pub sfbs: i64,
    pub sfs: i64,
    pub stretches: i64,
    pub sft: i64,
    pub inroll: i64,
    pub outroll: i64,
    pub alternate: i64,
    pub redirect: i64,
    pub onehandin: i64,
    pub onehandout: i64,
    pub full_scissors: i64,
    pub half_scissors: i64,
    pub full_scissors_skip: i64,
    pub half_scissors_skip: i64,
    /// Penalty for finger usage imbalance. Higher = penalize load on weak fingers more.
    pub finger_usage: i64,
    /// Penalty per active magic rule (negative = penalize). Applied per non-repeat rule.
    pub magic_rule_penalty: i64,
    /// Penalty per active repeat magic rule (leader → same key). Typically less harsh.
    pub magic_repeat_penalty: i64,
    pub fingers: FingerWeights,
}

impl Weights {
    pub const fn has_bigram_weights(&self) -> bool {
        self.sfbs != 0 || self.sfs != 0
    }

    pub const fn has_trigram_weights(&self) -> bool {
        self.sft != 0
            || self.inroll != 0
            || self.outroll != 0
            || self.alternate != 0
            || self.redirect != 0
            || self.onehandin != 0
            || self.onehandout != 0
    }

    pub const fn has_stretch_weights(&self) -> bool {
        self.stretches != 0
    }

    pub const fn has_scissors_weights(&self) -> bool {
        self.full_scissors != 0
            || self.half_scissors != 0
            || self.full_scissors_skip != 0
            || self.half_scissors_skip != 0
    }

    /// Compute scale factors from corpus data so that weight=1 for each metric
    /// produces roughly the same score magnitude.
    pub fn compute_scale_factors(&self, data: &crate::data::Data) -> ScaleFactors {
        // Bigram-based scores scale with bigram_total × avg_finger_weight
        // Trigram-based scores scale with trigram_total
        // The ratio gives us the trigram scale factor.
        let bg_total = data.bigram_total.max(1) as f64;
        let tg_total = data.trigram_total.max(1) as f64;

        // Average finger weight (used by SFB scoring)
        let avg_finger: f64 = {
            let sum: i64 = libdof::dofinitions::Finger::FINGERS.iter()
                .map(|&f| self.fingers.get(f))
                .sum();
            sum as f64 / 10.0
        };

        // Bigram magnitude estimate: bg_total * avg_finger_weight * avg_metric_weight
        let avg_bg_weight = (self.sfbs.abs() + self.sfs.abs()).max(1) as f64 / 2.0;
        let bg_mag = bg_total * avg_finger * avg_bg_weight;

        // Trigram magnitude estimate: tg_total * avg_reward_weight
        // Only use reward weights (not redirect penalty) to compute the scale factor.
        // Otherwise raising the redirect penalty shrinks the scale factor, weakening
        // ALL trigram weights including the rewards — the opposite of what the user wants.
        let avg_tg_weight = {
            let reward_weights = [self.inroll, self.outroll, self.alternate,
                                  self.onehandin, self.onehandout];
            let sum: i64 = reward_weights.iter().map(|w| w.abs()).sum();
            let count = reward_weights.iter().filter(|&&w| w != 0).count().max(1);
            sum as f64 / count as f64
        };
        let tg_mag = tg_total * avg_tg_weight;

        let trigram_scale = if tg_mag > 0.0 { (bg_mag / tg_mag).max(1.0) as i64 } else { 1 };

        // Magic penalty scale: penalty=10 should cost roughly 1% of total score
        let total_mag = bg_mag + tg_mag * trigram_scale as f64;
        let magic_penalty_scale = (total_mag / 300.0).max(1.0) as i64; // ~30 positions × 10 penalty units

        ScaleFactors {
            trigram_scale,
            magic_penalty_scale,
        }
    }
}

/// Pre-computed scale factors for normalizing score contributions across metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleFactors {
    /// Multiplier for trigram weights to match bigram magnitude.
    pub trigram_scale: i64,
    /// Multiplier for magic rule penalty.
    pub magic_penalty_scale: i64,
}

impl Default for ScaleFactors {
    fn default() -> Self {
        Self { trigram_scale: 1, magic_penalty_scale: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FingerWeights {
    pub lp: i64,
    pub lr: i64,
    pub lm: i64,
    pub li: i64,
    pub lt: i64,
    pub rt: i64,
    pub ri: i64,
    pub rm: i64,
    pub rr: i64,
    pub rp: i64,
}

impl FingerWeights {
    #[inline]
    pub const fn get(&self, f: Finger) -> i64 {
        use Finger::*;

        match f {
            LP => self.lp,
            LR => self.lr,
            LM => self.lm,
            LI => self.li,
            LT => self.lt,
            RT => self.rt,
            RI => self.ri,
            RM => self.rm,
            RR => self.rr,
            RP => self.rp,
        }
    }
}

impl Default for FingerWeights {
    fn default() -> Self {
        Self {
            lp: 1,
            lr: 1,
            lm: 1,
            li: 1,
            lt: 1,
            rt: 1,
            ri: 1,
            rm: 1,
            rr: 1,
            rp: 1,
        }
    }
}

pub fn dummy_weights() -> Weights {
    Weights {
        sfbs: 7,
        sfs: 2,
        stretches: 3,
        sft: 12,
        inroll: 7,
        outroll: 4,
        alternate: 4,
        redirect: 4,
        onehandin: 2,
        onehandout: 0,
        full_scissors: 5,
        half_scissors: 1,
        full_scissors_skip: 2,
        half_scissors_skip: 1,
        finger_usage: 0,
        magic_rule_penalty: 0,
        magic_repeat_penalty: 0,
        fingers: FingerWeights {
            lp: 77,
            lr: 32,
            lm: 24,
            li: 21,
            lt: 46,
            rt: 46,
            ri: 21,
            rm: 24,
            rr: 32,
            rp: 77,
        },
    }
}
