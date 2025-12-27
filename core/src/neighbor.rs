// The difference between two neighboring layouts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Neighbor {
    KeySwap(PosPair),
    MagicRule(MagicRule),
}

impl Neighbor {
    pub fn default() -> Self {
        Neighbor::KeySwap(PosPair(0, 0))
    }

    pub fn revert(&self, cache: &CachedLayout) -> Neighbor {
        match self {
            Neighbor::KeySwap(_) => *self,
            Neighbor::MagicRule(rule) => {
                let output = cache
                    .magic
                    .rules
                    .get(&rule.0)
                    .unwrap()
                    .get(&rule.1)
                    .unwrap();
                Neighbor::MagicRule(MagicRule(rule.0, rule.1, *output))
            }
        }
    }
}
