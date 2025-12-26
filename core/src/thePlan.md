# Refactoring Plan: analyze.rs & cached_layout.rs

## Executive Summary

**IMPORTANT REVISION**: CachedLayout keeps its name. Immutable data (fingers, keyboard, shape, char_mapping) moves to Analyzer. CachedLayout becomes purely mutable state.

This document outlines a major refactoring of the layout analysis and optimization system. The goal is to simplify the codebase while maintaining performance through:

1. **Clear data ownership**: Immutable layout metadata in Analyzer, mutable scores in CachedLayout
2. **Unified cache initialization** via `add_key/remove_key` primitives (single source of truth)
3. **Full trigram integration** into optimization scoring (currently only used for stats)
4. **Internal state management** with current + working CachedLayouts in Analyzer
5. **Lazy neighbor generation** to reduce memory overhead
6. **Simplified cache structures** by removing duplication while keeping performance optimizations

## Original Plan Issues Identified

### 1. API Surface Contradiction
**Problem**: Original plan stated "CachedLayout is now an implementation detail, and is not exposed publicly" but `stats.rs` heavily depends on CachedLayout for calculating statistics.

**Resolution REVISED**: Keep CachedLayout name. Move immutable data (fingers, keyboard, shape, char_mapping) to Analyzer. CachedLayout becomes purely mutable cache state. Stats access Analyzer's internal current/working CachedLayout.

### 2. Missing Trigram Integration
**Problem**: Trigrams are calculated separately in `trigrams()` but never contribute to optimization scoring. The plan mentions trigrams in `addKey/removeKey` but `score()` only sums SFBs and stretches.

**Resolution**: Add TrigramCache to CachedLayout and include trigram scores in scoring.

### 3. Per-Finger Cache Removal Unclear
**Problem**: Plan suggests removing "intermediate caching steps (per_finger, magic cache, etc.)" but per_finger is critical for O(1) keyswap updates.

**Resolution**: Keep per_finger cache but simplify by removing weighted vs unweighted duplication.

### 4. copy_to() Not Well Justified
**Problem**: Plan proposes `copy_to(CachedLayout)` but doesn't explain advantage over `clone()`.

**Resolution**: Use `clone()` for Arc-wrapped immutable data (cheap) and `copy_from()` for mutable caches (explicit slice copies).

### 5. Neighbor Generation Inefficiency
**Problem**: `possible_neighbors` stored in CachedLayout takes O(n²) memory and is regenerated for every layout.

**Resolution**: Generate neighbors lazily via iterator, computed on-demand from Analyzer.

### 6. addKey/removeKey Symmetry
**Problem**: Current implementation doesn't support explicit add/remove - only neighbor diffs.

**Resolution**: Implement `add_key_to_state()` and `remove_key_from_state()` as fundamental primitives used everywhere.

### 7. getBigramFrequency Logic
**Problem**: Pseudocode has confusing "return 0 if there's a rule" that seems to lose data.

**Resolution**: Pre-compute all frequency adjustments in magic cache during initialization and rule changes.

### 8. Stretch Calculation Correctness
**Problem**: `stretch_neighbor_update()` has TODO comment questioning its correctness.

**Resolution**: Rewrite stretch updates using add/remove primitives for clarity.

### 9. apply_neighbor Return Value
**Problem**: Original plan shows apply_neighbor returning void.

**Resolution**: Return i64 score for optimizer convenience.

---

## Amended Architecture

### Core Structures

```rust
// Main analyzer with internal state (REVISED)
pub struct Analyzer {
    pub data: AnalyzerData,       // Const: char/bigram/trigram frequencies
    pub weights: Weights,          // Const: scoring weights
    
    // Immutable layout metadata (moved FROM CachedLayout)
    fingers: Box<[Finger]>,
    keyboard: Box<[PhysicalKey]>,
    shape: Shape,
    char_mapping: Arc<CharMapping>,
    
    // Immutable indices (moved FROM caches - depend only on layout structure)
    sfb_indices: SfbIndices,       // Pre-computed SFB distances
    stretch_pairs: Box<[BigramPair]>, // Pre-computed stretch pairs
    trigram_indices: TrigramIndices,  // Pre-computed trigram groupings
    
    // Mutable state (two copies for check_neighbor pattern)
    current: CachedLayout,         // Current layout state
    working: CachedLayout,         // Temporary for check_neighbor
    
    analyze_bigrams: bool,
    analyze_stretches: bool,
    analyze_trigrams: bool,
}

// CachedLayout contains ONLY mutable state (REVISED)
pub struct CachedLayout {
    name: String,
    keys: Box<[u8]>,              // Current layout keys
    
    // Score caches (ONLY mutable scores - no indices!)
    sfb: SFBCache,
    stretch: StretchCache,
    trigram: TrigramCache,
    magic: MagicCache,
    
    // Temporary - will be removed in Phase 4
    possible_neighbors: Box<[Neighbor]>,
}

// Simplified cache structures (ONLY mutable data)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SFBCache {
    per_finger: Box<[i64; 10]>,   // Per-finger scores (SFB + SFS combined)
    total: i64,                    // Total score
    // indices moved to Analyzer!
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StretchCache {
    total: i64,                    // Total score
    // all_pairs moved to Analyzer!
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrigramCache {
    per_type: Box<[i64; 10]>,     // Scores per TrigramType
    total: i64,                    // Total score
    // indices moved to Analyzer!
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MagicCache {
    rules: HashMap<u8, HashMap<u8, u8>>,  // magic_key → (leader → output)
    // bigrams/skipgrams/trigrams arrays REMOVED - compute on-demand!
}
```

### Public API

```rust
impl Analyzer {
    // Initialization
    pub fn new(data: Data, weights: Weights) -> Self;
    
    // Layout management
    pub fn load_layout(&mut self, layout: Layout, pins: &[usize]);
    pub fn get_layout(&self) -> Layout;
    
    // Note: Immutable layout data (fingers, keyboard, shape) stored in Analyzer
    // CachedLayout only contains mutable state (keys, scores)
    
    // Scoring operations
    pub fn score(&self) -> i64;
    pub fn check_neighbor(&mut self, neighbor: Neighbor) -> i64;
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) -> i64;
    
    // Neighbor generation
    pub fn possible_neighbors(&self, pins: &[usize]) -> impl Iterator<Item=Neighbor>;
    
    // High-level optimization (backwards compatible)
    pub fn greedy_improve(&mut self, layout: Layout, pins: &[usize]) -> (Layout, i64);
    
    // Stats (tightly coupled - accesses state directly)
    pub fn stats(&self) -> Stats;
    pub fn finger_use(&self) -> [i64; 10];
    pub fn sfbs(&self) -> i64;
    pub fn sfs(&self) -> i64;
    pub fn stretches(&self) -> i64;
    pub fn trigrams(&self) -> TrigramData;
    // etc...
}
```

---

## Magic Key Frequency Calculation (REVISED)

**Key Change**: Instead of pre-computing adjusted frequencies in giant arrays, we compute them on-demand.

### How Magic Keys Work

When a magic key has rule `leader → magic_key → output`:
- The bigram `leader → output` is **never typed** (user types `leader → magic_key` instead)
- We calculate the effective frequency by checking if a rule applies

### On-Demand Frequency Calculation

```rust
impl Analyzer {
    fn get_bigram_frequency(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
        let base_freq = self.data.get_bigram_u([a, b]);
        
        // Check if this bigram is affected by any magic rule
        for (magic_key, rules) in &cache.magic.rules {
            // If 'b' is the magic key and there's a rule for 'a'
            if b == *magic_key {
                if let Some(output) = rules.get(&a) {
                    // The bigram a→output is never typed, instead a→magic_key is typed
                    // So we add the frequency that a→output would have had
                    // This is approximated by summing trigrams containing a→output
                    
                    // For simplicity: if exact match a→magic_key, use trigram sums
                    // Otherwise return base frequency
                }
            }
            
            // If 'a' is typing something that becomes magic output
            if let Some(output) = rules.get(&a) {
                if b == *output {
                    // This bigram wouldn't be typed directly
                    return 0;
                }
            }
        }
        
        base_freq
    }
}
```

This approach:
- ✅ No massive O(n²) or O(n³) arrays to copy
- ✅ Only stores the actual rules (small HashMap)
- ✅ Computes frequencies on-demand when needed
- ✅ Much faster `copy_from()` operation

---

## Implementation Summary

Execute in **6 phases**, each testable independently:

### Phase 1: Restructure Data Ownership (REVISED)
**Goal**: Separate immutable structure data from mutable scores

**Changes**:
1. **Move from CachedLayout to Analyzer**:
   - Layout metadata: `fingers`, `keyboard`, `shape`, `char_mapping`
   - Immutable indices: `SfbIndices`, `stretch_pairs`, `TrigramIndices`
   
2. **Simplify CachedLayout**:
   - Remove all indices (stored in Analyzer now)
   - Remove `bigrams`, `skipgrams`, `trigrams` from MagicCache (compute on-demand)
   - Keep only mutable scores: `per_finger`, `per_type`, `total`, `rules`

3. **Add to Analyzer**:
   - `current: CachedLayout` and `working: CachedLayout` fields
   - `sfb_indices: SfbIndices` (immutable)
   - `stretch_pairs: Box<[BigramPair]>` (immutable)
   - `trigram_indices: TrigramIndices` (immutable)

**Files Modified**:
- `core/src/cached_layout.rs`: Strip to mutable-only data
- `core/src/analyze.rs`: Add immutable fields + current/working
- All scoring methods: Use `self.sfb_indices` instead of `cache.sfb.indices`

**Benefits**:
- `copy_from()` is now MUCH faster (no giant arrays)
- Clear separation: Analyzer = structure, CachedLayout = scores
- Magic frequencies computed on-demand (cleaner, no huge arrays)

**Testing**: Run `cargo test` - everything should still pass

### Phase 2: Core Primitives (add_key/remove_key)
- Add `add_key_to_cache(cache, pos, key)` - updates all score caches
- Add `remove_key_from_cache(cache, pos)` - reverses all updates
- Add helper methods on Analyzer: `finger_positions()`, `stretch_positions()`, `trigram_positions()`
- Add distance getters: `SfbIndices::get_distance()`, `StretchCache::get_distance()`
- Refactor initialization to build CachedLayout using ONLY `add_key`
- **Testing**: Verify `add_key` → `remove_key` restores score to 0

### Phase 3: Trigram Integration
- Create `core/src/trigram_cache.rs` with TrigramCache structure
- Add `trigram: TrigramCache` field to CachedLayout
- Update `add_key_to_cache()` to update trigram scores
- Update `remove_key_from_cache()` to reverse trigram scores
- Update `score()` method to include `cache.trigram.total`
- Add `Weights::get_trigram_weight(TrigramType)` method
- **Testing**: Verify trigrams contribute to optimization, compare stats

### Phase 4: Modernize Neighbor Operations
- Add `load_layout(layout, pins)` - initializes `self.current` and `self.working`
- Add `score()` - returns `self.current` score
- Add `check_neighbor(neighbor)` - uses `self.working` without mutating `self.current`
- Add `apply_neighbor(neighbor) → i64` - modifies `self.current` and returns new score
- Add `possible_neighbors(pins)` - lazy iterator on Analyzer
- Add `get_layout()` - extracts Layout from `self.current` + Analyzer's immutable fields
- Add `CachedLayout::copy_from()` for fast shallow copy
- Remove `possible_neighbors` field from CachedLayout
- Rewrite `greedy_improve()` using new API
- **Testing**: Verify check/apply consistency, state isolation

### Phase 5: API Cleanup
- Make CachedLayout fully private (remove from `pub use` in lib.rs if present)
- Update all stats methods to access `self.current` directly
- Remove or deprecate `cached_layout()` and `score_cache()` public methods
- Ensure only Layout and scores are exposed publicly
- **Testing**: Verify no external breakage

### Phase 6: Testing & Validation
- **Symmetry tests**: add_key/remove_key restore original score
- **Consistency tests**: check_neighbor score == apply_neighbor score
- **Magic tests**: Frequency conservation, rule changes work correctly
- **Isolation tests**: check_neighbor doesn't mutate state
- **Accuracy tests**: Cached scores match full recalculations
- **Performance tests**: Benchmark before/after refactor

---

## Critical Correctness Invariants

These must hold at all times:

1. **add_key/remove_key symmetry**: `add_key(p, k); remove_key(p);` → score returns to original

2. **Magic frequency correctness**: On-demand frequency calculation must correctly account for all active magic rules

3. **State isolation**: `check_neighbor()` must not modify `self.current`

4. **Score consistency**: `score()` must equal sum of all cache totals in `self.current`

5. **Check/apply equivalence**: `check_neighbor(n)` score must equal `apply_neighbor(n)` score

---

## Detailed Implementation Reference

### Core Primitives Implementation

```rust
impl Analyzer {
    /// Add a key to empty position, updating all affected scores
    fn add_key_to_cache(&self, cache: &mut CachedLayout, pos: u8, key: u8) {
        assert_eq!(cache.keys[pos as usize], EMPTY);
        cache.keys[pos as usize] = key;
        
        let finger = self.fingers[pos as usize];  // From Analyzer, not cache!
        
        // Update SFB/SFS for this finger
        if self.analyze_bigrams {
            let mut delta = 0i64;
            for other_pos in self.finger_positions(finger) {
                if other_pos == pos { continue; }
                let other_key = cache.keys[other_pos as usize];
                if other_key == EMPTY { continue; }
                
                // Get distance from Analyzer's immutable indices
                let dist = self.sfb_indices.get_distance(pos, other_pos);
                
                // Compute frequencies on-demand (no pre-computed arrays!)
                let bg = self.get_bigram_frequency(cache, key, other_key)
                    + self.get_bigram_frequency(cache, other_key, key);
                delta += bg * dist * self.weights.sfbs;
                
                let sg = self.get_skipgram_frequency(cache, key, other_key)
                    + self.get_skipgram_frequency(cache, other_key, key);
                delta += sg * dist * self.weights.sfs;
            }
            cache.sfb.per_finger[finger as usize] += delta;
            cache.sfb.total += delta;
        }
        
        // Update stretches
        if self.analyze_stretches {
            let mut delta = 0i64;
            for pair in &self.stretch_pairs {  // From Analyzer's immutable data!
                if pair.pair.0 != pos && pair.pair.1 != pos { continue; }
                
                let other_pos = if pair.pair.0 == pos { pair.pair.1 } else { pair.pair.0 };
                let other_key = cache.keys[other_pos as usize];
                if other_key == EMPTY { continue; }
                
                let bg = self.get_bigram_frequency(cache, key, other_key)
                    + self.get_bigram_frequency(cache, other_key, key);
                let sg = self.get_skipgram_frequency(cache, key, other_key)
                    + self.get_skipgram_frequency(cache, other_key, key);
                
                let sfb_over_sfs = self.weights.sfbs as f64 / self.weights.sfs as f64;
                delta += (bg + (sg as f64 * sfb_over_sfs) as i64)
                    * self.weights.stretches
                    * pair.dist;
            }
            cache.stretch.total += delta;
        }
        
        // Update trigrams
        if self.analyze_trigrams {
            for tpos in self.trigram_indices.affected_by(pos as usize) {  // From Analyzer!
                let [p1, p2, p3] = tpos.pos;
                let k1 = cache.keys[p1 as usize];
                let k2 = cache.keys[p2 as usize];
                let k3 = cache.keys[p3 as usize];
                
                if k1 != EMPTY && k2 != EMPTY && k3 != EMPTY {
                    let freq = self.get_trigram_frequency(cache, k1, k2, k3);
                    let weight = self.weights.get_trigram_weight(tpos.ttype);
                    cache.trigram.per_type[tpos.ttype as usize] += freq * weight;
                    cache.trigram.total += freq * weight;
                }
            }
        }
    }
    
    /// Remove a key from position, reversing all score contributions
    fn remove_key_from_cache(&self, cache: &mut CachedLayout, pos: u8) {
        let key = cache.keys[pos as usize];
        assert_ne!(key, EMPTY);
        
        // All operations are exact reversal of add_key
        // Use self.fingers, self.sfb_indices, self.stretch_pairs, self.trigram_indices
        // Negate all deltas before removing the key
        
        // ... (same structure as add_key but subtracting instead of adding)
        
        cache.keys[pos as usize] = EMPTY;
    }
    
    // On-demand frequency calculation (no pre-computed arrays!)
    fn get_bigram_frequency(&self, cache: &CachedLayout, a: u8, b: u8) -> i64 {
        let mut freq = self.data.get_bigram_u([a, b]);
        
        // Check if any magic rule affects this bigram
        for (magic_key, rules) in &cache.magic.rules {
            if let Some(&output) = rules.get(&a) {
                if b == *magic_key {
                    // a→magic_key typed instead of a→output
                    // Gain frequency from trigrams where output follows
                    for &c in cache.keys.iter() {
                        if c == EMPTY { continue; }
                        freq += self.data.get_trigram_u([a, output, c]);
                    }
                }
                if b == output {
                    // a→output never typed directly (typed as a→magic_key)
                    return 0;
                }
            }
        }
        
        freq
    }
}
```

### Neighbor Operations Implementation

```rust
impl Analyzer {
    pub fn check_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        // Copy current → working (fast shallow copy)
        self.working.copy_from(&self.current);
        
        // Apply neighbor to working cache
        self.apply_neighbor_to_cache(&mut self.working, neighbor);
        
        // Return score without modifying self.current
        self.score_cache(&self.working)
    }
    
    pub fn apply_neighbor(&mut self, neighbor: Neighbor) -> i64 {
        self.apply_neighbor_to_cache(&mut self.current, neighbor);
        self.score_cache(&self.current)
    }
    
    fn apply_neighbor_to_cache(&self, cache: &mut CachedLayout, neighbor: Neighbor) {
        match neighbor {
            Neighbor::KeySwap(PosPair(a, b)) => {
                if a == b { return; }
                
                let key_a = cache.keys[a as usize];
                let key_b = cache.keys[b as usize];
                
                self.remove_key_from_cache(cache, a);
                self.remove_key_from_cache(cache, b);
                self.add_key_to_cache(cache, a, key_b);
                self.add_key_to_cache(cache, b, key_a);
            }
            Neighbor::MagicRule(MagicRule(magic_key, leader, new_output)) => {
                // Handle magic rule change
                // ... implementation details
            }
        }
    }
}
```

### Trigram Cache Structure

```rust
// New file: core/src/trigram_cache.rs

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrigramCache {
    pub per_type: Box<[i64; 10]>,
    pub total: i64,
    pub(crate) indices: TrigramIndices,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct TrigramIndices {
    by_first: Vec<Vec<TrigramPos>>,
    by_second: Vec<Vec<TrigramPos>>,
    by_third: Vec<Vec<TrigramPos>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrigramPos {
    pub pos: [u8; 3],
    pub ttype: TrigramType,
}

impl TrigramIndices {
    fn compute(fingers: &[Finger]) -> Self {
        let n = fingers.len();
        let mut by_first = vec![Vec::new(); n];
        let mut by_second = vec![Vec::new(); n];
        let mut by_third = vec![Vec::new(); n];
        
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let ttype = TRIGRAMS[
                        fingers[i] as usize * 100 
                        + fingers[j] as usize * 10 
                        + fingers[k] as usize
                    ];
                    
                    let tpos = TrigramPos {
                        pos: [i as u8, j as u8, k as u8],
                        ttype,
                    };
                    
                    by_first[i].push(tpos);
                    by_second[j].push(tpos);
                    by_third[k].push(tpos);
                }
            }
        }
        
        Self { by_first, by_second, by_third }
    }
    
    pub fn affected_by(&self, pos: usize) -> impl Iterator<Item = &TrigramPos> {
        self.by_first[pos].iter()
            .chain(self.by_second[pos].iter())
            .chain(self.by_third[pos].iter())
    }
}
```

---

## End Notes

This refactoring achieves:

✅ **Simplified codebase** - Single source of truth for cache updates  
✅ **Better performance** - Kept all critical optimizations (per_finger, lazy neighbors)  
✅ **Feature complete** - Trigrams fully integrated into optimization  
✅ **Clean API** - Internal state hidden, public API is minimal and clear  
✅ **Maintainable** - Easy to add new metrics (just update add_key/remove_key)  
✅ **Correct** - Well-defined invariants, comprehensive test suite  

Implementation can proceed phase-by-phase with testing at each step. The architecture is designed to be incremental - each phase maintains a working codebase.
