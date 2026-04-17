---
inclusion: auto
---

# Layout Design Notes

## User's Layout: my-layout

A colstag layout with a magic key and 2 thumb keys (apostrophe, space). 32 active positions.

```json
{
    "board": "colstag",
    "layers": {
        "main": [
            "row1: 10 keys",
            "row2: 10 keys",
            "row3: 10 keys (includes &mag)",
            "thumb: 2 keys + 5 unused (~)"
        ]
    }
}
```

## Layout Revisions

The user iterated through multiple revisions, tuning weights between each:

- **rev1**: High magic rule count (16-25), 2x trigram flow vs sturdy, but 2.5x worse SFBs
- **rev2**: Higher sfbs weight (15). Best SFBs (0.539%) with 25 magic rules. Stretches still bad.
- **rev3**: magic_rule_penalty=1. Only 2-4 rules. oxey-v8 standout: 0.628% SFBs, 3676% alternates.
- **rev4**: Nailed stretches (2.247, better than sturdy) and redirects (302%). oxey-v10 is the best overall.
- **rev5**: Latest iteration with further tuning.

## Reference Layouts for Comparison

- **Sturdy**: The gold standard. 0.584% SFBs, 1032% inrolls, 2.483 stretches, 251% redirects.
- **nrts-oxey**: Good trigram flow, slightly worse SFBs than sturdy.
- **Noctum**: Lowest SFBs (0.556%) among non-magic layouts.
- **Canary**: Consistently 4th place.
- **Colemak-DH**: Consistently last. Older design.

## Weight Tuning Philosophy

1. Start with goal-seek to find weights where a target layout (sturdy) ranks #1
2. Run gen with those weights, compare output stats to target
3. Adjust weights to close gaps (e.g., bump inroll if output has too few inrolls)
4. Iterate: gen → compare → adjust → gen

Key insight: the optimizer will always find layouts that beat existing ones under the given weights. The goal is to find weights that produce layouts with the *characteristics* you want (high inrolls, low SFBs, etc.), not weights that make a specific layout win.

## Colstag Board

- 3 rows of 10 keys + thumb row with 7 positions (use ~ for unused)
- Fingering: "traditional"
- Supports magic keys via `&mag` in the layer definition
- Extra pinky columns not supported — use thumb row for extra keys

## .dof File Format

JSON format parsed by libdof. Magic rules defined in a "magic" section:
```json
{
    "magic": {
        "mag": {
            "a": "b",    // typing 'a' then magic produces 'b'
            "t": "t"     // repeat rule: typing 't' then magic produces 't'
        }
    }
}
```

libdof doesn't expose magic rules via public API, so Layout::load parses them from raw JSON separately.
