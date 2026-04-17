---
inclusion: auto
---

# User Preferences & Working Style

## Code Style
- Backwards compatibility is NOT a requirement
- Create additional functions for new techniques rather than refactoring existing methods (preserve progress during iteration)
- NEVER use inline/nested functions in Python
- For Python debugging: create temp .py files, never use `python -c`
- After completing a task: `git add -A && git commit` with conventional commits

## Git
- NEVER use -C in git commands
- Always cd into the directory first
- Conventional commits format

## Performance Philosophy
- Don't care about memory consumption or pre-computation startup time
- Optimize the hot loop
- Benchmarks should segregate init time from hot loop time
- All numbers in progress displays should use `fmt_num` (human-readable K/M/B/T)
- Time remaining should use human-readable format

## Scoring Conventions
- Less negative score = better layout (closer to 0 is better)
- Thumbs are valid keys (not just 8 fingers)
- All weights should be positive (code handles sign convention)
- fmt_num uses 3 significant figures

## Layout Preferences
- Colstag board with magic key and thumb row
- Interested in magic rules but wants control over complexity (magic_rule_penalty)
- Repeat rules (t→t) are valued — easy to remember, eliminate SFBs
- Prefers sturdy-like characteristics: high inrolls, low redirects, low SFBs

## Workflow
- Iterative: generate layouts, compare stats, adjust weights, repeat
- Saves layout history in my-layout_revN files
- Creates .dof files for best layouts (oxey-vN.dof)
- Uses benchmark scripts to compare approaches empirically
- Prefers data-driven decisions over theoretical arguments
