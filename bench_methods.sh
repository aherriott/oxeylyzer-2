#!/bin/bash
# Benchmark optimization methods: gen (random-restart greedy), sa, da, mcts
# Runs each method at several time budgets, multiple repetitions, outputs CSV

set -u

BINARY="./target/release/main"
LAYOUT="my-layout"
REPS=3
THREADS=3

# Time budgets in seconds
TIMES=(30 120 600)

OUT="bench_methods_results.csv"
echo "method,time_secs,rep,score,duration_secs" > "$OUT"

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null || { echo "Build failed"; exit 1; }

parse_score() {
    # Parse score value from lines like "#1, score: -992B" or "Best score: -1.21T"
    # We need the number AFTER "score:", not the "#1" rank.
    local line="$1"
    # Strip everything before "score:" to isolate the score value
    local after=$(echo "$line" | sed 's/.*score:[[:space:]]*//')
    local num=$(echo "$after" | grep -oE '\-?[0-9]+\.?[0-9]*[KMBT]?' | head -1)
    # Convert to raw i64
    python3 -c "
s = '$num'
if not s: print(0); exit()
mult = 1
if s.endswith('T'): mult = 10**12; s = s[:-1]
elif s.endswith('B'): mult = 10**9; s = s[:-1]
elif s.endswith('M'): mult = 10**6; s = s[:-1]
elif s.endswith('K'): mult = 10**3; s = s[:-1]
print(int(float(s) * mult))
"
}

run_method() {
    local method="$1"
    local time_secs="$2"
    local rep="$3"
    local cmd="$4"

    echo "  [$method t=${time_secs}s rep=$rep]"
    local start=$(date +%s)
    local out
    out=$(printf "$cmd\nq\n" | RAYON_NUM_THREADS=$THREADS $BINARY 2>&1)
    local end=$(date +%s)
    local duration=$((end - start))

    # Extract the best score — first line after "score:" or similar
    local best_line=$(echo "$out" | grep -E "^#1|^Best score:" | head -1)
    local score=$(parse_score "$best_line")

    echo "$method,$time_secs,$rep,$score,$duration" >> "$OUT"
    echo "    score=$score  duration=${duration}s"
}

for t in "${TIMES[@]}"; do
    for rep in $(seq 1 $REPS); do
        # gen (random-restart greedy, runs for t seconds)
        run_method "gen" "$t" "$rep" "gen $LAYOUT -t $t -n 1"

        # sa: 1M iters ~ 1s, so sa_iters = t * 1M
        sa_iters=$((t * 1000000))
        run_method "sa" "$t" "$rep" "sa $LAYOUT 1 -s $sa_iters"

        # da: dual annealing with time limit
        run_method "da" "$t" "$rep" "da $LAYOUT -t $t"

        # mcts: time-limited
        run_method "mcts" "$t" "$rep" "mcts $LAYOUT -t $t"
    done
done

echo ""
echo "Results written to $OUT"
echo ""
echo "=== Summary ==="
python3 -c "
import csv
from collections import defaultdict

by_key = defaultdict(list)
with open('$OUT') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['method'], row['time_secs'])
        by_key[key].append(int(row['score']))

# Format: method | 30s | 120s | 600s
methods = ['gen', 'sa', 'da', 'mcts']
times = [30, 120, 600]

# Use human-readable numbers
def fmt(n):
    n = abs(n)
    if n >= 1e12: return f'-{n/1e12:.2f}T'
    if n >= 1e9: return f'-{n/1e9:.2f}B'
    if n >= 1e6: return f'-{n/1e6:.2f}M'
    if n >= 1e3: return f'-{n/1e3:.2f}K'
    return f'-{n}'

print(f\"{'method':<8} {'30s':>12} {'120s':>12} {'600s':>12}\")
print('-' * 52)
for m in methods:
    row = [m]
    for t in times:
        scores = by_key.get((m, str(t)), [])
        if scores:
            # Best of reps (least negative score)
            best = max(scores)
            row.append(fmt(best))
        else:
            row.append('--')
    print(f\"{row[0]:<8} {row[1]:>12} {row[2]:>12} {row[3]:>12}\")
"
