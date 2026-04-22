#!/bin/bash
# One-at-a-time (OAT) sweep to tune basin hopping parameters.
# Starts with defaults, finds best value for each param in sequence.

set -u

DURATION=${1:-120}  # seconds per run (default 2 min)
LAYOUT="my-layout"
THREADS=3
REPS=2  # reps per config to reduce noise

BINARY="./target/release/main"
OUT="bh_tune_results.csv"
LOG_DIR="bh_tune_logs"

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null || { echo "Build failed"; exit 1; }

mkdir -p "$LOG_DIR"
echo "stage,param_name,param_value,rep,final_score,iterations,restarts,improves" > "$OUT"

# Current best params (start at defaults)
BEST_SWAPS=4
BEST_TEMP=0.1
BEST_COOL=0.9999
BEST_STALE=100

# Parse: "Best score: -1.21T" → raw i64
parse_score() {
    grep -oE 'Best score: -?[0-9]+\.?[0-9]*[KMBT]?' "$1" | tail -1 | \
        sed 's/Best score: //' | \
        python3 -c "
import sys
s = sys.stdin.read().strip()
if not s: print(0); exit()
mult = 1
if s.endswith('T'): mult = 10**12; s = s[:-1]
elif s.endswith('B'): mult = 10**9; s = s[:-1]
elif s.endswith('M'): mult = 10**6; s = s[:-1]
elif s.endswith('K'): mult = 10**3; s = s[:-1]
print(int(float(s) * mult))
"
}

# Parse: "42 restarts, 100 accepts, 8 improvements"
parse_stats() {
    local line=$(grep -E "restarts," "$1" | head -1)
    local restarts=$(echo "$line" | grep -oE "[0-9]+ restarts" | grep -oE "[0-9]+")
    local improves=$(echo "$line" | grep -oE "[0-9]+ improvements" | grep -oE "[0-9]+")
    # Iterations: from "iter XXX | restarts..." in progress
    local iter=$(grep -oE "iter [0-9.KMBT]+" "$1" | tail -1 | grep -oE "[0-9.KMBT]+" | python3 -c "
import sys
s = sys.stdin.read().strip()
if not s: print(0); exit()
mult = 1
if s.endswith('T'): mult = 10**12; s = s[:-1]
elif s.endswith('B'): mult = 10**9; s = s[:-1]
elif s.endswith('M'): mult = 10**6; s = s[:-1]
elif s.endswith('K'): mult = 10**3; s = s[:-1]
print(int(float(s) * mult))
")
    echo "$iter|$restarts|$improves"
}

run_bh() {
    local stage="$1"
    local param_name="$2"
    local param_value="$3"
    local rep="$4"
    local swaps="$5"
    local temp="$6"
    local cool="$7"
    local stale="$8"

    local log_file="$LOG_DIR/${stage}_${param_name}_${param_value}_rep${rep}.log"

    echo "  [$(date '+%H:%M:%S')] $stage: $param_name=$param_value rep=$rep (swaps=$swaps, temp=$temp, cool=$cool, stale=$stale)"

    printf "bh $LAYOUT -t $DURATION -k $swaps --temp $temp --cool $cool --stale $stale\nq\n" | \
        RAYON_NUM_THREADS=$THREADS $BINARY > "$log_file" 2>&1

    local score=$(parse_score "$log_file")
    local stats=$(parse_stats "$log_file")
    local iter=$(echo "$stats" | cut -d'|' -f1)
    local restarts=$(echo "$stats" | cut -d'|' -f2)
    local improves=$(echo "$stats" | cut -d'|' -f3)

    echo "$stage,$param_name,$param_value,$rep,$score,$iter,$restarts,$improves" >> "$OUT"
    echo "    score=$score iters=$iter restarts=$restarts improves=$improves"
}

# Helper: find best param value for a given stage from the CSV
pick_best() {
    local stage="$1"
    local param_name="$2"
    python3 <<EOF
import csv
from collections import defaultdict
by_val = defaultdict(list)
with open("$OUT") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['stage'] == "$stage" and row['param_name'] == "$param_name":
            try:
                by_val[row['param_value']].append(int(row['final_score']))
            except ValueError:
                pass
# Best value = highest mean score
best_val = None
best_mean = None
for val, scores in by_val.items():
    m = sum(scores) / len(scores)
    if best_mean is None or m > best_mean:
        best_mean = m
        best_val = val
print(best_val)
EOF
}

# Format for display
fmt() {
    python3 -c "
n = abs($1)
if n >= 1e12: print(f'-{n/1e12:.2f}T')
elif n >= 1e9: print(f'-{n/1e9:.2f}B')
elif n >= 1e6: print(f'-{n/1e6:.2f}M')
else: print(f'-{n}')
"
}

START_TIME=$(date +%s)
TOTAL_RUNS=$(( (5+3+3+3) * REPS ))
ESTIMATED_SECS=$(( TOTAL_RUNS * DURATION ))
echo "Total runs: $TOTAL_RUNS × ${DURATION}s = ${ESTIMATED_SECS}s"
echo ""

# === Stage 1: perturbation_swaps ===
echo "=== Stage 1: perturbation_swaps ==="
for swaps in 2 4 6 8 12; do
    for rep in $(seq 1 $REPS); do
        run_bh "stage1" "swaps" "$swaps" "$rep" "$swaps" "$BEST_TEMP" "$BEST_COOL" "$BEST_STALE"
    done
done
BEST_SWAPS=$(pick_best "stage1" "swaps")
echo ">>> Best perturbation_swaps: $BEST_SWAPS"
echo ""

# === Stage 2: initial_temp ===
# Range 0.01 to 1.0 — BH iterates less than SA, so exp(-delta/T) needs to be more selective.
# At |delta|/|score| ≈ 0.3 (typical basin-to-basin change):
#   T=1.0: 74% accept — too permissive (random walk)
#   T=0.1: 5% accept  — reasonable selectivity
#   T=0.01: ~0% accept — pure greedy
echo "=== Stage 2: initial_temp ==="
for temp in 0.01 0.1 0.3 1.0; do
    for rep in $(seq 1 $REPS); do
        run_bh "stage2" "temp" "$temp" "$rep" "$BEST_SWAPS" "$temp" "$BEST_COOL" "$BEST_STALE"
    done
done
BEST_TEMP=$(pick_best "stage2" "temp")
echo ">>> Best initial_temp: $BEST_TEMP"
echo ""

# === Stage 3: cooling_rate ===
echo "=== Stage 3: cooling_rate ==="
for cool in 0.999 0.9999 0.99995; do
    for rep in $(seq 1 $REPS); do
        run_bh "stage3" "cool" "$cool" "$rep" "$BEST_SWAPS" "$BEST_TEMP" "$cool" "$BEST_STALE"
    done
done
BEST_COOL=$(pick_best "stage3" "cool")
echo ">>> Best cooling_rate: $BEST_COOL"
echo ""

# === Stage 4: restart_after_stale ===
echo "=== Stage 4: restart_after_stale ==="
for stale in 20 100 500; do
    for rep in $(seq 1 $REPS); do
        run_bh "stage4" "stale" "$stale" "$rep" "$BEST_SWAPS" "$BEST_TEMP" "$BEST_COOL" "$stale"
    done
done
BEST_STALE=$(pick_best "stage4" "stale")
echo ">>> Best restart_after_stale: $BEST_STALE"
echo ""

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "=== DONE ==="
echo "Elapsed: ${ELAPSED}s"
echo ""
echo "Tuned BH parameters:"
echo "  perturbation_swaps = $BEST_SWAPS"
echo "  initial_temp       = $BEST_TEMP"
echo "  cooling_rate       = $BEST_COOL"
echo "  restart_after_stale= $BEST_STALE"
echo ""
echo "Suggested command:"
echo "  bh $LAYOUT -t <secs> -k $BEST_SWAPS --temp $BEST_TEMP --cool $BEST_COOL --stale $BEST_STALE"
echo ""

# Summary table
python3 <<EOF
import csv
from collections import defaultdict

by_key = defaultdict(list)
with open("$OUT") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            score = int(row['final_score'])
            by_key[(row['stage'], row['param_name'], row['param_value'])].append(score)
        except ValueError:
            pass

def fmt(n):
    n = abs(n)
    if n >= 1e12: return f'-{n/1e12:.2f}T'
    if n >= 1e9: return f'-{n/1e9:.2f}B'
    if n >= 1e6: return f'-{n/1e6:.2f}M'
    return f'-{n}'

print("\\n=== Results by stage ===")
current_stage = None
for (stage, param, val), scores in sorted(by_key.items()):
    if stage != current_stage:
        print(f"\\n{stage} — {param}:")
        current_stage = stage
    mean = sum(scores) / len(scores)
    best = max(scores)
    print(f"  {val:>10}: mean={fmt(int(mean))}  best={fmt(best)}  (n={len(scores)})")
EOF
