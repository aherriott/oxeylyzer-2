#!/bin/bash
# Benchmark optimization methods by running each for a long duration and sampling
# the best-so-far score at regular intervals. Produces convergence curves.
#
# Usage: ./bench_methods.sh [duration_secs]
#   default duration: 2400s (40 min per run)
#
# Total wall-clock time: 3 methods * 3 reps * duration
#   For 8 hours with 3 reps * 3 methods, duration = 8*3600 / 9 ≈ 3200s (~53 min per run)

set -u

DURATION_SECS=${1:-2400}  # default: 40 min per run → 8h total for 4 methods × 3 reps
SAMPLE_INTERVAL=30   # seconds between samples
REPS=3
THREADS=3
LAYOUT="my-layout"

BINARY="./target/release/main"
OUT="bench_convergence.csv"
LOG_DIR="bench_logs"

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null || { echo "Build failed"; exit 1; }

mkdir -p "$LOG_DIR"
echo "method,rep,elapsed_secs,best_score" > "$OUT"

total_runs=$((4 * REPS))
total_secs=$((DURATION_SECS * total_runs))
total_hours=$(echo "scale=1; $total_secs / 3600" | bc)
echo "Running 4 methods x $REPS reps at ${DURATION_SECS}s each = ${total_secs}s (${total_hours}h)"
echo ""

# Parse best score from a progress line like "  X layouts | best: -1.23T | ..."
# or "  iter X | ... | best: -1.23T | ..."
# or "  X rollouts | best: -1.23T | ..."
extract_score() {
    # Reads the last progress line from stdin, extracts the number after "best:"
    # Outputs raw i64 value
    grep -oE 'best: -?[0-9]+\.?[0-9]*[KMBT]?' | tail -1 | \
        sed 's/best: //' | \
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

run_and_sample() {
    local method="$1"
    local rep="$2"
    local cmd="$3"

    local log_file="$LOG_DIR/${method}_rep${rep}.log"
    local run_id="${method}_rep${rep}"
    echo "[$(date '+%H:%M:%S')] Starting $run_id (duration: ${DURATION_SECS}s)"

    # Start the method in background, capturing both stdout and stderr to log file
    printf "$cmd\nq\n" | RAYON_NUM_THREADS=$THREADS $BINARY > "$log_file" 2>&1 &
    local pid=$!

    # Sampling loop: every SAMPLE_INTERVAL, read latest progress line from log
    local elapsed=0
    while [ $elapsed -lt $DURATION_SECS ]; do
        sleep $SAMPLE_INTERVAL
        elapsed=$((elapsed + SAMPLE_INTERVAL))

        if ! kill -0 $pid 2>/dev/null; then
            echo "  [$(date '+%H:%M:%S')] $run_id finished at ${elapsed}s"
            break
        fi

        # Extract best score from log file
        local score
        score=$(cat "$log_file" | tr '\r' '\n' | extract_score)
        if [ -n "$score" ] && [ "$score" != "0" ]; then
            echo "$method,$rep,$elapsed,$score" >> "$OUT"
            echo "  [$(date '+%H:%M:%S')] $run_id @ ${elapsed}s: $score"
        fi
    done

    # Ensure process is stopped
    if kill -0 $pid 2>/dev/null; then
        kill -INT $pid 2>/dev/null
        sleep 2
        kill -KILL $pid 2>/dev/null
    fi
    wait $pid 2>/dev/null

    # Final score from log
    local final_score
    final_score=$(cat "$log_file" | tr '\r' '\n' | extract_score)
    if [ -n "$final_score" ] && [ "$final_score" != "0" ]; then
        echo "$method,$rep,$DURATION_SECS,$final_score" >> "$OUT"
        echo "  [$(date '+%H:%M:%S')] $run_id final: $final_score"
    fi
}

for rep in $(seq 1 $REPS); do
    echo ""
    echo "=== Rep $rep of $REPS ==="

    # gen: random-restart greedy, runs for DURATION_SECS
    run_and_sample "gen" "$rep" "gen $LAYOUT -t $DURATION_SECS -n 1"

    # sa: time-limited SA+greedy with continuous restarts
    run_and_sample "sa" "$rep" "sa $LAYOUT -t $DURATION_SECS"

    # da: dual annealing with time limit
    run_and_sample "da" "$rep" "da $LAYOUT -t $DURATION_SECS"

    # mcts: time-limited
    run_and_sample "mcts" "$rep" "mcts $LAYOUT -t $DURATION_SECS"
done

echo ""
echo "=== Done ==="
echo "Results: $OUT"
echo "Logs: $LOG_DIR/"
echo ""

# Summary
python3 <<EOF
import csv
from collections import defaultdict

data = defaultdict(list)  # (method, rep) -> [(elapsed, score)]
with open("$OUT") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['method'], row['rep'])
        data[key].append((int(row['elapsed_secs']), int(row['best_score'])))

def fmt(n):
    n = abs(n)
    if n >= 1e12: return f'-{n/1e12:.2f}T'
    if n >= 1e9: return f'-{n/1e9:.2f}B'
    if n >= 1e6: return f'-{n/1e6:.2f}M'
    if n >= 1e3: return f'-{n/1e3:.2f}K'
    return f'-{n}'

# Show best final score per method
print("\\n=== Best final score per method ===")
by_method = defaultdict(list)
for (method, rep), samples in data.items():
    if samples:
        final = max(samples, key=lambda s: s[0])  # latest sample
        by_method[method].append(final[1])

for method in ['gen', 'sa', 'da', 'mcts']:
    scores = by_method.get(method, [])
    if scores:
        best = max(scores)
        mean = sum(scores) / len(scores)
        print(f"  {method:<8}: best={fmt(best)}  mean={fmt(int(mean))}  reps={len(scores)}")

# Show convergence milestones
print("\\n=== Convergence milestones (best score by time) ===")
milestones = [60, 300, 900, 1800, 3600]
for method in ['gen', 'sa', 'da', 'mcts']:
    print(f"\\n  {method}:")
    for t in milestones:
        scores_at_t = []
        for (m, rep), samples in data.items():
            if m != method: continue
            # Find best score at or before time t
            valid = [s for (e, s) in samples if e <= t]
            if valid:
                scores_at_t.append(max(valid))
        if scores_at_t:
            best = max(scores_at_t)
            print(f"    @{t:>5}s: best={fmt(best)}  (n={len(scores_at_t)})")
EOF
