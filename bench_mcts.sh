#!/bin/bash
# MCTS parameter sweep benchmark
# Runs each configuration 5 times for 5 minutes (300s) each
# Outputs results to mcts_bench_results.csv

set -e

BINARY="./target/release/main"
LAYOUT="nrts-oxey"
TIME=120  # 5 minutes per run
RUNS=5
CSV="mcts_bench_results.csv"

# Build release first
echo "Building release..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "layout,sa_iters,greedy,tree_depth,explore,run,elapsed_s,rollouts,best_score,rollouts_per_s" > "$CSV"

run_test() {
    local sa="$1"
    local greedy="$2"
    local depth="$3"
    local explore="$4"
    local run_num="$5"

    local cmd="mcts $LAYOUT -s $sa -g $greedy -d $depth -c $explore -t $TIME"
    echo "  Run $run_num: $cmd"

    local output
    output=$(echo "$cmd
q" | "$BINARY" 2>/dev/null)

    local elapsed rollouts best

    elapsed=$(echo "$output" | grep "MCTS completed" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
    rollouts=$(echo "$output" | grep "MCTS completed" | sed -E 's/.*\(([^ ]+) rollouts\).*/\1/')
    best=$(echo "$output" | grep "^#1:" | sed -E 's/.*score ([-0-9.]+[A-Za-z]*)/\1/')

    # Convert human-readable numbers back to raw values
    local rollouts_raw best_raw
    rollouts_raw=$(echo "$rollouts" | sed -E '
        s/([0-9.]+)T$/\1e12/
        s/([0-9.]+)B$/\1e9/
        s/([0-9.]+)M$/\1e6/
        s/([0-9.]+)K$/\1e3/
    ')
    rollouts_raw=$(echo "$rollouts_raw" | awk '{printf "%.0f", $1}')

    best_raw=$(echo "$best" | sed -E '
        s/([0-9.]+)T$/\1e12/
        s/([0-9.]+)B$/\1e9/
        s/([0-9.]+)M$/\1e6/
        s/([0-9.]+)K$/\1e3/
        s/^-//' | awk '{printf "%.0f", $1}')
    best_raw="-$best_raw"

    local rate
    rate=$(echo "$rollouts_raw $elapsed" | awk '{if ($2 > 0) printf "%.1f", $1/$2; else print "0"}')

    echo "$LAYOUT,$sa,$greedy,$depth,$explore,$run_num,$elapsed,$rollouts_raw,$best_raw,$rate" >> "$CSV"
    echo "    -> best=$best, rollouts=$rollouts, rate=${rate}/s"
}

# Focused parameter grid — test the most impactful dimensions
# Each config: (sa_iters, greedy, tree_depth, explore)
configs=(
    # Vary SA iterations (greedy=0, depth=0, explore=1.41)
    "100,0,0,1.41"
    "1000,0,0,1.41"
    "10000,0,0,1.41"
    "100000,0,0,1.41"

    # Vary SA iterations with greedy=1 (depth=0, explore=1.41)
    "100,1,0,1.41"
    "1000,1,0,1.41"
    "10000,1,0,1.41"
    "100000,1,0,1.41"

    # Greedy only (no SA)
    "0,1,0,1.41"

    # Vary tree depth (sa=10000, greedy=0, explore=1.41)
    "10000,0,1,1.41"
    "10000,0,3,1.41"
    "10000,0,5,1.41"
    "10000,0,10,1.41"

    # Vary tree depth with greedy (sa=10000, greedy=1, explore=1.41)
    "10000,1,1,1.41"
    "10000,1,3,1.41"
    "10000,1,5,1.41"

    # Vary exploration constant (sa=10000, greedy=1, depth=1)
    "10000,1,1,0.5"
    "10000,1,1,1.0"
    "10000,1,1,2.0"
    "10000,1,1,3.0"

    # Best SA + greedy combos at depth=1
    "1000,1,1,1.41"
    "100000,1,1,1.41"
)

total=$((${#configs[@]} * RUNS))
est_hours=$(echo "$total * $TIME / 3600" | bc -l | xargs printf "%.1f")
echo "Configs: ${#configs[@]}, Runs per config: $RUNS, Time per run: ${TIME}s"
echo "Total test runs: $total (estimated time: ${est_hours} hours)"
echo ""

current=0
for config in "${configs[@]}"; do
    IFS=',' read -r sa greedy depth explore <<< "$config"
    echo "Config: sa=$sa greedy=$greedy depth=$depth explore=$explore"
    for run in $(seq 1 $RUNS); do
        current=$((current + 1))
        echo -n "  [$current/$total] "
        run_test "$sa" "$greedy" "$depth" "$explore" "$run"
    done
    echo ""
done

echo "Done! Results saved to $CSV"
echo ""
echo "=== Summary (sorted by mean best score, less negative = better) ==="
echo "sa,greedy,depth,explore | mean_best | std | mean_rate/s"
echo "---"
awk -F',' 'NR>1 {
    key=$2","$3","$4","$5
    scores[key] = scores[key] " " $9
    rates[key] = rates[key] " " $10
    n[key]++
}
END {
    for (k in n) {
        split(scores[k], s, " ")
        split(rates[k], r, " ")
        sum=0; rsum=0; cnt=0
        for (i in s) { if (s[i] != "") { sum += s[i]; cnt++ } }
        for (i in r) { if (r[i] != "") { rsum += r[i] } }
        mean = sum / cnt
        rmean = rsum / cnt
        var=0
        for (i in s) { if (s[i] != "") { var += (s[i] - mean)^2 } }
        std = sqrt(var / cnt)
        printf "%s | %.0f | %.0f | %.1f\n", k, mean, std, rmean
    }
}' "$CSV" | sort -t'|' -k2 -n -r
