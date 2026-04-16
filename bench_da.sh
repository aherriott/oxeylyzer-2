#!/bin/bash
# DA parameter sweep benchmark
# Varies exploration parameters, 3 parallel runs per config, 30s each

set -e

BINARY="./target/release/main"
LAYOUT="my-layout"
TIME=30
RUNS=3
CSV="da_bench_results.csv"

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "layout,label,temp,qv,qa,restart,swaps,run,elapsed_s,best_score,iters,iters_per_s,restarts" > "$CSV"

run_test() {
    local label="$1" temp="$2" qv="$3" qa="$4" restart="$5" swaps="$6" run_num="$7"

    local cmd="da $LAYOUT --temp $temp --qv $qv --qa $qa --restart $restart --swaps $swaps -t $TIME"

    local output
    output=$(echo "$cmd
q" | "$BINARY" 2>/dev/null)

    local best elapsed iters restarts
    best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
    elapsed=$(echo "$output" | grep "completed in" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
    restarts=$(echo "$output" | grep "completed in" | sed -E 's/.*\(([0-9]+) restarts\).*/\1/')
    iters=$(echo "$output" | grep "iter " | tail -1 | sed -E 's/.*iter ([0-9.]+[A-Za-z]*).*/\1/')
    local iters_raw
    iters_raw=$(echo "$iters" | sed -E 's/([0-9.]+)K$/\1e3/; s/([0-9.]+)M$/\1e6/; s/([0-9.]+)B$/\1e9/' | awk '{printf "%.0f", $1}')
    local rate
    rate=$(echo "$iters_raw $elapsed" | awk '{if ($2>0) printf "%.1f", $1/$2; else print "0"}')
    local best_raw
    best_raw=$(echo "$best" | sed -E 's/([0-9.]+)T$/\1e12/; s/([0-9.]+)B$/\1e9/; s/([0-9.]+)M$/\1e6/; s/([0-9.]+)K$/\1e3/; s/^-//' | awk '{printf "%.0f", $1}')
    best_raw="-$best_raw"

    echo "$LAYOUT,$label,$temp,$qv,$qa,$restart,$swaps,$run_num,$elapsed,$best_raw,$iters_raw,$rate,$restarts" >> "$CSV"
    echo "  $label [$run_num]: best=$best rate=${rate}/s restarts=$restarts"
}

configs=(
    # Baseline
    "baseline,5230,2.62,-5.0,2e-5,8"

    # Vary initial temperature
    "temp=100,100,2.62,-5.0,2e-5,8"
    "temp=1000,1000,2.62,-5.0,2e-5,8"
    "temp=10000,10000,2.62,-5.0,2e-5,8"
    "temp=50000,50000,2.62,-5.0,2e-5,8"

    # Vary qv
    "qv=1.5,5230,1.5,-5.0,2e-5,8"
    "qv=2.0,5230,2.0,-5.0,2e-5,8"
    "qv=2.8,5230,2.8,-5.0,2e-5,8"
    "qv=2.95,5230,2.95,-5.0,2e-5,8"

    # Vary qa
    "qa=-5,5230,2.62,-5.0,2e-5,8"
    "qa=-10,5230,2.62,-10.0,2e-5,8"
    "qa=-50,5230,2.62,-50.0,2e-5,8"
    "qa=-500,5230,2.62,-500.0,2e-5,8"

    # Vary restart ratio
    "restart=1e-3,5230,2.62,-5.0,1e-3,8"
    "restart=1e-4,5230,2.62,-5.0,1e-4,8"
    "restart=1e-6,5230,2.62,-5.0,1e-6,8"

    # Vary max swaps
    "swaps=1,5230,2.62,-5.0,2e-5,1"
    "swaps=2,5230,2.62,-5.0,2e-5,2"
    "swaps=4,5230,2.62,-5.0,2e-5,4"
    "swaps=12,5230,2.62,-5.0,2e-5,12"
    "swaps=16,5230,2.62,-5.0,2e-5,16"
)

total=$((${#configs[@]} * RUNS))
est_min=$(echo "${#configs[@]} * $TIME / 60" | bc)
echo "Configs: ${#configs[@]}, Runs: $RUNS (parallel), Time/run: ${TIME}s"
echo "Total: $total runs (~${est_min} min wall clock)"
echo ""

for config_str in "${configs[@]}"; do
    IFS=',' read -r label temp qv qa restart swaps <<< "$config_str"
    echo "Config: $label"
    for run in $(seq 1 $RUNS); do
        run_test "$label" "$temp" "$qv" "$qa" "$restart" "$swaps" "$run" &
    done
    wait
    echo ""
done

echo "Done! Results in $CSV"
echo ""
echo "=== Summary (sorted by mean best score) ==="
awk -F',' 'NR>1 {
    key=$2
    scores[key] = scores[key] " " $10
    rates[key] = rates[key] " " $12
    n[key]++
}
END {
    for (k in n) {
        split(scores[k], s, " ")
        split(rates[k], r, " ")
        sum=0; rsum=0; cnt=0
        for (i in s) { if (s[i]!="") { sum+=s[i]; cnt++ } }
        for (i in r) { if (r[i]!="") { rsum+=r[i] } }
        mean=sum/cnt; rmean=rsum/cnt
        printf "%-20s mean_best=%15.0f  rate=%.1f/s\n", k, mean, rmean
    }
}' "$CSV" | sort -t= -k2 -n -r
