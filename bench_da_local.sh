#!/bin/bash
# DA local optimizer benchmark
# Compares greedy-only vs SA+greedy with various SA temperatures
# Tests on my-layout (magic key layout)

set -e

BINARY="./target/release/main"
LAYOUT="my-layout"
TIME=120  # 2 minutes per run
RUNS=3
CSV="da_local_bench.csv"

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "layout,label,sa_iters,sa_temp,sa_final,greedy,run,elapsed_s,best_score,iters,rate,restarts" > "$CSV"

run_test() {
    local label="$1" sa="$2" sa_temp="$3" sa_final="$4" greedy="$5" run_num="$6"

    local cmd="da $LAYOUT -s $sa --sa-temp $sa_temp --sa-final $sa_final -g $greedy -t $TIME"
    echo -n "  [$run_num/$RUNS] "

    local output
    output=$(echo "$cmd
q" | "$BINARY" 2>/dev/null)

    local best elapsed restarts iters
    best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
    elapsed=$(echo "$output" | grep "completed in" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
    restarts=$(echo "$output" | grep "completed in" | sed -E 's/.*\(([0-9]+) restarts\).*/\1/')
    iters=$(echo "$output" | grep "iter " | tail -1 | sed -E 's/.*iter ([0-9.]+[A-Za-z]*).*/\1/')
    local iters_raw
    iters_raw=$(echo "$iters" | sed -E 's/([0-9.]+)K$/\1e3/; s/([0-9.]+)M$/\1e6/' | awk '{printf "%.0f", $1}')
    local rate
    rate=$(echo "$iters_raw $elapsed" | awk '{if ($2>0) printf "%.1f", $1/$2; else print "0"}')
    local best_raw
    best_raw=$(echo "$best" | sed -E 's/([0-9.]+)T$/\1e12/; s/([0-9.]+)B$/\1e9/; s/([0-9.]+)M$/\1e6/; s/^-//' | awk '{printf "%.0f", $1}')
    best_raw="-$best_raw"

    echo "$LAYOUT,$label,$sa,$sa_temp,$sa_final,$greedy,$run_num,$elapsed,$best_raw,$iters_raw,$rate,$restarts" >> "$CSV"
    echo "$label: best=$best iters=$iters rate=${rate}/s"
}

# Configs: (label, sa_iters, sa_temp, sa_final, greedy)
configs=(
    # Greedy only (baseline)
    "greedy-only,0,10,1e-5,1"

    # Low-temp SA + greedy (exploit-focused)
    "sa100-t0.1,100,0.1,1e-5,1"
    "sa100-t0.01,100,0.01,1e-5,1"
    "sa100-t0.001,100,0.001,1e-7,1"
    "sa500-t0.1,500,0.1,1e-5,1"
    "sa500-t0.01,500,0.01,1e-5,1"
    "sa500-t0.001,500,0.001,1e-7,1"
    "sa1000-t0.1,1000,0.1,1e-5,1"
    "sa1000-t0.01,1000,0.01,1e-5,1"
    "sa1000-t0.001,1000,0.001,1e-7,1"

    # Medium-temp SA + greedy
    "sa1000-t1.0,1000,1.0,1e-5,1"
    "sa1000-t0.5,1000,0.5,1e-5,1"

    # Higher iters, low temp
    "sa5000-t0.01,5000,0.01,1e-5,1"
    "sa5000-t0.1,5000,0.1,1e-5,1"

    # Original hot SA (for comparison)
    "sa1000-t10,1000,10.0,1e-5,1"
    "sa10000-t10,10000,10.0,1e-5,1"

    # SA only, no greedy
    "sa1000-t0.01-ng,1000,0.01,1e-5,0"
    "sa5000-t0.01-ng,5000,0.01,1e-5,0"
)

total=$((${#configs[@]} * RUNS))
est_min=$(echo "$total * $TIME / 60" | bc)
echo "Configs: ${#configs[@]}, Runs: $RUNS, Time/run: ${TIME}s"
echo "Total: $total runs (~${est_min} minutes)"
echo ""

current=0
for config_str in "${configs[@]}"; do
    IFS=',' read -r label sa sa_temp sa_final greedy <<< "$config_str"
    echo "Config: $label"
    for run in $(seq 1 $RUNS); do
        current=$((current + 1))
        run_test "$label" "$sa" "$sa_temp" "$sa_final" "$greedy" "$run"
    done
    echo ""
done

echo "Done! Results in $CSV"
echo ""
echo "=== Summary (sorted by mean best score, less negative = better) ==="
awk -F',' 'NR>1 {
    key=$2
    scores[key] = scores[key] " " $9
    rates[key] = rates[key] " " $11
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
        printf "%-25s mean_best=%15.0f  rate=%.1f/s\n", k, mean, rmean
    }
}' "$CSV" | sort -t= -k2 -n -r
