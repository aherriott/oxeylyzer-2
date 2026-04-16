#!/bin/bash
# Compare: DA vs random-restart with same local optimizer
# Same 30s time budget, which approach finds better layouts?

set -e

BINARY="./target/release/main"
LAYOUT="my-layout"
TIME=30
RUNS=5

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "4 methods, $RUNS runs each, ${TIME}s per run"
echo ""

echo "=== 1. DA with SA-1K@0.001 + greedy (default) ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "da $LAYOUT -t $TIME
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
        echo "  DA [$run]: $best"
    ) &
done
wait

echo ""
echo "=== 2. Random restart: SA-1K@0.001 + greedy (same local as DA, no global) ==="
# Each SA+greedy takes ~0.15s, so ~200 in 30s
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 200 -s 1000 --sa-temp 0.001 --sa-final 1e-7
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  SA-1K [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "=== 3. Random restart: greedy only (no SA) ==="
# Each greedy takes ~0.05s, so ~600 in 30s
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 600 -s 0
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  Greedy [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "=== 4. Random restart: SA-10M@hot + greedy (old approach) ==="
# Each takes ~30s, so 1 layout per run
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 1 -s 10000000
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  SA-10M [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "Done!"
