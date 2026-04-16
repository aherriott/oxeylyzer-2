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
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 9999 -s 1000 --sa-temp 0.001 --sa-final 1e-7
q" | timeout $((TIME + 2)) "$BINARY" 2>/dev/null || true)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        count=$(echo "$output" | grep "| score:" | wc -l | tr -d ' ')
        echo "  SA-1K [$run]: $best ($count layouts)"
    ) &
done
wait

echo ""
echo "=== 3. Random restart: greedy only (no SA) ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 9999 -s 0
q" | timeout $((TIME + 2)) "$BINARY" 2>/dev/null || true)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        count=$(echo "$output" | grep "| score:" | wc -l | tr -d ' ')
        echo "  Greedy [$run]: $best ($count layouts)"
    ) &
done
wait

echo ""
echo "=== 4. Random restart: SA-10M@hot + greedy (old approach) ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 9999 -s 10000000
q" | timeout $((TIME + 2)) "$BINARY" 2>/dev/null || true)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        if [ -z "$best" ]; then
            best=$(echo "$output" | grep "| score:" | tail -1 | sed -E 's/.*score: ([^ ]+).*/\1/')
        fi
        count=$(echo "$output" | grep "| score:" | wc -l | tr -d ' ')
        echo "  SA-10M [$run]: $best ($count layouts)"
    ) &
done
wait

echo ""
echo "Done!"
