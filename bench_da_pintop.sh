#!/bin/bash
# Test DA with pin-top-K: global SA moves all keys, local greedy pins top K
# Compare different K values and high swap counts

set -e

BINARY="./target/release/main"
LAYOUT="my-layout"
TIME=30
RUNS=3

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "Pin-top-K experiment, $RUNS runs each, ${TIME}s per run"
echo ""

# Baseline: no pinning, greedy only
echo "=== Baseline: DA greedy, no pin-top ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "da $LAYOUT -s 0 -t $TIME --swaps 16
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
        echo "  no-pin [$run]: $best"
    ) &
done
wait

echo ""
echo "=== Random restart greedy × 20 (reference) ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 20 -s 0
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        echo "  rand-greedy [$run]: $best"
    ) &
done
wait

# Test different K values with high swaps
for k in 3 5 8 10 15; do
    echo ""
    echo "=== pin-top=$k, swaps=16 ==="
    for run in $(seq 1 $RUNS); do
        (
            output=$(echo "da $LAYOUT -s 0 -t $TIME --swaps 16 --pin-top $k
q" | "$BINARY" 2>/dev/null)
            best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
            echo "  k=$k [$run]: $best"
        ) &
    done
    wait
done

echo ""
echo "Done!"
