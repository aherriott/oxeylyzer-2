#!/bin/bash
# Compare: DA vs random-restart with same local optimizer
# ~30s time budget per method

set -e

BINARY="./target/release/main"
LAYOUT="my-layout"
RUNS=5

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null

echo "4 methods, $RUNS runs each"
echo ""

echo "=== 1. DA 30s ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "da $LAYOUT -t 30
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "Best score:" | sed -E 's/Best score: (.*)/\1/')
        echo "  DA [$run]: $best"
    ) &
done
wait

echo ""
echo "=== 2. SA-1K@0.001 + greedy × 20 ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 20 -s 1000 --sa-temp 0.001 --sa-final 1e-7
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  SA-1K [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "=== 3. Greedy only × 20 ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 20 -s 0
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  Greedy [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "=== 4. SA-10M@hot + greedy × 1 ==="
for run in $(seq 1 $RUNS); do
    (
        output=$(echo "sa $LAYOUT 1
q" | "$BINARY" 2>/dev/null)
        best=$(echo "$output" | grep "^#1, score:" | sed -E 's/#1, score: ([^ ]+).*/\1/')
        elapsed=$(echo "$output" | grep "SA completed" | sed -E 's/SA completed in ([0-9.]+)s/\1/')
        echo "  SA-10M [$run]: $best (${elapsed}s)"
    ) &
done
wait

echo ""
echo "Done!"
