#!/bin/bash
# Weight sensitivity experiment
# Analyzes reference layouts under varied weight configs
# Then runs a few gen experiments to see what the optimizer produces

set -e

BINARY="./target/release/main"
CONFIG="./analyzer-config.toml"
CONFIG_BAK="./analyzer-config.toml.bak"
CSV="weight_experiment.csv"
REF_LAYOUTS="nrts-oxey canary colemak-dh sturdy noctum"
GEN_BASE="nrts-oxey"
GEN_TIME=30

echo "Building..."
cargo build -p oxeylyzer-repl --release 2>/dev/null
cp "$CONFIG" "$CONFIG_BAK"

parse_stats() {
    local output="$1"
    local score sfbs sfs stretches inroll outroll alternate redirect onehandin onehandout

    score=$(echo "$output" | grep "^score:" | sed -E 's/score:[[:space:]]+(-?[0-9]+)[[:space:]].*/\1/')
    sfbs=$(echo "$output" | grep "^sfbs:" | sed -E 's/sfbs:[[:space:]]+([0-9.]+)%.*/\1/')
    sfs=$(echo "$output" | grep "^sfs:" | sed -E 's/sfs:[[:space:]]+([0-9.]+)%.*/\1/')
    stretches=$(echo "$output" | grep "^stretches:" | sed -E 's/stretches:[[:space:]]+([0-9.]+)/\1/')
    inroll=$(echo "$output" | grep "^Inroll:" | sed -E 's/Inroll:[[:space:]]+([0-9.]+)%.*/\1/')
    outroll=$(echo "$output" | grep "^Outroll:" | sed -E 's/Outroll:[[:space:]]+([0-9.]+)%.*/\1/')
    alternate=$(echo "$output" | grep "^Alternate:" | sed -E 's/Alternate:[[:space:]]+([0-9.]+)%.*/\1/')
    redirect=$(echo "$output" | grep "^Redirect:" | sed -E 's/Redirect:[[:space:]]+([0-9.]+)%.*/\1/')
    onehandin=$(echo "$output" | grep "^Onehand In:" | sed -E 's/Onehand In:[[:space:]]+([0-9.]+)%.*/\1/')
    onehandout=$(echo "$output" | grep "^Onehand Out:" | sed -E 's/Onehand Out:[[:space:]]+([0-9.]+)%.*/\1/')

    echo "$score,$sfbs,$sfs,$stretches,$inroll,$outroll,$alternate,$redirect,$onehandin,$onehandout"
}

write_config() {
    cat > "$CONFIG" << EOF
corpus = "./data/english.json"
layouts = ["./core/test-layouts", "./layouts"]

[weights]
sfbs = $1
sfs = $2
stretches = $3
sft = -12
inroll = $4
outroll = $5
alternate = $6
redirect = $7
onehandin = $8
onehandout = $9
thumb = 0
full_scissors = -5
half_scissors = -3
full_scissors_skip = -2
half_scissors_skip = -1
magic_rule_penalty = 0
magic_repeat_penalty = 0

[weights.fingers]
lp = 77
lr = 32
lm = 24
li = 21
lt = 46
rt = 46
ri = 21
rm = 24
rr = 32
rp = 77
EOF
}

echo "type,config,layout,sfbs_w,sfs_w,stretches_w,inroll_w,outroll_w,alternate_w,redirect_w,onehandin_w,onehandout_w,score,sfbs,sfs,stretches,inroll,outroll,alternate,redirect,onehandin,onehandout" > "$CSV"

# Analyze all reference layouts with default weights
echo "=== Reference layouts (default weights) ==="
for layout in $REF_LAYOUTS; do
    echo "  $layout"
    output=$(echo "a $layout
q" | "$BINARY" 2>/dev/null)
    stats=$(parse_stats "$output")
    echo "reference,default,$layout,-7,-1,-3,5,4,4,1,1,0,$stats" >> "$CSV"
done

# Weight configs to test: (label, sfbs, sfs, stretches, inroll, outroll, alternate, redirect, onehandin, onehandout)
configs=(
    # Baseline
    "baseline,-7,-1,-3,5,4,4,1,1,0"
    # Heavy SFB penalty
    "heavy-sfb,-15,-1,-3,5,4,4,1,1,0"
    "extreme-sfb,-25,-1,-3,5,4,4,1,1,0"
    # Heavy SFS
    "heavy-sfs,-7,-5,-3,5,4,4,1,1,0"
    # Low SFB (trigrams dominate)
    "low-sfb,-2,-1,-3,5,4,4,1,1,0"
    # Heavy inroll
    "heavy-inroll,-7,-1,-3,15,4,4,1,1,0"
    # Heavy alternate
    "heavy-alt,-7,-1,-3,5,4,15,1,1,0"
    # Heavy redirect penalty
    "heavy-redirect,-7,-1,-3,5,4,4,-10,1,0"
    # Balanced trigrams
    "balanced-tg,-7,-1,-3,8,8,8,-3,2,1"
    # Inroll focused
    "inroll-focus,-5,-1,-2,20,2,2,-2,1,0"
    # Alternate focused
    "alt-focus,-5,-1,-2,2,2,20,-5,1,0"
    # Low penalty, high reward
    "low-pen-high-rew,-3,0,-1,10,8,8,0,2,1"
    # Everything cranked
    "all-high,-15,-5,-8,15,12,12,-8,5,3"
)

echo ""
echo "=== Analyzing reference layouts under different weight configs ==="
for config_str in "${configs[@]}"; do
    IFS=',' read -r label sfbs sfs stretches inroll outroll alternate redirect onehandin onehandout <<< "$config_str"
    echo "Config: $label"
    write_config "$sfbs" "$sfs" "$stretches" "$inroll" "$outroll" "$alternate" "$redirect" "$onehandin" "$onehandout"

    for layout in $REF_LAYOUTS; do
        output=$(echo "r
a $layout
q" | "$BINARY" 2>/dev/null)
        stats=$(parse_stats "$output")
        echo "scored,$label,$layout,$sfbs,$sfs,$stretches,$inroll,$outroll,$alternate,$redirect,$onehandin,$onehandout,$stats" >> "$CSV"
    done
done

echo ""
echo "=== Generating optimized layouts under select weight configs ==="
gen_configs=(
    "baseline,-7,-1,-3,5,4,4,1,1,0"
    "heavy-sfb,-15,-1,-3,5,4,4,1,1,0"
    "heavy-inroll,-7,-1,-3,15,4,4,1,1,0"
    "heavy-alt,-7,-1,-3,5,4,15,1,1,0"
    "balanced-tg,-7,-1,-3,8,8,8,-3,2,1"
    "inroll-focus,-5,-1,-2,20,2,2,-2,1,0"
    "all-high,-15,-5,-8,15,12,12,-8,5,3"
)

for config_str in "${gen_configs[@]}"; do
    IFS=',' read -r label sfbs sfs stretches inroll outroll alternate redirect onehandin onehandout <<< "$config_str"
    echo "Gen: $label (${GEN_TIME}s)"
    write_config "$sfbs" "$sfs" "$stretches" "$inroll" "$outroll" "$alternate" "$redirect" "$onehandin" "$onehandout"

    gen_output=$(echo "r
gen $GEN_BASE 1 -t $GEN_TIME
q" | "$BINARY" 2>/dev/null)

    # Extract the gen score
    gen_score=$(echo "$gen_output" | grep "^#1, score:" | sed -E 's/#1, score: [^ ]+ *//')
    echo "  -> $gen_score"

    # Parse gen stats by analyzing the base layout with these weights (approximation)
    analyze_output=$(echo "r
a $GEN_BASE
q" | "$BINARY" 2>/dev/null)
    stats=$(parse_stats "$analyze_output")
    echo "generated,$label,$GEN_BASE,$sfbs,$sfs,$stretches,$inroll,$outroll,$alternate,$redirect,$onehandin,$onehandout,$stats" >> "$CSV"
done

# Restore
cp "$CONFIG_BAK" "$CONFIG"
rm "$CONFIG_BAK"

echo ""
echo "Done! Results in $CSV"
echo ""
echo "=== Reference layout rankings by config ==="
echo "(Higher score = better under that config)"
echo ""
awk -F',' 'NR>1 && $1=="scored" {
    printf "%-20s %-15s %s\n", $2, $3, $13
}' "$CSV" | sort -k1,1 -k3,3rn
