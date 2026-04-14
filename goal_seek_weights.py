#!/usr/bin/env python3
"""
Goal-seek weights to maximize sturdy's ranking among reference layouts.

Objective: find weights where sturdy has the highest score among
nrts-oxey, canary, colemak-dh, sturdy, noctum.

Uses Nelder-Mead simplex optimization on the weight vector.
"""

import subprocess
import re
import os
import shutil
import numpy as np
from scipy.optimize import minimize

BINARY = "./target/release/main"
CONFIG = "./analyzer-config.toml"
CONFIG_BAK = "./analyzer-config.toml.bak"
TARGET = "sturdy"
LAYOUTS = ["nrts-oxey", "canary", "colemak-dh", "sturdy", "noctum"]

# Weight names and their initial values
WEIGHT_NAMES = [
    "sfbs", "sfs", "stretches", "inroll", "outroll",
    "alternate", "redirect", "onehandin", "onehandout",
    "full_scissors", "half_scissors",
]
INITIAL = np.array([7, 1, 3, 5, 4, 4, 1, 1, 0, 5, 3], dtype=float)

# Fixed weights (not optimized)
FIXED = {
    "sft": 12,
    "thumb": 0,
    "full_scissors_skip": 2,
    "half_scissors_skip": 1,
    "magic_rule_penalty": 0,
    "magic_repeat_penalty": 0,
}

FINGER_WEIGHTS = {
    "lp": 77, "lr": 32, "lm": 24, "li": 21, "lt": 46,
    "rt": 46, "ri": 21, "rm": 24, "rr": 32, "rp": 77,
}

eval_count = 0
best_obj = float("inf")
best_weights = None


def write_config(weights_vec):
    w = {name: max(0, val) for name, val in zip(WEIGHT_NAMES, weights_vec)}
    w.update(FIXED)

    lines = [
        'corpus = "./data/english.json"',
        'layouts = ["./core/test-layouts", "./layouts"]',
        "",
        "[weights]",
    ]
    for k, v in w.items():
        lines.append(f"{k} = {int(round(v))}")

    lines.append("")
    lines.append("[weights.fingers]")
    for k, v in FINGER_WEIGHTS.items():
        lines.append(f"{k} = {v}")
    lines.append("")

    with open(CONFIG, "w") as f:
        f.write("\n".join(lines))


def score_layouts(weights_vec):
    write_config(weights_vec)

    cmds = "\n".join([f"r\na {layout}" for layout in LAYOUTS]) + "\nq\n"
    result = subprocess.run(
        [BINARY],
        input=cmds,
        capture_output=True,
        text=True,
        timeout=120,
    )

    scores = {}
    current_layout = None
    for line in result.stdout.split("\n"):
        # Detect which layout we're analyzing from the command echo
        for layout in LAYOUTS:
            if line.strip().startswith(f"> a {layout}") or line.strip() == f"a {layout}":
                current_layout = layout

        m = re.match(r"score:\s+(-?\d+)", line)
        if m and current_layout:
            scores[current_layout] = int(m.group(1))
            current_layout = None

    return scores


def objective(weights_vec):
    global eval_count, best_obj, best_weights
    eval_count += 1

    # Clamp to positive
    weights_vec = np.maximum(weights_vec, 0)

    scores = score_layouts(weights_vec)
    if TARGET not in scores or len(scores) < len(LAYOUTS):
        return 1e18  # failed

    target_score = scores[TARGET]
    # Objective: minimize the gap between target and the best non-target layout
    other_scores = [s for name, s in scores.items() if name != TARGET]
    best_other = max(other_scores)

    # We want target_score > best_other (target_score is negative, less negative = better)
    # gap > 0 means target is winning
    gap = target_score - best_other

    # Primary: maximize gap (target beats others)
    # Secondary: prefer smaller weights (regularization)
    reg = 0.001 * np.sum(weights_vec ** 2)
    obj = -gap + reg

    if obj < best_obj:
        best_obj = obj
        best_weights = weights_vec.copy()
        w_str = ", ".join(f"{n}={int(round(v))}" for n, v in zip(WEIGHT_NAMES, weights_vec))
        rank = sorted(scores.items(), key=lambda x: -x[1])
        rank_str = " > ".join(f"{n}({s:,.0f})" for n, s in rank)
        print(f"[{eval_count}] gap={gap:,.0f} obj={obj:.1f} | {w_str}", flush=True)
        print(f"       {rank_str}", flush=True)
    elif eval_count % 20 == 0:
        print(f"[{eval_count}] no improvement (gap={gap:,.0f})", flush=True)

    return obj


def main():
    global best_weights

    shutil.copy(CONFIG, CONFIG_BAK)

    print(f"Goal-seeking weights to maximize {TARGET}'s ranking")
    print(f"Layouts: {', '.join(LAYOUTS)}")
    print(f"Initial weights: {dict(zip(WEIGHT_NAMES, INITIAL))}")
    print()

    # First eval with initial weights
    print("=== Initial evaluation ===")
    scores = score_layouts(INITIAL)
    rank = sorted(scores.items(), key=lambda x: -x[1])
    for name, score in rank:
        marker = " <-- TARGET" if name == TARGET else ""
        print(f"  {name:<15} {score:>20,}{marker}")
    print()

    print("=== Optimizing (Nelder-Mead) ===")
    result = minimize(
        objective,
        INITIAL,
        method="Nelder-Mead",
        options={
            "maxiter": 500,
            "xatol": 0.5,
            "fatol": 1e6,
            "adaptive": True,
        },
    )

    print()
    print("=== Result ===")
    final_weights = np.maximum(result.x, 0)
    print(f"Evaluations: {eval_count}")
    print(f"Optimal weights:")
    for name, val in zip(WEIGHT_NAMES, final_weights):
        print(f"  {name} = {int(round(val))}")

    print()
    print("Final ranking:")
    scores = score_layouts(final_weights)
    rank = sorted(scores.items(), key=lambda x: -x[1])
    for name, score in rank:
        marker = " <-- TARGET" if name == TARGET else ""
        print(f"  {name:<15} {score:>20,}{marker}")

    # Restore config
    shutil.copy(CONFIG_BAK, CONFIG)
    os.remove(CONFIG_BAK)

    print()
    print("Config to use:")
    for name, val in zip(WEIGHT_NAMES, final_weights):
        print(f"{name} = {int(round(val))}")


if __name__ == "__main__":
    main()
