#!/usr/bin/env python3
"""
Goal-seek: find weights that produce a magic-key layout beating sturdy on every metric.

Loop:
  1. Pick a weight vector (Bayesian-ish: random perturbation of best known)
  2. Write analyzer-config.toml
  3. Run `gen my-layout -t 120` (2 min per trial)
  4. Parse top 3 layouts from gen output
  5. Create temp .dof files, analyze each
  6. Score against sturdy on all metrics
  7. Log results, keep best

Sturdy targets (from analyzer):
  SFBs:      0.584%
  SFS:       3.702%
  Stretches: 2.705
  Inroll:    24.330%
  Outroll:   25.958%
  Alternate: 24.268%
  Redirect:   7.462%
"""

import subprocess
import re
import os
import sys
import json
import shutil
import time
import random
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path

BINARY = "./target/release/main"
CONFIG = "./analyzer-config.toml"
CONFIG_BAK = "./analyzer-config.toml.bak"
LAYOUT_DIR = "./layouts"
GEN_LAYOUT = "my-layout"
GEN_TIME = 120  # seconds per gen run
TOP_N = 3       # parse top N from gen
LOG_FILE = "goal_seek_results.csv"
STATE_FILE = "goal_seek_state.json"

# Sturdy reference metrics (will be measured at startup)
STURDY_METRICS = None


@dataclass
class LayoutMetrics:
    name: str = ""
    score: int = 0
    sfbs: float = 0.0
    sfs: float = 0.0
    stretches: float = 0.0
    inroll: float = 0.0
    outroll: float = 0.0
    alternate: float = 0.0
    redirect: float = 0.0
    onehandin: float = 0.0
    onehandout: float = 0.0
    magic_rules: int = 0


@dataclass
class WeightSet:
    sfbs: int = 25
    sfs: int = 3
    stretches: int = 19
    sft: int = 5
    inroll: int = 3
    outroll: int = 3
    alternate: int = 3
    redirect: int = 35
    onehandin: int = 2
    onehandout: int = 0
    full_scissors: int = 5
    half_scissors: int = 1
    full_scissors_skip: int = 2
    half_scissors_skip: int = 1
    finger_usage: int = 35
    magic_rule_penalty: int = 2
    magic_repeat_penalty: int = 0


# Weights we optimize (the rest stay fixed)
TUNABLE = [
    "sfbs", "sfs", "stretches", "inroll", "outroll",
    "alternate", "redirect", "onehandin", "finger_usage",
    "magic_rule_penalty",
]

FINGER_WEIGHTS = {
    "lp": 77, "lr": 32, "lm": 24, "li": 21, "lt": 46,
    "rt": 46, "ri": 21, "rm": 24, "rr": 32, "rp": 77,
}


def write_config(ws: WeightSet):
    """Write analyzer-config.toml from a WeightSet."""
    lines = [
        'corpus = "./data/english.json"',
        'layouts = ["./core/test-layouts", "./layouts"]',
        "",
        "[weights]",
    ]
    for k, v in asdict(ws).items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[weights.fingers]")
    for k, v in FINGER_WEIGHTS.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    with open(CONFIG, "w") as f:
        f.write("\n".join(lines))


def run_repl(commands: str, timeout: int = 300) -> str:
    """Run REPL with given commands, return stdout."""
    result = subprocess.run(
        [BINARY],
        input=commands,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def measure_sturdy() -> LayoutMetrics:
    """Analyze sturdy and return its metrics."""
    output = run_repl("a sturdy\nq\n", timeout=120)
    metrics = parse_analyze_output(output, "sturdy")
    if metrics is None:
        print("ERROR: Could not parse sturdy metrics!")
        sys.exit(1)
    return metrics


def parse_analyze_output(output: str, name: str) -> LayoutMetrics:
    """Parse the analyze command output into LayoutMetrics."""
    m = LayoutMetrics(name=name)

    for line in output.split("\n"):
        line = line.strip()

        score_match = re.match(r"score:\s+(-?[\d,]+)", line)
        if score_match:
            m.score = int(score_match.group(1).replace(",", ""))

        sfbs_match = re.match(r"sfbs:\s+([\d.]+)%", line)
        if sfbs_match:
            m.sfbs = float(sfbs_match.group(1))

        sfs_match = re.match(r"sfs:\s+([\d.]+)%", line)
        if sfs_match:
            m.sfs = float(sfs_match.group(1))

        stretch_match = re.match(r"stretches:\s+([\d.]+)", line)
        if stretch_match:
            m.stretches = float(stretch_match.group(1))

        inroll_match = re.match(r"Inroll:\s+([\d.]+)%", line)
        if inroll_match:
            m.inroll = float(inroll_match.group(1))

        outroll_match = re.match(r"Outroll:\s+([\d.]+)%", line)
        if outroll_match:
            m.outroll = float(outroll_match.group(1))

        alt_match = re.match(r"Alternate:\s+([\d.]+)%", line)
        if alt_match:
            m.alternate = float(alt_match.group(1))

        redir_match = re.match(r"Redirect:\s+([\d.]+)%", line)
        if redir_match:
            m.redirect = float(redir_match.group(1))

        ohin_match = re.match(r"Onehand In:\s+([\d.]+)%", line)
        if ohin_match:
            m.onehandin = float(ohin_match.group(1))

        ohout_match = re.match(r"Onehand Out:\s+([\d.]+)%", line)
        if ohout_match:
            m.onehandout = float(ohout_match.group(1))

    return m



def parse_gen_output(output: str) -> list:
    """Parse gen output into list of (rank, score_str, layout_lines, magic_line)."""
    layouts = []
    current_rank = None
    current_score = None
    current_lines = []
    current_magic = ""

    for line in output.split("\n"):
        rank_match = re.match(r"#(\d+), score: (.+)", line)
        if rank_match:
            if current_rank is not None:
                layouts.append((current_rank, current_score, current_lines, current_magic))
            current_rank = int(rank_match.group(1))
            current_score = rank_match.group(2).strip()
            current_lines = []
            current_magic = ""
            continue

        if current_rank is not None:
            magic_match = re.match(r"magic:\s+(.+)", line)
            if magic_match:
                current_magic = magic_match.group(1).strip()
            elif line.strip() and not line.startswith("gen ") and not line.startswith("  "):
                current_lines.append(line)

    if current_rank is not None:
        layouts.append((current_rank, current_score, current_lines, current_magic))

    return layouts[:TOP_N]


def gen_layout_to_dof(rank: int, lines: list, magic_str: str, trial: int) -> str:
    """Convert parsed gen output to a .dof file. Returns the layout name."""
    name = f"goalseed-t{trial}-r{rank}"
    dof_path = os.path.join(LAYOUT_DIR, f"{name}.dof")

    # Parse the key rows from gen output
    # Gen output has rows like: "v r c k z . u o y f"
    # We need to map back to .dof format with &mag and spc
    rows = []
    thumb_key = None
    for line in lines:
        chars = line.strip().split()
        if len(chars) <= 2 and len(chars) >= 1:
            # This is the thumb key line (single char like "␣" or "e")
            thumb_key = chars[0]
            continue
        row_chars = []
        for c in chars:
            if c == "◊":
                row_chars.append("&mag")
            elif c == "␣":
                row_chars.append("spc")
            else:
                row_chars.append(c)
        rows.append(row_chars)

    if len(rows) < 3:
        return None

    # Build the 3 main rows (10 chars each, space in middle)
    main_rows = []
    for row in rows[:3]:
        if len(row) >= 10:
            left = " ".join(row[:5])
            right = " ".join(row[5:10])
            main_rows.append(f"{left}  {right}")
        else:
            main_rows.append(" ".join(row))

    # Thumb row: the single key goes in a specific position
    # Based on my-layout.dof pattern: " ~ <key> ~  ~ ~ ~ "
    if thumb_key == "␣":
        thumb_key_dof = "spc"
    elif thumb_key:
        thumb_key_dof = thumb_key
    else:
        thumb_key_dof = "spc"
    thumb_row = f"  ~ {thumb_key_dof} ~  ~ ~ ~     "

    # Parse magic rules from "a→c b→s ..." format
    magic_rules = {}
    if magic_str:
        for rule in magic_str.split():
            parts = rule.split("→")
            if len(parts) == 2:
                leader = parts[0]
                output = parts[1]
                # Map special chars
                if leader == "␣":
                    leader = " "
                if output == "␣":
                    output = " "
                magic_rules[leader] = output

    dof = {
        "name": name,
        "board": "colstag",
        "layers": {
            "main": main_rows + [thumb_row]
        },
        "fingering": "traditional",
        "magic": {
            "mag": magic_rules
        }
    }

    with open(dof_path, "w") as f:
        json.dump(dof, f, indent=4)

    return name


def analyze_layout(name: str) -> LayoutMetrics:
    """Run analyze on a layout and return metrics."""
    output = run_repl(f"a {name}\nq\n", timeout=120)
    metrics = parse_analyze_output(output, name)

    # Count magic rules from the output
    for line in output.split("\n"):
        if line.strip().startswith("magic:"):
            rules = line.strip().replace("magic:", "").strip().split()
            metrics.magic_rules = len([r for r in rules if "→" in r])
            break

    return metrics


def beats_sturdy(m: LayoutMetrics, sturdy: LayoutMetrics) -> dict:
    """Check which metrics beat sturdy. Returns dict of metric -> (value, sturdy_value, beats)."""
    comparisons = {}
    # Lower is better
    for metric in ["sfbs", "sfs", "stretches", "redirect", "onehandin", "onehandout"]:
        val = getattr(m, metric)
        sval = getattr(sturdy, metric)
        comparisons[metric] = (val, sval, val < sval)
    # Higher is better
    for metric in ["inroll", "outroll", "alternate"]:
        val = getattr(m, metric)
        sval = getattr(sturdy, metric)
        comparisons[metric] = (val, sval, val > sval)
    return comparisons


def compute_fitness(m: LayoutMetrics, sturdy: LayoutMetrics) -> float:
    """
    Fitness score: how close to beating sturdy on all metrics.
    0 = beats sturdy on everything. Positive = worse.
    Each metric contributes a penalty if it doesn't beat sturdy.
    """
    penalty = 0.0
    comps = beats_sturdy(m, sturdy)
    for metric, (val, sval, beats) in comps.items():
        if not beats:
            if metric in ["sfbs", "sfs", "stretches", "redirect", "onehandin", "onehandout"]:
                # Lower is better — penalty = how much worse we are (as ratio)
                if sval > 0:
                    penalty += (val - sval) / sval
                else:
                    penalty += val
            else:
                # Higher is better — penalty = how much worse we are (as ratio)
                if sval > 0:
                    penalty += (sval - val) / sval
                else:
                    penalty += 1.0
    return penalty


def perturb_weights(base: WeightSet, temperature: float = 0.3) -> WeightSet:
    """Create a new weight set by perturbing the base."""
    ws = WeightSet(**asdict(base))
    for name in TUNABLE:
        val = getattr(ws, name)
        # Perturbation proportional to current value
        delta = max(1, int(val * temperature))
        new_val = val + random.randint(-delta, delta)
        new_val = max(0, new_val)
        setattr(ws, name, new_val)
    return ws


def cleanup_dof_files(trial: int):
    """Remove temporary .dof files from a trial."""
    for f in Path(LAYOUT_DIR).glob(f"goalseed-t{trial}-*.dof"):
        f.unlink()


def format_comparison(m: LayoutMetrics, sturdy: LayoutMetrics) -> str:
    """Format a comparison table row."""
    comps = beats_sturdy(m, sturdy)
    wins = sum(1 for _, _, b in comps.values() if b)
    total = len(comps)

    parts = [f"  {m.name}: {wins}/{total} metrics beat sturdy"]
    for metric, (val, sval, beats) in comps.items():
        marker = "✓" if beats else "✗"
        parts.append(f"    {marker} {metric}: {val:.3f} vs {sval:.3f}")
    return "\n".join(parts)



def save_state(state: dict):
    """Save current optimization state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    """Load optimization state from disk, or return default."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def log_result(trial: int, ws: WeightSet, metrics: LayoutMetrics, fitness: float,
               wins: int, total: int):
    """Append a result row to the CSV log."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = (
                ["trial", "fitness", "wins", "total", "layout_name", "magic_rules"]
                + ["sfbs", "sfs", "stretches", "inroll", "outroll", "alternate",
                   "redirect", "onehandin", "onehandout"]
                + [f"w_{name}" for name in TUNABLE]
            )
            writer.writerow(header)
        row = (
            [trial, f"{fitness:.4f}", wins, total, metrics.name, metrics.magic_rules]
            + [f"{metrics.sfbs:.3f}", f"{metrics.sfs:.3f}", f"{metrics.stretches:.3f}",
               f"{metrics.inroll:.3f}", f"{metrics.outroll:.3f}", f"{metrics.alternate:.3f}",
               f"{metrics.redirect:.3f}", f"{metrics.onehandin:.3f}", f"{metrics.onehandout:.3f}"]
            + [getattr(ws, name) for name in TUNABLE]
        )
        writer.writerow(row)


def main():
    global STURDY_METRICS

    # Backup config
    shutil.copy(CONFIG, CONFIG_BAK)

    print("=" * 70)
    print("GOAL SEEK: Find magic layout that beats sturdy on every metric")
    print("=" * 70)
    print(f"Gen time per trial: {GEN_TIME}s")
    print(f"Top layouts per trial: {TOP_N}")
    print()

    # Build release binary
    print("Building release binary...")
    result = subprocess.run(
        ["cargo", "build", "-p", "oxeylyzer-repl", "--release"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return
    print("Build complete.")
    print()

    # Measure sturdy baseline
    print("Measuring sturdy baseline...")
    STURDY_METRICS = measure_sturdy()
    print(f"  SFBs:      {STURDY_METRICS.sfbs:.3f}%")
    print(f"  SFS:       {STURDY_METRICS.sfs:.3f}%")
    print(f"  Stretches: {STURDY_METRICS.stretches:.3f}")
    print(f"  Inroll:    {STURDY_METRICS.inroll:.3f}%")
    print(f"  Outroll:   {STURDY_METRICS.outroll:.3f}%")
    print(f"  Alternate: {STURDY_METRICS.alternate:.3f}%")
    print(f"  Redirect:  {STURDY_METRICS.redirect:.3f}%")
    print(f"  OhIn:      {STURDY_METRICS.onehandin:.3f}%")
    print(f"  OhOut:     {STURDY_METRICS.onehandout:.3f}%")
    print()

    # Load or initialize state
    state = load_state()
    if state:
        print(f"Resuming from trial {state['trial']} (best fitness: {state['best_fitness']:.4f})")
        best_weights = WeightSet(**state["best_weights"])
        best_fitness = state["best_fitness"]
        best_wins = state["best_wins"]
        start_trial = state["trial"] + 1
        temperature = state.get("temperature", 0.3)
    else:
        best_weights = WeightSet()  # defaults
        best_fitness = float("inf")
        best_wins = 0
        start_trial = 1
        temperature = 0.5

    print(f"Starting temperature: {temperature:.2f}")
    print()

    try:
        for trial in range(start_trial, 10000):
            print(f"{'=' * 70}")
            print(f"TRIAL {trial} | best so far: {best_wins}/9 wins, fitness={best_fitness:.4f}")
            print(f"{'=' * 70}")

            # Generate new weights
            if trial == 1:
                ws = best_weights
            else:
                ws = perturb_weights(best_weights, temperature)

            # Print weights being tested
            print(f"Weights: {', '.join(f'{n}={getattr(ws, n)}' for n in TUNABLE)}")

            # Write config and run gen
            write_config(ws)
            print(f"Running gen my-layout -t {GEN_TIME}...")
            gen_start = time.time()

            try:
                gen_output = run_repl(
                    f"gen {GEN_LAYOUT} -t {GEN_TIME}\nq\n",
                    timeout=GEN_TIME + 60,
                )
            except subprocess.TimeoutExpired:
                print("  gen timed out, skipping trial")
                cleanup_dof_files(trial)
                continue

            gen_elapsed = time.time() - gen_start
            print(f"  gen completed in {gen_elapsed:.0f}s")

            # Parse gen output
            parsed = parse_gen_output(gen_output)
            if not parsed:
                print("  No layouts parsed from gen output, skipping")
                cleanup_dof_files(trial)
                continue

            print(f"  Parsed {len(parsed)} layouts")

            # Create .dof files and analyze each
            trial_best_fitness = float("inf")
            trial_best_metrics = None
            trial_best_name = None

            for rank, score_str, lines, magic_str in parsed:
                dof_name = gen_layout_to_dof(rank, lines, magic_str, trial)
                if dof_name is None:
                    print(f"  Failed to create .dof for rank {rank}")
                    continue

                metrics = analyze_layout(dof_name)
                fitness = compute_fitness(metrics, STURDY_METRICS)
                comps = beats_sturdy(metrics, STURDY_METRICS)
                wins = sum(1 for _, _, b in comps.values() if b)
                total = len(comps)

                print(f"\n  #{rank} ({score_str}): {wins}/{total} wins, fitness={fitness:.4f}")
                for metric, (val, sval, beats) in comps.items():
                    marker = "✓" if beats else "✗"
                    print(f"    {marker} {metric}: {val:.3f} vs {sval:.3f}")

                log_result(trial, ws, metrics, fitness, wins, total)

                if fitness < trial_best_fitness:
                    trial_best_fitness = fitness
                    trial_best_metrics = metrics
                    trial_best_name = dof_name

                # Check for victory!
                if wins == total:
                    print()
                    print("!" * 70)
                    print(f"  VICTORY! {dof_name} beats sturdy on ALL {total} metrics!")
                    print("!" * 70)
                    print(f"\n  Weights: {', '.join(f'{n}={getattr(ws, n)}' for n in TUNABLE)}")
                    print(f"\n  Layout file: {LAYOUT_DIR}/{dof_name}.dof")
                    # Restore config but keep the winning .dof
                    shutil.copy(CONFIG_BAK, CONFIG)
                    os.remove(CONFIG_BAK)
                    return

            # Update best if this trial improved
            if trial_best_fitness < best_fitness:
                best_fitness = trial_best_fitness
                best_weights = ws
                comps = beats_sturdy(trial_best_metrics, STURDY_METRICS)
                best_wins = sum(1 for _, _, b in comps.values() if b)
                print(f"\n  ★ New best! {best_wins}/9 wins, fitness={best_fitness:.4f}")
                print(f"    Layout: {trial_best_name}")
                # Keep the best .dof, clean others
                temperature = max(0.1, temperature * 0.95)  # cool down on improvement
            else:
                print(f"\n  No improvement (trial fitness={trial_best_fitness:.4f})")
                temperature = min(0.8, temperature * 1.02)  # heat up on stagnation

            # Clean up trial .dof files (except if it's the best ever)
            cleanup_dof_files(trial)

            # Save state
            save_state({
                "trial": trial,
                "best_fitness": best_fitness,
                "best_wins": best_wins,
                "best_weights": asdict(best_weights),
                "temperature": temperature,
            })

            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        # Restore config
        if os.path.exists(CONFIG_BAK):
            shutil.copy(CONFIG_BAK, CONFIG)
            os.remove(CONFIG_BAK)
            print("Config restored.")

        print(f"\nBest result: {best_wins}/9 wins, fitness={best_fitness:.4f}")
        print(f"Best weights: {', '.join(f'{n}={getattr(best_weights, n)}' for n in TUNABLE)}")
        print(f"Results logged to {LOG_FILE}")


if __name__ == "__main__":
    main()
