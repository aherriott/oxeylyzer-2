#!/usr/bin/env python3
"""
Goal-seek: find weights that produce a magic-key layout that beats sturdy
on every metric by the widest possible margin.

Two-phase approach:
  Phase 1: Find any weight set that produces all-9-wins (already solved)
  Phase 2: Starting from winning weights, maximize the total margin of victory

Fitness (lower = better):
  - If not all 9 wins: large penalty (1000 + sum of miss ratios)
  - If all 9 wins: negative margin score (more negative = bigger margin = better)
    margin = sum of (improvement_ratio) across all 9 metrics

Loop:
  1. Perturb weights from best known
  2. Write analyzer-config.toml
  3. Run `gen my-layout -t <GEN_TIME>`
  4. Parse top layouts, create .dof, analyze each
  5. Pick the layout with best fitness
  6. Log results, update best if improved
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
from dataclasses import dataclass, asdict
from pathlib import Path

BINARY = "./target/release/main"
CONFIG = "./analyzer-config.toml"
CONFIG_BAK = "./analyzer-config.toml.bak"
LAYOUT_DIR = "./layouts"
GEN_LAYOUT = "my-layout"
GEN_TIME = 120  # seconds per gen run
TOP_N = 5       # parse top N from gen
LOG_FILE = "goal_seek_results.csv"
STATE_FILE = "goal_seek_state.json"
BEST_DOF_DIR = "./layouts/goalseed-best"

# Metrics where lower is better
LOWER_IS_BETTER = ["sfbs", "sfs", "stretches", "redirect", "onehandin", "onehandout"]
# Metrics where higher is better
HIGHER_IS_BETTER = ["inroll", "outroll", "alternate"]
ALL_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER

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
    # Seeded from the t12 victory weights
    sfbs: int = 39
    sfs: int = 6
    stretches: int = 23
    sft: int = 5
    inroll: int = 3
    outroll: int = 2
    alternate: int = 0
    redirect: int = 16
    onehandin: int = 1
    onehandout: int = 0
    full_scissors: int = 5
    half_scissors: int = 1
    full_scissors_skip: int = 2
    half_scissors_skip: int = 1
    finger_usage: int = 19
    magic_rule_penalty: int = 1
    magic_repeat_penalty: int = 0


# Weights we optimize
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
    if metrics is None or metrics.sfbs == 0.0:
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

    rows = []
    thumb_key = None
    for line in lines:
        chars = line.strip().split()
        if len(chars) <= 2 and len(chars) >= 1:
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

    main_rows = []
    for row in rows[:3]:
        if len(row) >= 10:
            left = " ".join(row[:5])
            right = " ".join(row[5:10])
            main_rows.append(f"{left}  {right}")
        else:
            main_rows.append(" ".join(row))

    if thumb_key == "␣":
        thumb_key_dof = "spc"
    elif thumb_key:
        thumb_key_dof = thumb_key
    else:
        thumb_key_dof = "spc"
    thumb_row = f"  ~ {thumb_key_dof} ~  ~ ~ ~     "

    magic_rules = {}
    if magic_str:
        for rule in magic_str.split():
            parts = rule.split("→")
            if len(parts) == 2:
                leader = parts[0]
                output_char = parts[1]
                if leader == "␣":
                    leader = " "
                if output_char == "␣":
                    output_char = " "
                magic_rules[leader] = output_char

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

    for line in output.split("\n"):
        if line.strip().startswith("magic:"):
            rules = line.strip().replace("magic:", "").strip().split()
            metrics.magic_rules = len([r for r in rules if "→" in r])
            break

    return metrics


def beats_sturdy(m: LayoutMetrics, sturdy: LayoutMetrics) -> dict:
    """Check which metrics beat sturdy. Returns dict of metric -> (value, sturdy_value, beats)."""
    comparisons = {}
    for metric in LOWER_IS_BETTER:
        val = getattr(m, metric)
        sval = getattr(sturdy, metric)
        comparisons[metric] = (val, sval, val < sval)
    for metric in HIGHER_IS_BETTER:
        val = getattr(m, metric)
        sval = getattr(sturdy, metric)
        comparisons[metric] = (val, sval, val > sval)
    return comparisons



def compute_fitness(m: LayoutMetrics, sturdy: LayoutMetrics) -> float:
    """
    Fitness score (lower = better):
    - If not all 9 wins: 1000 + sum of miss penalties (ensures any all-win beats any non-all-win)
    - If all 9 wins: negative total margin (more negative = bigger margin = better)

    Margin for each metric = improvement ratio:
      lower-is-better:  (sturdy - ours) / sturdy   (positive when we're better)
      higher-is-better: (ours - sturdy) / sturdy    (positive when we're better)
    """
    comps = beats_sturdy(m, sturdy)
    wins = sum(1 for _, _, b in comps.values() if b)
    total = len(comps)

    if wins < total:
        # Phase 1: penalty for missing metrics
        penalty = 1000.0
        for metric, (val, sval, beats) in comps.items():
            if not beats:
                if sval > 0:
                    if metric in LOWER_IS_BETTER:
                        penalty += (val - sval) / sval
                    else:
                        penalty += (sval - val) / sval
                else:
                    penalty += 1.0
        return penalty

    # Phase 2: all 9 wins — maximize margin
    # Return negative margin so lower fitness = bigger margin
    total_margin = 0.0
    for metric, (val, sval, beats) in comps.items():
        if sval > 0:
            if metric in LOWER_IS_BETTER:
                total_margin += (sval - val) / sval  # positive = we're lower = better
            else:
                total_margin += (val - sval) / sval  # positive = we're higher = better
    return -total_margin  # negate so lower = better


def compute_margin_details(m: LayoutMetrics, sturdy: LayoutMetrics) -> list:
    """Return list of (metric, our_val, sturdy_val, margin_pct, beats) for display."""
    details = []
    comps = beats_sturdy(m, sturdy)
    for metric in ALL_METRICS:
        val, sval, beats = comps[metric]
        if sval > 0:
            if metric in LOWER_IS_BETTER:
                margin_pct = (sval - val) / sval * 100  # positive = better
            else:
                margin_pct = (val - sval) / sval * 100  # positive = better
        else:
            margin_pct = 0.0
        details.append((metric, val, sval, margin_pct, beats))
    return details


def perturb_weights(base: WeightSet, temperature: float = 0.3) -> WeightSet:
    """Create a new weight set by perturbing the base."""
    ws = WeightSet(**asdict(base))
    for name in TUNABLE:
        val = getattr(ws, name)
        delta = max(1, int(val * temperature))
        new_val = val + random.randint(-delta, delta)
        new_val = max(0, new_val)
        setattr(ws, name, new_val)
    return ws


def cleanup_dof_files(trial: int):
    """Remove temporary .dof files from a trial."""
    for f in Path(LAYOUT_DIR).glob(f"goalseed-t{trial}-*.dof"):
        f.unlink()


def save_best_layout(trial: int, rank: int, name: str):
    """Copy the best layout .dof to the best directory for safekeeping."""
    os.makedirs(BEST_DOF_DIR, exist_ok=True)
    src = os.path.join(LAYOUT_DIR, f"{name}.dof")
    dst = os.path.join(BEST_DOF_DIR, f"{name}.dof")
    if os.path.exists(src):
        shutil.copy(src, dst)


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
               wins: int, total: int, margin: float):
    """Append a result row to the CSV log."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = (
                ["trial", "fitness", "margin", "wins", "total",
                 "layout_name", "magic_rules"]
                + ALL_METRICS
                + [f"w_{name}" for name in TUNABLE]
            )
            writer.writerow(header)
        row = (
            [trial, f"{fitness:.4f}", f"{margin:.4f}", wins, total,
             metrics.name, metrics.magic_rules]
            + [f"{getattr(metrics, m):.3f}" for m in ALL_METRICS]
            + [getattr(ws, name) for name in TUNABLE]
        )
        writer.writerow(row)



def main():
    global STURDY_METRICS

    shutil.copy(CONFIG, CONFIG_BAK)

    print("=" * 70)
    print("GOAL SEEK: Beat sturdy on all metrics, maximize margin")
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

    # Measure sturdy baseline (use fixed weights for consistent measurement)
    # We need sturdy measured with the SAME weights each time for fair comparison.
    # Use the trial weights — sturdy's stats (sfbs%, inroll%, etc.) are weight-independent.
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
        print(f"Resuming from trial {state['trial']}")
        best_weights = WeightSet(**state["best_weights"])
        best_fitness = state["best_fitness"]
        best_wins = state["best_wins"]
        best_margin = state.get("best_margin", 0.0)
        start_trial = state["trial"] + 1
        temperature = state.get("temperature", 0.3)
        best_layout_name = state.get("best_layout_name", "")
        print(f"  Best: {best_wins}/9 wins, fitness={best_fitness:.4f}, margin={best_margin:.2f}%")
    else:
        best_weights = WeightSet()  # seeded from t12 victory
        best_fitness = float("inf")
        best_wins = 0
        best_margin = 0.0
        start_trial = 1
        temperature = 0.3  # start smaller since we're near a good region
        best_layout_name = ""

    print(f"Starting temperature: {temperature:.2f}")
    print()

    try:
        for trial in range(start_trial, 10000):
            phase = "MAXIMIZE" if best_wins == 9 else "FIND WINS"
            print(f"{'=' * 70}")
            print(f"TRIAL {trial} [{phase}] | best: {best_wins}/9, "
                  f"margin={best_margin:.1f}%, fitness={best_fitness:.4f}")
            print(f"{'=' * 70}")

            # Generate new weights
            if trial == start_trial and not state:
                ws = best_weights
            else:
                ws = perturb_weights(best_weights, temperature)

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

            # Analyze each layout, pick the best
            trial_best_fitness = float("inf")
            trial_best_metrics = None
            trial_best_name = None
            trial_best_margin = 0.0

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

                # Compute margin details
                details = compute_margin_details(metrics, STURDY_METRICS)
                total_margin_pct = sum(d[3] for d in details)

                print(f"\n  #{rank} ({score_str}): {wins}/{total} wins, "
                      f"margin={total_margin_pct:+.1f}%, fitness={fitness:.4f}")
                for metric, val, sval, margin_pct, beats in details:
                    marker = "✓" if beats else "✗"
                    print(f"    {marker} {metric:12s}: {val:7.3f} vs {sval:7.3f}  "
                          f"({margin_pct:+.1f}%)")

                log_result(trial, ws, metrics, fitness, wins, total, total_margin_pct)

                if fitness < trial_best_fitness:
                    trial_best_fitness = fitness
                    trial_best_metrics = metrics
                    trial_best_name = dof_name
                    trial_best_margin = total_margin_pct

            # Update best if improved
            if trial_best_fitness < best_fitness:
                best_fitness = trial_best_fitness
                best_weights = ws
                comps = beats_sturdy(trial_best_metrics, STURDY_METRICS)
                best_wins = sum(1 for _, _, b in comps.values() if b)
                best_margin = trial_best_margin
                best_layout_name = trial_best_name

                print(f"\n  ★ New best! {best_wins}/9 wins, margin={best_margin:+.1f}%, "
                      f"fitness={best_fitness:.4f}")
                print(f"    Layout: {trial_best_name}")

                # Save the best layout permanently
                save_best_layout(trial, 0, trial_best_name)

                temperature = max(0.1, temperature * 0.95)
            else:
                print(f"\n  No improvement (fitness={trial_best_fitness:.4f} vs "
                      f"best={best_fitness:.4f})")
                temperature = min(0.6, temperature * 1.02)

            # Clean up trial .dof files
            cleanup_dof_files(trial)

            # Save state
            save_state({
                "trial": trial,
                "best_fitness": best_fitness,
                "best_wins": best_wins,
                "best_margin": best_margin,
                "best_weights": asdict(best_weights),
                "temperature": temperature,
                "best_layout_name": best_layout_name,
            })

            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        if os.path.exists(CONFIG_BAK):
            shutil.copy(CONFIG_BAK, CONFIG)
            os.remove(CONFIG_BAK)
            print("Config restored.")

        print(f"\n{'=' * 70}")
        print(f"FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Best: {best_wins}/9 wins, margin={best_margin:+.1f}%")
        print(f"Best layout: {best_layout_name}")
        print(f"Best weights: {', '.join(f'{n}={getattr(best_weights, n)}' for n in TUNABLE)}")
        print(f"Results logged to {LOG_FILE}")
        if best_layout_name:
            best_path = os.path.join(BEST_DOF_DIR, f"{best_layout_name}.dof")
            if os.path.exists(best_path):
                print(f"Best layout saved to: {best_path}")


if __name__ == "__main__":
    main()
