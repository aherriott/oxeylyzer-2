#!/usr/bin/env python3
"""
Goal-seek: find weights that produce a magic-key layout that beats
sturdy, graphite, AND canary on every metric, then maximize the margin.

Tiered fitness (lower = better):
  - Tier 0: Beats all 3 targets on all metrics → negative margin (maximize)
  - Tier 1: Beats sturdy + graphite on all metrics → 1000 + canary miss penalty
  - Tier 2: Beats sturdy on all metrics → 2000 + graphite miss penalty
  - Tier 3: Doesn't beat sturdy on all → 3000 + sturdy miss penalty

Within each tier, lower fitness = better.
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

LOWER_IS_BETTER = ["sfbs", "sfs", "stretches", "redirect", "onehandin", "onehandout"]
HIGHER_IS_BETTER = ["inroll", "outroll", "alternate"]
ALL_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER

# Target layouts in priority order
TARGETS = ["sturdy", "graphite", "canary"]


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
    # Seeded from t12 victory weights (beat sturdy on all 9)
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
    result = subprocess.run(
        [BINARY], input=commands,
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "RAYON_NUM_THREADS": "3"},
    )
    return result.stdout + result.stderr


def parse_analyze_output(output: str, name: str) -> LayoutMetrics:
    m = LayoutMetrics(name=name)
    for line in output.split("\n"):
        line = line.strip()
        if mat := re.match(r"score:\s+(-?[\d,]+)", line):
            m.score = int(mat.group(1).replace(",", ""))
        elif mat := re.match(r"sfbs:\s+([\d.]+)%", line):
            m.sfbs = float(mat.group(1))
        elif mat := re.match(r"sfs:\s+([\d.]+)%", line):
            m.sfs = float(mat.group(1))
        elif mat := re.match(r"stretches:\s+([\d.]+)", line):
            m.stretches = float(mat.group(1))
        elif mat := re.match(r"Inroll:\s+([\d.]+)%", line):
            m.inroll = float(mat.group(1))
        elif mat := re.match(r"Outroll:\s+([\d.]+)%", line):
            m.outroll = float(mat.group(1))
        elif mat := re.match(r"Alternate:\s+([\d.]+)%", line):
            m.alternate = float(mat.group(1))
        elif mat := re.match(r"Redirect:\s+([\d.]+)%", line):
            m.redirect = float(mat.group(1))
        elif mat := re.match(r"Onehand In:\s+([\d.]+)%", line):
            m.onehandin = float(mat.group(1))
        elif mat := re.match(r"Onehand Out:\s+([\d.]+)%", line):
            m.onehandout = float(mat.group(1))
    return m


def measure_layout(name: str) -> LayoutMetrics:
    output = run_repl(f"a {name}\nq\n", timeout=120)
    return parse_analyze_output(output, name)


def parse_gen_output(output: str) -> list:
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
    name = f"goalseed-t{trial}-r{rank}"
    dof_path = os.path.join(LAYOUT_DIR, f"{name}.dof")

    rows = []
    thumb_key = None
    for line in lines:
        chars = line.strip().split()
        if len(chars) <= 2:
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
            main_rows.append(f"{' '.join(row[:5])}  {' '.join(row[5:10])}")
        else:
            main_rows.append(" ".join(row))

    thumb_key_dof = "spc" if (thumb_key == "␣" or not thumb_key) else thumb_key
    thumb_row = f"  ~ {thumb_key_dof} ~  ~ ~ ~     "

    magic_rules = {}
    if magic_str:
        for rule in magic_str.split():
            parts = rule.split("→")
            if len(parts) == 2:
                leader = " " if parts[0] == "␣" else parts[0]
                output_char = " " if parts[1] == "␣" else parts[1]
                magic_rules[leader] = output_char

    dof = {
        "name": name, "board": "colstag",
        "layers": {"main": main_rows + [thumb_row]},
        "fingering": "traditional",
        "magic": {"mag": magic_rules}
    }
    with open(dof_path, "w") as f:
        json.dump(dof, f, indent=4)
    return name


def analyze_layout(name: str) -> LayoutMetrics:
    output = run_repl(f"a {name}\nq\n", timeout=120)
    metrics = parse_analyze_output(output, name)
    for line in output.split("\n"):
        if line.strip().startswith("magic:"):
            rules = line.strip().replace("magic:", "").strip().split()
            metrics.magic_rules = len([r for r in rules if "→" in r])
            break
    return metrics



def beats_target(m: LayoutMetrics, target: LayoutMetrics) -> dict:
    """Returns dict of metric -> (our_val, target_val, beats_bool)."""
    comps = {}
    for metric in LOWER_IS_BETTER:
        val, tval = getattr(m, metric), getattr(target, metric)
        comps[metric] = (val, tval, val < tval)
    for metric in HIGHER_IS_BETTER:
        val, tval = getattr(m, metric), getattr(target, metric)
        comps[metric] = (val, tval, val > tval)
    return comps


def miss_penalty(m: LayoutMetrics, target: LayoutMetrics) -> float:
    """Sum of ratio penalties for metrics that don't beat the target."""
    penalty = 0.0
    comps = beats_target(m, target)
    for metric, (val, tval, beats) in comps.items():
        if not beats and tval > 0:
            if metric in LOWER_IS_BETTER:
                penalty += (val - tval) / tval
            else:
                penalty += (tval - val) / tval
    return penalty


def total_margin(m: LayoutMetrics, target: LayoutMetrics) -> float:
    """Sum of improvement ratios across all metrics (positive = better than target)."""
    margin = 0.0
    comps = beats_target(m, target)
    for metric, (val, tval, _) in comps.items():
        if tval > 0:
            if metric in LOWER_IS_BETTER:
                margin += (tval - val) / tval
            else:
                margin += (val - tval) / tval
    return margin


def win_count(m: LayoutMetrics, target: LayoutMetrics) -> tuple:
    comps = beats_target(m, target)
    wins = sum(1 for _, _, b in comps.values() if b)
    return wins, len(comps)


def compute_fitness(m: LayoutMetrics, targets: dict) -> float:
    """
    Tiered fitness (lower = better):
      Tier 0: beats all 3 → negative combined margin
      Tier 1: beats sturdy+graphite → 1000 + canary miss
      Tier 2: beats sturdy → 2000 + graphite miss
      Tier 3: doesn't beat sturdy → 3000 + sturdy miss
    """
    sturdy = targets["sturdy"]
    graphite = targets["graphite"]
    canary = targets["canary"]

    s_wins, s_total = win_count(m, sturdy)
    g_wins, g_total = win_count(m, graphite)
    c_wins, c_total = win_count(m, canary)

    beats_sturdy_all = (s_wins == s_total)
    beats_graphite_all = (g_wins == g_total)
    beats_canary_all = (c_wins == c_total)

    if beats_sturdy_all and beats_graphite_all and beats_canary_all:
        # Tier 0: maximize combined margin over all 3
        combined = total_margin(m, sturdy) + total_margin(m, graphite) + total_margin(m, canary)
        return -combined  # negative so lower = bigger margin

    if beats_sturdy_all and beats_graphite_all:
        # Tier 1: beat canary next
        return 1000.0 + miss_penalty(m, canary)

    if beats_sturdy_all:
        # Tier 2: beat graphite next
        return 2000.0 + miss_penalty(m, graphite)

    # Tier 3: beat sturdy first
    return 3000.0 + miss_penalty(m, sturdy)


def perturb_weights(base: WeightSet, temperature: float = 0.3) -> WeightSet:
    ws = WeightSet(**asdict(base))
    for name in TUNABLE:
        val = getattr(ws, name)
        delta = max(1, int(val * temperature))
        new_val = max(0, val + random.randint(-delta, delta))
        setattr(ws, name, new_val)
    return ws


def cleanup_dof_files(trial: int, keep: set = None):
    """Remove temporary .dof files from a trial, except those in keep set."""
    if keep is None:
        keep = set()
    for f in Path(LAYOUT_DIR).glob(f"goalseed-t{trial}-*.dof"):
        if f.name not in keep:
            f.unlink()


def save_best_layout(name: str):
    os.makedirs(BEST_DOF_DIR, exist_ok=True)
    src = os.path.join(LAYOUT_DIR, f"{name}.dof")
    dst = os.path.join(BEST_DOF_DIR, f"{name}.dof")
    if os.path.exists(src):
        shutil.copy(src, dst)


def next_oxey_version() -> int:
    """Find the next available oxey-vN.dof version number."""
    existing = list(Path(LAYOUT_DIR).glob("oxey-v*.dof"))
    max_v = 0
    for f in existing:
        m = re.match(r"oxey-v(\d+)\.dof", f.name)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def print_layout_dof(dof_path: str):
    """Print a .dof file's layout rows and magic rules."""
    if not os.path.exists(dof_path):
        print(f"    (file not found: {dof_path})")
        return
    with open(dof_path) as f:
        dof = json.load(f)
    for row in dof.get("layers", {}).get("main", []):
        print(f"    {row}")
    rules = dof.get("magic", {}).get("mag", {})
    if rules:
        rule_strs = [f"{k}→{v}" for k, v in sorted(rules.items())]
        print(f"    magic: {' '.join(rule_strs)}")


class TopN:
    """Track the top N results across all trials."""

    def __init__(self, n: int = 3):
        self.n = n
        self.entries = []  # list of (fitness, name, metrics, weights_dict)

    def maybe_add(self, fitness: float, name: str, metrics: LayoutMetrics, ws: WeightSet):
        """Add if it's in the top N. Returns True if added."""
        entry = (fitness, name, metrics, asdict(ws))
        # Don't add duplicates by name
        if any(e[1] == name for e in self.entries):
            return False
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e[0])
        if len(self.entries) > self.n:
            # Remove the worst and delete its .dof from goalseed-best
            removed = self.entries.pop()
            removed_dof = os.path.join(BEST_DOF_DIR, f"{removed[1]}.dof")
            if os.path.exists(removed_dof):
                os.remove(removed_dof)
            return removed[1] != name  # True if we kept the new one
        return True

    def best_fitness(self) -> float:
        if self.entries:
            return self.entries[0][0]
        return float("inf")

    def to_json(self) -> list:
        return [
            {"fitness": f, "name": n, "weights": w}
            for f, n, _, w in self.entries
        ]

    def from_json(self, data: list):
        # Restore from saved state (metrics not saved, will be None)
        self.entries = [
            (e["fitness"], e["name"], None, e["weights"])
            for e in data
        ]


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def log_result(trial: int, ws: WeightSet, metrics: LayoutMetrics, fitness: float,
               s_wins: int, g_wins: int, c_wins: int):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["trial", "fitness", "s_wins", "g_wins", "c_wins",
                 "layout_name", "magic_rules"]
                + ALL_METRICS
                + [f"w_{n}" for n in TUNABLE]
            )
        writer.writerow(
            [trial, f"{fitness:.4f}", s_wins, g_wins, c_wins,
             metrics.name, metrics.magic_rules]
            + [f"{getattr(metrics, m):.3f}" for m in ALL_METRICS]
            + [getattr(ws, n) for n in TUNABLE]
        )


def tier_label(fitness: float) -> str:
    if fitness < 0:
        return "ALL 3 ✓✓✓"
    if fitness < 1000:
        return "ALL 3 ✓✓✓"
    if fitness < 2000:
        return "sturdy✓ graphite✓ canary✗"
    if fitness < 3000:
        return "sturdy✓ graphite✗"
    return "sturdy✗"


def print_vs_target(m: LayoutMetrics, target: LayoutMetrics, target_name: str):
    comps = beats_target(m, target)
    wins = sum(1 for _, _, b in comps.values() if b)
    margin = total_margin(m, target)
    print(f"    vs {target_name}: {wins}/9 wins, margin={margin:+.1f}")
    for metric in ALL_METRICS:
        val, tval, beats = comps[metric]
        marker = "✓" if beats else "✗"
        if tval > 0:
            if metric in LOWER_IS_BETTER:
                pct = (tval - val) / tval * 100
            else:
                pct = (val - tval) / tval * 100
        else:
            pct = 0.0
        print(f"      {marker} {metric:12s}: {val:7.3f} vs {tval:7.3f}  ({pct:+.1f}%)")



def main():
    shutil.copy(CONFIG, CONFIG_BAK)

    print("=" * 70)
    print("GOAL SEEK: Beat sturdy → graphite → canary, maximize margin")
    print("=" * 70)
    print(f"Gen time per trial: {GEN_TIME}s, top {TOP_N} per trial")
    print()

    # Build
    print("Building release binary...")
    result = subprocess.run(
        ["cargo", "build", "-p", "oxeylyzer-repl", "--release"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return
    print("Build complete.\n")

    # Measure targets
    print("Measuring target layouts...")
    targets = {}
    for name in TARGETS:
        targets[name] = measure_layout(name)
        m = targets[name]
        print(f"  {name:12s}: sfbs={m.sfbs:.3f}% sfs={m.sfs:.3f}% stretch={m.stretches:.3f} "
              f"inroll={m.inroll:.1f}% outroll={m.outroll:.1f}% alt={m.alternate:.1f}% "
              f"redir={m.redirect:.1f}%")
    print()

    # Load or init state
    state = load_state()
    top3 = TopN(3)
    if state:
        best_weights = WeightSet(**state["best_weights"])
        best_fitness = state["best_fitness"]
        start_trial = state["trial"] + 1
        temperature = state.get("temperature", 0.1)
        best_layout_name = state.get("best_layout_name", "")
        if "top3" in state:
            top3.from_json(state["top3"])
        print(f"Resuming from trial {start_trial}, best fitness={best_fitness:.4f} "
              f"[{tier_label(best_fitness)}]")
    else:
        best_weights = WeightSet()
        best_fitness = float("inf")
        start_trial = 1
        temperature = 0.1
        best_layout_name = ""

    print(f"Temperature: {temperature:.3f}\n")

    try:
        for trial in range(start_trial, 10000):
            phase = tier_label(best_fitness)
            print(f"{'=' * 70}")
            print(f"TRIAL {trial} [{phase}] | fitness={best_fitness:.4f}, temp={temperature:.3f}")
            print(f"{'=' * 70}")

            if trial == start_trial and not state:
                ws = best_weights
            else:
                ws = perturb_weights(best_weights, temperature)

            print(f"Weights: {', '.join(f'{n}={getattr(ws, n)}' for n in TUNABLE)}")

            write_config(ws)
            print(f"Running gen my-layout -t {GEN_TIME}...")
            gen_start = time.time()

            try:
                gen_output = run_repl(
                    f"gen {GEN_LAYOUT} -t {GEN_TIME}\nq\n",
                    timeout=GEN_TIME + 60,
                )
            except subprocess.TimeoutExpired:
                print("  gen timed out, skipping")
                cleanup_dof_files(trial)
                continue

            print(f"  gen completed in {time.time() - gen_start:.0f}s")

            parsed = parse_gen_output(gen_output)
            if not parsed:
                print("  No layouts parsed, skipping")
                cleanup_dof_files(trial)
                continue

            print(f"  Parsed {len(parsed)} layouts")

            trial_best_fitness = float("inf")
            trial_best_name = None

            for rank, score_str, lines, magic_str in parsed:
                dof_name = gen_layout_to_dof(rank, lines, magic_str, trial)
                if dof_name is None:
                    continue

                metrics = analyze_layout(dof_name)
                fitness = compute_fitness(metrics, targets)

                s_wins, _ = win_count(metrics, targets["sturdy"])
                g_wins, _ = win_count(metrics, targets["graphite"])
                c_wins, _ = win_count(metrics, targets["canary"])

                print(f"\n  #{rank} ({score_str}): S={s_wins}/9 G={g_wins}/9 C={c_wins}/9 "
                      f"fitness={fitness:.4f} [{tier_label(fitness)}]")

                # Show details for the current bottleneck target
                if s_wins < 9:
                    print_vs_target(metrics, targets["sturdy"], "sturdy")
                elif g_wins < 9:
                    print_vs_target(metrics, targets["graphite"], "graphite")
                elif c_wins < 9:
                    print_vs_target(metrics, targets["canary"], "canary")
                else:
                    # All 3 beaten — show margins
                    for tname in TARGETS:
                        m_val = total_margin(metrics, targets[tname])
                        w, _ = win_count(metrics, targets[tname])
                        print(f"    vs {tname}: {w}/9, margin={m_val:+.1f}")

                log_result(trial, ws, metrics, fitness, s_wins, g_wins, c_wins)

                if fitness < trial_best_fitness:
                    trial_best_fitness = fitness
                    trial_best_name = dof_name

                # Track global top 3
                if top3.maybe_add(fitness, dof_name, metrics, ws):
                    save_best_layout(dof_name)

            if trial_best_fitness < best_fitness:
                best_fitness = trial_best_fitness
                best_weights = ws
                best_layout_name = trial_best_name
                print(f"\n  ★ New best! fitness={best_fitness:.4f} [{tier_label(best_fitness)}]")
                print(f"    Layout: {trial_best_name}")
                temperature = max(0.05, temperature * 0.95)
            else:
                print(f"\n  No improvement ({trial_best_fitness:.4f} vs {best_fitness:.4f})")
                temperature = min(0.5, temperature * 1.02)

            # Keep .dof files that are in the top 3
            top3_names = {e[1] for e in top3.entries}
            keep_files = {f"{n}.dof" for n in top3_names}
            cleanup_dof_files(trial, keep=keep_files)

            save_state({
                "trial": trial,
                "best_fitness": best_fitness,
                "best_weights": asdict(best_weights),
                "temperature": temperature,
                "best_layout_name": best_layout_name,
                "top3": top3.to_json(),
            })
            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    finally:
        if os.path.exists(CONFIG_BAK):
            shutil.copy(CONFIG_BAK, CONFIG)
            os.remove(CONFIG_BAK)
            print("Config restored.")

        print(f"\n{'=' * 70}")
        print(f"TOP 3 RESULTS")
        print(f"{'=' * 70}")

        next_v = next_oxey_version()
        saved_oxey = []

        for i, (fitness, name, metrics, ws_dict) in enumerate(top3.entries):
            oxey_name = f"oxey-v{next_v + i}"
            src_dof = os.path.join(BEST_DOF_DIR, f"{name}.dof")
            dst_dof = os.path.join(LAYOUT_DIR, f"{oxey_name}.dof")

            print(f"\n  #{i+1}: fitness={fitness:.4f} [{tier_label(fitness)}]")
            print(f"    Weights: {', '.join(f'{n}={ws_dict[n]}' for n in TUNABLE)}")

            # Copy to oxey-v*.dof with updated name
            if os.path.exists(src_dof):
                with open(src_dof) as f:
                    dof = json.load(f)
                dof["name"] = oxey_name
                with open(dst_dof, "w") as f:
                    json.dump(dof, f, indent=4)
                saved_oxey.append(dst_dof)
                print(f"    Saved as: {dst_dof}")
                print_layout_dof(dst_dof)
            else:
                print(f"    (.dof not found at {src_dof})")

        if saved_oxey:
            print(f"\nSaved {len(saved_oxey)} layouts: {', '.join(saved_oxey)}")
        print(f"Results: {LOG_FILE}")


if __name__ == "__main__":
    main()
