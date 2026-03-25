#!/usr/bin/env python3
"""
Phase 2: Planning Budget Reduction Sweep (Staged)

Two-stage sweep that finds the minimum viable planning budget in ~45-90 min
instead of exhaustively testing 42 configs over 8-15 hours.

  Stage A (Screen):  8 gate-candidate configs × 20 episodes  (~15 min)
  Stage B (Confirm): Top configs from A     × 50 episodes  (~15-30 min)

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness
    python scripts/sweep_budget.py --policy pusht/lejepa          # auto: A then B
    python scripts/sweep_budget.py --policy pusht/lejepa --stage screen   # A only
    python scripts/sweep_budget.py --policy pusht/lejepa --stage confirm  # B only (reads A results)
    python scripts/sweep_budget.py --policy pusht/lejepa --dry-run        # print plan

Estimated runtime: 45-90 minutes on RTX 4090.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─── Stage A: Screen Configs ──────────────────────────────────────────
# Gate: ≥93% success at ≤9,000 forward passes (5x reduction from 45,000).
# All configs below are ≤3,200 FP. We test both iCEM (expected winner)
# and CEM (comparison) across the likely gate boundary.
#
# Forward passes = num_samples × n_steps × horizon (horizon=5 from pusht.yaml)

SCREEN_CONFIGS = [
    # iCEM — expected to dominate CEM at equal budget
    ("icem", 16, 5),    #    400 FP — very aggressive
    ("icem", 32, 5),    #    800 FP — aggressive
    ("icem", 32, 10),   #  1,600 FP — moderate
    ("icem", 64, 5),    #  1,600 FP — moderate (more samples, fewer iters)
    ("icem", 64, 10),   #  3,200 FP — comfortable
    # CEM — comparison at matched budgets
    ("cem", 32, 10),    #  1,600 FP
    ("cem", 64, 5),     #  1,600 FP
    ("cem", 64, 10),    #  3,200 FP
]

SCREEN_EPISODES = 20
CONFIRM_EPISODES = 50

# Screen threshold: configs scoring ≥ this advance to confirm stage.
# Set below the 93% gate to account for noise at 20 episodes.
# At 20 eps, a true-93% config has ~68% chance of scoring ≥90%.
# At 20 eps, a true-80% config has ~40% chance of scoring ≥90%.
# Using 80% to be generous — we'd rather confirm a few extra than miss the winner.
SCREEN_ADVANCE_THRESHOLD = 80.0

# Phase 2 gate
BASELINE_SR = 98.0
GATE_SR = 93.0       # within 5 percentage points of baseline
GATE_MAX_FP = 9000   # 5x reduction from 45,000


def parse_metrics(output: str) -> dict:
    """Extract success_rate from eval.py output."""
    metrics = {}
    sr_match = re.search(r"['\"]success_rate['\"]:\s*([\d.]+)", output)
    if sr_match:
        metrics["success_rate"] = float(sr_match.group(1))
    else:
        sr_match2 = re.search(r"success_rate.*?(\d+\.?\d*)", output)
        if sr_match2:
            metrics["success_rate"] = float(sr_match2.group(1))
    return metrics


def run_eval(solver: str, num_samples: int, n_steps: int, policy: str,
             num_eval: int, seed: int) -> dict:
    """Run a single eval configuration and return results."""
    horizon_steps = 5
    forward_passes = num_samples * n_steps * horizon_steps

    cmd = [
        sys.executable, "eval.py",
        f"--config-name=pusht",
        f"policy={policy}",
        f"solver={solver}",
        f"solver.num_samples={num_samples}",
        f"solver.n_steps={n_steps}",
        f"solver.topk={max(1, min(num_samples // 5, 30))}",
        f"eval.num_eval={num_eval}",
        f"seed={seed}",
    ]

    if solver == "icem":
        n_elite_keep = max(1, min(5, num_samples // 10))
        cmd.append(f"solver.n_elite_keep={n_elite_keep}")

    config_str = f"{solver} samples={num_samples} iters={n_steps}"
    print(f"\n{'='*60}")
    print(f"Running: {config_str} ({num_eval} episodes)")
    print(f"Forward passes: {forward_passes:,}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = {
        "solver": solver,
        "num_samples": num_samples,
        "n_steps": n_steps,
        "forward_passes": forward_passes,
        "num_eval": num_eval,
        "success_rate": None,
        "eval_time_s": None,
        "ms_per_step": None,
        "error": None,
    }

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=os.getcwd(),
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        elapsed = time.time() - start
        result["eval_time_s"] = round(elapsed, 1)

        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            result["error"] = f"exit code {proc.returncode}"
            print(f"  ERROR: exit code {proc.returncode}")
            print(f"  stderr (last 500 chars): {proc.stderr[-500:]}")
        else:
            print(f"  Completed in {elapsed:.0f}s")

        metrics = parse_metrics(output)
        if "success_rate" in metrics:
            result["success_rate"] = metrics["success_rate"]
            eval_budget = 50  # from pusht.yaml
            total_steps = num_eval * eval_budget
            result["ms_per_step"] = round((elapsed / total_steps) * 1000, 0)
            print(f"  Success rate: {result['success_rate']}%")
            print(f"  Planning latency: ~{result['ms_per_step']:.0f} ms/step")
        else:
            print(f"  WARNING: Could not parse success_rate from output")
            result["error"] = (result.get("error") or "") + " | no success_rate parsed"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        result["eval_time_s"] = round(elapsed, 1)
        result["error"] = "timeout (>1hr)"
        print(f"  TIMEOUT after {elapsed:.0f}s")

    except Exception as e:
        elapsed = time.time() - start
        result["eval_time_s"] = round(elapsed, 1)
        result["error"] = str(e)
        print(f"  EXCEPTION: {e}")

    return result


def csv_fieldnames():
    return [
        "stage", "solver", "num_samples", "n_steps", "forward_passes",
        "num_eval", "success_rate", "eval_time_s", "ms_per_step", "error",
        "timestamp",
    ]


def write_result(path: Path, result: dict, stage: str, write_header: bool = False):
    """Append a single result row to CSV (crash-safe)."""
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        if write_header:
            writer.writeheader()
        writer.writerow({
            "stage": stage,
            "solver": result["solver"],
            "num_samples": result["num_samples"],
            "n_steps": result["n_steps"],
            "forward_passes": result["forward_passes"],
            "num_eval": result["num_eval"],
            "success_rate": result["success_rate"],
            "eval_time_s": result["eval_time_s"],
            "ms_per_step": result["ms_per_step"],
            "error": result["error"],
            "timestamp": datetime.now().isoformat(),
        })


def load_results(path: Path, stage_filter: str = None) -> list:
    """Load results from CSV, optionally filtering by stage."""
    results = []
    if not path.exists():
        return results
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if stage_filter and row.get("stage") != stage_filter:
                continue
            # Convert types
            row["num_samples"] = int(row["num_samples"])
            row["n_steps"] = int(row["n_steps"])
            row["forward_passes"] = int(row["forward_passes"])
            row["num_eval"] = int(row["num_eval"])
            row["success_rate"] = float(row["success_rate"]) if row["success_rate"] else None
            row["eval_time_s"] = float(row["eval_time_s"]) if row["eval_time_s"] else None
            row["ms_per_step"] = float(row["ms_per_step"]) if row["ms_per_step"] else None
            results.append(row)
    return results


def completed_keys(results: list) -> set:
    """Return set of (solver, num_samples, n_steps, num_eval) already in results."""
    return {
        (r["solver"], r["num_samples"], r["n_steps"], r["num_eval"])
        for r in results
    }


def print_results_table(results: list, title: str):
    """Print a formatted results table."""
    successful = [r for r in results if r.get("success_rate") is not None]
    if not successful:
        print(f"\n{title}: No successful results.")
        return

    successful.sort(key=lambda r: r["forward_passes"])

    print(f"\n{title}")
    print(f"{'Solver':>6s} {'Samp':>5s} {'Iter':>4s} {'FwdPass':>8s} "
          f"{'Eps':>4s} {'SR%':>6s} {'ms/step':>7s} {'Gate':>6s}")
    print("-" * 55)

    for r in successful:
        sr = r["success_rate"]
        fp = r["forward_passes"]
        passes_gate = sr >= GATE_SR and fp <= GATE_MAX_FP
        marker = "PASS" if passes_gate else ""
        ms = f"{r['ms_per_step']:.0f}" if r.get("ms_per_step") else "?"
        eps = r.get("num_eval", "?")
        print(f"{r['solver']:>6s} {r['num_samples']:>5d} {r['n_steps']:>4d} "
              f"{fp:>8,} {eps:>4} {sr:>6.1f} {ms:>7s} {marker:>6s}")


def select_confirm_configs(screen_results: list) -> list:
    """Select configs from screen results that should advance to confirm stage."""
    candidates = []
    for r in screen_results:
        if r.get("success_rate") is None:
            continue
        if r["forward_passes"] > GATE_MAX_FP:
            continue
        if r["success_rate"] >= SCREEN_ADVANCE_THRESHOLD:
            candidates.append((r["solver"], r["num_samples"], r["n_steps"]))

    # Also include the best config below threshold as a boundary probe
    below = [
        r for r in screen_results
        if r.get("success_rate") is not None
        and r["forward_passes"] <= GATE_MAX_FP
        and r["success_rate"] < SCREEN_ADVANCE_THRESHOLD
    ]
    if below:
        best_below = max(below, key=lambda r: r["success_rate"])
        boundary = (best_below["solver"], best_below["num_samples"], best_below["n_steps"])
        if boundary not in candidates:
            candidates.append(boundary)

    return candidates


def run_stage(stage_name: str, configs: list, num_eval: int, policy: str,
              seed: int, output_path: Path, resume: bool) -> list:
    """Run a list of configs and return results."""
    # Load existing results for resume
    existing = load_results(output_path)
    done = completed_keys(existing)

    # Filter out already-completed configs
    to_run = []
    for solver, samples, iters in configs:
        if resume and (solver, samples, iters, num_eval) in done:
            continue
        to_run.append((solver, samples, iters))

    skipped = len(configs) - len(to_run)
    if skipped:
        print(f"  Resuming: {skipped} configs already completed, {len(to_run)} remaining")

    if not to_run:
        print(f"  All {stage_name} configs already completed.")
        return [r for r in existing if r.get("stage") == stage_name]

    # If not resuming and file exists, truncate to avoid duplicate rows.
    # Preserve rows from OTHER stages (e.g., keep screen results when re-running confirm).
    if not resume and output_path.exists():
        other_stage = [r for r in existing if r.get("stage") != stage_name]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
            writer.writeheader()
            for r in other_stage:
                writer.writerow(r)

    # Write header only if file doesn't exist
    write_header = not output_path.exists()

    results = []
    start_time = time.time()

    for i, (solver, samples, iters) in enumerate(to_run):
        print(f"\n[{stage_name} {i+1}/{len(to_run)}] ", end="")
        result = run_eval(solver, samples, iters, policy, num_eval, seed)
        results.append(result)

        write_result(output_path, result, stage_name, write_header=write_header)
        write_header = False  # only on first write

        # Progress
        elapsed = time.time() - start_time
        remaining = len(to_run) - (i + 1)
        if i > 0:
            eta = (elapsed / (i + 1)) * remaining
            print(f"  Progress: {i+1}/{len(to_run)} | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Planning Budget Sweep (Staged)")
    parser.add_argument("--policy", default="pusht/lejepa", help="Policy checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="/workspace/data/results/phase2_sweep.csv",
                        help="Output CSV path")
    parser.add_argument("--stage", choices=["auto", "screen", "confirm"], default="auto",
                        help="Which stage to run (default: auto runs both)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in the output CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LeHarness Phase 2: Planning Budget Sweep (Staged)")
    print(f"{'='*60}")
    print(f"Policy: {args.policy}")
    print(f"Stage: {args.stage}")
    print(f"Output: {args.output}")
    print(f"Baseline: 98.0% success at 45,000 forward passes (300×30×5)")
    print(f"Gate: ≥{GATE_SR}% success at ≤{GATE_MAX_FP:,} forward passes")

    # ─── Dry Run ──────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n--- Stage A: Screen ({SCREEN_EPISODES} episodes each) ---")
        for solver, samples, iters in SCREEN_CONFIGS:
            fp = samples * iters * 5
            print(f"  {solver:5s} samples={samples:3d} iters={iters:2d} → {fp:5,} FP")
        print(f"  Total: {len(SCREEN_CONFIGS)} configs × {SCREEN_EPISODES} eps")
        print(f"\n--- Stage B: Confirm ({CONFIRM_EPISODES} episodes each) ---")
        print(f"  Auto-selected from Stage A (configs scoring ≥{SCREEN_ADVANCE_THRESHOLD}%)")
        print(f"  + boundary probe (best config below threshold)")
        print(f"\nEstimated total: ~45-90 min on RTX 4090")
        return

    # ─── Stage A: Screen ──────────────────────────────────────────
    run_screen = args.stage in ("auto", "screen")
    run_confirm = args.stage in ("auto", "confirm")

    if run_screen:
        print(f"\n{'='*60}")
        print(f"STAGE A: Screen ({len(SCREEN_CONFIGS)} configs × {SCREEN_EPISODES} episodes)")
        print(f"{'='*60}")

        screen_results = run_stage(
            "screen", SCREEN_CONFIGS, SCREEN_EPISODES,
            args.policy, args.seed, output_path, args.resume,
        )

        # Merge with any previously saved screen results
        all_screen = load_results(output_path, stage_filter="screen")
        print_results_table(all_screen, "Stage A Results (Screen)")

    # ─── Stage B: Confirm ─────────────────────────────────────────
    if run_confirm:
        all_screen = load_results(output_path, stage_filter="screen")
        if not all_screen:
            print("\nERROR: No screen results found. Run --stage screen first.")
            sys.exit(1)

        confirm_configs = select_confirm_configs(all_screen)

        if not confirm_configs:
            print(f"\nNo configs scored ≥{SCREEN_ADVANCE_THRESHOLD}% in screen stage.")
            print("The planning budget may be too aggressive for this model.")
            print("Consider: (a) testing higher budgets, or (b) fine-tuning the predictor.")
            print_results_table(all_screen, "All Screen Results")
            return

        print(f"\n{'='*60}")
        print(f"STAGE B: Confirm ({len(confirm_configs)} configs × {CONFIRM_EPISODES} episodes)")
        print(f"{'='*60}")
        for solver, samples, iters in confirm_configs:
            fp = samples * iters * 5
            print(f"  {solver:5s} samples={samples:3d} iters={iters:2d} → {fp:5,} FP")

        confirm_results = run_stage(
            "confirm", confirm_configs, CONFIRM_EPISODES,
            args.policy, args.seed, output_path, args.resume,
        )

        # Final results from confirm stage
        all_confirm = load_results(output_path, stage_filter="confirm")
        print_results_table(all_confirm, "Stage B Results (Confirm, 50 episodes)")

    # ─── Gate Check ───────────────────────────────────────────────
    all_results = load_results(output_path)
    # Use confirm results if available, otherwise screen
    confirm_results = load_results(output_path, stage_filter="confirm")
    gate_source = confirm_results if confirm_results else all_results

    gate_passed = False
    best_passing = None
    for r in gate_source:
        if r.get("success_rate") is None:
            continue
        sr = r["success_rate"]
        fp = r["forward_passes"]
        if sr >= GATE_SR and fp <= GATE_MAX_FP:
            gate_passed = True
            if best_passing is None or fp < best_passing["forward_passes"]:
                best_passing = r

    print(f"\n{'='*60}")
    if gate_passed:
        bp = best_passing
        print(f"PHASE 2 GATE: PASS")
        print(f"  Minimum viable budget: {bp['solver']} "
              f"samples={bp['num_samples']} iters={bp['n_steps']}")
        print(f"  Forward passes: {bp['forward_passes']:,} "
              f"({45000 / bp['forward_passes']:.0f}x reduction from baseline)")
        print(f"  Success rate: {bp['success_rate']}% "
              f"(baseline: {BASELINE_SR}%)")
        if bp.get("ms_per_step"):
            print(f"  Planning latency: ~{bp['ms_per_step']:.0f} ms/step "
                  f"(baseline: 1310 ms/step)")
        print(f"\nNext: Run convergence logging on this config:")
        print(f"  python scripts/log_convergence.py "
              f"--solver {bp['solver']} "
              f"--num-samples {bp['num_samples']} "
              f"--n-steps {bp['n_steps']} "
              f"--policy {args.policy}")
    else:
        print(f"PHASE 2 GATE: FAIL")
        print(f"  No config achieved ≥{GATE_SR}% at ≤{GATE_MAX_FP:,} forward passes.")
        print(f"  Action: Consider fine-tuning the predictor or testing higher budgets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
