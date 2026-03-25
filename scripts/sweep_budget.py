#!/usr/bin/env python3
"""
Phase 2: Planning Budget Reduction Sweep

Sweeps (solver, num_samples, n_steps) configurations and records
success rate + timing for each. Outputs CSV to the network volume.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness
    python scripts/sweep_budget.py --policy pusht/lejepa

The sweep runs all configs sequentially. Use tmux so you can disconnect.
Estimated runtime: 8-15 hours on RTX 4090 depending on grid size.
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


# ─── Sweep Grid ───────────────────────────────────────────────────────
# Phase 1 showed depth-5 ratio = 0.139 (3.6x below threshold).
# This means we can be aggressive with budget reduction.

SWEEP_CONFIGS = []

# Step 1: Sweep samples with fixed 30 iterations (CEM baseline)
for samples in [300, 128, 64, 32, 16]:
    SWEEP_CONFIGS.append(("cem", samples, 30))

# Step 2: Sweep iterations with fixed 300 samples (CEM baseline)
for iters in [15, 10, 5, 3]:
    SWEEP_CONFIGS.append(("cem", 300, iters))

# Step 3: Joint sweep at promising intersections (CEM)
for samples, iters in [(128, 15), (128, 10), (128, 5), (64, 15), (64, 10), (64, 5), (64, 3), (32, 15), (32, 10), (32, 5), (16, 10), (16, 5)]:
    SWEEP_CONFIGS.append(("cem", samples, iters))

# Step 4: iCEM at the same grid (this is where we expect the wins)
for samples in [300, 128, 64, 32, 16]:
    SWEEP_CONFIGS.append(("icem", samples, 30))

for iters in [15, 10, 5, 3]:
    SWEEP_CONFIGS.append(("icem", 300, iters))

for samples, iters in [(128, 15), (128, 10), (128, 5), (64, 15), (64, 10), (64, 5), (64, 3), (32, 15), (32, 10), (32, 5), (16, 10), (16, 5)]:
    SWEEP_CONFIGS.append(("icem", samples, iters))

# Deduplicate while preserving order
seen = set()
SWEEP_GRID = []
for cfg in SWEEP_CONFIGS:
    if cfg not in seen:
        seen.add(cfg)
        SWEEP_GRID.append(cfg)


def parse_metrics(output: str) -> dict:
    """Extract success_rate from eval.py output."""
    metrics = {}

    # Look for success_rate in the printed metrics dict
    # Format: {'success_rate': 98.0, ...} or similar
    sr_match = re.search(r"['\"]success_rate['\"]:\s*([\d.]+)", output)
    if sr_match:
        metrics["success_rate"] = float(sr_match.group(1))

    # Also try to catch it from the results file output
    sr_match2 = re.search(r"success_rate.*?(\d+\.?\d*)", output)
    if "success_rate" not in metrics and sr_match2:
        metrics["success_rate"] = float(sr_match2.group(1))

    return metrics


def run_eval(solver: str, num_samples: int, n_steps: int, policy: str,
             num_eval: int, seed: int) -> dict:
    """Run a single eval configuration and return results."""
    # Compute forward passes: samples * iterations * horizon_steps (5)
    horizon_steps = 5
    forward_passes = num_samples * n_steps * horizon_steps

    # Build the command
    cmd = [
        sys.executable, "eval.py",
        f"--config-name=pusht",
        f"policy={policy}",
        f"solver={solver}",
        f"solver.num_samples={num_samples}",
        f"solver.n_steps={n_steps}",
        f"solver.topk={max(1, min(num_samples // 5, 30))}",  # ~20% of samples, capped at 30
        f"eval.num_eval={num_eval}",
        f"seed={seed}",
    ]

    # For iCEM, also scale n_elite_keep proportionally
    if solver == "icem":
        n_elite_keep = max(1, min(5, num_samples // 10))
        cmd.append(f"solver.n_elite_keep={n_elite_keep}")

    config_str = f"{solver} samples={num_samples} iters={n_steps}"
    print(f"\n{'='*60}")
    print(f"Running: {config_str}")
    print(f"Forward passes: {forward_passes:,}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = {
        "solver": solver,
        "num_samples": num_samples,
        "n_steps": n_steps,
        "forward_passes": forward_passes,
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
            timeout=3600,  # 1 hour max per config
            cwd=os.getcwd(),
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        elapsed = time.time() - start
        result["eval_time_s"] = round(elapsed, 1)

        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            result["error"] = f"exit code {proc.returncode}"
            # Try to extract partial results anyway
            print(f"  ERROR: exit code {proc.returncode}")
            print(f"  stderr (last 500 chars): {proc.stderr[-500:]}")
        else:
            print(f"  Completed in {elapsed:.0f}s")

        # Parse metrics
        metrics = parse_metrics(output)
        if "success_rate" in metrics:
            result["success_rate"] = metrics["success_rate"]
            # Estimate ms/step: total time / (num_eval * eval_budget)
            eval_budget = 50  # from pusht.yaml
            total_steps = num_eval * eval_budget
            result["ms_per_step"] = round((elapsed / total_steps) * 1000, 0)
            print(f"  Success rate: {result['success_rate']}%")
            print(f"  Planning latency: ~{result['ms_per_step']:.0f} ms/step")
        else:
            print(f"  WARNING: Could not parse success_rate from output")
            result["error"] = result.get("error", "") + " | no success_rate parsed"

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


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Planning Budget Sweep")
    parser.add_argument("--policy", default="pusht/lejepa", help="Policy checkpoint path")
    parser.add_argument("--num-eval", type=int, default=50, help="Episodes per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="/workspace/data/results/phase2_sweep.csv",
                        help="Output CSV path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already in the output CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    completed = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["solver"], int(row["num_samples"]), int(row["n_steps"]))
                completed.add(key)
        print(f"Resuming: {len(completed)} configs already completed")

    # Filter grid
    grid = [(s, n, i) for s, n, i in SWEEP_GRID if (s, n, i) not in completed]

    print(f"\n{'='*60}")
    print(f"LeHarness Phase 2: Planning Budget Sweep")
    print(f"{'='*60}")
    print(f"Policy: {args.policy}")
    print(f"Episodes per config: {args.num_eval}")
    print(f"Total configs: {len(grid)} (of {len(SWEEP_GRID)}, {len(completed)} already done)")
    print(f"Output: {args.output}")
    print(f"Baseline: 98.0% success at 45,000 forward passes (300x30x5)")
    print(f"Target: ≥93% success (within 5%) at ≤9,000 forward passes (5x reduction)")
    print()

    if args.dry_run:
        print("DRY RUN — configs that would be tested:")
        for solver, samples, iters in grid:
            fp = samples * iters * 5
            print(f"  {solver:5s} samples={samples:3d} iters={iters:2d} → {fp:6,} forward passes")
        print(f"\nTotal: {len(grid)} configs")
        return

    # Write CSV header if new file
    write_header = not output_path.exists() or not args.resume
    if write_header:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "solver", "num_samples", "n_steps", "forward_passes",
                "success_rate", "eval_time_s", "ms_per_step", "error",
                "timestamp",
            ])

    # Run sweep
    start_time = time.time()
    results = []

    for i, (solver, samples, iters) in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}] ", end="")
        result = run_eval(solver, samples, iters, args.policy, args.num_eval, args.seed)
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)

        # Append to CSV immediately (crash-safe)
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                result["solver"], result["num_samples"], result["n_steps"],
                result["forward_passes"], result["success_rate"],
                result["eval_time_s"], result["ms_per_step"],
                result["error"], result["timestamp"],
            ])

        # Progress summary
        elapsed_total = time.time() - start_time
        configs_done = i + 1
        configs_remaining = len(grid) - configs_done
        if configs_done > 0:
            avg_time = elapsed_total / configs_done
            eta = avg_time * configs_remaining
            print(f"  Progress: {configs_done}/{len(grid)} | "
                  f"Elapsed: {elapsed_total/3600:.1f}h | "
                  f"ETA: {eta/3600:.1f}h")

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Configs tested: {len(results)}")
    print(f"Results saved to: {args.output}")
    print()

    # Print results table sorted by forward passes
    successful = [r for r in results if r["success_rate"] is not None]
    if successful:
        successful.sort(key=lambda r: r["forward_passes"])

        baseline_sr = 98.0
        threshold_sr = baseline_sr - 5.0  # within 5 percentage points

        print(f"{'Solver':>6s} {'Samples':>7s} {'Iters':>5s} {'FwdPass':>8s} "
              f"{'SR%':>6s} {'ms/step':>7s} {'Gate':>6s}")
        print("-" * 55)

        gate_passed = False
        for r in successful:
            sr = r["success_rate"]
            fp = r["forward_passes"]
            passes_gate = sr >= threshold_sr and fp <= 9000
            marker = "PASS" if passes_gate else ""
            if passes_gate:
                gate_passed = True
            ms = f"{r['ms_per_step']:.0f}" if r["ms_per_step"] else "?"
            print(f"{r['solver']:>6s} {r['num_samples']:>7d} {r['n_steps']:>5d} "
                  f"{fp:>8,} {sr:>6.1f} {ms:>7s} {marker:>6s}")

        print()
        if gate_passed:
            print("PHASE 2 GATE: PASS — at least one config achieves ≥93% at ≤9,000 forward passes")
        else:
            print("PHASE 2 GATE: FAIL — no config achieves ≥93% at ≤9,000 forward passes")
            print("Action: Consider fine-tuning the predictor or reducing the planning horizon.")


if __name__ == "__main__":
    main()
