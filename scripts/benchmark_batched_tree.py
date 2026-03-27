#!/usr/bin/env python3
"""
N1: Benchmark Batched vs Sequential Dream Tree

Compares latency and success rate of:
  - Sequential tree (CUDA graphs, B=1 per CEM call)
  - Batched tree (flexible compile, B=K per CEM call)

Tests different compile modes to find the best balance.

Usage (on RunPod):
    export STABLEWM_HOME=/workspace/data
    export MUJOCO_GL=egl
    cd /workspace/le-harness

    python scripts/benchmark_batched_tree.py --policy tworoom/lewm --config-name tworoom
    python scripts/benchmark_batched_tree.py --policy tworoom/lewm --config-name tworoom --compile-mode default
"""

import argparse
import json
import os
import time

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

from harness.pipeline import PlanningPipeline
from harness.dream_tree import DreamTreePlanner


def benchmark_latency(planner, num_steps: int = 20):
    """Run planner on dummy inputs to measure latency."""
    dummy_obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_goal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        planner.plan(dummy_obs, dummy_goal)
    planner.reset_timing()
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_steps):
        planner.plan(dummy_obs, dummy_goal)
    torch.cuda.synchronize()

    return planner.get_timing_summary()


def main():
    parser = argparse.ArgumentParser(description="N1: Benchmark Batched Tree")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["reduce-overhead", "default", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--num-roots", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of planning steps for latency benchmark")
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    results = {}

    # --- Sequential baseline (always uses reduce-overhead for CUDA graphs) ---
    print(f"\n{'='*60}")
    print(f"Loading pipeline with compile_mode='reduce-overhead' (sequential baseline)...")
    print(f"{'='*60}")

    pipeline_seq = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
        compile_mode="reduce-overhead",
    )
    pipeline_seq.warmup()

    tree_seq = DreamTreePlanner(
        pipeline_seq,
        num_roots=args.num_roots,
        max_depth=2,
        cheap_depth=False,
        batched=False,
    )

    print(f"\nBenchmarking sequential tree ({args.num_steps} steps)...")
    seq_timing = benchmark_latency(tree_seq, args.num_steps)
    results["sequential"] = {
        "compile_mode": "reduce-overhead",
        "batched": False,
        **seq_timing,
    }
    print(f"  Sequential: {seq_timing['total_ms']['mean']:.0f}ms "
          f"({seq_timing['effective_hz']:.1f} Hz)")
    print(f"    Root: {seq_timing['root_ms']['mean']:.0f}ms, "
          f"Expansion: {seq_timing['expansion_ms']['mean']:.0f}ms")

    # --- Batched (with requested compile mode) ---
    if args.compile_mode != "reduce-overhead":
        print(f"\n{'='*60}")
        print(f"Loading pipeline with compile_mode='{args.compile_mode}' (batched)...")
        print(f"{'='*60}")

        pipeline_batch = PlanningPipeline(
            policy_name=args.policy,
            num_samples=128,
            n_steps=15,
            horizon=5,
            topk=25,
            compile_mode=args.compile_mode,
        )
        pipeline_batch.warmup()
    else:
        # Try batched with reduce-overhead anyway — may fail on shape mismatch
        print(f"\n{'='*60}")
        print(f"Testing batched with compile_mode='reduce-overhead' (may fail)...")
        print(f"{'='*60}")
        pipeline_batch = pipeline_seq

    tree_batch = DreamTreePlanner(
        pipeline_batch,
        num_roots=args.num_roots,
        max_depth=2,
        cheap_depth=False,
        batched=True,
    )

    print(f"\nBenchmarking batched tree ({args.num_steps} steps)...")
    try:
        batch_timing = benchmark_latency(tree_batch, args.num_steps)
        results["batched"] = {
            "compile_mode": args.compile_mode,
            "batched": True,
            **batch_timing,
        }
        print(f"  Batched: {batch_timing['total_ms']['mean']:.0f}ms "
              f"({batch_timing['effective_hz']:.1f} Hz)")
        print(f"    Root: {batch_timing['root_ms']['mean']:.0f}ms, "
              f"Expansion: {batch_timing['expansion_ms']['mean']:.0f}ms")

        # Speedup
        speedup = seq_timing['total_ms']['mean'] / batch_timing['total_ms']['mean']
        results["speedup"] = round(speedup, 2)
        print(f"\n  Speedup: {speedup:.2f}x")

    except Exception as e:
        print(f"  FAILED: {e}")
        results["batched"] = {"error": str(e), "compile_mode": args.compile_mode}

    # --- Also test batched with mode='default' if not already tested ---
    if args.compile_mode not in ("default",):
        print(f"\n{'='*60}")
        print(f"Loading pipeline with compile_mode='default' (batched fallback)...")
        print(f"{'='*60}")

        try:
            pipeline_default = PlanningPipeline(
                policy_name=args.policy,
                num_samples=128,
                n_steps=15,
                horizon=5,
                topk=25,
                compile_mode="default",
            )
            pipeline_default.warmup()

            tree_default = DreamTreePlanner(
                pipeline_default,
                num_roots=args.num_roots,
                max_depth=2,
                cheap_depth=False,
                batched=True,
            )

            print(f"\nBenchmarking batched tree with mode='default' ({args.num_steps} steps)...")
            default_timing = benchmark_latency(tree_default, args.num_steps)
            results["batched_default"] = {
                "compile_mode": "default",
                "batched": True,
                **default_timing,
            }
            print(f"  Batched (default): {default_timing['total_ms']['mean']:.0f}ms "
                  f"({default_timing['effective_hz']:.1f} Hz)")

            speedup_default = seq_timing['total_ms']['mean'] / default_timing['total_ms']['mean']
            results["speedup_default"] = round(speedup_default, 2)
            print(f"  Speedup vs sequential: {speedup_default:.2f}x")

        except Exception as e:
            print(f"  FAILED: {e}")
            results["batched_default"] = {"error": str(e)}

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"N1 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Sequential (reduce-overhead): {seq_timing['total_ms']['mean']:.0f}ms "
          f"({seq_timing['effective_hz']:.1f} Hz)")

    for key in ["batched", "batched_default"]:
        if key in results and "error" not in results[key]:
            t = results[key]
            print(f"{key} ({t['compile_mode']}): {t['total_ms']['mean']:.0f}ms "
                  f"({t['effective_hz']:.1f} Hz)")

    gate_passed = False
    for key in ["batched", "batched_default"]:
        if key in results and "total_ms" in results[key]:
            if results[key]["total_ms"]["mean"] < 300:
                gate_passed = True
                print(f"\nGATE: PASS — {key} achieves <300ms")
                break

    if not gate_passed:
        print(f"\nGATE: FAIL — no batched config achieved <300ms")

    # Save
    out_dir = os.path.join(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"n1_benchmark_{args.compile_mode}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
