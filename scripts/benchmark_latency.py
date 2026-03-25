#!/usr/bin/env python3
"""
Phase 5: Benchmark planning latency breakdown.

Profiles a single CEM planning step to measure time spent in each component:
encoder, predictor, action encoder, CEM overhead.

Usage:
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness
    python scripts/benchmark_latency.py --policy pusht/lejepa
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def benchmark_components(model, num_samples=128, n_steps=15, n_warmup=3, n_trials=10):
    """Benchmark individual model components with CUDA event timing."""
    device = next(model.parameters()).device

    # Prepare dummy inputs matching CEM planning shapes
    # Encoder: single image
    dummy_pixels = torch.randn(1, 1, 3, 224, 224, device=device)
    # Predictor: batch of samples, history=3
    BS = num_samples  # flattened B*S where B=1
    dummy_emb = torch.randn(BS, 3, 192, device=device)
    dummy_act_emb = torch.randn(BS, 3, 192, device=device)
    # Action encoder
    dummy_actions = torch.randn(BS, 3, 10, device=device)  # action_dim=10 (2*action_block)
    # Full rollout input
    dummy_info = {
        "pixels": torch.randn(1, num_samples, 1, 3, 224, 224, device=device),
        "goal": torch.randn(1, num_samples, 1, 3, 224, 224, device=device),
        "action": torch.randn(1, num_samples, 1, 10, device=device),
    }
    dummy_candidates = torch.randn(1, num_samples, 6, 10, device=device)  # horizon+hist=6

    results = {}

    # ─── Encoder ──────────────────────────────────────────────
    print("Benchmarking encoder...")
    for _ in range(n_warmup):
        model.encode({"pixels": dummy_pixels})
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]

    for i in range(n_trials):
        start_events[i].record()
        model.encode({"pixels": dummy_pixels})
        end_events[i].record()
    torch.cuda.synchronize()

    encoder_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["encoder_ms"] = {
        "mean": np.mean(encoder_times),
        "std": np.std(encoder_times),
        "calls_per_step": 1,
    }

    # ─── Predictor (single call) ─────────────────────────────
    print("Benchmarking predictor (single call)...")
    for _ in range(n_warmup):
        model.predict(dummy_emb, dummy_act_emb)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]

    for i in range(n_trials):
        start_events[i].record()
        model.predict(dummy_emb, dummy_act_emb)
        end_events[i].record()
    torch.cuda.synchronize()

    pred_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    # In CEM: called (horizon=5 + 1 final) times per CEM iteration, across n_steps iterations
    # But actually, the rollout loop calls predict n_steps times (n_steps = horizon - history + 1 ≈ 5)
    # And the CEM loop runs n_steps=15 CEM iterations, each calling rollout once
    # So total predict calls = 15 * (5+1) = 90... wait let me recalculate.
    # rollout: for t in range(n_steps=5): predict() → 5 calls + 1 final = 6 calls
    # CEM does n_steps=15 iterations, each calling get_cost which calls rollout
    # Total: 15 * 6 = 90 predict calls per planning step
    # But each call has batch_size = num_samples = 128
    calls_per_step = n_steps * 6  # 15 CEM iters × 6 predict calls per rollout
    results["predictor_single_ms"] = {
        "mean": np.mean(pred_times),
        "std": np.std(pred_times),
        "batch_size": BS,
        "calls_per_step": calls_per_step,
        "total_per_step_ms": np.mean(pred_times) * calls_per_step,
    }

    # ─── Action Encoder ──────────────────────────────────────
    print("Benchmarking action encoder...")
    for _ in range(n_warmup):
        model.action_encoder(dummy_actions)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]

    for i in range(n_trials):
        start_events[i].record()
        model.action_encoder(dummy_actions)
        end_events[i].record()
    torch.cuda.synchronize()

    act_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["action_encoder_ms"] = {
        "mean": np.mean(act_times),
        "std": np.std(act_times),
        "calls_per_step": calls_per_step,  # same as predictor
        "total_per_step_ms": np.mean(act_times) * calls_per_step,
    }

    # ─── Full get_cost (one CEM iteration) ───────────────────
    print("Benchmarking full get_cost (one CEM iteration)...")
    for _ in range(n_warmup):
        info_copy = {k: v.clone() for k, v in dummy_info.items()}
        model.get_cost(info_copy, dummy_candidates)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]

    for i in range(n_trials):
        info_copy = {k: v.clone() for k, v in dummy_info.items()}
        start_events[i].record()
        model.get_cost(info_copy, dummy_candidates)
        end_events[i].record()
    torch.cuda.synchronize()

    cost_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    results["get_cost_ms"] = {
        "mean": np.mean(cost_times),
        "std": np.std(cost_times),
        "calls_per_step": n_steps,  # 15 CEM iterations
        "total_per_step_ms": np.mean(cost_times) * n_steps,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Benchmark Latency")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=15)
    args = parser.parse_args()

    # Load model
    model = swm.policy.AutoCostModel(args.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f"Model loaded. Benchmarking CEM {args.num_samples}×{args.n_steps}...\n")

    results = benchmark_components(model, args.num_samples, args.n_steps)

    # Print summary
    print(f"\n{'='*60}")
    print(f"LATENCY BREAKDOWN (CEM {args.num_samples}×{args.n_steps})")
    print(f"{'='*60}")

    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Per-call: {data['mean']:.2f} ± {data['std']:.2f} ms")
        if "calls_per_step" in data:
            print(f"  Calls/step: {data['calls_per_step']}")
        if "total_per_step_ms" in data:
            print(f"  Total/step: {data['total_per_step_ms']:.1f} ms")

    # Estimate total
    encoder_total = results["encoder_ms"]["mean"]
    pred_total = results["predictor_single_ms"]["total_per_step_ms"]
    act_total = results["action_encoder_ms"]["total_per_step_ms"]
    cost_total = results["get_cost_ms"]["total_per_step_ms"]

    print(f"\n{'='*60}")
    print(f"ESTIMATED TOTAL PER PLANNING STEP:")
    print(f"  Encoder:        {encoder_total:6.1f} ms")
    print(f"  Predictor:      {pred_total:6.1f} ms (dominant)")
    print(f"  Action encoder: {act_total:6.1f} ms")
    print(f"  Full get_cost × {args.n_steps}: {cost_total:6.1f} ms (includes above + overhead)")
    print(f"{'='*60}")

    gate_pass = cost_total < 100
    print(f"\nPhase 5 gate (<100ms): {'ALREADY PASSING' if gate_pass else 'NEEDS OPTIMIZATION'} ({cost_total:.0f}ms)")


if __name__ == "__main__":
    main()
