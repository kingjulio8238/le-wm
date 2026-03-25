#!/usr/bin/env python3
"""
Phase 3: Evaluate adaptive early stopping.

Runs eval with AdaptiveCEMSolver wrapping the base CEM solver.
Reports success rate, mean iterations used, and iteration reduction.

Usage:
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness
    python scripts/eval_adaptive.py \
        --solver cem --num-samples 128 --n-steps 15 \
        --epsilon 0.10 --min-steps 3 --patience 1 \
        --policy pusht/lejepa --num-eval 50
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from harness.adaptive_solver import AdaptiveCEMSolver


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Adaptive Stopping Eval")
    parser.add_argument("--solver", default="cem", choices=["cem", "icem"])
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    # Load config via Hydra compose
    from hydra import compose, initialize_config_dir

    config_dir = str(Path("./config/eval").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        overrides = [
            f"policy={args.policy}",
            f"solver={args.solver}",
            f"solver.num_samples={args.num_samples}",
            f"solver.n_steps={args.n_steps}",
            f"solver.topk={max(1, min(args.num_samples // 5, 30))}",
            f"eval.num_eval={args.num_eval}",
            f"seed={args.seed}",
        ]
        if args.solver == "icem":
            n_elite_keep = max(1, min(5, args.num_samples // 10))
            overrides.append(f"solver.n_elite_keep={n_elite_keep}")

        cfg = compose(config_name="pusht", overrides=overrides)

    # Load model
    print(f"Loading model: {args.policy}")
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # Set up transforms and dataset
    transform = {
        "pixels": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]),
        "goal": transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]),
    }

    dataset_path = Path(cfg.get("cache_dir", None) or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # Create base solver and wrap with adaptive stopping
    config = swm.PlanConfig(**cfg.plan_config)
    base_solver = hydra.utils.instantiate(cfg.solver, model=model)
    adaptive_solver = AdaptiveCEMSolver(
        base_solver,
        epsilon=args.epsilon,
        min_steps=args.min_steps,
        patience=args.patience,
    )

    print(f"Config: {args.solver} samples={args.num_samples} iters={args.n_steps}")
    print(f"Adaptive: epsilon={args.epsilon} min_steps={args.min_steps} patience={args.patience}")

    wm_policy = swm.policy.WorldModelPolicy(
        solver=adaptive_solver, config=config, process=process, transform=transform
    )

    # Create world
    cfg_world = OmegaConf.to_container(cfg.world, resolve=True, throw_on_missing=False)
    cfg_world["max_episode_steps"] = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg_world, image_shape=(224, 224))

    # Sample episodes
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_indices = np.unique(episode_idx)

    episode_len = []
    for ep_id in ep_indices:
        episode_len.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    episode_len = np.array(episode_len)

    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(args.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=args.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(wm_policy)

    # Run eval
    print(f"\nRunning {args.num_eval} episodes...")
    start = time.time()

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
    )

    elapsed = time.time() - start
    summary = adaptive_solver.get_summary()

    # Print results
    print(f"\n{'='*60}")
    print(f"PHASE 3: Adaptive Stopping Results")
    print(f"{'='*60}")
    print(f"Success rate: {metrics['success_rate']}%")
    print(f"Eval time: {elapsed:.0f}s")
    print(f"")
    print(f"Adaptive stopping stats:")
    print(f"  Max iterations (configured): {summary['max_iterations']}")
    print(f"  Mean iterations used: {summary['mean_iterations']:.1f}")
    print(f"  Median iterations: {summary['median_iterations']:.0f}")
    print(f"  P95 iterations: {summary['p95_iterations']:.0f}")
    print(f"  Early stop rate: {summary['early_stop_rate']*100:.0f}%")
    print(f"  Iteration reduction: {summary['iteration_reduction']*100:.1f}%")

    # Compute effective forward passes
    fp_per_iter = args.num_samples * 5  # samples × horizon
    effective_fp = summary['mean_iterations'] * fp_per_iter
    baseline_fp = args.n_steps * fp_per_iter

    print(f"")
    print(f"Forward passes:")
    print(f"  Fixed budget: {baseline_fp:,.0f}")
    print(f"  Adaptive mean: {effective_fp:,.0f}")
    print(f"  Reduction: {(1 - effective_fp/baseline_fp)*100:.1f}%")

    # Gate check
    fixed_sr = 92.0  # Phase 2 best for CEM 128×15
    sr = metrics['success_rate']
    iter_reduction = summary['iteration_reduction']

    print(f"\n{'='*60}")
    sr_pass = sr >= fixed_sr - 2.0
    iter_pass = iter_reduction >= 0.30
    gate_pass = sr_pass and iter_pass

    print(f"PHASE 3 GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(f"  SR: {sr}% {'≥' if sr_pass else '<'} {fixed_sr - 2.0}% (fixed - 2%) {'PASS' if sr_pass else 'FAIL'}")
    print(f"  Iter reduction: {iter_reduction*100:.1f}% {'≥' if iter_pass else '<'} 30% {'PASS' if iter_pass else 'FAIL'}")
    print(f"{'='*60}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase3_adaptive_eps{args.epsilon}.json"

    result = {
        "config": {
            "solver": args.solver,
            "num_samples": args.num_samples,
            "n_steps": args.n_steps,
            "epsilon": args.epsilon,
            "min_steps": args.min_steps,
            "patience": args.patience,
        },
        "metrics": {
            "success_rate": float(sr),
            "eval_time_s": round(elapsed, 1),
        },
        "adaptive_stats": summary,
        "effective_forward_passes": effective_fp,
        "gate": {
            "sr_pass": sr_pass,
            "iter_pass": iter_pass,
            "gate_pass": gate_pass,
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
