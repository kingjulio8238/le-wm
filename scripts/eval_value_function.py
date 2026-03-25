#!/usr/bin/env python3
"""
Phase 4: Evaluate value function cost vs MSE baseline.

Runs CEM planning with the learned value function as cost model
and compares against the default MSE embedding distance.

Usage:
    export STABLEWM_HOME=/workspace/data
    cd /workspace/le-harness
    python scripts/eval_value_function.py \
        --checkpoint /workspace/data/checkpoints/value_ensemble.pt \
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
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from harness.value_cost import ValueCostModel
from harness.value_function import ValueEnsemble


def run_eval(cfg, model, policy_name, num_eval, seed):
    """Run standard eval and return metrics + time."""
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

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)

    wm_policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

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

    g = np.random.default_rng(seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(wm_policy)

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

    return metrics, elapsed


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Value Function Eval")
    parser.add_argument("--checkpoint", default="/workspace/data/checkpoints/value_ensemble.pt")
    parser.add_argument("--policy", default="pusht/lejepa")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="pusht", overrides=[
            f"policy={args.policy}",
            "solver=cem",
            f"solver.num_samples={args.num_samples}",
            f"solver.n_steps={args.n_steps}",
            f"solver.topk={max(1, min(args.num_samples // 5, 30))}",
            f"eval.num_eval={args.num_eval}",
            f"seed={args.seed}",
        ])

    # Load base model
    print(f"Loading base model: {args.policy}")
    base_model = swm.policy.AutoCostModel(cfg.policy)
    base_model = base_model.to("cuda").eval()
    base_model.requires_grad_(False)
    base_model.interpolate_pos_encoding = True

    # Load value ensemble
    print(f"Loading value ensemble: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=True)
    ensemble = ValueEnsemble(**ckpt["config"])
    ensemble.load_state_dict(ckpt["state_dict"])
    ensemble = ensemble.to("cuda").eval()
    print(f"  {ensemble.param_count():,} params, {ckpt['config']['n_members']} members")

    # Create value cost model
    value_model = ValueCostModel(base_model, ensemble)

    # ─── Run MSE baseline ────────────────────────────────────────
    print(f"\n--- MSE Baseline (CEM {args.num_samples}×{args.n_steps}) ---")
    mse_metrics, mse_time = run_eval(cfg, base_model, args.policy, args.num_eval, args.seed)
    print(f"  Success rate: {mse_metrics['success_rate']}%")
    print(f"  Time: {mse_time:.0f}s")

    # ─── Run Value Function ──────────────────────────────────────
    print(f"\n--- Value Function (CEM {args.num_samples}×{args.n_steps}) ---")
    val_metrics, val_time = run_eval(cfg, value_model, args.policy, args.num_eval, args.seed)
    print(f"  Success rate: {val_metrics['success_rate']}%")
    print(f"  Time: {val_time:.0f}s")

    # ─── Also test at half budget ────────────────────────────────
    half_steps = max(3, args.n_steps // 2)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg_half = compose(config_name="pusht", overrides=[
            f"policy={args.policy}",
            "solver=cem",
            f"solver.num_samples={args.num_samples}",
            f"solver.n_steps={half_steps}",
            f"solver.topk={max(1, min(args.num_samples // 5, 30))}",
            f"eval.num_eval={args.num_eval}",
            f"seed={args.seed}",
        ])

    print(f"\n--- Value Function at half budget (CEM {args.num_samples}×{half_steps}) ---")
    val_half_metrics, val_half_time = run_eval(cfg_half, value_model, args.policy, args.num_eval, args.seed)
    print(f"  Success rate: {val_half_metrics['success_rate']}%")
    print(f"  Time: {val_half_time:.0f}s")

    # ─── Summary ─────────────────────────────────────────────────
    fp_full = args.num_samples * args.n_steps * 5
    fp_half = args.num_samples * half_steps * 5

    print(f"\n{'='*60}")
    print(f"PHASE 4 COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<30s} {'FP':>8s} {'SR%':>6s}")
    print(f"{'-'*50}")
    print(f"{'MSE baseline':<30s} {fp_full:>8,} {mse_metrics['success_rate']:>6.1f}")
    print(f"{'Value function':<30s} {fp_full:>8,} {val_metrics['success_rate']:>6.1f}")
    print(f"{'Value function (half budget)':<30s} {fp_half:>8,} {val_half_metrics['success_rate']:>6.1f}")

    # Gate check
    sr_mse = mse_metrics['success_rate']
    sr_val = val_metrics['success_rate']
    sr_val_half = val_half_metrics['success_rate']

    higher_at_equal = sr_val > sr_mse
    equal_at_half = sr_val_half >= sr_mse - 2.0

    print(f"\n{'='*60}")
    if higher_at_equal:
        print(f"PHASE 4 GATE: PASS (higher SR at equal budget: {sr_val}% vs {sr_mse}%)")
    elif equal_at_half:
        print(f"PHASE 4 GATE: PASS (equal SR at half budget: {sr_val_half}% vs {sr_mse}%)")
    else:
        print(f"PHASE 4 GATE: FAIL")
        print(f"  Value at equal budget: {sr_val}% vs MSE {sr_mse}%")
        print(f"  Value at half budget: {sr_val_half}% vs MSE {sr_mse}%")
    print(f"{'='*60}")

    # Save results
    out_path = Path(args.output_dir) / "phase4_value_function.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "mse_baseline": {"success_rate": sr_mse, "forward_passes": fp_full},
            "value_function": {"success_rate": sr_val, "forward_passes": fp_full},
            "value_half_budget": {"success_rate": sr_val_half, "forward_passes": fp_half},
            "gate_pass": higher_at_equal or equal_at_half,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
