#!/usr/bin/env python3
"""
S4: End-to-End S1.5 Demo Evaluation

Compares baseline planning (flat CEM, receding-horizon) vs S1.5 control loop
(confidence-based replanning + drift detection) on TwoRoom.

Runs two modes on the same episodes:
  1. Baseline: pipeline.plan() per step → execute (standard CEM eval)
  2. S1.5: full control loop with MockVLM replanning on low confidence/drift

Measures: success rate, replan frequency, drift events, mean confidence.

Usage (on-pod with GPU):
    python scripts/eval_s15_demo.py --config-name tworoom
    python scripts/eval_s15_demo.py --num-eval 50 --drift-threshold 0.5
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.pipeline import PlanningPipeline
from harness.s15_loop import S15ControlLoop, MockVLM, MockMotorPolicy


def load_eval_config(config_name):
    from hydra import compose, initialize_config_dir
    config_dir = str(Path("./config/eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg


def setup_eval_env(cfg, args):
    """Set up dataset, world env, action scaler, and episode sampling.
    Follows the pattern from eval_dream_tree.py."""
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    # Action scaler (same pattern as eval_dream_tree.py)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # World environment
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True, throw_on_missing=False)
    world_cfg["max_episode_steps"] = 2 * args.eval_budget
    world_cfg["num_envs"] = 1
    world = swm.World(**world_cfg, image_shape=(224, 224))

    # Sample episodes
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    unique_eps = np.unique(episode_idx)

    episode_len = []
    for ep_id in unique_eps:
        episode_len.append(int(np.max(step_idx[episode_idx == ep_id]) + 1))
    episode_len = np.array(episode_len)

    goal_offset = cfg.eval.goal_offset_steps
    max_start_idx = episode_len - goal_offset - 1
    max_start_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(unique_eps)}
    max_start_per_row = np.array(
        [max_start_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(valid_indices) - 1, size=args.num_eval, replace=False)
    sample_indices = np.sort(valid_indices[sample_indices])

    eval_episodes = dataset.get_row_data(sample_indices)[col_name]
    eval_start_steps = dataset.get_row_data(sample_indices)["step_idx"]

    return dataset, process, world, cfg, episode_idx, col_name, \
        eval_episodes, eval_start_steps, goal_offset


def reset_env_for_episode(world, cfg, dataset, episode_idx, col_name,
                          ep_id, start_step, goal_offset):
    """Reset environment for an episode. Returns (start_pixels, goal_pixels)."""
    ep_mask = episode_idx == ep_id
    ep_indices = np.where(ep_mask)[0]

    start_row = dataset.get_row_data(int(ep_indices[start_step]))
    goal_step = min(start_step + goal_offset, len(ep_indices) - 1)
    goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

    start_pixels = start_row["pixels"]
    goal_pixels = goal_row["pixels"]

    for pix_name in ("start_pixels", "goal_pixels"):
        pix = locals()[pix_name]
        if isinstance(pix, np.ndarray) and pix.dtype != np.uint8:
            pix = (pix * 255).astype(np.uint8) if pix.max() <= 1.0 else pix.astype(np.uint8)
            if pix_name == "start_pixels":
                start_pixels = pix
            else:
                goal_pixels = pix

    # Reset env and apply callables
    world.envs.reset()
    unwrapped_env = world.envs.envs[0].unwrapped

    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) \
        if cfg.eval.get("callables") else []
    for spec in callables:
        method_name = spec["method"]
        if not hasattr(unwrapped_env, method_name):
            continue
        method = getattr(unwrapped_env, method_name)
        prepared_args = {}
        for arg_name, arg_data in spec.get("args", {}).items():
            value_key = arg_data.get("value", None)
            if value_key is None:
                continue
            if value_key.startswith("goal_"):
                col = value_key[5:]
                if col in goal_row:
                    prepared_args[arg_name] = goal_row[col]
            else:
                if value_key in start_row:
                    prepared_args[arg_name] = start_row[value_key]
        if prepared_args:
            method(**prepared_args)

    return start_pixels, goal_pixels


def run_baseline_episode(pipeline, obs_image, goal_pixels, world, process, args):
    """Run a baseline episode (standard receding-horizon CEM, no S1.5 feedback)."""
    raw_action_dim = process["action"].scale_.shape[0] if "action" in process else 2
    action_block = pipeline._action_dim // raw_action_dim

    pipeline.set_goal(goal_pixels)
    obs = obs_image
    planning_times = []
    env_step = 0

    while env_step < args.eval_budget:
        t0 = time.perf_counter()
        result = pipeline.plan(obs)
        planning_times.append((time.perf_counter() - t0) * 1000)

        sub_actions = result.reshape(action_block, raw_action_dim)
        for sub_action in sub_actions:
            if env_step >= args.eval_budget:
                break
            if "action" in process:
                sub_action = process["action"].inverse_transform(
                    sub_action.reshape(1, -1)
                ).squeeze()
            try:
                obs_dict, reward, terminated, truncated, info = world.envs.step(
                    np.array([sub_action])
                )
                env_step += 1
                obs = world.envs.render()[0]

                if isinstance(terminated, (list, np.ndarray)):
                    terminated = bool(terminated[0])
                if isinstance(info, (list, tuple)):
                    step_info = info[0] if info else {}
                elif isinstance(info, dict):
                    step_info = info
                else:
                    step_info = {}

                if terminated or step_info.get("is_success", False):
                    return True, env_step, np.mean(planning_times)
            except Exception as e:
                print(f"  env error: {e}")
                break

    return False, env_step, np.mean(planning_times)


def run_s15_episode(pipeline, obs_image, goal_pixels, world, process, args):
    """Run an S1.5 episode (confidence + drift replanning)."""
    raw_action_dim = process["action"].scale_.shape[0] if "action" in process else 2
    action_block = pipeline._action_dim // raw_action_dim

    goal_tensor = pipeline.preprocess(goal_pixels)
    goal_emb = pipeline.encode(goal_tensor)

    vlm = MockVLM(goal_embedding=goal_emb, replan_strategy="same")
    motor = MockMotorPolicy()

    env_success = False

    def get_next_obs(action):
        nonlocal env_success
        sub_actions = action.reshape(action_block, raw_action_dim)
        for sub_action in sub_actions:
            if "action" in process:
                sub_action = process["action"].inverse_transform(
                    sub_action.reshape(1, -1)
                ).squeeze()
            obs_dict, reward, terminated, truncated, info = world.envs.step(
                np.array([sub_action])
            )
            if isinstance(terminated, (list, np.ndarray)):
                terminated = bool(terminated[0])
            if isinstance(info, (list, tuple)):
                step_info = info[0] if info else {}
            elif isinstance(info, dict):
                step_info = info
            else:
                step_info = {}
            if terminated or step_info.get("is_success", False):
                env_success = True

        obs = world.envs.render()[0]
        return obs

    loop = S15ControlLoop(
        pipeline=pipeline,
        vlm=vlm,
        motor=motor,
        drift_threshold=args.drift_threshold,
        drift_window=5,
        max_replans_per_episode=args.max_replans,
    )

    stats = loop.run_episode(
        initial_obs=obs_image,
        get_next_obs=get_next_obs,
        max_steps=args.eval_budget,
        success_fn=lambda obs: env_success,
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="S1.5 End-to-End Demo Eval")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-budget", type=int, default=100)
    parser.add_argument("--drift-threshold", type=float, default=0.5)
    parser.add_argument("--max-replans", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/s15_demo_eval.json")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    dataset, process, world, cfg, episode_idx, col_name, \
        eval_episodes, eval_start_steps, goal_offset = setup_eval_env(cfg=load_eval_config(args.config_name), args=args)

    # Build pipeline
    print("Building pipeline...")
    pipeline = PlanningPipeline(
        args.policy,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
    )
    pipeline.warmup()

    # Run baseline episodes
    print(f"\n=== Baseline (receding-horizon CEM) ===")
    baseline_results = []
    for i in range(args.num_eval):
        ep_id = int(eval_episodes[i])
        start_step = int(eval_start_steps[i])

        start_pix, goal_pix = reset_env_for_episode(
            world, cfg, dataset, episode_idx, col_name, ep_id, start_step, goal_offset
        )

        success, steps, mean_ms = run_baseline_episode(
            pipeline, start_pix, goal_pix, world, process, args
        )
        baseline_results.append({"success": success, "steps": steps, "mean_ms": float(mean_ms)})
        status = "OK" if success else "FAIL"
        print(f"  Episode {i+1}/{args.num_eval}: {status} ({steps} steps)")

    baseline_rate = sum(r["success"] for r in baseline_results) / args.num_eval * 100

    # Run S1.5 episodes (same episodes, re-reset)
    print(f"\n=== S1.5 (confidence + drift replanning) ===")
    s15_results = []
    for i in range(args.num_eval):
        ep_id = int(eval_episodes[i])
        start_step = int(eval_start_steps[i])

        start_pix, goal_pix = reset_env_for_episode(
            world, cfg, dataset, episode_idx, col_name, ep_id, start_step, goal_offset
        )

        stats = run_s15_episode(pipeline, start_pix, goal_pix, world, process, args)
        s15_results.append(stats)
        status = "OK" if stats.success else "FAIL"
        replans = f"R={stats.total_replans}" if stats.total_replans > 0 else ""
        print(f"  Episode {i+1}/{args.num_eval}: {status} ({stats.steps} steps) {replans}")

    s15_rate = sum(s.success for s in s15_results) / args.num_eval * 100

    # Summary
    print(f"\n=== Results ===")
    print(f"  Baseline: {baseline_rate:.0f}% ({sum(r['success'] for r in baseline_results)}/{args.num_eval})")
    print(f"  S1.5:     {s15_rate:.0f}% ({sum(s.success for s in s15_results)}/{args.num_eval})")
    print(f"  S1.5 replans (confidence): {sum(s.replans_confidence for s in s15_results)}")
    print(f"  S1.5 replans (drift):      {sum(s.replans_drift for s in s15_results)}")
    print(f"  S1.5 drift events:         {sum(s.drift_events for s in s15_results)}")
    print(f"  S1.5 mean confidence:      {np.mean([s.mean_confidence for s in s15_results]):.3f}")

    # Save results
    drift_mses = [s.mean_drift_mse for s in s15_results if s.drift_mses]
    output = {
        "baseline": {
            "success_rate": baseline_rate,
            "num_eval": args.num_eval,
            "results": baseline_results,
        },
        "s15": {
            "success_rate": s15_rate,
            "num_eval": args.num_eval,
            "total_replans_confidence": sum(s.replans_confidence for s in s15_results),
            "total_replans_drift": sum(s.replans_drift for s in s15_results),
            "total_drift_events": sum(s.drift_events for s in s15_results),
            "mean_confidence": float(np.mean([s.mean_confidence for s in s15_results])),
            "mean_planning_cost": float(np.mean([s.mean_planning_cost for s in s15_results])),
            "mean_drift_mse": float(np.mean(drift_mses)) if drift_mses else 0.0,
        },
        "config": {
            "policy": args.policy,
            "num_samples": args.num_samples,
            "n_steps": args.n_steps,
            "drift_threshold": args.drift_threshold,
            "max_replans": args.max_replans,
            "eval_budget": args.eval_budget,
            "seed": args.seed,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
