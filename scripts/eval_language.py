#!/usr/bin/env python3
"""
N2: Evaluate language-conditioned planning vs image-conditioned planning.

Runs the same TwoRoom episodes with:
  1. Image goal (baseline): plan(obs, goal_image)
  2. Coord text goal: plan_from_text(obs, "navigate to (x, y)")
  3. CLIP text goal: plan_from_text(obs, "go to the upper left area")

Gate: text-conditioned success rate >= 80% of image-conditioned.

Usage:
    export STABLEWM_HOME=/workspace/data
    export MUJOCO_GL=egl
    python scripts/eval_language.py --policy tworoom/lewm --config-name tworoom
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing

from harness.pipeline import PlanningPipeline

# Import eval helpers from dream_tree script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_dream_tree import load_eval_config, setup_eval_env


def get_region_description(x_norm, y_norm):
    """Same as training - maps (x,y) to spatial description."""
    if x_norm < 0.25:
        h = "far left"
    elif x_norm < 0.45:
        h = "left"
    elif x_norm < 0.55:
        h = "center"
    elif x_norm < 0.75:
        h = "right"
    else:
        h = "far right"

    if y_norm < 0.25:
        v = "top"
    elif y_norm < 0.45:
        v = "upper"
    elif y_norm < 0.55:
        v = "middle"
    elif y_norm < 0.75:
        v = "lower"
    else:
        v = "bottom"

    return f"go to the {v} {h} area"


def run_eval(args):
    print(f"\n{'='*60}")
    print(f"N2: Language Conditioning Evaluation")
    print(f"{'='*60}")

    cfg = load_eval_config(args.config_name)

    # Build pipeline
    print("\nLoading pipeline...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        num_samples=128,
        n_steps=15,
        horizon=5,
        topk=25,
        compile_mode="default",
    )
    pipeline.warmup()

    # Set up eval environment
    (dataset, process, world, episode_idx, col_name,
     eval_episodes, eval_start_steps, goal_offset) = setup_eval_env(cfg, args)

    # Also load position data for generating text goals
    import h5py
    cache_dir = Path(swm.data.utils.get_cache_dir())
    h5_path = cache_dir / f"{cfg.eval.dataset_name}.h5"
    with h5py.File(h5_path, "r") as f:
        pos_target_all = f["pos_target"][:]

    # Action processing
    raw_action_dim = process["action"].scale_.shape[0]

    results = {}

    # --- Mode loop ---
    modes_to_run = args.modes.split(",")

    for mode in modes_to_run:
        mode = mode.strip()
        print(f"\n--- Mode: {mode} ---")

        # Load language encoder for text modes
        if mode == "coord":
            pipeline.load_language_encoder(args.projection_path, mode="coord")
        elif mode == "clip":
            pipeline.load_language_encoder(args.projection_path, mode="clip")

        successes = []
        planning_times = []

        for ep_i in range(args.num_eval):
            ep_id = int(eval_episodes[ep_i])
            start_step = int(eval_start_steps[ep_i])

            ep_mask = episode_idx == ep_id
            ep_indices = np.where(ep_mask)[0]

            start_row = dataset.get_row_data(int(ep_indices[start_step]))
            goal_step = min(start_step + goal_offset, len(ep_indices) - 1)
            goal_row = dataset.get_row_data(int(ep_indices[goal_step]))

            start_pixels = start_row["pixels"]
            goal_pixels = goal_row["pixels"]

            if isinstance(start_pixels, np.ndarray) and start_pixels.dtype != np.uint8:
                start_pixels = (start_pixels * 255).astype(np.uint8) if start_pixels.max() <= 1.0 else start_pixels.astype(np.uint8)
            if isinstance(goal_pixels, np.ndarray) and goal_pixels.dtype != np.uint8:
                goal_pixels = (goal_pixels * 255).astype(np.uint8) if goal_pixels.max() <= 1.0 else goal_pixels.astype(np.uint8)

            # Get goal position for text modes
            goal_global_idx = int(ep_indices[goal_step])
            goal_pos = pos_target_all[goal_global_idx]
            x_norm, y_norm = goal_pos[0] / 224.0, goal_pos[1] / 224.0

            # Generate text goal
            if mode == "coord":
                goal_text = f"navigate to ({x_norm:.2f}, {y_norm:.2f})"
            elif mode == "clip":
                goal_text = get_region_description(x_norm, y_norm)
            else:
                goal_text = None

            # Reset environment
            world.envs.reset()
            unwrapped_env = world.envs.envs[0].unwrapped

            # Apply callables
            callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) if cfg.eval.get("callables") else []
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

            episode_success = False
            obs_image = start_pixels
            action_block = pipeline._action_dim // raw_action_dim

            env_step = 0
            while env_step < args.eval_budget:
                t0 = time.perf_counter()

                if mode == "image":
                    raw_action = pipeline.plan(obs_image, goal_pixels)
                else:
                    raw_action = pipeline.plan_from_text(obs_image, goal_text)

                planning_times.append((time.perf_counter() - t0) * 1000)

                sub_actions = raw_action.reshape(action_block, raw_action_dim)
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
                        obs_image = world.envs.render()[0]

                        if isinstance(terminated, (list, np.ndarray)):
                            terminated = bool(terminated[0])
                        if isinstance(info, (list, tuple)):
                            step_info = info[0] if info else {}
                        elif isinstance(info, dict):
                            step_info = info
                        else:
                            step_info = {}

                        if terminated or step_info.get("is_success", False):
                            episode_success = True
                            break
                    except Exception as e:
                        print(f"  Episode {ep_i} step {env_step}: env error: {e}")
                        break

                if episode_success:
                    break

            successes.append(episode_success)

            if (ep_i + 1) % 10 == 0 or ep_i == 0:
                sr = np.mean(successes) * 100
                mean_ms = np.mean(planning_times[-args.eval_budget:]) if planning_times else 0
                print(f"  [{mode}] Episode {ep_i+1}/{args.num_eval}: "
                      f"success_rate={sr:.1f}%, mean_planning={mean_ms:.0f}ms")

        sr = np.mean(successes) * 100
        mean_ms = np.mean(planning_times) if planning_times else 0
        results[mode] = {
            "success_rate": float(sr),
            "num_successes": int(sum(successes)),
            "num_eval": args.num_eval,
            "mean_planning_ms": float(mean_ms),
        }
        print(f"\n  {mode}: {sr:.1f}% ({int(sum(successes))}/{args.num_eval})")

    # Summary
    print(f"\n{'='*60}")
    print(f"N2 EVALUATION SUMMARY")
    print(f"{'='*60}")
    for mode, r in results.items():
        print(f"  {mode:8s}: {r['success_rate']:.1f}% ({r['num_successes']}/{r['num_eval']})")

    if "image" in results and any(m in results for m in ("coord", "clip")):
        image_sr = results["image"]["success_rate"]
        for text_mode in ("coord", "clip"):
            if text_mode in results:
                text_sr = results[text_mode]["success_rate"]
                ratio = text_sr / image_sr * 100 if image_sr > 0 else 0
                gate = "PASS" if ratio >= 80 else "FAIL"
                print(f"\n  {text_mode} vs image: {ratio:.0f}% — GATE: {gate}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "n2_language_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--config-name", default="tworoom")
    parser.add_argument("--projection-path", default="/workspace/data/language_projection_v3.pt")
    parser.add_argument("--modes", default="image,coord,clip",
                        help="Comma-separated modes to evaluate")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/workspace/data/results")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
