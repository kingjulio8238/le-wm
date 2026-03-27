#!/usr/bin/env python3
"""
N2: Generate (caption, target_embedding) training pairs for language projection.

For each sampled goal state from TwoRoom:
  1. Extract pos_target (x, y) → generate synthetic captions
  2. Encode goal_pixels through LeWM encoder → target embedding (192-dim)
  3. Encode captions through CLIP text encoder → CLIP features (512-dim)

Output: text_goal_pairs.pt containing:
  - clip_features: (N, 512) pre-computed CLIP text embeddings
  - target_embeddings: (N, 192) LeWM goal image embeddings
  - captions: list of N caption strings (for inspection)

Usage:
    export STABLEWM_HOME=/workspace/data
    export MUJOCO_GL=egl
    python scripts/generate_text_pairs.py --num-pairs 8000
"""

import argparse
import os
import random

os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import torch

from harness.language_encoder import LanguageEncoder
from harness.pipeline import PlanningPipeline

# Caption templates — varied phrasing for same (x, y)
TEMPLATES = [
    "navigate to position ({x:.2f}, {y:.2f})",
    "go to ({x:.2f}, {y:.2f})",
    "move to the target at ({x:.2f}, {y:.2f})",
    "reach position ({x:.2f}, {y:.2f})",
    "target location is ({x:.2f}, {y:.2f})",
    "move towards ({x:.2f}, {y:.2f})",
    "head to coordinates ({x:.2f}, {y:.2f})",
    "the goal is at ({x:.2f}, {y:.2f})",
]


def normalize_pos(x, y, grid_size=224.0):
    """Normalize pixel coordinates to [0, 1]."""
    return x / grid_size, y / grid_size


def generate_caption(x_norm, y_norm):
    """Generate a random caption for normalized (x, y)."""
    template = random.choice(TEMPLATES)
    return template.format(x=x_norm, y=y_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/data/tworoom.h5")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--num-pairs", type=int, default=8000)
    parser.add_argument("--output", default="/workspace/data/text_goal_pairs.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"

    # Load dataset
    print("Loading dataset...")
    with h5py.File(args.dataset, "r") as f:
        pos_target = f["pos_target"][:]  # (N, 2)
        pixels = f["pixels"]  # (N, 224, 224, 3) — lazy load
        ep_idx = f["ep_idx"][:]
        num_total = len(pos_target)

        # Sample unique goal states: one per episode at a random step
        unique_eps = np.unique(ep_idx)
        selected_indices = []
        for ep in unique_eps:
            ep_mask = np.where(ep_idx == ep)[0]
            # Pick a random step from the latter half of the episode (closer to goal)
            mid = len(ep_mask) // 2
            idx = np.random.choice(ep_mask[mid:])
            selected_indices.append(idx)

        # Subsample if needed
        if len(selected_indices) > args.num_pairs:
            selected_indices = np.random.choice(
                selected_indices, args.num_pairs, replace=False
            )
        selected_indices = sorted(selected_indices)

        print(f"Selected {len(selected_indices)} goal states from {len(unique_eps)} episodes")

        # Generate captions and load pixels
        captions = []
        goal_pixels_list = []
        for idx in selected_indices:
            x, y = pos_target[idx]
            x_norm, y_norm = normalize_pos(x, y)
            caption = generate_caption(x_norm, y_norm)
            captions.append(caption)
            goal_pixels_list.append(pixels[idx])

    goal_pixels_arr = np.array(goal_pixels_list)  # (N, 224, 224, 3)
    print(f"Generated {len(captions)} captions")
    print(f"Example captions: {captions[:3]}")

    # Encode goal images through LeWM
    print("\nEncoding goal images through LeWM...")
    pipeline = PlanningPipeline(
        policy_name=args.policy,
        compile_mode="default",
    )

    target_embeddings = []
    for i in range(0, len(goal_pixels_arr), args.batch_size):
        batch = goal_pixels_arr[i : i + args.batch_size]
        batch_embs = []
        for img in batch:
            img_tensor = pipeline.preprocess(img)
            emb = pipeline.encode(img_tensor)  # (1, 1, 192)
            batch_embs.append(emb.squeeze(0).squeeze(0))  # (192,)
        target_embeddings.extend(batch_embs)
        if (i // args.batch_size) % 10 == 0:
            print(f"  Encoded {min(i + args.batch_size, len(goal_pixels_arr))}/{len(goal_pixels_arr)}")

    target_embeddings = torch.stack(target_embeddings)  # (N, 192)
    print(f"Target embeddings: {target_embeddings.shape}")

    # Encode captions through CLIP
    print("\nEncoding captions through CLIP...")
    lang_enc = LanguageEncoder(projection_path=None, device=device)

    clip_features = []
    for i in range(0, len(captions), args.batch_size):
        batch_captions = captions[i : i + args.batch_size]
        feats = lang_enc.get_clip_features(batch_captions)  # (B, 512)
        clip_features.append(feats.cpu())
        if (i // args.batch_size) % 10 == 0:
            print(f"  Encoded {min(i + args.batch_size, len(captions))}/{len(captions)}")

    clip_features = torch.cat(clip_features)  # (N, 512)
    print(f"CLIP features: {clip_features.shape}")

    # Save
    torch.save(
        {
            "clip_features": clip_features,
            "target_embeddings": target_embeddings.cpu(),
            "captions": captions,
        },
        args.output,
    )
    print(f"\nSaved {len(captions)} pairs to {args.output}")


if __name__ == "__main__":
    main()
