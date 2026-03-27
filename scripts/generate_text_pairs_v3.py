#!/usr/bin/env python3
"""
N2 v3: Generate training pairs using CORRECT goal images.

Key fix: Use frames where the agent HAS REACHED the target (terminated=True),
so the image actually shows what the target location looks like.

Previous versions used mid-episode frames where the agent was ~78px from the
target — the image didn't match the caption at all.
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


def get_region_description(x_norm, y_norm):
    """Generate natural language description of position."""
    # Horizontal
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

    # Vertical (y=0 is top)
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

    return h, v


def get_compass(x_norm, y_norm):
    if x_norm < 0.33:
        ew = "west"
    elif x_norm < 0.67:
        ew = ""
    else:
        ew = "east"
    if y_norm < 0.33:
        ns = "north"
    elif y_norm < 0.67:
        ns = ""
    else:
        ns = "south"
    if ns and ew:
        return f"{ns}{ew}"
    return ns or ew or "center"


TEMPLATES = [
    lambda h, v, c: f"go to the {v} {h} area",
    lambda h, v, c: f"navigate to the {v} {h} region",
    lambda h, v, c: f"move to the {c} part of the room",
    lambda h, v, c: f"the target is in the {v} {h} area",
    lambda h, v, c: f"reach the {c} area",
    lambda h, v, c: f"head {c}",
    lambda h, v, c: f"go {c} to the target",
    lambda h, v, c: f"the goal is {c}",
    lambda h, v, c: f"move towards the {c} section",
    lambda h, v, c: f"navigate {c} to find the target",
    lambda h, v, c: f"the destination is in the {v} {h} zone",
    lambda h, v, c: f"travel to the {v} {h} zone",
]


def generate_caption(x_norm, y_norm):
    h, v = get_region_description(x_norm, y_norm)
    compass = get_compass(x_norm, y_norm)
    template = random.choice(TEMPLATES)
    return template(h, v, compass)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/data/tworoom.h5")
    parser.add_argument("--policy", default="tworoom/lewm")
    parser.add_argument("--output", default="/workspace/data/text_goal_pairs_v3.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda"

    print("Loading dataset...")
    with h5py.File(args.dataset, "r") as f:
        pos_agent = f["pos_agent"][:]
        pos_target = f["pos_target"][:]
        terminated = f["terminated"][:]
        ep_idx = f["ep_idx"][:]
        pixels = f["pixels"]

        # Strategy: use TERMINAL frames where agent ≈ target
        # These show what the target location actually looks like
        term_indices = np.where(terminated)[0]
        print(f"Terminated frames: {len(term_indices)}")

        # For each terminal frame, the image shows agent at target
        # Use pos_target for the caption (agent is there now)
        captions = []
        goal_pixels_list = []
        positions = []

        for idx in term_indices:
            target = pos_target[idx]
            x_norm, y_norm = target[0] / 224.0, target[1] / 224.0
            caption = generate_caption(x_norm, y_norm)
            captions.append(caption)
            goal_pixels_list.append(pixels[idx])
            positions.append([x_norm, y_norm])

        # Also augment with near-terminal frames (agent close to target)
        # to increase dataset size and diversity
        close_mask = np.where(
            (~terminated) & (f["distance_to_target"][:] < 20)
        )[0]
        print(f"Near-target frames (<20px): {len(close_mask)}")

        # Sample some close frames
        n_close = min(len(close_mask), 4000)
        close_sample = np.random.choice(close_mask, n_close, replace=False)

        for idx in close_sample:
            target = pos_target[idx]
            x_norm, y_norm = target[0] / 224.0, target[1] / 224.0
            caption = generate_caption(x_norm, y_norm)
            captions.append(caption)
            goal_pixels_list.append(pixels[idx])
            positions.append([x_norm, y_norm])

    goal_pixels_arr = np.array(goal_pixels_list)
    positions = np.array(positions)

    print(f"\nTotal pairs: {len(captions)}")
    print(f"Example captions: {captions[:5]}")

    # Encode goal images through LeWM
    print("\nEncoding goal images through LeWM...")
    pipeline = PlanningPipeline(policy_name=args.policy, compile_mode="default")

    target_embeddings = []
    for i in range(0, len(goal_pixels_arr), args.batch_size):
        batch = goal_pixels_arr[i : i + args.batch_size]
        for img in batch:
            img_tensor = pipeline.preprocess(img)
            emb = pipeline.encode(img_tensor)
            target_embeddings.append(emb.squeeze(0).squeeze(0))
        if (i // args.batch_size) % 20 == 0:
            print(f"  Encoded {min(i + args.batch_size, len(goal_pixels_arr))}/{len(goal_pixels_arr)}")

    target_embeddings = torch.stack(target_embeddings)
    print(f"Target embeddings: {target_embeddings.shape}")

    # Check: do nearby positions now have similar embeddings?
    pos_t = torch.tensor(positions, dtype=torch.float32)
    tgt_t = target_embeddings.cpu()

    # Find pairs of points within 0.1 distance
    n_check = min(2000, len(pos_t))
    dists = torch.cdist(pos_t[:n_check], pos_t[:n_check])
    close_pairs = (dists < 0.05) & (dists > 0)
    if close_pairs.any():
        emb_cos = torch.nn.functional.cosine_similarity(
            tgt_t[:n_check].unsqueeze(1), tgt_t[:n_check].unsqueeze(0), dim=2
        )
        close_cos = emb_cos[close_pairs].mean()
        all_cos = emb_cos[~torch.eye(n_check, dtype=bool)].mean()
        print(f"\nEmbedding cosine sim for nearby positions (<0.05): {close_cos:.4f}")
        print(f"Embedding cosine sim overall: {all_cos:.4f}")
        print(f"Nearby positions have {'MORE' if close_cos > all_cos else 'LESS'} similar embeddings")

    # Encode captions through CLIP
    print("\nEncoding captions through CLIP...")
    lang_enc = LanguageEncoder(projection_path=None, device=device)

    clip_features = []
    for i in range(0, len(captions), args.batch_size):
        batch_captions = captions[i : i + args.batch_size]
        feats = lang_enc.get_clip_features(batch_captions)
        clip_features.append(feats.cpu())
    clip_features = torch.cat(clip_features)
    print(f"CLIP features: {clip_features.shape}")

    # Save everything
    torch.save(
        {
            "clip_features": clip_features,
            "target_embeddings": target_embeddings.cpu(),
            "captions": captions,
            "positions": positions,
        },
        args.output,
    )
    print(f"\nSaved {len(captions)} pairs to {args.output}")


if __name__ == "__main__":
    main()
