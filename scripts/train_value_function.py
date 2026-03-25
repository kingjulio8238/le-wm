#!/usr/bin/env python3
"""
Phase 4: Train and save the value function ensemble.

Usage:
    python scripts/train_value_function.py \
        --data /workspace/data/value_train_data.pt \
        --output /workspace/data/checkpoints/value_ensemble.pt
"""

import argparse
import json
from pathlib import Path

import torch

from harness.value_function import ValueEnsemble, train_ensemble


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train Value Function Ensemble")
    parser.add_argument("--data", default="/workspace/data/value_train_data.pt")
    parser.add_argument("--output", default="/workspace/data/checkpoints/value_ensemble.pt")
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}")
    data = torch.load(args.data, weights_only=True)
    z_t = data["z_t"]
    z_goal = data["z_goal"]
    progress = data["progress"]
    embed_dim = data["embed_dim"]

    print(f"  Samples: {len(z_t)}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Progress: mean={progress.mean():.3f} std={progress.std():.3f}")

    # Create ensemble
    ensemble = ValueEnsemble(
        n_members=args.n_members,
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
    )
    print(f"  Ensemble: {args.n_members} members, {ensemble.param_count():,} params total")

    # Train
    history = train_ensemble(
        ensemble, z_t, z_goal, progress,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cuda",
        verbose=True,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": ensemble.state_dict(),
        "config": {
            "n_members": args.n_members,
            "embed_dim": embed_dim,
            "hidden_dim": args.hidden_dim,
        },
        "history": {
            "final_val_loss": [losses[-1] for losses in history["val_loss"]],
        },
    }, output_path)

    print(f"\nSaved ensemble to {output_path}")
    print(f"  Final val losses: {[f'{l:.4f}' for l in [losses[-1] for losses in history['val_loss']]]}")


if __name__ == "__main__":
    main()
