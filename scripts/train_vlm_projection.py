#!/usr/bin/env python3
"""
Train a VLM→LeWM projection MLP.

Generic script that works for any VLM family (SigLIP, T5, Eagle, PaliGemma).
Trains a VLMProjection MLP to minimize MSE between projected VLM embeddings
and LeWM ViT goal embeddings.

Data format: a .pt file with:
    - "vlm_features": (N, in_dim) tensor of VLM encoder outputs
    - "target_embeddings": (N, 192) tensor of LeWM ViT encoder outputs
Both computed from the same goal images.

Usage:
    # Train SigLIP projection (768 → 192)
    python scripts/train_vlm_projection.py --source siglip --data pairs.pt

    # Train Eagle projection (1536 → 192)
    python scripts/train_vlm_projection.py --source eagle --data pairs.pt --epochs 200

    # Custom dimensions
    python scripts/train_vlm_projection.py --in-dim 1024 --data pairs.pt
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add parent to path for harness imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.projections import VLMProjection

# Known VLM embedding dimensions
VLM_DIMS = {
    "clip": 512,
    "siglip": 768,
    "t5": 768,
    "eagle": 1536,
    "paligemma": 4608,
}


def main():
    parser = argparse.ArgumentParser(description="Train VLM→LeWM projection")
    parser.add_argument("--source", type=str, default=None,
                        help=f"VLM family name: {list(VLM_DIMS.keys())}")
    parser.add_argument("--in-dim", type=int, default=None,
                        help="Input dim (overrides --source lookup)")
    parser.add_argument("--data", required=True,
                        help="Path to .pt file with vlm_features and target_embeddings")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <source>_projection.pt)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Hidden dim for MLP (default: auto based on in_dim)")
    args = parser.parse_args()

    # Determine input dimension
    if args.in_dim is not None:
        in_dim = args.in_dim
    elif args.source is not None:
        if args.source not in VLM_DIMS:
            parser.error(f"Unknown source '{args.source}'. Known: {list(VLM_DIMS.keys())}")
        in_dim = VLM_DIMS[args.source]
    else:
        parser.error("Must specify --source or --in-dim")

    # Output path
    if args.output is None:
        name = args.source or f"vlm{in_dim}"
        args.output = f"{name}_projection.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-computed features
    data = torch.load(args.data, map_location=device, weights_only=True)
    vlm_features = data["vlm_features"].float().to(device)
    target_embeddings = data["target_embeddings"].float().to(device)

    assert vlm_features.shape[0] == target_embeddings.shape[0], \
        f"Mismatched: {vlm_features.shape[0]} vs {target_embeddings.shape[0]} samples"
    assert vlm_features.shape[1] == in_dim, \
        f"VLM features dim {vlm_features.shape[1]} != expected {in_dim}"
    assert target_embeddings.shape[1] == 192, \
        f"Target embeddings dim {target_embeddings.shape[1]} != 192"

    print(f"Source: {args.source or 'custom'} (in_dim={in_dim})")
    print(f"Data: {vlm_features.shape[0]} pairs")
    print(f"  VLM features: {vlm_features.shape}")
    print(f"  Target embeddings: {target_embeddings.shape}")

    # Train/val split
    dataset = TensorDataset(vlm_features, target_embeddings)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Model: VLMProjection MLP
    hidden_dim = args.hidden_dim or (1024 if in_dim > 1024 else 512)
    projection = VLMProjection(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    n_params = sum(p.numel() for p in projection.parameters())
    print(f"  Projection: {n_params:,} params (hidden_dim={hidden_dim})")

    optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        # Train
        projection.train()
        train_loss = 0.0
        for vlm_batch, target_batch in train_loader:
            pred = projection(vlm_batch)
            loss = criterion(pred, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(vlm_batch)
        train_loss /= train_size
        scheduler.step()

        # Validate
        projection.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vlm_batch, target_batch in val_loader:
                pred = projection(vlm_batch)
                loss = criterion(pred, target_batch)
                val_loss += loss.item() * len(vlm_batch)
        val_loss /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in projection.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}  val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # Save with metadata
    save_dict = {
        "state_dict": best_state,
        "source": args.source,
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "out_dim": 192,
        "val_loss": best_val_loss,
    }
    torch.save(save_dict, args.output)
    print(f"Saved projection to {args.output}")

    # Quality check
    projection.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    projection.eval()
    with torch.no_grad():
        all_pred = projection(vlm_features)
        cos_sim = nn.functional.cosine_similarity(all_pred, target_embeddings, dim=1)
        mse = criterion(all_pred, target_embeddings)
        print(f"\nQuality check (all data):")
        print(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
        print(f"  MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
