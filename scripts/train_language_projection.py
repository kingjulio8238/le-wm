#!/usr/bin/env python3
"""
N2: Train the CLIP→LeWM projection layer.

Trains a Linear(512→192) to minimize MSE between projected CLIP text
features and LeWM image embeddings.

Usage:
    python scripts/train_language_projection.py
    python scripts/train_language_projection.py --epochs 100 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/data/text_goal_pairs.pt")
    parser.add_argument("--output", default="/workspace/data/language_projection.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda"

    # Load pre-computed features
    data = torch.load(args.data, weights_only=True)
    clip_features = data["clip_features"].float().to(device)      # (N, 512)
    target_embeddings = data["target_embeddings"].float().to(device)  # (N, 192)

    print(f"Data: {clip_features.shape[0]} pairs")
    print(f"  CLIP features: {clip_features.shape}")
    print(f"  Target embeddings: {target_embeddings.shape}")

    # Train/val split
    dataset = TensorDataset(clip_features, target_embeddings)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Model: single linear layer
    projection = nn.Linear(512, 192).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        # Train
        projection.train()
        train_loss = 0.0
        for clip_batch, target_batch in train_loader:
            pred = projection(clip_batch)
            loss = criterion(pred, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(clip_batch)
        train_loss /= train_size

        # Validate
        projection.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clip_batch, target_batch in val_loader:
                pred = projection(clip_batch)
                loss = criterion(pred, target_batch)
                val_loss += loss.item() * len(clip_batch)
        val_loss /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in projection.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # Save best model
    torch.save(best_state, args.output)
    print(f"Saved projection to {args.output}")

    # Quick quality check: cosine similarity between projected and target
    projection.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    projection.eval()
    with torch.no_grad():
        all_pred = projection(clip_features)
        cos_sim = nn.functional.cosine_similarity(all_pred, target_embeddings, dim=1)
        print(f"\nQuality check (all data):")
        print(f"  Cosine similarity: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")
        print(f"  MSE: {criterion(all_pred, target_embeddings):.6f}")


if __name__ == "__main__":
    main()
