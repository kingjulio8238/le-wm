"""
N2: Language Conditioning for LeHarness

Maps natural language goal descriptions to LeWM's 192-dim embedding space.

Two modes:
  1. CLIP mode: text → CLIP text encoder → MLP → (1, 1, 192)
  2. Coordinate mode: text with coords → parse (x,y) → MLP → (1, 1, 192)

The coordinate mode is more precise (cos_sim=0.61 vs 0.44) because CLIP
doesn't encode spatial coordinates well. In practice, use coordinate mode
for precise goals and CLIP mode for qualitative commands.

Usage:
    # Coordinate mode (precise)
    enc = LanguageEncoder.from_coordinates(projection_path="coord_projection.pt")
    goal_emb = enc.encode_text("navigate to (0.43, 0.57)")  # (1, 1, 192)

    # CLIP mode (qualitative)
    enc = LanguageEncoder.from_clip(projection_path="clip_projection.pt")
    goal_emb = enc.encode_text("go to the upper left area")  # (1, 1, 192)
"""

import re

import torch
import torch.nn as nn
import open_clip


class CoordProjection(nn.Module):
    """MLP mapping (x, y) coordinates to LeWM embedding space."""

    def __init__(self, hidden_dim=256, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPProjection(nn.Module):
    """MLP mapping CLIP text features to LeWM embedding space."""

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _parse_coordinates(text: str) -> tuple[float, float] | None:
    """Extract (x, y) from text like 'navigate to (0.43, 0.57)'."""
    match = re.search(r"\((\d+\.?\d*),\s*(\d+\.?\d*)\)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


class LanguageEncoder(nn.Module):
    """Language encoder with CLIP and/or coordinate projection."""

    def __init__(
        self,
        mode: str = "clip",
        projection_path: str | None = None,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.mode = mode

        if mode in ("clip", "both"):
            model, _, _ = open_clip.create_model_and_transforms(
                clip_model, pretrained=clip_pretrained
            )
            self.clip = model.eval()
            self.clip.requires_grad_(False)
            self.tokenizer = open_clip.get_tokenizer(clip_model)
            self.clip_projection = CLIPProjection()
        else:
            self.clip = None
            self.tokenizer = None
            self.clip_projection = None

        if mode in ("coord", "both"):
            self.coord_projection = CoordProjection()
        else:
            self.coord_projection = None

        if projection_path is not None:
            self._load_projection(projection_path)

        self.to(device)

    def _load_projection(self, path: str):
        """Load projection weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        if self.mode == "coord" and "coord_state" in ckpt:
            self.coord_projection.load_state_dict(ckpt["coord_state"])
        elif self.mode == "clip" and "clip_state" in ckpt:
            self.clip_projection.load_state_dict(ckpt["clip_state"])
        elif self.mode == "both":
            if "coord_state" in ckpt:
                self.coord_projection.load_state_dict(ckpt["coord_state"])
            if "clip_state" in ckpt:
                self.clip_projection.load_state_dict(ckpt["clip_state"])
        elif isinstance(ckpt, dict) and "state_dict" not in ckpt:
            # Legacy: plain state_dict for linear/MLP
            if self.clip_projection is not None:
                self.clip_projection.load_state_dict(ckpt)
            elif self.coord_projection is not None:
                self.coord_projection.load_state_dict(ckpt)

    @classmethod
    def from_coordinates(cls, projection_path: str, device: str = "cuda"):
        """Create encoder using coordinate parsing (no CLIP needed)."""
        return cls(mode="coord", projection_path=projection_path, device=device)

    @classmethod
    def from_clip(cls, projection_path: str, device: str = "cuda"):
        """Create encoder using CLIP text features."""
        return cls(mode="clip", projection_path=projection_path, device=device)

    @torch.inference_mode()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Encode text to LeWM-compatible goal embedding.

        In coord mode: parses coordinates from text.
        In clip mode: encodes via CLIP + projection.
        In both mode: tries coord first, falls back to clip.

        Returns:
            (B, 1, 192) tensor
        """
        if isinstance(text, str):
            text = [text]

        if self.mode == "coord":
            return self._encode_coords(text)
        elif self.mode == "clip":
            return self._encode_clip(text)
        else:
            # Try coord parsing first, fall back to CLIP
            coords = [_parse_coordinates(t) for t in text]
            if all(c is not None for c in coords):
                return self._encode_coords(text)
            return self._encode_clip(text)

    def _encode_coords(self, texts: list[str]) -> torch.Tensor:
        coords = []
        for t in texts:
            parsed = _parse_coordinates(t)
            if parsed is None:
                raise ValueError(f"Could not parse coordinates from: {t}")
            coords.append(parsed)
        coord_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        projected = self.coord_projection(coord_tensor)
        return projected.unsqueeze(1)

    def _encode_clip(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        clip_features = self.clip.encode_text(tokens).float()
        projected = self.clip_projection(clip_features)
        return projected.unsqueeze(1)

    def get_clip_features(self, text: str | list[str]) -> torch.Tensor:
        """Get raw CLIP features (for training)."""
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text).to(self.device)
        with torch.inference_mode():
            features = self.clip.encode_text(tokens)
        return features.float()
