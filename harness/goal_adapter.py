"""
GoalAdapter: Unified goal ingestion for S2 (VLM) integration.

Maps any upstream goal format — images, text, VLM embeddings, or
subgoal sequences — to LeWM's 192-dim planning space.

Usage:
    adapter = GoalAdapter(pipeline)

    # From image (existing path)
    goal_emb = adapter.from_image(goal_image_np)

    # From text (existing path — requires language encoder loaded)
    goal_emb = adapter.from_text("navigate to (0.43, 0.57)")

    # From VLM embedding (NEW — for S2 integration)
    adapter.load_projection("siglip", "path/to/siglip_projection.pt")
    goal_emb = adapter.from_vlm_embedding(siglip_emb, source="siglip")

    # From raw 192-dim embedding (direct injection)
    goal_emb = adapter.from_raw_embedding(emb_192)

    # From subgoal sequence
    goals = adapter.from_subgoals([img1, img2, img3], format="image")
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from harness.projections import VLMProjection


# Known VLM embedding dimensions
VLM_DIMS = {
    "clip": 512,
    "siglip": 768,
    "t5": 768,
    "eagle": 1536,
    "paligemma": 4608,
}


class GoalAdapter:
    """Maps any upstream VLM goal format to 192-dim planning space.

    Holds a set of projection MLPs (one per VLM family) and provides
    a unified interface for goal conversion.
    """

    def __init__(self, pipeline, device: str = None):
        """
        Args:
            pipeline: PlanningPipeline instance (used for image encoding)
            device: torch device (defaults to pipeline's device)
        """
        self.pipeline = pipeline
        self.device = device or pipeline.device
        self._projections: dict[str, VLMProjection] = {}

    def load_projection(self, source: str, projection_path: str):
        """Load a trained VLM projection.

        Args:
            source: VLM family name ("siglip", "t5", "eagle", "paligemma", "clip")
            projection_path: path to saved projection weights
        """
        if source not in VLM_DIMS:
            raise ValueError(
                f"Unknown VLM source '{source}'. Known: {list(VLM_DIMS.keys())}"
            )

        in_dim = VLM_DIMS[source]
        hidden_dim = 1024 if in_dim > 1024 else 512
        proj = VLMProjection(in_dim=in_dim, hidden_dim=hidden_dim)

        ckpt = torch.load(projection_path, map_location=self.device, weights_only=True)
        if "state_dict" in ckpt:
            proj.load_state_dict(ckpt["state_dict"])
        else:
            proj.load_state_dict(ckpt)

        proj = proj.to(self.device).eval()
        proj.requires_grad_(False)
        self._projections[source] = proj

    def register_projection(self, source: str, projection: nn.Module):
        """Register an already-instantiated projection module.

        Args:
            source: VLM family name
            projection: nn.Module that maps (B, in_dim) -> (B, 192)
        """
        projection = projection.to(self.device).eval()
        projection.requires_grad_(False)
        self._projections[source] = projection

    @torch.inference_mode()
    def from_image(self, image_np: np.ndarray) -> torch.Tensor:
        """Encode a goal image via the pipeline's ViT encoder.

        Args:
            image_np: (H, W, 3) uint8 numpy array

        Returns:
            (1, 1, 192) goal embedding tensor
        """
        tensor = self.pipeline.preprocess(image_np)
        return self.pipeline.encode(tensor)

    @torch.inference_mode()
    def from_text(self, text: str) -> torch.Tensor:
        """Encode a text goal via the pipeline's language encoder.

        Requires pipeline.load_language_encoder() to have been called.

        Args:
            text: goal description string

        Returns:
            (1, 1, 192) goal embedding tensor
        """
        assert self.pipeline.language_encoder is not None, (
            "Call pipeline.load_language_encoder(projection_path) first"
        )
        return self.pipeline.language_encoder.encode_text(text)

    @torch.inference_mode()
    def from_vlm_embedding(
        self, emb: torch.Tensor, source: str
    ) -> torch.Tensor:
        """Project a VLM embedding to LeWM's 192-dim space.

        Args:
            emb: (B, D) or (D,) tensor from the VLM encoder
            source: VLM family name ("siglip", "t5", "eagle", "paligemma", "clip")

        Returns:
            (B, 1, 192) goal embedding tensor
        """
        if source not in self._projections:
            raise ValueError(
                f"No projection loaded for '{source}'. "
                f"Call load_projection('{source}', path) first. "
                f"Loaded: {list(self._projections.keys())}"
            )

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # (D,) → (1, D)

        emb = emb.to(self.device).float()
        proj = self._projections[source]
        projected = proj(emb)  # (B, 192)
        return projected.unsqueeze(1)  # (B, 1, 192)

    @torch.inference_mode()
    def from_raw_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        """Directly inject a 192-dim embedding (no projection needed).

        Args:
            emb: (1, 1, 192) or (1, 192) or (192,) tensor

        Returns:
            (1, 1, 192) goal embedding tensor
        """
        emb = emb.to(self.device).float()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)  # (192,) → (1, 1, 192)
        elif emb.dim() == 2:
            emb = emb.unsqueeze(1)  # (1, 192) → (1, 1, 192)
        return emb

    def from_subgoals(
        self,
        goals: list,
        format: str = "image",
        source: str = None,
    ) -> list[torch.Tensor]:
        """Convert a sequence of goals to ordered 192-dim embeddings.

        Args:
            goals: list of goals (images, text strings, or VLM embeddings)
            format: "image", "text", "vlm_embedding", or "raw_embedding"
            source: VLM family name (required when format="vlm_embedding")

        Returns:
            list of (1, 1, 192) embedding tensors
        """
        encode_fn = {
            "image": self.from_image,
            "text": self.from_text,
            "raw_embedding": self.from_raw_embedding,
        }

        if format == "vlm_embedding":
            if source is None:
                raise ValueError("source is required when format='vlm_embedding'")
            return [self.from_vlm_embedding(g, source=source) for g in goals]

        if format not in encode_fn:
            raise ValueError(f"Unknown format '{format}'. Use: {list(encode_fn.keys()) + ['vlm_embedding']}")

        fn = encode_fn[format]
        return [fn(g) for g in goals]
