"""
Phase 4: Value Function Cost Model for CEM Solver Integration

Wraps a ValueEnsemble to conform to the get_cost() interface expected
by the CEM solver. Replaces the default MSE embedding distance with
learned per-step value estimates.

Cost = negative mean ensemble prediction, summed over rollout steps,
with 2x weight on the terminal step.
"""

import torch
import torch.nn.functional as F
from typing import Any


class ValueCostModel:
    """Cost model that uses a learned value function instead of MSE distance.

    Wraps an existing JEPA model (for encode + rollout) and a ValueEnsemble
    (for scoring). Conforms to the Costable protocol expected by CEM solvers.

    The get_cost() method follows the exact same flow as JEPA.get_cost():
    1. Encode goal from info_dict (taking first sample's data)
    2. Rollout predicted embeddings using base model
    3. Score with value function instead of MSE criterion
    """

    def __init__(self, base_model, value_ensemble, terminal_weight: float = 2.0):
        self.base_model = base_model
        self.value_ensemble = value_ensemble
        self.terminal_weight = terminal_weight
        self.interpolate_pos_encoding = getattr(
            base_model, "interpolate_pos_encoding", False
        )

    def encode(self, info_dict: dict) -> dict:
        return self.base_model.encode(info_dict)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute cost using learned value function.

        Follows the same interface as JEPA.get_cost(). The solver passes:
            info_dict: keys have shape (B, S, T, ...) where S = num_samples
            action_candidates: shape (B, S, T, action_dim)

        Returns:
            costs: shape (B, S) — lower is better
        """
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.base_model.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # Step 1: Encode goal — same as JEPA.get_cost()
        # Take first sample ([:, 0]) to get (B, T, ...) for encoding
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.base_model.encode(goal)
        goal_emb = goal["emb"]  # (B, T_goal, embed_dim)

        # Step 2: Rollout — delegates to base model which handles (B, S, T, ...) shapes
        info_dict["goal_emb"] = goal_emb
        info_dict = self.base_model.rollout(info_dict, action_candidates)

        # info_dict now has "predicted_emb": (B, S, T_pred, embed_dim)
        pred_emb = info_dict["predicted_emb"]
        B, S = pred_emb.shape[0], pred_emb.shape[1]
        T_pred = pred_emb.shape[2]

        # Step 3: Score with value function instead of MSE
        # Expand goal_emb to match: (B, 1, embed_dim) -> (B, S, T_pred, embed_dim)
        goal_expanded = goal_emb[:, -1:, :].unsqueeze(1).expand(B, S, T_pred, -1)

        # Flatten for value function
        pred_flat = pred_emb.reshape(-1, pred_emb.shape[-1])
        goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])

        with torch.no_grad():
            values = self.value_ensemble(pred_flat, goal_flat)
            values = values.reshape(B, S, T_pred)

        # Step 4: Cost = negative weighted value sum
        weights = torch.ones(T_pred, device=values.device)
        weights[-1] = self.terminal_weight
        weights = weights / weights.sum()

        cost = -(values * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return cost

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to base model."""
        if name in ("base_model", "value_ensemble", "terminal_weight",
                     "interpolate_pos_encoding"):
            raise AttributeError(name)
        return getattr(self.base_model, name)
