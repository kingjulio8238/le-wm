"""
PlanResult: Structured return type for LeHarness planning calls.

Wraps the planned action alongside confidence signals that enable
S2 (VLM) integration — the VLM can use these signals to decide
when replanning is needed.

Backward compatible: PlanResult supports numpy array protocol,
so existing code doing `raw_action = pipeline.plan(obs, goal)`
followed by `raw_action.reshape(...)` works unchanged.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PlanResult:
    """Structured result from a planning call.

    Attributes:
        action: (action_dim,) numpy array — the planned action for S1.
        planning_cost: Raw MSE cost of the best CEM trajectory (lower = better).
        confidence: 0.0-1.0 normalized confidence (1.0 = best possible plan).
        terminal_embedding: (1, 1, D) tensor — predicted state after executing plan.
        planability: Float score from _score_state() — how easy is it to keep
            planning from the predicted terminal state (lower = easier).
        planning_ms: Wall-clock time for this planning call in milliseconds.
        replan_threshold: Confidence below which needs_replan triggers.
    """

    action: np.ndarray
    planning_cost: float
    confidence: float
    terminal_embedding: torch.Tensor
    planability: float
    planning_ms: float
    replan_threshold: float = 0.3

    @property
    def needs_replan(self) -> bool:
        """True if confidence is below the replan threshold."""
        return self.confidence < self.replan_threshold

    # --- Numpy array protocol for backward compatibility ---

    def __array__(self, dtype=None, copy=None):
        arr = np.array(self.action, dtype=dtype, copy=copy)
        return arr

    def reshape(self, *args, **kwargs):
        return self.action.reshape(*args, **kwargs)

    def __getitem__(self, key):
        return self.action[key]

    def __len__(self):
        return len(self.action)

    @property
    def shape(self):
        return self.action.shape

    @property
    def dtype(self):
        return self.action.dtype
