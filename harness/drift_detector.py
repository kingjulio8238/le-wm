"""
DriftDetector: Monitors prediction accuracy to trigger S2 replanning.

After each action execution, compares where the world model predicted
it would end up (terminal_embedding from PlanResult) against where it
actually is (encoded observation). Large or increasing drift indicates
the world model is systematically wrong — escalate to S2 for replanning.

This is the feedback signal that no other S1/S2 architecture provides.
GR00T N1, Helix, OpenHelix — none have S1→S2 feedback. LeHarness can
close that loop because CEM planning produces predictive embeddings.

Usage:
    detector = DriftDetector(threshold=0.1, window=5)

    # In control loop:
    result = pipeline.plan(obs)
    execute(result.action)
    new_obs = camera.capture()

    signal = detector.check(
        predicted=result.terminal_embedding,
        actual_obs=new_obs,
        pipeline=pipeline,
    )
    if signal.escalate_to_s2:
        vlm.replan(...)
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DriftSignal:
    """Result of a drift check between predicted and actual state.

    Attributes:
        drift_mse: MSE between predicted terminal embedding and actual
            observation embedding. Lower = prediction was accurate.
        drift_exceeded: True if drift_mse > threshold.
        trend_increasing: True if drift is getting worse over the recent window.
        escalate_to_s2: True if drift is both high AND increasing — signals
            the VLM should replan because the world model is systematically wrong.
    """

    drift_mse: float
    drift_exceeded: bool
    trend_increasing: bool
    escalate_to_s2: bool


class DriftDetector:
    """Monitors prediction vs. reality divergence over time.

    Compares predicted terminal embeddings (from PlanResult) against
    actual observation embeddings to detect when the world model's
    predictions are systematically wrong.
    """

    def __init__(self, threshold: float = 0.1, window: int = 5):
        """
        Args:
            threshold: MSE above which drift is considered significant.
            window: Number of recent steps to use for trend analysis.
        """
        self.threshold = threshold
        self.window = window
        self._history: list[float] = []

    @torch.inference_mode()
    def check(
        self,
        predicted: torch.Tensor,
        actual_obs: np.ndarray = None,
        actual_emb: torch.Tensor = None,
        pipeline=None,
    ) -> DriftSignal:
        """Compare predicted terminal state against actual observation.

        Provide either (actual_obs + pipeline) for automatic encoding,
        or actual_emb for pre-encoded embeddings.

        Args:
            predicted: (1, 1, D) predicted terminal embedding from PlanResult.
            actual_obs: (H, W, 3) uint8 image — encoded via pipeline if provided.
            actual_emb: (1, 1, D) pre-encoded observation embedding.
            pipeline: PlanningPipeline instance (required if actual_obs is given).

        Returns:
            DriftSignal with drift metrics and escalation recommendation.
        """
        if actual_emb is None:
            if actual_obs is None or pipeline is None:
                raise ValueError(
                    "Provide either actual_emb or (actual_obs + pipeline)"
                )
            actual_emb = pipeline.encode(pipeline.preprocess(actual_obs))

        # Compute MSE between predicted and actual
        drift_mse = float(
            ((predicted.float() - actual_emb.float()) ** 2).sum().item()
        )
        self._history.append(drift_mse)

        drift_exceeded = drift_mse > self.threshold
        trend_increasing = self._trend_increasing()

        return DriftSignal(
            drift_mse=drift_mse,
            drift_exceeded=drift_exceeded,
            trend_increasing=trend_increasing,
            escalate_to_s2=drift_exceeded and trend_increasing,
        )

    def _trend_increasing(self) -> bool:
        """Check if drift is increasing over the recent window.

        Uses linear regression slope on the last `window` drift values.
        Returns True if slope > 0 (drift getting worse).
        Requires at least 2 data points; returns False otherwise.
        """
        if len(self._history) < 2:
            return False

        recent = self._history[-self.window:]
        n = len(recent)
        if n < 2:
            return False

        # Linear regression slope: sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
        x = np.arange(n, dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        return float(slope) > 0

    def reset(self):
        """Clear all drift history."""
        self._history.clear()

    def get_history(self) -> list[float]:
        """Return the full drift MSE history."""
        return list(self._history)

    @property
    def last_drift(self) -> float | None:
        """Most recent drift MSE value, or None if no checks yet."""
        return self._history[-1] if self._history else None

    @property
    def mean_drift(self) -> float | None:
        """Mean drift over the recent window."""
        if not self._history:
            return None
        recent = self._history[-self.window:]
        return float(np.mean(recent))
