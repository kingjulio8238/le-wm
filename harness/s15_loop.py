"""
S1.5 Control Loop: Full VLM → LeHarness → Motor Policy integration.

Orchestrates the three-layer robotics stack:
  S2 (VLM) provides goals → S1.5 (LeHarness) plans → S1 (motor policy) executes

With closed-loop feedback:
  - Low confidence → ask VLM to replan
  - Drift detected → ask VLM to replan

Usage:
    from harness.s15_loop import S15ControlLoop, MockVLM, MockMotorPolicy

    vlm = MockVLM(goal_image=goal_img)
    motor = MockMotorPolicy()
    loop = S15ControlLoop(pipeline, vlm, motor)

    stats = loop.run_episode(
        initial_obs=obs_img,
        max_steps=100,
        success_fn=lambda obs: check_success(obs),
    )
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from harness.drift_detector import DriftDetector
from harness.plan_result import PlanResult


# ==================== Mock Components ====================


class MockVLM:
    """Mock S2 VLM that provides goals and handles replan requests.

    For testing the S1.5 control loop without a real VLM.
    Provides an initial goal and optionally adjusts goals on replan.
    """

    def __init__(
        self,
        goal_image: np.ndarray = None,
        goal_embedding: torch.Tensor = None,
        replan_strategy: str = "same",
    ):
        """
        Args:
            goal_image: (H, W, 3) uint8 goal image.
            goal_embedding: (1, 1, D) pre-computed goal embedding.
                Provide one of goal_image or goal_embedding.
            replan_strategy: How to respond to replan requests.
                "same" — return the same goal (VLM insists).
                "noisy" — add noise to the goal embedding.
                "callback" — use a custom callback (set via on_replan).
        """
        self.goal_image = goal_image
        self.goal_embedding = goal_embedding
        self.replan_strategy = replan_strategy

        self._replan_count = 0
        self._replan_history: list[dict] = []
        self._replan_callback = None

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def replan_history(self) -> list[dict]:
        return self._replan_history

    def on_replan(self, callback):
        """Register a custom replan callback: fn(reason, obs, **kwargs) -> goal."""
        self._replan_callback = callback
        self.replan_strategy = "callback"

    def get_initial_goal(self):
        """Return the initial goal (image or embedding)."""
        if self.goal_embedding is not None:
            return {"type": "embedding", "value": self.goal_embedding}
        return {"type": "image", "value": self.goal_image}

    def replan(self, reason: str, obs: np.ndarray = None, **kwargs):
        """Handle a replan request from S1.5.

        Args:
            reason: "low_confidence" or "drift_detected"
            obs: current observation image
            **kwargs: additional context (planning_cost, drift_mse, etc.)

        Returns:
            dict with "type" and "value" keys (same format as get_initial_goal)
        """
        self._replan_count += 1
        self._replan_history.append({"reason": reason, "step": kwargs.get("step"), **kwargs})

        if self.replan_strategy == "same":
            return self.get_initial_goal()

        elif self.replan_strategy == "noisy" and self.goal_embedding is not None:
            noise = torch.randn_like(self.goal_embedding) * 0.01
            noisy = self.goal_embedding + noise
            return {"type": "embedding", "value": noisy}

        elif self.replan_strategy == "callback" and self._replan_callback is not None:
            result = self._replan_callback(reason, obs, **kwargs)
            return result

        return self.get_initial_goal()

    def reset(self):
        self._replan_count = 0
        self._replan_history.clear()


class MockMotorPolicy:
    """Mock S1 motor policy that consumes actions.

    Records execution history for analysis. In a real system, this would
    convert LeHarness actions to joint commands and send to the robot.
    """

    def __init__(self):
        self._history: list[np.ndarray] = []

    def execute(self, action: np.ndarray):
        """Execute an action (record it)."""
        self._history.append(action.copy())

    @property
    def execution_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[np.ndarray]:
        return self._history

    def reset(self):
        self._history.clear()


# ==================== Episode Statistics ====================


@dataclass
class EpisodeStats:
    """Statistics from a single S1.5 episode."""

    steps: int = 0
    success: bool = False
    replans_confidence: int = 0
    replans_drift: int = 0
    drift_events: int = 0
    mean_confidence: float = 0.0
    mean_planning_cost: float = 0.0
    mean_drift_mse: float = 0.0
    total_planning_ms: float = 0.0

    # Per-step history
    confidences: list = field(default_factory=list)
    planning_costs: list = field(default_factory=list)
    drift_mses: list = field(default_factory=list)

    @property
    def total_replans(self) -> int:
        return self.replans_confidence + self.replans_drift

    def finalize(self):
        """Compute summary stats from per-step data."""
        if self.confidences:
            self.mean_confidence = float(np.mean(self.confidences))
        if self.planning_costs:
            self.mean_planning_cost = float(np.mean(self.planning_costs))
        if self.drift_mses:
            self.mean_drift_mse = float(np.mean(self.drift_mses))


# ==================== Control Loop ====================


class S15ControlLoop:
    """S1.5 control loop orchestrator.

    Runs the full three-layer stack:
        S2 (VLM) → GoalAdapter → S1.5 (LeHarness) → PlanResult → S1 (motor)
    with closed-loop confidence and drift feedback to S2.
    """

    def __init__(
        self,
        pipeline,
        vlm: MockVLM,
        motor: MockMotorPolicy,
        drift_threshold: float = 0.1,
        drift_window: int = 5,
        max_replans_per_episode: int = 10,
    ):
        """
        Args:
            pipeline: PlanningPipeline (real or mock).
            vlm: VLM that provides goals and handles replanning.
            motor: Motor policy that executes actions.
            drift_threshold: MSE threshold for drift detection.
            drift_window: Window size for drift trend analysis.
            max_replans_per_episode: Safety limit on replanning.
        """
        self.pipeline = pipeline
        self.vlm = vlm
        self.motor = motor
        self.drift_detector = DriftDetector(
            threshold=drift_threshold, window=drift_window
        )
        self.max_replans = max_replans_per_episode

    def _set_goal(self, goal_dict: dict):
        """Set pipeline goal from VLM output."""
        if goal_dict["type"] == "embedding":
            self.pipeline.set_goal_embedding(goal_dict["value"])
        elif goal_dict["type"] == "image":
            self.pipeline.set_goal(goal_dict["value"])
        else:
            raise ValueError(f"Unknown goal type: {goal_dict['type']}")

    def run_episode(
        self,
        initial_obs: np.ndarray,
        get_next_obs,
        max_steps: int = 100,
        success_fn=None,
    ) -> EpisodeStats:
        """Run a single S1.5 episode.

        Args:
            initial_obs: (H, W, 3) starting observation image.
            get_next_obs: callable(action) -> obs_image. Executes action
                in the environment and returns the next observation.
            max_steps: maximum planning steps per episode.
            success_fn: callable(obs) -> bool. Checks if task is complete.
                If None, episode runs to max_steps.

        Returns:
            EpisodeStats with per-step tracking data.
        """
        stats = EpisodeStats()
        self.vlm.reset()
        self.motor.reset()
        self.drift_detector.reset()

        # S2: Get initial goal
        goal_dict = self.vlm.get_initial_goal()
        self._set_goal(goal_dict)

        obs = initial_obs
        prev_result = None

        for step in range(max_steps):
            # S1.5: Plan
            result = self.pipeline.plan(obs)

            # Track stats
            stats.steps += 1
            stats.confidences.append(result.confidence)
            stats.planning_costs.append(result.planning_cost)
            stats.total_planning_ms += result.planning_ms

            # Check confidence → replan if needed
            if result.needs_replan and self.vlm.replan_count < self.max_replans:
                stats.replans_confidence += 1
                new_goal = self.vlm.replan(
                    reason="low_confidence",
                    obs=obs,
                    step=step,
                    planning_cost=result.planning_cost,
                    confidence=result.confidence,
                )
                self._set_goal(new_goal)
                continue  # re-plan with new goal before executing

            # S1: Execute
            self.motor.execute(result.action)

            # Environment: get next observation
            next_obs = get_next_obs(result.action)

            # Drift detection (compare predicted terminal vs actual)
            if prev_result is not None:
                drift_signal = self.drift_detector.check(
                    predicted=prev_result.terminal_embedding,
                    actual_emb=self.pipeline.encode(
                        self.pipeline.preprocess(next_obs)
                    ),
                )
                stats.drift_mses.append(drift_signal.drift_mse)

                if drift_signal.drift_exceeded:
                    stats.drift_events += 1

                if (drift_signal.escalate_to_s2
                        and self.vlm.replan_count < self.max_replans):
                    stats.replans_drift += 1
                    new_goal = self.vlm.replan(
                        reason="drift_detected",
                        obs=next_obs,
                        step=step,
                        drift_mse=drift_signal.drift_mse,
                    )
                    self._set_goal(new_goal)

            prev_result = result
            obs = next_obs

            # Check success
            if success_fn is not None and success_fn(obs):
                stats.success = True
                break

        stats.finalize()
        return stats
