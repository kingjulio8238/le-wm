"""
SubgoalSequencer: Plans toward ordered list of VLM-provided subgoals.

Unlike Dream Chaining (D2) which failed because latent interpolation produces
unreachable states, subgoal sequencing uses VLM-provided waypoints that
correspond to real, semantically meaningful states.

Usage:
    from harness.subgoal_sequencer import SubgoalSequencer

    # VLM provides ordered subgoals
    subgoals = [goal_emb_1, goal_emb_2, goal_emb_3]  # list of (1, 1, 192) tensors

    sequencer = SubgoalSequencer(pipeline, subgoals)

    while not sequencer.is_complete:
        # Pipeline plans toward the current subgoal
        result = pipeline.plan(obs)

        # Check if we've arrived at current subgoal
        sequencer.step(result)

        # Execute action
        execute(result.action)
"""

import torch

from harness.plan_result import PlanResult


class SubgoalSequencer:
    """Sequences through an ordered list of goal embeddings.

    Tracks the current subgoal, detects arrival (planning cost drops
    below threshold), and advances to the next subgoal. Automatically
    sets the pipeline's goal embedding when advancing.
    """

    def __init__(
        self,
        pipeline,
        subgoals: list[torch.Tensor],
        arrival_threshold: float = 1.0,
        min_steps_per_subgoal: int = 3,
    ):
        """
        Args:
            pipeline: PlanningPipeline instance
            subgoals: ordered list of (1, 1, 192) goal embedding tensors
            arrival_threshold: planning_cost below which we consider arrived
            min_steps_per_subgoal: minimum steps before checking arrival
                (prevents premature advancement)
        """
        if not subgoals:
            raise ValueError("subgoals must be a non-empty list")

        self.pipeline = pipeline
        self.subgoals = subgoals
        self.arrival_threshold = arrival_threshold
        self.min_steps_per_subgoal = min_steps_per_subgoal

        self._current_idx = 0
        self._steps_at_current = 0
        self._history: list[dict] = []

        # Set the first subgoal
        self.pipeline.set_goal_embedding(self.subgoals[0])

    @property
    def current_index(self) -> int:
        """Index of the current subgoal (0-based)."""
        return self._current_idx

    @property
    def current_subgoal(self) -> torch.Tensor:
        """The current target subgoal embedding."""
        return self.subgoals[self._current_idx]

    @property
    def is_complete(self) -> bool:
        """True if all subgoals have been reached."""
        return self._current_idx >= len(self.subgoals)

    @property
    def num_subgoals(self) -> int:
        return len(self.subgoals)

    @property
    def progress(self) -> float:
        """Fraction of subgoals completed (0.0 to 1.0)."""
        return min(self._current_idx / len(self.subgoals), 1.0)

    def step(self, plan_result: PlanResult) -> bool:
        """Process a planning result and potentially advance to next subgoal.

        Args:
            plan_result: PlanResult from pipeline.plan()

        Returns:
            True if advanced to the next subgoal on this step
        """
        if self.is_complete:
            return False

        self._steps_at_current += 1

        self._history.append({
            "subgoal_idx": self._current_idx,
            "planning_cost": plan_result.planning_cost,
            "confidence": plan_result.confidence,
        })

        # Check arrival: cost below threshold AND minimum steps met
        arrived = (
            plan_result.planning_cost < self.arrival_threshold
            and self._steps_at_current >= self.min_steps_per_subgoal
        )

        if arrived:
            return self._advance()

        return False

    def _advance(self) -> bool:
        """Move to the next subgoal."""
        self._current_idx += 1
        self._steps_at_current = 0

        if not self.is_complete:
            self.pipeline.set_goal_embedding(self.subgoals[self._current_idx])
            return True

        return True  # advanced past the last subgoal → complete

    def reset(self):
        """Reset to the first subgoal."""
        self._current_idx = 0
        self._steps_at_current = 0
        self._history.clear()
        self.pipeline.set_goal_embedding(self.subgoals[0])

    def get_history(self) -> list[dict]:
        """Return step-by-step history of subgoal progression."""
        return self._history
