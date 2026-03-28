"""Tests for PlanResult dataclass and backward compatibility."""

import numpy as np
import pytest
import torch

from harness.plan_result import PlanResult


def _make_plan_result(**overrides):
    """Create a PlanResult with sensible defaults."""
    defaults = dict(
        action=np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.0, 0.4, -0.3]),
        planning_cost=2.5,
        confidence=0.75,
        terminal_embedding=torch.randn(1, 1, 192),
        planability=1.8,
        planning_ms=89.0,
        replan_threshold=0.3,
    )
    defaults.update(overrides)
    return PlanResult(**defaults)


class TestPlanResultFields:
    def test_action_shape(self):
        result = _make_plan_result()
        assert result.action.shape == (10,)

    def test_action_dtype(self):
        result = _make_plan_result()
        assert result.action.dtype == np.float64 or result.action.dtype == np.float32

    def test_confidence_range(self):
        result = _make_plan_result(confidence=0.75)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_zero(self):
        result = _make_plan_result(confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_one(self):
        result = _make_plan_result(confidence=1.0)
        assert result.confidence == 1.0

    def test_terminal_embedding_shape(self):
        result = _make_plan_result()
        assert result.terminal_embedding.shape == (1, 1, 192)

    def test_planning_cost_is_float(self):
        result = _make_plan_result()
        assert isinstance(result.planning_cost, float)

    def test_planability_is_float(self):
        result = _make_plan_result()
        assert isinstance(result.planability, float)

    def test_planning_ms_is_float(self):
        result = _make_plan_result()
        assert isinstance(result.planning_ms, float)


class TestNeedsReplan:
    def test_needs_replan_when_low_confidence(self):
        result = _make_plan_result(confidence=0.1, replan_threshold=0.3)
        assert result.needs_replan is True

    def test_no_replan_when_high_confidence(self):
        result = _make_plan_result(confidence=0.8, replan_threshold=0.3)
        assert result.needs_replan is False

    def test_no_replan_at_threshold(self):
        result = _make_plan_result(confidence=0.3, replan_threshold=0.3)
        assert result.needs_replan is False

    def test_needs_replan_just_below_threshold(self):
        result = _make_plan_result(confidence=0.299, replan_threshold=0.3)
        assert result.needs_replan is True

    def test_custom_threshold(self):
        result = _make_plan_result(confidence=0.5, replan_threshold=0.6)
        assert result.needs_replan is True


class TestNumpyBackwardCompat:
    """Test that PlanResult works as a drop-in replacement for np.ndarray
    in existing eval scripts that do raw_action.reshape(...)."""

    def test_reshape(self):
        action = np.zeros(10, dtype=np.float32)
        result = _make_plan_result(action=action)
        reshaped = result.reshape(5, 2)
        assert reshaped.shape == (5, 2)

    def test_reshape_matches_action(self):
        action = np.arange(10, dtype=np.float32)
        result = _make_plan_result(action=action)
        np.testing.assert_array_equal(result.reshape(5, 2), action.reshape(5, 2))

    def test_np_array_conversion(self):
        action = np.array([1.0, 2.0, 3.0])
        result = _make_plan_result(action=action)
        arr = np.array(result)
        np.testing.assert_array_equal(arr, action)

    def test_np_asarray(self):
        action = np.array([1.0, 2.0, 3.0])
        result = _make_plan_result(action=action)
        arr = np.asarray(result)
        np.testing.assert_array_equal(arr, action)

    def test_getitem(self):
        action = np.array([10.0, 20.0, 30.0])
        result = _make_plan_result(action=action)
        assert result[0] == 10.0
        assert result[2] == 30.0

    def test_getitem_slice(self):
        action = np.arange(10, dtype=np.float32)
        result = _make_plan_result(action=action)
        np.testing.assert_array_equal(result[:5], action[:5])

    def test_len(self):
        action = np.zeros(10)
        result = _make_plan_result(action=action)
        assert len(result) == 10

    def test_shape_property(self):
        action = np.zeros(10)
        result = _make_plan_result(action=action)
        assert result.shape == (10,)

    def test_dtype_property(self):
        action = np.zeros(10, dtype=np.float32)
        result = _make_plan_result(action=action)
        assert result.dtype == np.float32

    def test_eval_script_pattern(self):
        """Simulate the exact pattern used in eval scripts:
        raw_action = pipeline.plan(obs, goal)
        sub_actions = raw_action.reshape(action_block, raw_action_dim)
        for sub_action in sub_actions: ...
        """
        action = np.random.randn(10).astype(np.float32)
        result = _make_plan_result(action=action)

        # This is what eval scripts do:
        action_block = 5
        raw_action_dim = 2
        sub_actions = result.reshape(action_block, raw_action_dim)

        assert sub_actions.shape == (5, 2)
        for sub_action in sub_actions:
            assert sub_action.shape == (2,)
