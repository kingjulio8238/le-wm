"""Tests for GoalAdapter, VLMProjection, and SubgoalSequencer."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from harness.projections import VLMProjection, CLIPProjection, CoordProjection
from harness.goal_adapter import GoalAdapter, VLM_DIMS
from harness.subgoal_sequencer import SubgoalSequencer
from harness.plan_result import PlanResult


# --- Mock pipeline for testing without GPU/model ---

class MockPipeline:
    """Minimal pipeline mock for testing goal adapter and sequencer."""

    def __init__(self, device="cpu"):
        self.device = device
        self._goal_emb = None
        self.language_encoder = None

    def preprocess(self, image_np):
        return torch.randn(1, 1, 3, 224, 224)

    def encode(self, tensor):
        return torch.randn(1, 1, 192)

    def set_goal_embedding(self, emb):
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
        elif emb.dim() == 2:
            emb = emb.unsqueeze(1)
        self._goal_emb = emb.float()


def _make_plan_result(planning_cost=2.0, confidence=0.8, **overrides):
    defaults = dict(
        action=np.zeros(10, dtype=np.float32),
        planning_cost=planning_cost,
        confidence=confidence,
        terminal_embedding=torch.randn(1, 1, 192),
        planability=1.5,
        planning_ms=100.0,
        replan_threshold=0.3,
    )
    defaults.update(overrides)
    return PlanResult(**defaults)


# ==================== VLMProjection Tests ====================

class TestVLMProjection:
    def test_output_shape(self):
        proj = VLMProjection(in_dim=768, out_dim=192)
        x = torch.randn(4, 768)
        out = proj(x)
        assert out.shape == (4, 192)

    def test_siglip_dims(self):
        proj = VLMProjection(in_dim=768, out_dim=192)
        x = torch.randn(2, 768)
        assert proj(x).shape == (2, 192)

    def test_eagle_dims(self):
        proj = VLMProjection(in_dim=1536, out_dim=192)
        x = torch.randn(2, 1536)
        assert proj(x).shape == (2, 192)

    def test_paligemma_dims(self):
        proj = VLMProjection(in_dim=4608, hidden_dim=1024, out_dim=192)
        x = torch.randn(1, 4608)
        assert proj(x).shape == (1, 192)

    def test_single_sample(self):
        proj = VLMProjection(in_dim=512, out_dim=192)
        x = torch.randn(1, 512)
        assert proj(x).shape == (1, 192)

    def test_in_dim_stored(self):
        proj = VLMProjection(in_dim=768)
        assert proj.in_dim == 768

    def test_gradient_flow(self):
        proj = VLMProjection(in_dim=768, out_dim=192)
        x = torch.randn(2, 768, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ==================== GoalAdapter Tests ====================

class TestGoalAdapter:
    def test_from_image(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emb = adapter.from_image(img)
        assert emb.shape == (1, 1, 192)

    def test_from_raw_embedding_3d(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        emb = adapter.from_raw_embedding(torch.randn(1, 1, 192))
        assert emb.shape == (1, 1, 192)

    def test_from_raw_embedding_2d(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        emb = adapter.from_raw_embedding(torch.randn(1, 192))
        assert emb.shape == (1, 1, 192)

    def test_from_raw_embedding_1d(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        emb = adapter.from_raw_embedding(torch.randn(192))
        assert emb.shape == (1, 1, 192)

    def test_from_vlm_embedding(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")

        # Register a projection manually
        proj = VLMProjection(in_dim=768, out_dim=192)
        adapter.register_projection("siglip", proj)

        emb = adapter.from_vlm_embedding(torch.randn(768), source="siglip")
        assert emb.shape == (1, 1, 192)

    def test_from_vlm_embedding_batched(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        proj = VLMProjection(in_dim=768, out_dim=192)
        adapter.register_projection("siglip", proj)

        emb = adapter.from_vlm_embedding(torch.randn(3, 768), source="siglip")
        assert emb.shape == (3, 1, 192)

    def test_from_vlm_embedding_unknown_source(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        with pytest.raises(ValueError, match="No projection loaded"):
            adapter.from_vlm_embedding(torch.randn(768), source="siglip")

    def test_load_projection_unknown_source(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        with pytest.raises(ValueError, match="Unknown VLM source"):
            adapter.load_projection("unknown_model", "fake_path.pt")

    def test_from_subgoals_image(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        goals = adapter.from_subgoals(images, format="image")
        assert len(goals) == 3
        for g in goals:
            assert g.shape == (1, 1, 192)

    def test_from_subgoals_raw(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        embs = [torch.randn(192) for _ in range(4)]
        goals = adapter.from_subgoals(embs, format="raw_embedding")
        assert len(goals) == 4
        for g in goals:
            assert g.shape == (1, 1, 192)

    def test_from_subgoals_vlm(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        proj = VLMProjection(in_dim=768, out_dim=192)
        adapter.register_projection("siglip", proj)

        embs = [torch.randn(768) for _ in range(2)]
        goals = adapter.from_subgoals(embs, format="vlm_embedding", source="siglip")
        assert len(goals) == 2
        for g in goals:
            assert g.shape == (1, 1, 192)

    def test_from_subgoals_invalid_format(self):
        pipeline = MockPipeline()
        adapter = GoalAdapter(pipeline, device="cpu")
        with pytest.raises(ValueError, match="Unknown format"):
            adapter.from_subgoals(["a"], format="invalid")


# ==================== SubgoalSequencer Tests ====================

class TestSubgoalSequencer:
    def _make_subgoals(self, n=3):
        return [torch.randn(1, 1, 192) for _ in range(n)]

    def test_init_sets_first_goal(self):
        pipeline = MockPipeline()
        subgoals = self._make_subgoals()
        seq = SubgoalSequencer(pipeline, subgoals)
        assert pipeline._goal_emb is not None
        assert seq.current_index == 0
        assert not seq.is_complete

    def test_empty_subgoals_raises(self):
        pipeline = MockPipeline()
        with pytest.raises(ValueError, match="non-empty"):
            SubgoalSequencer(pipeline, [])

    def test_num_subgoals(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(pipeline, self._make_subgoals(5))
        assert seq.num_subgoals == 5

    def test_progress_starts_zero(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(pipeline, self._make_subgoals(4))
        assert seq.progress == 0.0

    def test_advance_on_low_cost(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(3),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )
        assert seq.current_index == 0

        # Step with low cost → should advance
        result = _make_plan_result(planning_cost=1.0)
        advanced = seq.step(result)
        assert advanced is True
        assert seq.current_index == 1

    def test_no_advance_on_high_cost(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(3),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )
        result = _make_plan_result(planning_cost=5.0)
        advanced = seq.step(result)
        assert advanced is False
        assert seq.current_index == 0

    def test_min_steps_respected(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(3),
            arrival_threshold=2.0,
            min_steps_per_subgoal=3,
        )
        # Low cost but not enough steps yet
        result = _make_plan_result(planning_cost=0.5)
        assert seq.step(result) is False  # step 1
        assert seq.step(result) is False  # step 2
        assert seq.step(result) is True   # step 3 → advance

    def test_full_sequence_completes(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(2),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )
        result = _make_plan_result(planning_cost=0.5)

        seq.step(result)  # advance to subgoal 1
        assert seq.current_index == 1
        assert not seq.is_complete

        seq.step(result)  # advance past subgoal 1 → complete
        assert seq.is_complete
        assert seq.progress == 1.0

    def test_step_after_complete_is_noop(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(1),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )
        result = _make_plan_result(planning_cost=0.5)
        seq.step(result)  # complete
        assert seq.is_complete

        advanced = seq.step(result)  # should be noop
        assert advanced is False

    def test_reset(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(3),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )
        result = _make_plan_result(planning_cost=0.5)
        seq.step(result)
        seq.step(result)
        assert seq.current_index == 2

        seq.reset()
        assert seq.current_index == 0
        assert not seq.is_complete
        assert len(seq.get_history()) == 0

    def test_history_tracking(self):
        pipeline = MockPipeline()
        seq = SubgoalSequencer(
            pipeline, self._make_subgoals(2),
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )

        seq.step(_make_plan_result(planning_cost=5.0))  # no advance
        seq.step(_make_plan_result(planning_cost=0.5))  # advance
        seq.step(_make_plan_result(planning_cost=0.3))  # advance → complete

        history = seq.get_history()
        assert len(history) == 3
        assert history[0]["subgoal_idx"] == 0
        assert history[0]["planning_cost"] == 5.0
        assert history[1]["subgoal_idx"] == 0
        assert history[2]["subgoal_idx"] == 1

    def test_pipeline_goal_updated_on_advance(self):
        pipeline = MockPipeline()
        subgoals = self._make_subgoals(3)
        seq = SubgoalSequencer(
            pipeline, subgoals,
            arrival_threshold=2.0,
            min_steps_per_subgoal=1,
        )

        # Initially set to subgoal 0
        initial_goal = pipeline._goal_emb.clone()

        # Advance to subgoal 1
        seq.step(_make_plan_result(planning_cost=0.5))
        assert seq.current_index == 1
        # Goal should have changed
        assert not torch.equal(pipeline._goal_emb, initial_goal)
