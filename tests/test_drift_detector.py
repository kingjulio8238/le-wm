"""Tests for DriftDetector and DriftSignal."""

import numpy as np
import pytest
import torch

from harness.drift_detector import DriftDetector, DriftSignal


# ==================== DriftSignal Tests ====================

class TestDriftSignal:
    def test_fields(self):
        sig = DriftSignal(drift_mse=0.05, drift_exceeded=False,
                          trend_increasing=False, escalate_to_s2=False)
        assert sig.drift_mse == 0.05
        assert sig.drift_exceeded is False
        assert sig.trend_increasing is False
        assert sig.escalate_to_s2 is False

    def test_escalate_requires_both(self):
        # Exceeded but not increasing → no escalation
        sig = DriftSignal(drift_mse=0.5, drift_exceeded=True,
                          trend_increasing=False, escalate_to_s2=False)
        assert sig.escalate_to_s2 is False

        # Increasing but not exceeded → no escalation
        sig2 = DriftSignal(drift_mse=0.01, drift_exceeded=False,
                           trend_increasing=True, escalate_to_s2=False)
        assert sig2.escalate_to_s2 is False

    def test_escalate_when_both(self):
        sig = DriftSignal(drift_mse=0.5, drift_exceeded=True,
                          trend_increasing=True, escalate_to_s2=True)
        assert sig.escalate_to_s2 is True


# ==================== DriftDetector Tests ====================

class TestDriftDetector:
    def test_identical_embeddings_zero_drift(self):
        """When predicted == actual, drift should be 0."""
        detector = DriftDetector(threshold=0.1)
        emb = torch.randn(1, 1, 192)
        signal = detector.check(predicted=emb, actual_emb=emb.clone())
        assert signal.drift_mse == pytest.approx(0.0, abs=1e-6)
        assert signal.drift_exceeded is False
        assert signal.escalate_to_s2 is False

    def test_different_embeddings_nonzero_drift(self):
        """When predicted != actual, drift should be > 0."""
        detector = DriftDetector(threshold=0.1)
        pred = torch.zeros(1, 1, 192)
        actual = torch.ones(1, 1, 192)
        signal = detector.check(predicted=pred, actual_emb=actual)
        assert signal.drift_mse > 0
        assert signal.drift_mse == pytest.approx(192.0, abs=0.01)

    def test_threshold_exceeded(self):
        detector = DriftDetector(threshold=1.0)
        pred = torch.zeros(1, 1, 192)
        actual = torch.ones(1, 1, 192)  # MSE = 192
        signal = detector.check(predicted=pred, actual_emb=actual)
        assert signal.drift_exceeded is True

    def test_threshold_not_exceeded(self):
        detector = DriftDetector(threshold=1000.0)
        pred = torch.zeros(1, 1, 192)
        actual = torch.ones(1, 1, 192)  # MSE = 192
        signal = detector.check(predicted=pred, actual_emb=actual)
        assert signal.drift_exceeded is False

    def test_trend_increasing(self):
        """Feed increasing drift values → trend should be increasing."""
        detector = DriftDetector(threshold=0.01, window=5)
        base = torch.zeros(1, 1, 192)

        # Increasing drift: each actual is farther from predicted
        for i in range(5):
            actual = base + (i + 1) * 0.1
            detector.check(predicted=base, actual_emb=actual)

        # The last check should show increasing trend
        signal = detector.check(predicted=base, actual_emb=base + 0.6)
        assert signal.trend_increasing is True

    def test_trend_decreasing(self):
        """Feed decreasing drift values → trend should NOT be increasing."""
        detector = DriftDetector(threshold=0.01, window=5)
        base = torch.zeros(1, 1, 192)

        # Decreasing drift: each actual is closer to predicted
        for i in range(5, 0, -1):
            actual = base + i * 0.1
            detector.check(predicted=base, actual_emb=actual)

        signal = detector.check(predicted=base, actual_emb=base + 0.01)
        assert signal.trend_increasing is False

    def test_trend_requires_min_2_points(self):
        """With only 1 data point, trend should be False."""
        detector = DriftDetector(threshold=0.01, window=5)
        emb = torch.randn(1, 1, 192)
        signal = detector.check(predicted=emb, actual_emb=emb + 1.0)
        assert signal.trend_increasing is False

    def test_escalate_high_and_increasing(self):
        """escalate_to_s2 = drift_exceeded AND trend_increasing."""
        detector = DriftDetector(threshold=0.5, window=3)
        base = torch.zeros(1, 1, 192)

        # Build up increasing drift above threshold
        for i in range(4):
            scale = (i + 1) * 0.05  # 0.05, 0.10, 0.15, 0.20 per dim
            actual = base + scale
            detector.check(predicted=base, actual_emb=actual)

        # This should exceed threshold and be increasing
        signal = detector.check(predicted=base, actual_emb=base + 0.25)
        # MSE for 0.25 per dim = 0.25^2 * 192 = 12.0
        assert signal.drift_exceeded is True
        assert signal.trend_increasing is True
        assert signal.escalate_to_s2 is True

    def test_no_escalate_stable_high_drift(self):
        """High but stable drift → exceeded but NOT increasing → no escalate."""
        detector = DriftDetector(threshold=0.5, window=5)
        base = torch.zeros(1, 1, 192)
        # Same high drift every time (flat trend)
        offset = torch.full((1, 1, 192), 0.1)
        for _ in range(6):
            detector.check(predicted=base, actual_emb=base + offset)

        signal = detector.check(predicted=base, actual_emb=base + offset)
        assert signal.drift_exceeded is True
        assert signal.trend_increasing is False
        assert signal.escalate_to_s2 is False

    def test_history_tracking(self):
        detector = DriftDetector(threshold=0.1)
        emb = torch.randn(1, 1, 192)
        for _ in range(5):
            detector.check(predicted=emb, actual_emb=emb + 0.01)

        history = detector.get_history()
        assert len(history) == 5
        assert all(isinstance(v, float) for v in history)

    def test_reset(self):
        detector = DriftDetector(threshold=0.1)
        emb = torch.randn(1, 1, 192)
        detector.check(predicted=emb, actual_emb=emb + 0.1)
        assert len(detector.get_history()) == 1

        detector.reset()
        assert len(detector.get_history()) == 0
        assert detector.last_drift is None
        assert detector.mean_drift is None

    def test_last_drift(self):
        detector = DriftDetector()
        assert detector.last_drift is None

        emb = torch.zeros(1, 1, 192)
        detector.check(predicted=emb, actual_emb=emb)
        assert detector.last_drift == pytest.approx(0.0, abs=1e-6)

    def test_mean_drift(self):
        detector = DriftDetector(window=3)
        base = torch.zeros(1, 1, 192)

        # 3 checks with known drift
        for scale in [0.0, 0.0, 0.0]:
            detector.check(predicted=base, actual_emb=base + scale)

        assert detector.mean_drift == pytest.approx(0.0, abs=1e-6)

    def test_window_limits_trend(self):
        """Old high-drift data outside window shouldn't affect trend."""
        detector = DriftDetector(threshold=0.01, window=3)
        base = torch.zeros(1, 1, 192)

        # First: 3 high-drift steps (old data)
        for _ in range(3):
            detector.check(predicted=base, actual_emb=base + 1.0)

        # Then: 3 decreasing steps (recent data within window)
        for i in [0.3, 0.2, 0.1]:
            detector.check(predicted=base, actual_emb=base + i)

        signal = detector.check(predicted=base, actual_emb=base + 0.05)
        # Recent window [0.2*192, 0.1*192, 0.05*192] is decreasing
        assert signal.trend_increasing is False

    def test_requires_actual_input(self):
        """Must provide either actual_emb or (actual_obs + pipeline)."""
        detector = DriftDetector()
        with pytest.raises(ValueError, match="actual_emb"):
            detector.check(predicted=torch.randn(1, 1, 192))

    def test_with_actual_obs_and_pipeline(self):
        """check() can encode raw images via pipeline."""

        class MockPipeline:
            def preprocess(self, img):
                return torch.randn(1, 1, 3, 224, 224)
            def encode(self, tensor):
                return torch.zeros(1, 1, 192)  # always returns zeros

        detector = DriftDetector(threshold=0.1)
        pipeline = MockPipeline()
        pred = torch.zeros(1, 1, 192)
        obs = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        signal = detector.check(predicted=pred, actual_obs=obs, pipeline=pipeline)
        assert signal.drift_mse == pytest.approx(0.0, abs=1e-6)
