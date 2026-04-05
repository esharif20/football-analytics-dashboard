"""Tests for pitch keypoint padding and detection robustness.

Verifies that variable-length output from sv.KeyPoints.from_inference (older
supervision versions return only detected keypoints, not zero-padded to 32) is
correctly padded to fixed length so downstream boolean filters never mismatch
CONFIG.vertices (always 32 elements).
"""

import sys
import os
from unittest.mock import MagicMock, patch

# ── Mock heavy CV dependencies before any pipeline imports ──────────────
_cv2_mock = MagicMock()
sys.modules.setdefault("cv2", _cv2_mock)
sys.modules.setdefault("supervision", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("torch", MagicMock())

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline", "src")
sys.path.insert(0, _src_dir)

import numpy as np
import pytest

# Import after mocks are installed
from utils.pitch_detector import _pad_inference_keypoints


# ── Helpers ─────────────────────────────────────────────────────────────

NUM_VERTICES = 32


def _make_result(n_preds: int, start_class_id: int = 0) -> object:
    """Build a mock Roboflow inference result matching the actual API structure.

    The Roboflow keypoint model response is:
        result.predictions[0].keypoints  — list of keypoint objects
    Each keypoint has .class_id, .x, .y, .confidence.

    Keypoints are placed at class_ids [start_class_id, start_class_id+n_preds).
    Each gets x=float(cid)*10, y=float(cid)*10+1, confidence=0.9.
    """
    result = MagicMock()
    kps = []
    for i in range(n_preds):
        cid = start_class_id + i
        kp = MagicMock()
        kp.class_id = cid
        kp.x = float(cid) * 10.0
        kp.y = float(cid) * 10.0 + 1.0
        kp.confidence = 0.9
        kps.append(kp)
    pred0 = MagicMock()
    pred0.keypoints = kps
    result.predictions = [pred0]
    return result


def _make_sv_keypoints(n: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return (xy, conf) arrays simulating sv.KeyPoints with n keypoints."""
    xy = np.random.rand(1, n, 2).astype(np.float32)
    conf = np.random.rand(1, n).astype(np.float32)
    return xy, conf


# ── Tests for _pad_inference_keypoints ──────────────────────────────────


class TestPadInferenceKeypoints:
    """Unit tests for the _pad_inference_keypoints helper."""

    def test_full_32_keypoints_passthrough(self):
        """When sv output already has 32 keypoints, arrays are returned unchanged."""
        xy, conf = _make_sv_keypoints(32)
        result = MagicMock()
        out_xy, out_conf = _pad_inference_keypoints(result, xy, conf, expected_n=32)
        assert out_xy is xy
        assert out_conf is conf

    def test_partial_keypoints_padded_to_32(self):
        """When sv output has 29 keypoints, output is padded to shape (1, 32, 2)."""
        xy_29, conf_29 = _make_sv_keypoints(29)
        result = _make_result(n_preds=29, start_class_id=0)
        out_xy, out_conf = _pad_inference_keypoints(result, xy_29, conf_29, expected_n=32)
        assert out_xy.shape == (1, 32, 2), f"Expected (1,32,2), got {out_xy.shape}"
        assert out_conf.shape == (1, 32), f"Expected (1,32), got {out_conf.shape}"

    def test_partial_keypoints_correct_placement(self):
        """Detections are placed at their class_id index; missing slots stay zero."""
        # Only detect keypoints 5, 10, 20
        result = MagicMock()
        kps = []
        for cid in [5, 10, 20]:
            kp = MagicMock()
            kp.class_id = cid
            kp.x = float(cid) * 3.0
            kp.y = float(cid) * 3.0 + 0.5
            kp.confidence = 0.8
            kps.append(kp)
        pred0 = MagicMock()
        pred0.keypoints = kps
        result.predictions = [pred0]

        # Give sv output of length 3 (not 32)
        xy_short = np.ones((1, 3, 2), dtype=np.float32)
        conf_short = np.ones((1, 3), dtype=np.float32)
        out_xy, out_conf = _pad_inference_keypoints(result, xy_short, conf_short, expected_n=32)

        # Detected indices have expected values
        for cid in [5, 10, 20]:
            assert out_xy[0, cid, 0] == pytest.approx(float(cid) * 3.0)
            assert out_xy[0, cid, 1] == pytest.approx(float(cid) * 3.0 + 0.5)
            assert out_conf[0, cid] == pytest.approx(0.8)

        # Undetected indices are zero
        for cid in [0, 1, 2, 15, 31]:
            assert out_xy[0, cid, 0] == 0.0
            assert out_conf[0, cid] == 0.0

    def test_empty_keypoints_returns_unchanged(self):
        """When xy is None or empty, function returns unchanged values without error."""
        result = MagicMock()
        result.predictions = []

        out_xy, out_conf = _pad_inference_keypoints(result, None, None, expected_n=32)
        assert out_xy is None
        assert out_conf is None

        empty_xy = np.empty((0, 2), dtype=np.float32)
        out_xy2, out_conf2 = _pad_inference_keypoints(result, empty_xy, None, expected_n=32)
        assert out_xy2 is empty_xy  # unchanged (size == 0)

    def test_no_predictions_attr_produces_zero_array(self):
        """If result has no predictions attribute, output is all-zeros of correct shape."""
        result = MagicMock(spec=[])  # no attributes
        xy_short, conf_short = _make_sv_keypoints(15)
        out_xy, out_conf = _pad_inference_keypoints(result, xy_short, conf_short, expected_n=32)
        assert out_xy.shape == (1, 32, 2)
        assert np.all(out_xy == 0.0)

    def test_out_of_range_class_id_ignored(self):
        """Keypoints with class_id outside [0, expected_n) are silently ignored."""
        kp_valid = MagicMock()
        kp_valid.class_id = 5
        kp_valid.x = 100.0
        kp_valid.y = 200.0
        kp_valid.confidence = 0.9

        kp_bad = MagicMock()
        kp_bad.class_id = 99  # out of range
        kp_bad.x = 999.0
        kp_bad.y = 999.0
        kp_bad.confidence = 0.9

        pred0 = MagicMock()
        pred0.keypoints = [kp_valid, kp_bad]
        result = MagicMock()
        result.predictions = [pred0]

        xy_short, conf_short = _make_sv_keypoints(2)
        out_xy, out_conf = _pad_inference_keypoints(result, xy_short, conf_short, expected_n=32)

        assert out_xy.shape == (1, 32, 2)
        assert out_xy[0, 5, 0] == pytest.approx(100.0)
        # Index 99 was ignored — no out-of-bounds write
        assert out_xy[0, 31, 0] == 0.0


# ── Tests for downstream consumer safety ────────────────────────────────


class TestConfMaskAlwaysMatchesVertices:
    """Verify that the conf_mask guard in all.py correctly handles any keypoint count."""

    def _run_detect_logic(self, xy: np.ndarray, conf: np.ndarray, num_vertices: int = 32):
        """Replicate the conf_mask logic from all.py:_detect_single_frame."""
        conf_mask = np.zeros(num_vertices, dtype=bool)
        frame_keypoints = np.empty((0, 2), dtype=np.float32)
        pitch_vertices = np.random.rand(num_vertices, 2).astype(np.float32)
        pitch_keypoints = np.empty((0, 2), dtype=np.float32)

        if conf is not None and len(conf) > 0:
            c = conf[0]
            if c.shape[0] == num_vertices:
                conf_mask = c > 0.5

        if conf_mask.any() and xy is not None and len(xy) > 0:
            x = xy[0]
            if x.shape[0] == num_vertices:
                frame_keypoints = x[conf_mask].astype(np.float32)
                pitch_keypoints = pitch_vertices[conf_mask]

        return conf_mask, frame_keypoints, pitch_keypoints

    @pytest.mark.parametrize("n_kp", [0, 1, 15, 29, 32])
    def test_no_index_error_for_any_keypoint_count(self, n_kp):
        """Boolean filter never raises IndexError regardless of keypoint count."""
        if n_kp == 0:
            xy = np.empty((1, 0, 2), dtype=np.float32)
            conf = np.empty((1, 0), dtype=np.float32)
        else:
            xy, conf = _make_sv_keypoints(n_kp)

        # Should not raise
        conf_mask, frame_kp, pitch_kp = self._run_detect_logic(xy, conf)
        assert conf_mask.shape == (32,)

    def test_padded_32_keypoints_pass_through_filter(self):
        """After padding to 32, the consumer processes detections correctly."""
        # Simulate: model detected 20 keypoints, after padding we have 32
        result = _make_result(n_preds=20, start_class_id=0)
        xy_short, conf_short = _make_sv_keypoints(20)
        # Set all confidences high so they pass the 0.5 threshold
        conf_padded_arr = np.zeros((1, 32), dtype=np.float32)
        conf_padded_arr[0, :20] = 0.9

        xy_padded = np.zeros((1, 32, 2), dtype=np.float32)
        xy_padded[0, :20] = xy_short[0]

        conf_mask, frame_kp, pitch_kp = self._run_detect_logic(xy_padded, conf_padded_arr)
        # 20 keypoints detected, so 20 should pass the threshold
        assert conf_mask.sum() == 20
        assert frame_kp.shape == (20, 2)
        assert pitch_kp.shape == (20, 2)
