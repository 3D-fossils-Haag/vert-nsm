"""Tests for NSM/alignment.py.

Strategy: apply a known similarity transform (scale, rotation, translation) to a point set
and check each function recovers or undoes it. No trained model or real meshes needed.
"""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from NSM.alignment import align_to_template, load_mrk_json, rms_err_pct, save_mrk_json


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def landmarks(rng):
    return rng.normal(size=(30, 3))


def random_rotation(seed):
    return Rotation.random(random_state=seed).as_matrix()


# --- align_to_template (Procrustes) ----------------------------------------------

def test_align_recovers_known_similarity(landmarks):
    R, s, t = random_rotation(1), 1.7, np.array([0.3, -2.0, 5.0])
    template = s * landmarks @ R + t

    s_hat, R_hat, t_hat = align_to_template(template, landmarks)

    assert s_hat == pytest.approx(s)
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-10)
    np.testing.assert_allclose(s_hat * landmarks @ R_hat + t_hat, template, atol=1e-10)


def test_align_without_scaling(landmarks):
    R, t = random_rotation(2), np.array([1.0, 2.0, 3.0])
    template = 2.0 * landmarks @ R + t
    s_hat, R_hat, _ = align_to_template(template, landmarks, allow_scaling=False)
    assert s_hat == 1.0
    np.testing.assert_allclose(R_hat, R, atol=1e-10)


def test_align_never_returns_a_reflection(landmarks):
    mirrored = landmarks * np.array([-1, 1, 1])
    _, R_hat, _ = align_to_template(mirrored, landmarks)
    assert np.linalg.det(R_hat) == pytest.approx(1.0)
    np.testing.assert_allclose(R_hat @ R_hat.T, np.eye(3), atol=1e-10)


def test_align_is_robust_to_small_noise(landmarks, rng):
    R, s, t = random_rotation(3), 0.8, np.array([0.0, 1.0, 0.0])
    template = s * landmarks @ R + t + rng.normal(scale=1e-3, size=landmarks.shape)
    s_hat, R_hat, _ = align_to_template(template, landmarks)
    assert s_hat == pytest.approx(s, rel=1e-2)
    np.testing.assert_allclose(R_hat, R, atol=1e-2)


# --- rms_err_pct ------------------------------------------------------------------

def test_rms_err_pct_zero_for_identical(landmarks):
    assert rms_err_pct(landmarks, landmarks) == 0.0


def test_rms_err_pct_known_value():
    orig = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)  # bbox diagonal = 5
    shifted = orig + np.array([0.05, 0, 0])               # every vertex moves 0.05
    assert rms_err_pct(shifted, orig) == pytest.approx(1.0)  # 0.05 / 5 = 1 %
    assert rms_err_pct(shifted, orig, size=1.0) == pytest.approx(5.0)


# --- .mrk.json landmark files -----------------------------------------------------

def test_mrk_json_round_trip(tmp_path, landmarks):
    path = tmp_path / "nested" / "out.mrk.json"
    save_mrk_json(landmarks, str(path))

    points, labels, idx = load_mrk_json(str(path), return_idx=True)
    np.testing.assert_allclose(points, landmarks.astype(np.float32))
    assert labels == [str(i) for i in range(len(landmarks))]
    assert idx == list(range(len(landmarks)))


def test_mrk_json_keeps_atlas_indices_and_template_labels(tmp_path):
    template = {"markups": [{"type": "Fiducial", "controlPoints": [
        {"id": "a", "label": "neural_spine", "position": [0, 0, 0], "description": "index:0"},
        {"id": "b", "label": "left_zyg", "position": [0, 0, 0], "description": "index:1"},
        {"id": "c", "label": "right_zyg", "position": [0, 0, 0], "description": "index:2"},
    ]}]}
    template_path = tmp_path / "template.mrk.json"
    template_path.write_text(json.dumps(template))
    out = tmp_path / "subset.mrk.json"

    save_mrk_json(np.array([[1.0, 2, 3], [4, 5, 6]]), str(out), template_path=str(template_path), idx=[2, 0])

    data = json.loads(out.read_text())
    assert data["markups"][0]["type"] == "Fiducial"  # template metadata kept
    points, labels, idx = load_mrk_json(str(out), return_idx=True)
    assert labels == ["right_zyg", "neural_spine"]
    assert idx == [2, 0]
    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])


def test_load_mrk_json_skips_unplaced_points(tmp_path):
    path = tmp_path / "partial.mrk.json"
    path.write_text(json.dumps({"markups": [{"controlPoints": [
        {"label": "A", "position": [1, 2, 3]},
        {"label": "B", "position": None},
        {"label": "C", "position": [4, 5, 6]},
    ]}]}))
    points, labels, idx = load_mrk_json(str(path), return_idx=True)
    assert labels == ["A", "C"]
    assert idx == [0, 1]  # falls back to position among placed points
    assert points.shape == (2, 3)


def test_load_mrk_json_without_markups_raises(tmp_path):
    path = tmp_path / "empty.mrk.json"
    path.write_text(json.dumps({"markups": []}))
    with pytest.raises(ValueError, match="No 'markups'"):
        load_mrk_json(str(path))