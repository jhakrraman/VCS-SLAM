"""Structured corruption of the 2D semantic priors (Tab. VIII)."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from vcs_slam.priors import CorruptedPriorDataset, corrupt_labels
from vcs_slam.semantics import LabelSpace


def scene_labels():
    """Wall (3) with a poster (20), floor (7), a big box (12) and two small objects (25, 30)."""
    labels = np.full((60, 80), 3, dtype=np.int64)
    labels[40:, :] = 7
    labels[10:20, 50:70] = 20
    labels[25:50, 10:40] = 12
    labels[44:47, 60:63] = 25          # 9 px  < 1% of 4800
    labels[52:55, 70:73] = 30          # 9 px, on the floor
    rgb = np.zeros((60, 80, 3))
    for class_id, color in {3: (200, 180, 160), 7: (60, 60, 70), 20: (190, 175, 165), 12: (200, 40, 40),
                            25: (230, 230, 20), 30: (50, 60, 80)}.items():
        rgb[labels == class_id] = color
    return labels, rgb


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_boundary_misalignment_moves_edges_by_k(k):
    labels, _ = scene_labels()
    out = corrupt_labels(labels, "boundary", k, rng())
    size = (2 * k + 1, 2 * k + 1)
    dilated = ndimage.grey_dilation(labels, size=size, mode="nearest")
    eroded = ndimage.grey_erosion(labels, size=size, mode="nearest")
    assert np.array_equal(out, dilated) or np.array_equal(out, eroded)
    changed = np.argwhere(out != labels)
    edges = ndimage.maximum_filter(labels, 3) != ndimage.minimum_filter(labels, 3)
    distance = ndimage.distance_transform_edt(~edges)
    assert distance[tuple(changed.T)].max() <= np.sqrt(2) * k + 1   # only pixels near a boundary change


def test_class_confusion_relabels_segments_to_the_most_similar_class():
    labels, rgb = scene_labels()
    assert np.array_equal(corrupt_labels(labels, "confusion", 0.0, rng(), rgb=rgb), labels)
    out = corrupt_labels(labels, "confusion", 1.0, rng(), rgb=rgb)
    assert np.all(out[labels == 20] == 3) and np.all(out[labels == 3] == 20)   # poster <-> wall
    assert np.all(out[labels == 7] == 30)                                       # floor -> dark object
    frac = [np.mean(corrupt_labels(labels, "confusion", 0.5, rng(s), rgb=rgb) != labels) for s in range(20)]
    assert 0.0 < np.mean(frac) < 1.0


def test_missed_detections_remove_small_objects_only():
    labels, _ = scene_labels()
    assert np.array_equal(corrupt_labels(labels, "missed", 0.0, rng()), labels)
    out = corrupt_labels(labels, "missed", 1.0, rng())
    assert not np.isin([25, 30], out).any()                 # both small objects missed ...
    assert np.all(out[labels == 25] == 7) and np.all(out[labels == 30] == 7)   # ... and filled with the floor
    big = ~np.isin(labels, [25, 30])
    assert np.array_equal(out[big], labels[big])             # large segments untouched


@pytest.mark.parametrize("r", [0.05, 0.1, 0.2])
def test_uniform_reassignment_rate(r):
    labels, _ = scene_labels()
    vocab = np.array([0, 3, 7, 12, 20, 25, 30])
    out = corrupt_labels(labels, "uniform", r, rng(), vocabulary=vocab)
    expected = r * (1 - 1 / len(vocab))                      # a redraw may hit the original class
    assert abs(np.mean(out != labels) - expected) < 0.02
    assert np.isin(out, vocab).all()


def test_corruption_is_deterministic():
    labels, _ = scene_labels()
    vocab = np.array([3, 7, 12])
    a = corrupt_labels(labels, "uniform", 0.1, rng(7), vocabulary=vocab)
    b = corrupt_labels(labels, "uniform", 0.1, rng(7), vocabulary=vocab)
    c = corrupt_labels(labels, "uniform", 0.1, rng(8), vocabulary=vocab)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_kind():
    with pytest.raises(ValueError):
        corrupt_labels(np.zeros((4, 4), dtype=np.int64), "blur", 1, rng())


class FrameSource:
    """Minimal dataset returning (color, depth, intrinsics, pose, semantic_id, semantic_color) like GradSLAMDataset."""

    def __init__(self, labels, rgb, space):
        self.labels, self.rgb, self.space = labels, rgb, space

    def __len__(self):
        return 3

    def __getitem__(self, index):
        ids = torch.from_numpy(self.labels)
        return (torch.from_numpy(self.rgb).float(), torch.ones(*self.labels.shape, 1), torch.eye(4), torch.eye(4),
                ids[..., None], self.space.palette(ids).float())


def test_corrupted_dataset_changes_priors_only():
    labels, rgb = scene_labels()
    ids = sorted(np.unique(labels).tolist())
    space = LabelSpace(ids, [(10 * i, 20 * i, 30 * i) for i in range(len(ids))])
    clean = FrameSource(labels, rgb, space)
    noisy = CorruptedPriorDataset(clean, "uniform", 0.2, seed=0, label_space=space)
    color, depth, _, _, sem_id, sem_color = noisy[1]
    assert torch.equal(color, clean[1][0]) and torch.equal(depth, clean[1][1])
    assert sem_id.shape == (60, 80, 1) and (sem_id[..., 0].numpy() != labels).mean() > 0.1
    assert torch.equal(sem_color, space.palette(sem_id[..., 0]).float())        # colours follow the labels
    assert torch.equal(noisy[1][4], sem_id) and not torch.equal(noisy[2][4], sem_id)   # per-frame seeds
    assert len(noisy) == 3 and noisy.labels is labels                            # attributes forwarded
