"""Semantic evaluation: Tab. V mIoU protocol and the Tab. VII mechanism metrics."""

import numpy as np
import pytest
import torch

from utils.eval_helpers import evaluate_miou, recolor_semantic_img
from vcs_slam.metrics import (ErrorDetectionAUROC, MechanismMetrics, boundary_f1, colors_to_ids,
                              label_boundaries, predict_visible_classes, visible_class_miou)
from vcs_slam.semantics import LabelSpace

PALETTE = {0: (0, 0, 0), 3: (192, 0, 128), 7: (0, 192, 0), 12: (128, 0, 192), 20: (224, 192, 0)}


def colorize(ids):
    rgb = np.zeros(ids.shape + (3,), dtype=np.float32)
    for k, c in PALETTE.items():
        rgb[ids == k] = c
    return torch.from_numpy(rgb / 255.0).permute(2, 0, 1)


def random_maps(seed, h=24, w=32):
    rng = np.random.default_rng(seed)
    gt = rng.choice([0, 3, 7, 12], size=(h, w))
    pred = np.where(rng.random((h, w)) < 0.7, gt, rng.choice([3, 7, 12, 20], size=(h, w)))
    return torch.from_numpy(pred), torch.from_numpy(gt)


@pytest.mark.parametrize("seed", range(5))
def test_label_miou_equals_sgs_slam_colour_protocol(seed):
    """visible_class_miou on class ids == SGS-SLAM's evaluate_miou on the same maps as colours."""
    pred, gt = random_maps(seed)
    assert visible_class_miou(pred, gt) == pytest.approx(evaluate_miou(colorize(pred.numpy()), colorize(gt.numpy())),
                                                         abs=1e-12)


def test_label_miou_by_hand():
    gt = torch.tensor([[3, 3, 7, 7], [0, 0, 7, 7]])
    pred = torch.tensor([[3, 7, 7, 7], [3, 3, 7, 3]])
    # unlabeled GT pixels are ignored; class 3: inter 1, union 3; class 7: inter 3, union 5
    assert visible_class_miou(pred, gt) == pytest.approx((1 / 3 + 3 / 5) / 2)
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    assert visible_class_miou(pred, gt, mask) == pytest.approx(1 / 2)
    assert np.isnan(visible_class_miou(pred, gt, torch.zeros_like(mask)))


def test_visible_class_prediction_and_colour_round_trip():
    space = LabelSpace(list(PALETTE), list(PALETTE.values()))
    logits = torch.zeros(space.num_classes, 4, 4)
    logits[space.to_index(torch.tensor(20))] = 5.0     # class 20 dominates but is absent from the GT frame
    logits[space.to_index(torch.tensor(7))] = 1.0
    gt = torch.full((4, 4), 3)
    gt[:, 2:] = 7
    assert set(predict_visible_classes(logits, gt, space).unique().tolist()) == {7}
    # SGS-SLAM colour path: recolour to the GT colours, then back to ids
    gt_rgb = colorize(gt.numpy())
    noisy = (gt_rgb + 0.02 * torch.randn_like(gt_rgb)).clamp(0, 1)
    assert torch.equal(colors_to_ids(recolor_semantic_img(noisy, gt_rgb), gt_rgb, gt), gt)


def test_label_space_lookup():
    space = LabelSpace([20, 3, 7], [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    assert space.class_ids.tolist() == [3, 7, 20]
    assert space.to_index(torch.tensor([7, 20, 5, 300, -1])).tolist() == [1, 2, -1, -1, -1]
    assert space.to_ids(torch.tensor([0, 2])).tolist() == [3, 20]
    assert space.palette(torch.tensor([[20, 5]])).tolist() == [[[1, 1, 1], [0, 0, 0]]]


def test_boundary_f1_tolerance_band():
    gt = np.zeros((40, 40), dtype=np.int64)
    gt[:, 20:] = 3
    assert boundary_f1(gt, gt, 3) == (1.0, 1.0, 1.0)
    shifted = np.zeros_like(gt)
    shifted[:, 22:] = 3                                     # boundary displaced by 2 px
    assert boundary_f1(shifted, gt, 3)[2] == 1.0
    assert boundary_f1(shifted, gt, 1) == (0.5, 0.5, 0.5)   # half of each edge band is within 1 px
    far = np.zeros_like(gt)
    far[:, 26:] = 3                                         # displaced by 6 px: outside the 3 px band
    assert boundary_f1(far, gt, 3)[2] == 0.0
    assert boundary_f1(np.zeros_like(gt), gt, 3) == (0.0, 0.0, 0.0)   # boundary missed entirely
    assert label_boundaries(gt).sum() == 2 * 40             # morphological edge: one pixel on each side


def exact_auroc(scores, errors):
    """Mann-Whitney AUROC with ties counted as 1/2."""
    pos, neg = scores[errors], scores[~errors]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (greater + 0.5 * ties) / (len(pos) * len(neg))


def test_error_detection_auroc():
    rng = np.random.default_rng(0)
    errors = rng.random(3000) < 0.3
    scores = 10 ** rng.uniform(-7, 0, 3000) * np.where(errors, 3.0, 1.0)   # errors have larger variance
    acc = ErrorDetectionAUROC()
    for chunk in np.array_split(np.arange(3000), 7):                       # pooled over "frames"
        acc.update(torch.from_numpy(scores[chunk]), torch.from_numpy(errors[chunk]))
    assert acc.compute() == pytest.approx(exact_auroc(scores, errors), abs=2e-3)
    assert np.isnan(ErrorDetectionAUROC().compute())


def test_mechanism_metrics_use_absolute_depth_threshold():
    gt = torch.full((6, 6), 3)
    gt[:, 3:] = 7
    pred = gt.clone()
    pred[0, 0] = 7                                             # one wrong pixel ...
    depth = torch.full((6, 6), 2.0)
    depth_obs = depth.clone()
    depth_obs[0, 0] = 2.06                                     # ... where |D - D_obs| = 0.06 m > 0.05 m
    depth_obs[5, 5] = 2.04                                     # 0.04 m: consistent
    depth_sq = depth ** 2
    depth_sq[0, 0] += 0.5                                      # high depth variance at the error
    m = MechanismMetrics()
    m.update(pred, gt, depth, depth_sq, depth_obs)
    s = m.summary()
    assert s["inconsistent_miou"] == 0.0                       # only (0, 0) is depth-inconsistent, and it is wrong
    assert s["error_auroc"] == 1.0
    assert s["inconsistent_depth_threshold_m"] == 0.05 and s["boundary_tolerance_px"] == 3
