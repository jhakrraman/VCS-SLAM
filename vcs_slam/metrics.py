"""Semantic evaluation.

* Tab. V: mIoU under the per-frame visible-class (subset) protocol of SGS-SLAM: each rendered
  pixel is assigned one of the classes present in that frame's ground truth, unlabeled ground-truth
  pixels are ignored, and IoU is averaged over the classes present in the frame.
* Tab. VII: the mechanism metrics of the three modules
    - VCSU: the same mIoU restricted to depth-inconsistent pixels, |D_p - D_p^obs| > 0.05 m
    - SCEA: boundary F1 within a 3-pixel tolerance band around ground-truth morphological edges
    - CAUW: AUROC of the per-pixel depth uncertainty for detecting pixels whose predicted class
      differs from the ground truth
"""

import math

import numpy as np
import torch
from scipy import ndimage

from vcs_slam.reliability import depth_variance

UNLABELED_ID = 0


def predict_visible_classes(sem_logits, gt_ids, label_space):
    """Argmax of the rendered logits restricted to the classes present in the GT frame -> class ids (H, W)."""
    present = label_space.to_index(torch.unique(gt_ids))
    present = present[present >= 0]
    if present.numel() == 0:
        return label_space.to_ids(sem_logits.argmax(dim=0))
    return label_space.to_ids(present[sem_logits[present].argmax(dim=0)])


def _color_keys(image):
    q = torch.round(image * 255.0).long()
    return (q[0] << 16) | (q[1] << 8) | q[2]


def colors_to_ids(recolored, gt_color, gt_ids):
    """Class ids of an SGS-SLAM semantic image recoloured to this frame's GT colours (-1 if unmatched)."""
    gt_keys = _color_keys(gt_color).reshape(-1)
    palette_keys, inverse = torch.unique(gt_keys, return_inverse=True)
    palette_ids = torch.zeros_like(palette_keys)
    palette_ids[inverse] = gt_ids.reshape(-1).long()
    pred_keys = _color_keys(recolored).reshape(-1)
    pos = torch.searchsorted(palette_keys, pred_keys).clamp(max=palette_keys.numel() - 1)
    matched = palette_keys[pos] == pred_keys
    pred_ids = torch.where(matched, palette_ids[pos], torch.full_like(pred_keys, -1))
    return pred_ids.reshape(gt_ids.shape)


def visible_class_miou(pred_ids, gt_ids, mask=None, unlabeled_id=UNLABELED_ID):
    """mIoU over the classes present in the (optionally masked) GT frame; NaN if nothing is left."""
    keep = gt_ids != unlabeled_id
    if mask is not None:
        keep = keep & mask
    gt = gt_ids[keep].long()
    pred = pred_ids[keep].long()
    if gt.numel() == 0:
        return float("nan")
    num = int(max(gt.max().item(), pred.max().item())) + 2
    pred = torch.where(pred < 0, torch.full_like(pred, num - 1), pred)
    confusion = torch.bincount(gt * num + pred, minlength=num * num).reshape(num, num)
    present = torch.unique(gt)
    intersection = confusion[present, present].double()
    union = confusion[present, :].sum(dim=1) + confusion[:, present].sum(dim=0) - confusion[present, present]
    return (intersection / union.double()).mean().item()


def label_boundaries(ids):
    """Morphological edges of a label image: pixels whose 3x3 neighbourhood contains another label."""
    return ndimage.maximum_filter(ids, size=3) != ndimage.minimum_filter(ids, size=3)


def boundary_f1(pred_ids, gt_ids, tolerance=3):
    """Boundary precision, recall and F1 with a ``tolerance``-pixel band (numpy (H, W) label images)."""
    gt_edges = label_boundaries(gt_ids)
    pred_edges = label_boundaries(pred_ids)
    num_gt, num_pred = int(gt_edges.sum()), int(pred_edges.sum())
    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0
    precision = recall = 0.0
    if num_gt and num_pred:
        precision = float((ndimage.distance_transform_edt(~gt_edges)[pred_edges] <= tolerance).mean())
        recall = float((ndimage.distance_transform_edt(~pred_edges)[gt_edges] <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class ErrorDetectionAUROC:
    """Pooled AUROC of an uncertainty score for detecting semantic errors, accumulated over frames.

    The score is the ray-level depth variance V (Eq. 9); the CAUW uncertainty 1 - C_p is a monotonic
    function of V, so both give the same AUROC. Scores are histogrammed on a fine log10 grid so that
    whole sequences can be pooled without keeping every pixel in memory.
    """

    def __init__(self, log10_min=-8.0, log10_max=2.0, bins=10000):
        self.log10_min, self.log10_max, self.bins = log10_min, log10_max, bins
        self.hist_err = np.zeros(bins + 2, dtype=np.int64)
        self.hist_ok = np.zeros(bins + 2, dtype=np.int64)

    def update(self, scores, is_error):
        scores = torch.log10(scores.double().clamp(min=1e-30))
        position = (scores - self.log10_min) / (self.log10_max - self.log10_min) * self.bins
        index = (torch.floor(position) + 1).clamp(0, self.bins + 1).long().cpu().numpy().reshape(-1)
        error = is_error.reshape(-1).cpu().numpy().astype(bool)
        self.hist_err += np.bincount(index[error], minlength=self.bins + 2)
        self.hist_ok += np.bincount(index[~error], minlength=self.bins + 2)

    def compute(self):
        num_err, num_ok = self.hist_err.sum(), self.hist_ok.sum()
        if num_err == 0 or num_ok == 0:
            return float("nan")
        ok_below = np.cumsum(self.hist_ok) - self.hist_ok
        wins = (self.hist_err * (ok_below + 0.5 * self.hist_ok)).sum()
        return float(wins / (float(num_err) * float(num_ok)))


class MechanismMetrics:
    """Per-frame accumulation of the Tab. VII metrics."""

    def __init__(self, inconsistent_depth=0.05, boundary_tolerance=3, unlabeled_id=UNLABELED_ID):
        self.inconsistent_depth = inconsistent_depth
        self.boundary_tolerance = boundary_tolerance
        self.unlabeled_id = unlabeled_id
        self.inconsistent_miou, self.bf_precision, self.bf_recall, self.bf_f1 = [], [], [], []
        self.auroc = ErrorDetectionAUROC()

    def update(self, pred_ids, gt_ids, depth, depth_sq, depth_obs):
        """All inputs are (H, W) tensors: predicted / GT class ids, rendered E[D], E[D^2] and sensor depth."""
        inconsistent = (depth_obs > 0) & (torch.abs(depth - depth_obs) > self.inconsistent_depth)
        miou = visible_class_miou(pred_ids, gt_ids, inconsistent, self.unlabeled_id)
        if not math.isnan(miou):
            self.inconsistent_miou.append(miou)
        p, r, f1 = boundary_f1(pred_ids.cpu().numpy(), gt_ids.cpu().numpy(), self.boundary_tolerance)
        self.bf_precision.append(p)
        self.bf_recall.append(r)
        self.bf_f1.append(f1)
        labeled = gt_ids != self.unlabeled_id
        self.auroc.update(depth_variance(depth, depth_sq)[labeled], (pred_ids != gt_ids)[labeled])

    def summary(self):
        def mean(values):
            return float(np.mean(values)) if values else float("nan")
        return {
            "inconsistent_miou": mean(self.inconsistent_miou),
            "boundary_precision": mean(self.bf_precision),
            "boundary_recall": mean(self.bf_recall),
            "boundary_f1": mean(self.bf_f1),
            "error_auroc": self.auroc.compute(),
            "num_frames": len(self.bf_f1),
            "inconsistent_depth_threshold_m": self.inconsistent_depth,
            "boundary_tolerance_px": self.boundary_tolerance,
        }
