"""Structured corruption of the per-frame 2D semantic priors (Sec. IV-F, Tab. VIII).

Only the labels that supervise the map are corrupted; evaluation always uses the clean ground
truth. Each frame draws from its own generator seeded with (seed, frame index), so a corrupted run
is reproducible and does not disturb the random stream of the SLAM loop.

    boundary   k px    boundary misalignment: morphological dilation or erosion of the label map
                       with a (2k+1) x (2k+1) square, chosen at random per frame
    confusion  p       each segment (connected component of one class) is relabelled with
                       probability p to the visually most similar class of the frame, i.e. the
                       class with the nearest mean RGB colour
    missed     q       each small segment (< ``small_object_frac`` of the image) is dropped with
                       probability q and filled with the dominant label around it
    uniform    r       control: a fraction r of the pixels is reassigned to classes drawn uniformly
                       from the label space
"""

import numpy as np
import torch
from scipy import ndimage

KINDS = ("boundary", "confusion", "missed", "uniform")


def _segments(labels, class_id):
    components, count = ndimage.label(labels == class_id)
    return components, ndimage.find_objects(components), count


def _boundary(labels, k, rng):
    size = 2 * int(round(k)) + 1
    if rng.random() < 0.5:
        return ndimage.grey_dilation(labels, size=(size, size), mode="nearest")
    return ndimage.grey_erosion(labels, size=(size, size), mode="nearest")


def _confusion(labels, p, rng, rgb, unlabeled_id):
    out = labels.copy()
    classes = [c for c in np.unique(labels) if c != unlabeled_id]
    if len(classes) < 2:
        return out
    means = np.stack([rgb[labels == c].mean(axis=0) for c in classes])
    for i, class_id in enumerate(classes):
        distance = np.linalg.norm(means - means[i], axis=1)
        distance[i] = np.inf
        similar = classes[int(np.argmin(distance))]
        components, boxes, count = _segments(labels, class_id)
        for j in range(count):
            if rng.random() < p:
                box = boxes[j]
                region = components[box] == j + 1
                out[box][region] = similar
    return out


def _missed(labels, q, rng, small_object_frac, unlabeled_id):
    out = labels.copy()
    max_area = small_object_frac * labels.size
    for class_id in [c for c in np.unique(labels) if c != unlabeled_id]:
        components, boxes, count = _segments(labels, class_id)
        for j in range(count):
            box = boxes[j]
            grown = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
            region = components[grown] == j + 1
            if region.sum() >= max_area or rng.random() >= q:
                continue
            ring = ndimage.binary_dilation(region) & ~region
            around = labels[grown][ring]
            around = around[around != class_id]
            fill = np.bincount(around).argmax() if around.size else unlabeled_id
            out[grown][region] = fill
    return out


def _uniform(labels, r, rng, vocabulary):
    out = labels.copy()
    flip = rng.random(labels.shape) < r
    out[flip] = rng.choice(np.asarray(vocabulary), size=int(flip.sum()))
    return out


def corrupt_labels(labels, kind, level, rng, rgb=None, vocabulary=None,
                   small_object_frac=0.01, unlabeled_id=0):
    """Return a corrupted copy of an (H, W) integer label image."""
    labels = np.asarray(labels, dtype=np.int64)
    if kind == "boundary":
        return _boundary(labels, level, rng)
    if kind == "confusion":
        if rgb is None:
            raise ValueError("Class confusion needs the RGB frame.")
        return _confusion(labels, level, rng, np.asarray(rgb, dtype=np.float64), unlabeled_id)
    if kind == "missed":
        return _missed(labels, level, rng, small_object_frac, unlabeled_id)
    if kind == "uniform":
        if vocabulary is None:
            raise ValueError("Uniform reassignment needs the label vocabulary.")
        return _uniform(labels, level, rng, vocabulary)
    raise ValueError(f"Unknown corruption {kind!r}; expected one of {KINDS}.")


class CorruptedPriorDataset:
    """View of a dataset whose semantic priors are corrupted; RGB-D frames and poses are untouched."""

    def __init__(self, dataset, kind, level, seed, label_space, small_object_frac=0.01, unlabeled_id=0):
        if kind not in KINDS:
            raise ValueError(f"Unknown corruption {kind!r}; expected one of {KINDS}.")
        self.dataset = dataset
        self.kind, self.level, self.seed = kind, level, seed
        self.label_space = label_space
        self.small_object_frac = small_object_frac
        self.unlabeled_id = unlabeled_id

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        color, depth, intrinsics, pose, semantic_id, semantic_color = self.dataset[index]
        rng = np.random.default_rng([int(self.seed), int(index)])
        rgb = color.cpu().numpy() if self.kind == "confusion" else None
        labels = corrupt_labels(semantic_id[..., 0].cpu().numpy(), self.kind, self.level, rng, rgb=rgb,
                                vocabulary=self.label_space.class_ids,
                                small_object_frac=self.small_object_frac, unlabeled_id=self.unlabeled_id)
        labels = torch.from_numpy(labels).to(semantic_id.device)
        semantic_color = self.label_space.palette(labels).to(semantic_color.dtype)
        return color, depth, intrinsics, pose, labels.to(semantic_id.dtype)[..., None], semantic_color
