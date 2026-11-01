"""Semantic channel of the Gaussian map (Sec. III-A).

VCS-SLAM stores a C-dimensional semantic logit vector s_i on every Gaussian. The logits are
alpha-composited into the semantic map S_p = sum_i s_i alpha_i prod_{j<i} (1 - alpha_j) (Eq. 1) and
supervised with per-pixel cross-entropy against one-hot labels. The CUDA rasterizer renders three
channels per pass, so S_p is rendered in ceil(C / 3) passes over the same geometry. Alpha blending
is linear in the per-Gaussian features, so this is exactly a C-channel render.

The SGS-SLAM baseline (``representation="color"``) keeps SGS-SLAM's 3-channel semantic colours.
"""

from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.slam_helpers import transformed_semantics2rendervar

REPRESENTATIONS = ("logits", "color")


def _read_label_ids(path):
    ids = np.asarray(imageio.imread(path))
    if ids.ndim == 3:
        ids = ids[..., 0]
    unique, first = np.unique(ids.reshape(-1).astype(np.int64), return_index=True)
    return unique, first


class LabelSpace:
    """The semantic classes of a sequence: dataset class id <-> logit channel, and a display colour per class."""

    def __init__(self, class_ids, colors):
        order = np.argsort(np.asarray(class_ids, dtype=np.int64))
        self.class_ids = np.asarray(class_ids, dtype=np.int64)[order]
        self.colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[order]
        if len(self.class_ids) == 0 or self.class_ids[0] < 0:
            raise ValueError("A label space needs at least one non-negative class id.")
        self._lut = np.full(int(self.class_ids[-1]) + 1, -1, dtype=np.int64)
        self._lut[self.class_ids] = np.arange(len(self.class_ids))
        self._device_cache = {}

    @property
    def num_classes(self):
        return len(self.class_ids)

    @classmethod
    def from_dataset(cls, dataset, num_workers=8):
        """Collect the classes that occur in the sequence's label images and their palette colours."""
        id_paths, color_paths = list(dataset.semantic_id_paths), list(dataset.semantic_color_paths)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            per_frame = list(pool.map(_read_label_ids, id_paths))
        palette = {}
        for (unique, first), color_path in zip(per_frame, color_paths):
            new = [(int(u), int(f)) for u, f in zip(unique, first) if int(u) not in palette]
            if not new:
                continue
            colors = np.asarray(imageio.imread(color_path))[..., :3].reshape(-1, 3)
            for class_id, flat_index in new:
                palette[class_id] = colors[flat_index]
        if not palette:
            raise RuntimeError("No semantic labels found; check the dataset's semantic_id folder.")
        ids = sorted(palette)
        return cls(ids, [palette[i] for i in ids])

    def state_dict(self):
        return {"label_space_ids": self.class_ids.copy(), "label_space_colors": self.colors.copy()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["label_space_ids"], state["label_space_colors"])

    def _tensors(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                torch.from_numpy(self._lut).to(device),
                torch.from_numpy(self.class_ids).to(device),
                torch.from_numpy(self.colors).to(device),
            )
        return self._device_cache[key]

    def to_index(self, ids):
        """Class ids (any shape, any numeric dtype) -> logit channel; -1 for ids outside the label space."""
        lut, _, _ = self._tensors(ids.device)
        ids = ids.long()
        inside = (ids >= 0) & (ids < lut.numel())
        index = torch.full_like(ids, -1)
        index[inside] = lut[ids[inside]]
        return index

    def to_ids(self, index):
        """Logit channel -> class id."""
        _, class_ids, _ = self._tensors(index.device)
        return class_ids[index]

    def palette(self, ids):
        """Class ids (H, W) -> uint8 RGB (H, W, 3); unknown ids are black."""
        _, _, colors = self._tensors(ids.device)
        index = self.to_index(ids)
        rgb = colors[index.clamp(min=0)]
        rgb[index < 0] = 0
        return rgb

    def colorize(self, ids):
        """Class ids (H, W) -> float RGB image (3, H, W) in [0, 1]."""
        return self.palette(ids).permute(2, 0, 1).float() / 255.0


def render_semantic_logits(params, transformed_pts, cam):
    """Render the semantic logit map S_p (Eq. 1) as a (C, H, W) tensor, three channels per pass."""
    logits = params["semantic_logits"]
    num_classes = logits.shape[1]
    pad = (-num_classes) % 3
    if pad:
        logits = F.pad(logits, (0, pad))
    shared = {
        "means3D": transformed_pts,
        "rotations": F.normalize(params["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(torch.tile(params["log_scales"], (1, 3))),
    }
    rasterizer = Renderer(raster_settings=cam)
    chunks = []
    for start in range(0, logits.shape[1], 3):
        means2D = torch.zeros_like(params["means3D"], requires_grad=True) + 0
        chunk, _, _ = rasterizer(colors_precomp=logits[:, start:start + 3].contiguous(), means2D=means2D, **shared)
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)[:num_classes]


class SemanticField:
    """How semantics are stored on the Gaussians, initialised, rendered and exported."""

    def __init__(self, representation="logits", label_space=None, init_logit=1.0, device="cuda"):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"semantics.representation must be one of {REPRESENTATIONS}, got {representation!r}")
        if representation == "logits" and label_space is None:
            raise ValueError("The logit representation needs a label space.")
        self.representation = representation
        self.label_space = label_space
        self.init_logit = float(init_logit)
        self.device = device

    @property
    def uses_logits(self):
        return self.representation == "logits"

    @property
    def param_key(self):
        return "semantic_logits" if self.uses_logits else "semantic_colors"

    @property
    def opt_exclude(self):
        """Per-Gaussian arrays that are carried along but not optimised."""
        return set() if self.uses_logits else {"semantic_ids"}

    def init_params(self, point_cld):
        """Semantic parameters of new Gaussians from a point cloud built by ``get_pointcloud``.

        Columns 6 and 7-9 hold the pixel's class id and semantic colour. A new Gaussian's logits
        start as ``init_logit`` times the one-hot vector of its pixel label.
        """
        if not self.uses_logits:
            return {"semantic_ids": point_cld[:, 6], "semantic_colors": point_cld[:, 7:10]}
        index = self.label_space.to_index(point_cld[:, 6].round())
        logits = torch.zeros((point_cld.shape[0], self.label_space.num_classes),
                             dtype=torch.float, device=point_cld.device)
        known = index >= 0
        logits[known, index[known]] = self.init_logit
        return {"semantic_logits": logits}

    def target(self, semantic_id):
        """Per-frame supervision: label-space indices (H, W) for the logit field."""
        return self.label_space.to_index(semantic_id[0])

    def render(self, params, transformed_pts, cam):
        if self.uses_logits:
            return render_semantic_logits(params, transformed_pts, cam)
        rendervar = transformed_semantics2rendervar(params, transformed_pts, device=self.device)
        rendered, _, _ = Renderer(raster_settings=cam)(**rendervar)
        return rendered

    def export(self, params):
        """Arrays stored with the final map so the viewers and the PLY export can show semantics."""
        if not self.uses_logits:
            return {"semantic_ids": params["semantic_ids"].type(torch.uint8)}
        with torch.no_grad():
            index = params["semantic_logits"].argmax(dim=1)
            ids = self.label_space.to_ids(index)
            colors = self.label_space.palette(ids).float() / 255.0
        dtype = torch.uint8 if int(self.label_space.class_ids[-1]) < 256 else torch.int32
        exported = {"semantic_ids": ids.to(dtype), "semantic_colors": colors}
        exported.update(self.label_space.state_dict())
        return exported
