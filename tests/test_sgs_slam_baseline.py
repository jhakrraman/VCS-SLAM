"""The SGS-SLAM baseline mode (configs/replica/sgs_slam.py) reproduces SGS-SLAM's loss bit for bit.

Losses and gradients of every parameter are compared with ``torch.equal`` against SGS-SLAM's original
``get_loss`` (tests/reference/sgs_slam_loss.py) for the three branches of the loss (tracking with and
without the silhouette mask, mapping), including a stage whose semantic weight is 0 (ScanNet tracking).
"""

import numpy as np
import pytest
import torch

from reference.sgs_slam_loss import get_loss as sgs_slam_get_loss
from scripts.slam import get_loss
from utils.recon_helpers import setup_camera
from vcs_slam.reliability import ReliabilityConfig
from vcs_slam.semantics import SemanticField

H, W, FRAMES = 12, 16, 3
K = np.array([[14.0, 0.0, 7.5], [0.0, 14.0, 5.5], [0.0, 0.0, 1.0]])


def make_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, 14), torch.linspace(-1.3, 1.3, 18), indexing="ij")
    n = xs.numel()
    depth = 2.0 + 0.5 * torch.rand(n, generator=g)
    means = torch.stack([xs.reshape(-1) * depth / 2, ys.reshape(-1) * depth / 2, depth], dim=1)
    params = {
        "means3D": means + 0.02 * torch.randn(n, 3, generator=g),
        "rgb_colors": torch.rand(n, 3, generator=g),
        "unnorm_rotations": torch.randn(n, 4, generator=g),
        "logit_opacities": 2.0 + torch.randn(n, 1, generator=g),
        "log_scales": torch.log(0.08 + 0.04 * torch.rand(n, 1, generator=g)),
        "semantic_ids": torch.randint(0, 5, (n,), generator=g).float(),
        "semantic_colors": torch.rand(n, 3, generator=g),
        "cam_unnorm_rots": torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 4, 1).repeat(1, 1, FRAMES)
                           + 0.01 * torch.randn(1, 4, FRAMES, generator=g),
        "cam_trans": 0.01 * torch.randn(1, 3, FRAMES, generator=g),
    }
    obs_depth = 2.0 + 0.5 * torch.rand(1, H, W, generator=g)
    obs_depth[0, :2, :3] = 0.0                                   # missing sensor depth
    data = {
        "cam": setup_camera(W, H, K, np.eye(4), device="cpu"),
        "im": torch.rand(3, H, W, generator=g),
        "depth": obs_depth,
        "id": 1,
        "intrinsics": torch.tensor(K).float(),
        "w2c": torch.eye(4),
        "semantic_id": torch.randint(0, 5, (1, H, W), generator=g),
        "semantic_color": torch.rand(3, H, W, generator=g),
    }
    return params, data


def run(loss_fn, params, data, **kwargs):
    p = {k: (v.clone() if k == "semantic_ids" else torch.nn.Parameter(v.clone())) for k, v in params.items()}
    variables = {k: torch.zeros(p["means3D"].shape[0]) for k in ("max_2D_radius", "means2D_gradient_accum", "denom")}
    loss, variables, losses = loss_fn(p, data, variables, 1, **kwargs)
    loss.backward()
    grads = {k: v.grad for k, v in p.items() if isinstance(v, torch.nn.Parameter)}
    return loss, losses, grads, variables


STAGES = {
    "tracking_sil": dict(loss_weights=dict(im=0.5, depth=1.0, seg=0.05), use_sil_for_loss=True, sil_thres=0.99,
                         use_l1=True, ignore_outlier_depth_loss=False, tracking=True),
    "tracking_outlier": dict(loss_weights=dict(im=0.5, depth=1.0, seg=0.05), use_sil_for_loss=False, sil_thres=0.99,
                             use_l1=True, ignore_outlier_depth_loss=True, tracking=True),
    "tracking_nomask": dict(loss_weights=dict(im=0.5, depth=1.0, seg=0.05), use_sil_for_loss=False, sil_thres=0.99,
                            use_l1=True, ignore_outlier_depth_loss=False, tracking=True),
    "tracking_no_semantics": dict(loss_weights=dict(im=0.5, depth=1.0, seg=0), use_sil_for_loss=True, sil_thres=0.99,
                                  use_l1=True, ignore_outlier_depth_loss=False, tracking=True),
    "mapping": dict(loss_weights=dict(im=0.5, depth=1.0, seg=0.1), use_sil_for_loss=False, sil_thres=0.5,
                    use_l1=True, ignore_outlier_depth_loss=False, mapping=True),
}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("stage", list(STAGES))
def test_bitwise_equal_to_sgs_slam(stage, seed):
    params, data = make_inputs(seed)
    kwargs = dict(STAGES[stage], device="cpu", load_semantics=True)
    ref_loss, ref_losses, ref_grads, ref_vars = run(sgs_slam_get_loss, params, data, **kwargs)
    loss, losses, grads, variables = run(get_loss, params, data, **kwargs,
                                         semantic_field=SemanticField("color", device="cpu"),
                                         reliability_cfg=ReliabilityConfig())
    assert ref_loss.item() > 0
    assert torch.equal(loss, ref_loss)
    for key in ("im", "depth", "seg"):
        if STAGES[stage]["loss_weights"]["seg"] or key != "seg":
            assert torch.equal(losses[key], ref_losses[key]), key
    assert set(grads) == set(ref_grads)
    for key, ref_grad in ref_grads.items():
        grad = grads[key]
        if ref_grad is None or grad is None:
            # A stage with semantic weight 0 is not rendered here; SGS-SLAM renders it and back-propagates
            # exact zeros. Both leave the parameters untouched.
            assert all(g is None or not g.any() for g in (grad, ref_grad)), key
        else:
            assert torch.equal(grad, ref_grad), key
    assert torch.equal(variables["max_2D_radius"], ref_vars["max_2D_radius"])
