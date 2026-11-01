"""The semantic objective (vcs_slam/reliability.py) against the paper's equations (tests/reference_equations.py)."""

import itertools

import numpy as np
import pytest
import torch

import reference_equations as ref
from vcs_slam import reliability as rel
from vcs_slam.reliability import ReliabilityConfig

C, H, W = 5, 9, 11


def make_view(seed=0):
    """A random rendered view in float64: logits, rendered depth moments, sensor depth with holes, labels."""
    rng = np.random.default_rng(seed)
    S = rng.normal(0.0, 2.0, (C, H, W))
    D = rng.uniform(0.5, 3.0, (H, W))
    D[0, 0] = 0.0                                                # no rendered depth
    D_obs = D + rng.normal(0.0, 0.08, (H, W))                     # residuals around sigma_vis
    D_obs[rng.random((H, W)) < 0.1] = 0.0                         # missing sensor depth
    D2 = D ** 2 + rng.uniform(0.0, 0.02, (H, W))                   # E[D^2] >= E[D]^2
    D2[1, 2] = D[1, 2] ** 2 - 1e-3                                 # negative variance -> clamped to eps
    y = rng.integers(-1, C, (H, W))
    return S, D, D2, D_obs, y


def to_torch(S, D, D2, D_obs, y):
    return (torch.tensor(S), torch.tensor(D)[None], torch.tensor(D2)[None], torch.tensor(D_obs)[None],
            torch.tensor(y))


def test_defaults_are_the_paper_values():
    cfg = ReliabilityConfig()
    assert (cfg.sigma_vis, cfg.tau_d, cfg.gamma, cfg.beta, cfg.lambda_edge, cfg.eps) == (0.05, 0.75, 100.0, 0.7, 0.02, 1e-8)
    assert cfg.vcsu and cfg.scea and cfg.cauw
    assert (cfg.vcsu_variant, cfg.scea_variant, cfg.cauw_variant) == ("soft", "edge", "variance")


@pytest.mark.parametrize("hard", [False, True])
def test_visibility_mask_eq5(hard):
    S, D, D2, D_obs, y = make_view(1)
    expected = ref.visibility_mask(D, D_obs, 0.05, hard)
    got = rel.visibility_mask(torch.tensor(D)[None], torch.tensor(D_obs)[None], 0.05, hard=hard)[0].numpy()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


def test_conflict_confidence_eqs9_10():
    S, D, D2, D_obs, y = make_view(2)
    expected = ref.conflict_confidence(D, D2, 100.0, 1e-8)
    got = rel.conflict_confidence(torch.tensor(D)[None], torch.tensor(D2)[None], 100.0, 1e-8)[0].numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
    assert got[1, 2] == pytest.approx(1.0 / (1.0 + 100.0 * 1e-8))  # negative variance clamped to eps


def test_entropy_confidence_tab9():
    S, *_ = make_view(3)
    np.testing.assert_allclose(rel.entropy_confidence(torch.tensor(S))[0].numpy(), ref.entropy_confidence(S),
                               rtol=1e-12, atol=0)


def test_sobel_magnitude_eq6():
    S, D, *_ = make_view(4)
    got = rel.sobel_magnitude(torch.tensor(S), 1e-8).numpy()
    expected = np.stack([ref.edge_magnitude(S[c], 1e-8) for c in range(C)])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_edge_alignment_loss_eqs6_8():
    S, D, *_ = make_view(5)
    got = rel.edge_alignment_loss(torch.tensor(S), torch.tensor(D)[None], 0.75, 1e-8).item()
    assert got == pytest.approx(ref.edge_loss(S, D, 0.75, 1e-8), rel=1e-12)


def test_edge_alignment_tolerance_tau_d():
    """Semantic edges are free where the normalised depth edge reaches tau_d (max(0, tau_d - E_D_hat) = 0)."""
    S = torch.zeros(2, 8, 8, dtype=torch.float64)
    S[0, :, 4:] = 3.0                                  # one vertical semantic boundary
    D = torch.ones(1, 8, 8, dtype=torch.float64)
    D[0, :, 4:] = 2.0                                  # coinciding depth discontinuity
    supported = rel.edge_alignment_loss(S, D, tau_d=0.75).item()
    assert supported < 1e-4                            # only the sqrt(eps) floor of flat pixels remains
    flat = torch.ones(1, 8, 8, dtype=torch.float64)    # no geometric support -> penalised by tau_d * E_S
    assert rel.edge_alignment_loss(S, flat, tau_d=0.75).item() == pytest.approx(0.75 * 12 * 2 / 8, abs=1e-4)


def test_total_variation_tab9():
    S, *_ = make_view(6)
    assert rel.total_variation(torch.tensor(S), 1e-8).item() == pytest.approx(ref.total_variation(S, 1e-8), rel=1e-12)


SETTINGS = [dict(vcsu=a, scea=b, cauw=c) for a, b, c in itertools.product([True, False], repeat=3)] + [
    dict(vcsu_variant="hard"), dict(scea_variant="tv"), dict(cauw_variant="entropy"),
    dict(sigma_vis=0.025, gamma=50.0, tau_d=0.5), dict(sigma_vis=0.1, gamma=200.0, tau_d=0.9),
]


@pytest.mark.parametrize("setting", SETTINGS, ids=lambda s: ",".join(f"{k}={v}" for k, v in s.items()))
def test_semantic_loss_eqs3_11_12(setting):
    """Full objective, every module combination of Tab. VI, the Tab. IX variants and the sensitivity grid."""
    S, D, D2, D_obs, y = make_view(7)
    expected = ref.semantic_loss(S, y, D, D2, D_obs, **setting)
    St, Dt, D2t, D_obst, yt = to_torch(S, D, D2, D_obs, y)
    got = rel.semantic_loss(St, yt, Dt, D2t, D_obst, ReliabilityConfig(**setting)).item()
    assert got == pytest.approx(expected, rel=1e-12)


def test_reliability_weights_are_constants():
    """rho_p and the depth-edge reference carry no gradient into the geometry; S_p receives it."""
    S, D, D2, D_obs, y = to_torch(*make_view(8))
    S.requires_grad_(True)
    D.requires_grad_(True)
    D2.requires_grad_(True)
    loss = rel.semantic_loss(S, y, D, D2, D_obs, ReliabilityConfig())
    grad_S, grad_D, grad_D2 = torch.autograd.grad(loss, (S, D, D2), allow_unused=True)
    assert grad_D is None and grad_D2 is None
    assert torch.isfinite(grad_S).all() and grad_S.abs().sum() > 0


def test_unreliable_pixels_get_smaller_semantic_gradients():
    """VCSU / CAUW down-weight the CE gradient at depth-inconsistent and high-variance pixels."""
    S = torch.zeros(3, 1, 3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1, 1, 1]])
    D = torch.full((1, 1, 3), 2.0, dtype=torch.float64)
    D_obs = torch.tensor([[[2.0, 2.5, 2.0]]], dtype=torch.float64)          # pixel 1 depth-inconsistent
    D2 = D ** 2 + torch.tensor([[[0.0, 0.0, 0.05]]], dtype=torch.float64)   # pixel 2 ray conflict
    loss = rel.semantic_loss(S, y, D, D2, D_obs, ReliabilityConfig(scea=False))
    (grad,) = torch.autograd.grad(loss, S)
    g = grad.abs().sum(dim=0)[0]
    assert g[1] < g[0] and g[2] < g[0]
    # (1 - beta) of the plain CE gradient always remains (Eq. 11)
    assert g[1] >= (1 - 0.7) * g[0] - 1e-12


def test_ignored_labels_and_nan_depth():
    S, D, D2, D_obs, y = to_torch(*make_view(9))
    y[:] = -1
    assert rel.semantic_loss(S, y, D, D2, D_obs, ReliabilityConfig(scea=False)).item() == 0.0
    S, D, D2, D_obs, y = to_torch(*make_view(9))
    D[0, 3, 3] = float("nan")
    assert torch.isfinite(rel.semantic_loss(S, y, D, D2, D_obs, ReliabilityConfig()))


def test_invalid_variant_rejected():
    with pytest.raises(ValueError):
        ReliabilityConfig(vcsu_variant="gaussian")
