"""Geometry-validated semantic evidence fusion (Sec. III-B to III-D).

All functions work on a single rendered view, using these tensors:

    sem_logits  rendered semantic logit map S_p (Eq. 1)              (C, H, W)
    depth       rendered expected depth D_p = E[D] (Eq. 2)          (1, H, W)
    depth_sq    rendered expected squared depth E[D^2]              (1, H, W)
    depth_obs   observed RGB-D sensor depth D_p^obs                 (1, H, W)
    labels      2D semantic prior y_p as label-space indices        (H, W), -1 = ignore

The reliability score rho_p = M_vis(p) * C_p (Eq. 3) only *weights* the semantic residuals,
so it is treated as a constant during back-propagation. Likewise, the normalised depth-edge
map E_D is a structural reference for SCEA, so no gradient flows through it into the geometry.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

_SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
_SOBEL_Y = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))


@dataclass(frozen=True)
class ReliabilityConfig:
    """Hyper-parameters of the semantic objective. Defaults are the values of the paper."""

    vcsu: bool = True            # visibility-consistent semantic updates (Sec. III-C1)
    scea: bool = True            # surface-coupled edge alignment (Sec. III-C2)
    cauw: bool = True            # conflict-aware uncertainty weighting (Sec. III-C3)
    sigma_vis: float = 0.05      # depth-gating bandwidth, Eq. 5 [m]
    tau_d: float = 0.75          # depth-edge support tolerance, Eq. 8
    gamma: float = 100.0         # depth-variance scaling, Eq. 10
    beta: float = 0.7            # reliability-gated / standard CE blend, Eq. 11
    lambda_edge: float = 0.02    # boundary penalty weight, Eq. 12
    eps: float = 1e-8            # numerical floor of Eqs. 7 and 9
    # Naive substitutes of each module, used for Tab. IX.
    vcsu_variant: str = "soft"       # "soft": Eq. 5   | "hard": M_vis = 1[|D - D_obs| < sigma_vis]
    scea_variant: str = "edge"       # "edge": Eq. 8   | "tv":   L_edge = isotropic TV of S
    cauw_variant: str = "variance"   # "variance": Eq. 10 | "entropy": C = 1 / (1 + H(softmax(S)))

    def __post_init__(self):
        if self.vcsu_variant not in ("soft", "hard"):
            raise ValueError(f"vcsu_variant must be 'soft' or 'hard', got {self.vcsu_variant!r}")
        if self.scea_variant not in ("edge", "tv"):
            raise ValueError(f"scea_variant must be 'edge' or 'tv', got {self.scea_variant!r}")
        if self.cauw_variant not in ("variance", "entropy"):
            raise ValueError(f"cauw_variant must be 'variance' or 'entropy', got {self.cauw_variant!r}")

    @classmethod
    def from_dict(cls, cfg):
        """Build from the ``semantics`` section of an experiment config (unrelated keys are ignored)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})


def sobel_magnitude(image, eps=0.0):
    """Per-channel Sobel edge magnitude E(I) = sqrt((grad_x * I)^2 + (grad_y * I)^2), Eq. 6.

    ``image`` is (C, H, W); the result has the same shape. Borders are replicate-padded so the
    image boundary is not detected as an edge. A small ``eps`` inside the root keeps the gradient
    finite on flat regions when the map is optimised (the semantic edges E_S).
    """
    channels = image.shape[0]
    kx = torch.tensor(_SOBEL_X, dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    ky = torch.tensor(_SOBEL_Y, dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    padded = F.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx.expand(channels, 1, 3, 3), groups=channels)
    gy = F.conv2d(padded, ky.expand(channels, 1, 3, 3), groups=channels)
    return torch.sqrt(gx ** 2 + gy ** 2 + eps)[0]


def visibility_mask(depth, depth_obs, sigma_vis, hard=False):
    """VCSU soft visibility mask M_vis = exp(-|D - D_obs|^2 / (2 sigma_vis^2)), Eq. 5.

    ``hard=True`` gives the hard gate of Tab. IX. Pixels without a valid sensor or rendered
    depth carry no geometric evidence and get weight 0.
    """
    valid = (depth_obs > 0) & (depth > 0)
    residual = torch.abs(depth - depth_obs)
    if hard:
        mask = (residual < sigma_vis).to(depth.dtype)
    else:
        mask = torch.exp(-residual ** 2 / (2.0 * sigma_vis ** 2))
    return torch.where(valid, mask, torch.zeros_like(mask))


def depth_variance(depth, depth_sq, eps=1e-8):
    """Ray-level depth variance V = max(E[D^2] - E[D]^2, eps), Eq. 9."""
    return torch.clamp(depth_sq - depth ** 2, min=eps)


def conflict_confidence(depth, depth_sq, gamma, eps=1e-8):
    """CAUW per-pixel confidence C_p = 1 / (1 + gamma * V), Eq. 10."""
    return 1.0 / (1.0 + gamma * depth_variance(depth, depth_sq, eps))


def entropy_confidence(sem_logits):
    """Tab. IX entropy-weighting baseline: C_p = 1 / (1 + H(softmax(S_p))), H in nats. Shape (1, H, W)."""
    log_p = F.log_softmax(sem_logits, dim=0)
    entropy = -(log_p.exp() * log_p).sum(dim=0, keepdim=True)
    return 1.0 / (1.0 + entropy)


def reliability_weight(sem_logits, depth, depth_sq, depth_obs, cfg):
    """Per-pixel reliability rho_p = M_vis(p) * C_p (Eq. 3) as an (H, W) constant.

    Returns ``None`` when both VCSU and CAUW are disabled, i.e. every observation counts fully.
    """
    rho = None
    if cfg.vcsu:
        rho = visibility_mask(depth, depth_obs, cfg.sigma_vis, hard=cfg.vcsu_variant == "hard")
    if cfg.cauw:
        if cfg.cauw_variant == "entropy":
            conf = entropy_confidence(sem_logits)
        else:
            conf = conflict_confidence(depth, depth_sq, cfg.gamma, cfg.eps)
        rho = conf if rho is None else rho * conf
    if rho is None:
        return None
    return torch.nan_to_num(rho.detach(), nan=0.0)[0]


def edge_alignment_loss(sem_logits, depth, tau_d, eps=1e-8):
    """SCEA surface-coupled edge penalty, Eqs. 6-8.

    L_edge = 1/N sum_p E_S(p) * max(0, tau_d - E_D_hat(p)), where E_S is the largest Sobel
    magnitude over the semantic channels and E_D_hat = E_D / (max(E_D) + eps) is the
    normalised depth-edge map (Eq. 7).
    """
    depth_ref = torch.nan_to_num(depth.detach(), nan=0.0)
    depth_edges = sobel_magnitude(depth_ref)[0]                     # E_D, Eq. 6
    depth_edges = depth_edges / (depth_edges.max() + eps)           # E_D_hat, Eq. 7
    sem_edges = sobel_magnitude(sem_logits, eps).amax(dim=0)        # E_S (differentiable)
    return (sem_edges * torch.clamp(tau_d - depth_edges, min=0.0)).mean()


def total_variation(sem_logits, eps=1e-8):
    """Tab. IX smoothness baseline: isotropic total variation of S_p (forward differences)."""
    dx = F.pad(sem_logits[:, :, 1:] - sem_logits[:, :, :-1], (0, 1, 0, 0))
    dy = F.pad(sem_logits[:, 1:, :] - sem_logits[:, :-1, :], (0, 0, 0, 1))
    return torch.sqrt((dx ** 2 + dy ** 2).sum(dim=0) + eps).mean()


def semantic_loss(sem_logits, labels, depth, depth_sq, depth_obs, cfg):
    """Final semantic objective L_sem^ours of Eq. 12 (weighted by lambda_S in Eq. 13).

    L_sem^fid = beta * 1/N sum_p rho_p CE(S_p, y_p) + (1 - beta) * 1/N sum_p CE(S_p, y_p)   (Eq. 11)
    L_sem^ours = L_sem^fid + lambda_edge * L_edge                                         (Eq. 12)

    N counts the pixels that carry a label (``labels >= 0``).
    """
    valid = labels >= 0
    num_valid = valid.sum().clamp(min=1)
    ce = F.cross_entropy(sem_logits.unsqueeze(0), labels.clamp(min=0).unsqueeze(0), reduction="none")[0]
    ce = torch.where(valid, ce, torch.zeros_like(ce))
    ce_mean = ce.sum() / num_valid

    rho = reliability_weight(sem_logits, depth, depth_sq, depth_obs, cfg)
    if rho is None:
        loss = ce_mean
    else:
        loss = cfg.beta * (rho * ce).sum() / num_valid + (1.0 - cfg.beta) * ce_mean

    if cfg.scea:
        if cfg.scea_variant == "tv":
            edge = total_variation(sem_logits, cfg.eps)
        else:
            edge = edge_alignment_loss(sem_logits, depth, cfg.tau_d, cfg.eps)
        loss = loss + cfg.lambda_edge * edge
    return loss
