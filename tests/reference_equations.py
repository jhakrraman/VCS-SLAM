"""Independent NumPy implementation of the semantic objective, written directly from the paper.

Each function transcribes one equation of Sec. III with plain loops and NumPy, without sharing code with
vcs_slam/reliability.py, so that tests/test_reliability.py can check the PyTorch implementation against it.
Shapes: S (C, H, W) rendered semantic logits, D / D2 / D_obs (H, W), labels y (H, W) with -1 = no label.
"""

import numpy as np

SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T


def _filter3x3(image, kernel):
    """3x3 cross-correlation with replicate ("edge") padding."""
    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel)
    return out


def edge_magnitude(image, eps=0.0):
    """Eq. 6: E(I) = sqrt((Sobel_x * I)^2 + (Sobel_y * I)^2); eps > 0 only where the root is differentiated."""
    return np.sqrt(_filter3x3(image, SOBEL_X) ** 2 + _filter3x3(image, SOBEL_Y) ** 2 + eps)


def visibility_mask(D, D_obs, sigma_vis, hard=False):
    """Eq. 5 (soft) and the hard gate of Tab. IX; zero where rendered or sensor depth is missing."""
    if hard:
        m = (np.abs(D - D_obs) < sigma_vis).astype(float)
    else:
        m = np.exp(-np.abs(D - D_obs) ** 2 / (2.0 * sigma_vis ** 2))
    return m * ((D_obs > 0) & (D > 0))


def conflict_confidence(D, D2, gamma, eps):
    """Eqs. 9-10: V = max(E[D^2] - E[D]^2, eps), C = 1 / (1 + gamma V)."""
    V = np.maximum(D2 - D ** 2, eps)
    return 1.0 / (1.0 + gamma * V)


def entropy_confidence(S):
    """Tab. IX: C = 1 / (1 + H(softmax(S)))."""
    p = np.exp(S - S.max(axis=0, keepdims=True))
    p /= p.sum(axis=0, keepdims=True)
    return 1.0 / (1.0 + (-(p * np.log(p)).sum(axis=0)))


def cross_entropy(S, y):
    """Per-pixel CE(S_p, y_p) with one-hot targets (0 where y < 0)."""
    C, H, W = S.shape
    ce = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if y[i, j] < 0:
                continue
            logits = S[:, i, j]
            log_norm = logits.max() + np.log(np.sum(np.exp(logits - logits.max())))
            ce[i, j] = log_norm - logits[y[i, j]]
    return ce


def edge_loss(S, D, tau_d, eps):
    """Eqs. 7-8: L_edge = 1/N sum E_S * max(0, tau_d - E_D / (max E_D + eps)), E_S = max over channels."""
    E_D = edge_magnitude(D)
    E_D_hat = E_D / (E_D.max() + eps)
    E_S = np.max(np.stack([edge_magnitude(S[c], eps) for c in range(S.shape[0])]), axis=0)
    return np.mean(E_S * np.maximum(0.0, tau_d - E_D_hat))


def total_variation(S, eps):
    """Tab. IX: isotropic TV of S with forward differences (zero at the last row / column)."""
    C, H, W = S.shape
    dx = np.zeros_like(S)
    dy = np.zeros_like(S)
    dx[:, :, :-1] = S[:, :, 1:] - S[:, :, :-1]
    dy[:, :-1, :] = S[:, 1:, :] - S[:, :-1, :]
    return np.mean(np.sqrt((dx ** 2 + dy ** 2).sum(axis=0) + eps))


def semantic_loss(S, y, D, D2, D_obs, vcsu=True, scea=True, cauw=True, sigma_vis=0.05, tau_d=0.75,
                  gamma=100.0, beta=0.7, lambda_edge=0.02, eps=1e-8, vcsu_variant="soft",
                  scea_variant="edge", cauw_variant="variance"):
    """Eqs. 3, 11, 12: beta * mean(rho * CE) + (1 - beta) * mean(CE) + lambda_edge * L_edge."""
    valid = y >= 0
    N = max(int(valid.sum()), 1)
    ce = cross_entropy(S, y)
    if not vcsu and not cauw:
        fid = ce[valid].sum() / N
    else:
        rho = np.ones_like(D)
        if vcsu:
            rho = rho * visibility_mask(D, D_obs, sigma_vis, hard=vcsu_variant == "hard")
        if cauw:
            rho = rho * (entropy_confidence(S) if cauw_variant == "entropy" else conflict_confidence(D, D2, gamma, eps))
        fid = beta * (rho * ce)[valid].sum() / N + (1.0 - beta) * ce[valid].sum() / N
    if scea:
        fid += lambda_edge * (total_variation(S, eps) if scea_variant == "tv" else edge_loss(S, D, tau_d, eps))
    return fid
