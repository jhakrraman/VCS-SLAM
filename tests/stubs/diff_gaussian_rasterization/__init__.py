"""CPU stand-in for diff-gaussian-rasterization (tests only).

Same interface as the CUDA rasterizer used by SplaTAM / SGS-SLAM: Gaussians are projected with the
camera's view / projection matrices, splatted as isotropic screen-space Gaussians and alpha-composited
front to back, so colours, depth and silhouettes depend differentiably on every Gaussian parameter.
It is O(N * H * W) and only meant for the tiny images of the test suite.
"""

import torch


class GaussianRasterizationSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GaussianRasterizer(torch.nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None,
                rotations=None, cov3D_precomp=None):
        s = self.raster_settings
        height, width = int(s.image_height), int(s.image_width)
        hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
        clip = hom @ s.projmatrix[0]
        depth = (hom @ s.viewmatrix[0])[:, 2]
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
        px = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5 + means2D[:, 0]
        py = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5 + means2D[:, 1]
        focal = width / (2.0 * s.tanfovx)
        sigma = (scales.mean(dim=1) * focal / depth.clamp(min=1e-3)).clamp(min=0.5, max=float(max(height, width)))
        visible = (depth > 0.01).to(means3D.dtype)

        ys = torch.arange(height, dtype=means3D.dtype, device=means3D.device)
        xs = torch.arange(width, dtype=means3D.dtype, device=means3D.device)
        d2 = (xs.view(1, 1, -1) - px.view(-1, 1, 1)) ** 2 + (ys.view(1, -1, 1) - py.view(-1, 1, 1)) ** 2
        alpha = opacities.view(-1, 1, 1) * torch.exp(-0.5 * d2 / sigma.view(-1, 1, 1) ** 2)
        alpha = alpha.clamp(max=0.99) * visible.view(-1, 1, 1)

        order = torch.argsort(depth.detach())
        alpha = alpha[order]
        transmittance = torch.cat([torch.ones_like(alpha[:1]), torch.cumprod(1.0 - alpha, dim=0)[:-1]], dim=0)
        weights = alpha * transmittance
        image = torch.einsum("nhw,nc->chw", weights, colors_precomp[order])
        depth_image = torch.einsum("nhw,n->hw", weights, depth[order]).unsqueeze(0)
        radii = (torch.ceil(3.0 * sigma) * visible).int()
        return image, radii, depth_image
