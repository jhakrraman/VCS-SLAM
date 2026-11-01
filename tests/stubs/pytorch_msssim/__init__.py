"""Test stand-in for pytorch_msssim: MS-SSIM needs images larger than 160 px, the tests use tiny frames."""

import torch


def ms_ssim(x, y, data_range=1.0, size_average=True):
    return 1.0 - torch.mean((x - y) ** 2) / (data_range ** 2)
