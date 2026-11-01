import torch


class LearnedPerceptualImagePatchSimilarity(torch.nn.Module):
    """Test stand-in: mean absolute difference instead of the learned AlexNet distance."""

    def __init__(self, net_type="alex", normalize=True):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))
