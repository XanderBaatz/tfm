# Originally from: https://github.com/frcnt/kldm/blob/main/src_kldm/data/dataset.py


import torch
from flow_matching.utils.manifolds import Manifold
from torch import Tensor
from torch_geometric.utils import scatter


def scatter_center(pos: Tensor, index: Tensor) -> Tensor:
    """Zero the center of gravity per graph."""
    # return pos - scatter_mean(pos, index=index, dim=0)[index]
    mean = scatter(pos, index=index, dim=0, reduce="mean")
    return pos - mean[index]


def wrap(manifold: Manifold, samples: int) -> Tensor:
    center = torch.zeros_like(samples)

    return manifold.expmap(center, samples)


def wrap(x, x_range: float = (2.0 * torch.pi)):
    return torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
