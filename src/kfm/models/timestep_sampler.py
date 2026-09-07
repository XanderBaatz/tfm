from abc import ABC, abstractmethod

import torch
from torch import nn


class TimestepSampler(nn.Module, ABC):
    """Abstract base class for sampling timesteps during training."""

    @abstractmethod
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps t for a given batch size."""


class UniformTimestepSampler(TimestepSampler):
    """Uniform timestep sampler t ~ Uniform(min_t, max_t)."""

    def __init__(
        self,
        min_t: float = 1e-5,
        max_t: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_t = min_t
        self.max_t = max_t

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """NOTE: invert bounds to get sample in (lb, ub], instead of [lb, ub)."""
        return torch.rand(batch_size, device=device) * (self.min_t - self.max_t) + self.max_t
