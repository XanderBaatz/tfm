from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.data import Batch, Data


class BasePrior(nn.Module, ABC):
    """Abstract base class for field-specific prior distributions."""

    @abstractmethod
    def sample_like(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        """Sample noise prior states matching the batch structure."""


class KineticPrior(BasePrior):
    """Prior p_0(x, v) for particle positions and velocities."""

    def __init__(self, sigma_v: float = 1.0, zero_velocity: bool = False) -> None:
        super().__init__()
        self.sigma_v = sigma_v
        self.zero_velocity = zero_velocity

    def sample_like(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        device = batch.pos.device

        # x_0 ~ U([0,1)^3)
        pos_0 = torch.rand_like(batch.pos, device=device)

        if self.zero_velocity:
            # v_0 ~ N(0, 1)
            v_0 = torch.zeros_like(batch.v, device=device)
        else:
            v_0 = torch.rand_like(batch.v, device=device) * self.sigma_v

        return {"pos": pos_0, "v": v_0}


class IsotropicLatticePrior(BasePrior):
    """l_0 ~ N(0, sigma_l^2 I_6). Baseline independent lattice prior."""

    def __init__(self, sigma_l: float = 1.0) -> None:
        super().__init__()
        self.sigma_l = sigma_l

    def sample_like(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        device = batch.l.device
        l_0 = torch.randn_like(batch.l, device=device) * self.sigma_l
        return {"l": l_0}


class SizeAwareLatticePrior(BasePrior):
    """MatterGen-style size-conditioned prior for the 6D lattice vector."""

    def __init__(
        self,
        limit_density: float = 0.05,
        length_std: float = 0.5,
        angle_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.limit_density = limit_density
        self.length_std = length_std
        self.angle_std = angle_std

    def sample_like(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        device = batch.l.device
        ref_l = batch.l

        num_atoms = torch.bincount(batch.batch).to(device=device, dtype=ref_l.dtype)
        expected_edge = (num_atoms / self.limit_density).clamp(min=1e-6) ** (1.0 / 3.0)
        mean_log_len = torch.log(expected_edge).unsqueeze(-1).expand(-1, 3)

        len_0 = mean_log_len + torch.randn_like(mean_log_len) * self.length_std
        ang_0 = torch.randn(num_atoms.shape[0], 3, device=device, dtype=ref_l.dtype) * self.angle_std
        l_0 = torch.cat([len_0, ang_0], dim=-1)

        return {"l": l_0}
