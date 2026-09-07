from abc import ABC, abstractmethod

import torch
from flow_matching.path.path import ProbPath
from torch import nn
from torch_geometric.data.batch import Batch, Data

from kfm.distributions.prior import BasePrior


class Flow(nn.Module, ABC):
    """Wraps a specific field's prior distribution and continuous flow path together."""

    def __init__(self, prior: BasePrior, path: ProbPath) -> None:
        super().__init__()
        self.prior = prior
        self.path = path

    def sample_prior(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        """Sample noise prior states matching the batch structure."""
        return self.prior.sample_like(batch)

    @abstractmethod
    def sample_path(
        self,
        batch: Batch | Data,
        t: torch.Tensor,
        state_0: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample intermediate latents state_t and target vector fields u_t at time t.

        Args:
            batch: PyG Batch or Data containing target data (x_1).
            t: Time tensor matching the entity resolution (node-level or graph-level).
            state_0: Prior data. If not provided it is automatically sampled from self.prior.

        Returns:
            latents: Dictionary of noisy intermediate tensors at time t (e.g. {'pos': pos_t, 'v': v_t}).
            targets: Dictionary of target vector fields to predict (e.g. {'dv': target_dv}).

        """


class KineticFlow(Flow):
    """Flow for particle positions and velocities."""

    # def __init__(self, prior: BasePrior, path: ProbPath) -> None:
    #    super().__init__(prior=prior, path=path)

    def sample_path(
        self,
        batch: Batch | Data,
        t: torch.Tensor,
        state_0: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if state_0 is None:
            state_0 = self.sample_prior(batch=batch)

        # Map graph-level t [B] -> node-level t [N] using PyG indexing
        t_node = t[batch.batch] if hasattr(batch, "batch") and batch.batch is not None else t

        sample_k = self.path.sample(
            x_0=state_0["pos"],
            x_1=batch.pos,
            v_0=state_0["v"],
            t=t_node,
        )

        latents = {"pos": sample_k.x_t, "v": sample_k.v_t}
        targets = {"dv": sample_k.dv_t}

        return latents, targets


class LatticeFlow(Flow):
    """Flow process for 6D lattice parameters.

    Expects state_0 to contain 'l', and target batch to have 'l'.
    Works directly with AffineProbPath/CondOTProbPath.
    """

    def sample_path(
        self,
        batch: Batch | Data,
        t: torch.Tensor,
        state_0: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if state_0 is None:
            state_0 = self.sample_prior(batch)

        sample_l = self.path.sample(
            x_0=state_0["l"],
            x_1=batch.l,
            t=t,
        )

        latents = {"l": sample_l.x_t}
        targets = {"dl": sample_l.dx_t}

        return latents, targets


class MultiFlow(nn.Module):
    """Aggregates multiple physical field flows (e.g., kinetic, lattice)."""

    def __init__(self, flows: dict[str, Flow]) -> None:
        super().__init__()
        self.flows = nn.ModuleDict(flows)

    def sample_priors(self, batch: Batch | Data) -> dict[str, torch.Tensor]:
        """Sample state_0 across all registered fields."""
        priors = {}

        for flow in self.flows.values():
            priors.update(flow.sample_prior(batch))

        return priors

    def sample_path(
        self,
        batch: Batch | Data,
        t: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample state_t and target vector fields across all fields."""
        latents = {}  # x_t, v_t, l_t
        targets = {}  # u_v, u_l

        # Distribute graph-level t to each flow (each flow handles node/graph expansion)
        for flow in self.flows.values():
            field_latents, field_targets = flow.sample_path(batch=batch, t=t)
            latents.update(field_latents)
            targets.update(field_targets)

        return latents, targets
