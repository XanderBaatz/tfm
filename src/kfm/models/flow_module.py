from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from kfm.data_types import FlowState, GraphBatch
from kfm.models.flow import MultiFlow
from kfm.models.timestep_sampler import TimestepSampler, UniformTimestepSampler
from kfm.models.vector_field import VectorFieldModel


def relative_mse(pred, target, eps=1e-8):
    """MSE normalized by per-dimension target variance -- comparable across
    runs regardless of whether targets were pre-scaled.
    """
    target_var = target.var(dim=0, keepdim=True).clamp(min=eps)
    return ((pred - target) ** 2 / target_var).mean()


class FlowModule(nn.Module, ABC):
    """Abstract Flow Module for a multi-flow state."""

    def __init__(
        self,
        vector_field_model: VectorFieldModel,
        multi_flow: MultiFlow,
        timestep_sampler: TimestepSampler | None = None,
        loss_fn: None = None,  # not yet implemented  # noqa: ARG002
    ) -> None:
        """Initialize Flow Module.

        Args:
            vector_field_model (VectorFieldModel): Network predicting the raw vector field(s) u_t.
                Distinct from a `Solver`'s `velocity_model`, which is this same network adapted
                (via a `flow_matching.utils.ModelWrapper`) to the solver's `(x, t, **extras)` call signature.
            multi_flow (MultiFlow): _description_
            timestep_sampler (TimestepSampler | None, optional): _description_. Defaults to None.
            loss_fn (None, optional): _description_. Defaults to None.

        """
        super().__init__()

        self.vector_field_model = vector_field_model
        self.multi_flow = multi_flow
        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-5,
            max_t=1.0 - 1e-5,
        )

    def sample_timesteps(self, batch: GraphBatch) -> torch.Tensor:
        """Sample timesteps."""
        num_graphs = getattr(batch, "num_graphs", 1)
        return self.timestep_sampler(batch_size=num_graphs, device=batch.pos.device)

    def calc_loss(self, batch: GraphBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = self.sample_timesteps(batch)

        latents, targets = self.multi_flow.sample_path(batch, t)

        preds = self.predict_velocity(batch=batch, latents=latents, t=t)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=batch.pos.device)

        for key, target_val in targets.items():
            field_loss = F.mse_loss(preds[key], target_val)

            loss_dict[f"loss_{key}"] = field_loss.detach()
            total_loss += field_loss

        loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict

    @abstractmethod
    def predict_velocity(
        self,
        batch: GraphBatch,
        latents: FlowState,
        t: torch.Tensor,
    ) -> FlowState:
        """Subclasses define how vector field network args are packed from batch/latents."""  # noqa: D401
