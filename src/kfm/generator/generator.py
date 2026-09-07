from abc import ABC, abstractmethod

import torch
from flow_matching.solver import Solver
from pymatgen.core import Structure

from kfm.data_types import FlowState, GraphBatch
from kfm.models.flow_module import FlowModule


class FlowGenerator(ABC, torch.nn.Module):
    """Abstract interface for generating structures from a flow model.

    Mirrors `FlowModule`'s training-time split (sample a path, predict a vector field) with
    its sampling-time counterpart: integrate the learned vector field from t=0 to t=1
    (`sample_state`), then decode the resulting state into domain objects (`state_to_structures`).
    """

    def __init__(
        self,
        flow_module: FlowModule | None = None,
        solver: Solver | None = None,
    ) -> None:
        super().__init__()

        self.flow_module = flow_module
        self.solver = solver

    @abstractmethod
    def sample_state(
        self,
        batch: GraphBatch,
        **kwargs,
    ) -> FlowState:
        """Integrate the flow's ODE/SDE from t=0 to t=1 and return the resulting state."""

    @abstractmethod
    def state_to_structures(
        self,
        state: FlowState,
        batch: GraphBatch,
    ) -> list[Structure]:
        """Convert generated state to pymatgen structures."""

    @abstractmethod
    def state_from_batch(
        self,
        batch: GraphBatch,
    ) -> FlowState:
        """Extract the ground-truth (t=1) flow state directly from a data batch, without integration."""

    def generate(
        self,
        batch: GraphBatch,
        **kwargs,
    ) -> list[Structure]:
        """Sample a state via `sample_state` and decode it into structures."""
        state = self.sample_state(batch, **kwargs)
        return self.state_to_structures(state, batch)

    def target_structures(
        self,
        batch: GraphBatch,
    ) -> list[Structure]:
        """Decode ground-truth structures directly from a data batch."""
        return self.state_to_structures(self.state_from_batch(batch), batch)
