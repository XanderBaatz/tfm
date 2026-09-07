from abc import ABC, abstractmethod

from torch import Tensor, nn

from kfm.data_types import FlowState


class VectorFieldModel(nn.Module, ABC):
    """A network predicting the vector field(s) u_t that drive a (Multi)Flow.

    Given the current latent state of every flow field at time t, a `VectorFieldModel`
    predicts the corresponding target field(s) (e.g. velocity, acceleration, lattice
    rate-of-change). Its predictions are consumed by `FlowModule.calc_loss` for training,
    and by a `Solver` (wrapped in a `flow_matching.utils.ModelWrapper`) for generation.

    Subclasses are free to define whatever keyword arguments they need (e.g. graph
    connectivity); this base class only fixes the shared contract that `forward` takes
    the time `t` and returns a `FlowState` dict of predicted fields.
    """

    @abstractmethod
    def forward(self, t: Tensor, **state: Tensor) -> FlowState:
        """Predict the vector field(s) for the given time and latent state."""
