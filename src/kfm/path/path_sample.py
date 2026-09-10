from dataclasses import dataclass, field

from flow_matching.path.path_sample import PathSample
from torch import Tensor


@dataclass
class KineticPathSample(PathSample):
    """Represents a sample of a second-order kinetic probability path on G x g.

    Attributes:
        x_1 (Tensor): target spatial position X_1.
        x_0 (Tensor): source spatial position X_0.
        v_0 (Tensor): initial source velocity V_0 ~ p_0(V).
        v_1 (Tensor): final terminal velocity V_1 ~ p_1(V).
        v_t (Tensor): velocity state V_t along the path.
        t (Tensor): time samples t.
        x_t (Tensor): position state X_t along the path.
        dx_t (Tensor).

    """

    v_0: Tensor = field(metadata={"help": "source velocity sample V_0 ~ p_0(v), shape (batch_size, ...)."})
    v_1: Tensor = field(metadata={"help": "target velocity sample V_1 ~ p_1(v), shape (batch_size, ...)."})
    v_t: Tensor = field(metadata={"help": "velocity sample V_t at time t, shape (batch_size, ...)."})
    dv_t: Tensor = field(metadata={"help": "target acceleration field u_{t, v} = dV_t/dt"})

    def __post_init__(self) -> None:
        # Keep dx_t and dv_t synchronized so inherited code works seamlessly
        if self.dv_t is not None and self.dx_t is None:
            self.dx_t = self.dv_t
        elif self.dx_t is not None and self.dv_t is None:
            self.dv_t = self.dx_t
