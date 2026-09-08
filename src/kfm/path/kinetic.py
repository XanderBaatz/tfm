from flow_matching.path import ProbPath
from flow_matching.utils import expand_tensor_like
from torch import Tensor

from kfm.path.path_sample import KineticPathSample
from kfm.utils.manifolds.torus import UnitFlatTorus


class KineticTorusProbPath(ProbPath):
    """Implements the second-order Kinetic Flow Matching boundary path on Torus T^n.

    Satisfies the paper's cubic Hermite boundary conditions:
        - At t=0:
            x_0 = source_x
            v_0 = source_v ~ p_0(v)
        - At t=1:
            x_1 = target_x
            v_1 = 0

    The conditional target dx_t stored in PathSample corresponds to the target acceleration field u_{t, v}.
    """

    def __init__(
        self,
        manifold: UnitFlatTorus = None,
        simplified: bool = True,
    ) -> None:
        self.manifold = manifold if manifold is not None else UnitFlatTorus(scale=1.0)
        self.simplified = simplified

    def sample(
        self,
        x_0: Tensor,
        x_1: Tensor,
        v_0: Tensor,
        t: Tensor,
    ) -> KineticPathSample:
        r"""Sample from the Kinetic Torus path.

        Args:
            x_0 (Tensor): Source position in [0, 1)^n, shape (batch_size, ...).
            x_1 (Tensor): Target position in [0, 1)^n, shape (batch_size, ...).
            t (Tensor): Times in [0, 1], shape (batch_size).
            v_0 (Tensor): Source velocity ~ p_0(v), shape matching x_0.

        Returns:
            KineticPathSample: Fully populated second-order kinetic sample.

        """
        self.assert_sample_shape(x_0, x_1, t)

        t_exp = expand_tensor_like(input_tensor=t, expand_to=x_1)  # [batch_size, ...]

        # wrapped displacement
        d = self.manifold.logmap(x_0, x_1)  # (x_1 - x_0 + 0.5) % 1.0 - 0.5

        # time powers
        t2 = t_exp**2
        t3 = t_exp**3

        if self.simplified:
            # v_0 = 0 simplification
            omega_t = (3 * t2 - 2 * t3) * d
            v_t = (6 * t_exp - 6 * t2) * d  # velocity path
            u_t_v = (6 - 12 * t_exp) * d  # target acceleration field
        else:
            omega_t = (3 * t2 - 2 * t3) * d + (t_exp - 2 * t2 + t3) * v_0
            v_t = (6 * t_exp - 6 * t2) * d + (1 - 4 * t_exp + 3 * t2) * v_0  # velocity path
            u_t_v = (6 - 12 * t_exp) * d + (-4 + 6 * t_exp) * v_0  # target acceleration field

        x_t = self.manifold.expmap(x=x_0, u=omega_t)

        return KineticPathSample(
            x_0=x_0,
            x_1=x_1,
            v_0=v_0,
            x_t=x_t,
            v_t=v_t,
            dx_t=d if self.simplified else u_t_v,  # target displacement field or acceleration field
            dv_t=u_t_v,  # Target acceleration field u_{t, v}
            t=t,
        )
