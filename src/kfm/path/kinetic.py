import torch
from flow_matching.path import ProbPath
from flow_matching.utils import expand_tensor_like
from torch import Tensor

from kfm.path.path_sample import KineticPathSample
from kfm.utils.manifolds.torus import UnitFlatTorus


class KineticTorusProbPath(ProbPath):
    """Implements the cubic second-order Kinetic Flow Matching boundary path on Torus T^n.

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
        *,
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
        # print(d.var(dim=0))

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

    def get_acceleration_coeff(self, t: Tensor) -> Tensor:
        """Return the multiplier converting target displacement d into target acceleration u_{t,v}."""
        # Derivative weight for cubic path: (6 - 12*t)
        return 6.0 - 12.0 * t

    def reconstruct_acceleration(self, pred_target: Tensor, t: Tensor) -> Tensor:
        r"""Convert network prediction (e.g., \hat{d}) into target acceleration u_{t,v}."""
        if self.simplified:
            # pred_target is \hat{d}
            multiplier = self.get_acceleration_coeff(t)
            return multiplier * pred_target
        # If not simplified, the model predicts u_{t,v} directly
        return pred_target


class KineticTorusQuinticProbPath(ProbPath):
    """Quintic C^2 smooth boundary path on Torus T^n."""

    def __init__(
        self,
        manifold: UnitFlatTorus = None,
        *,
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
        self.assert_sample_shape(x_0, x_1, t)

        t_exp = expand_tensor_like(input_tensor=t, expand_to=x_1)  # [batch_size, ...]

        # Wrapped displacement on the flat torus
        d = self.manifold.logmap(x_0, x_1)

        # Precompute higher time powers
        t2 = t_exp**2
        t3 = t_exp**3
        t4 = t_exp**4
        t5 = t_exp**5

        if self.simplified:
            # Boundary conditions with v_0 = 0
            omega_t = (10 * t3 - 15 * t4 + 6 * t5) * d
            v_t = (30 * t2 - 60 * t3 + 30 * t4) * d
            u_t_v = (60 * t_exp - 180 * t2 + 120 * t3) * d
        else:
            # General case with non-zero v_0
            omega_t = (10 * t3 - 15 * t4 + 6 * t5) * d + (t_exp - 6 * t3 + 8 * t4 - 3 * t5) * v_0
            v_t = (30 * t2 - 60 * t3 + 30 * t4) * d + (1 - 18 * t2 + 32 * t3 - 15 * t4) * v_0
            u_t_v = (60 * t_exp - 180 * t2 + 120 * t3) * d + (-36 * t_exp + 96 * t2 - 60 * t3) * v_0

        x_t = self.manifold.expmap(x=x_0, u=omega_t)

        return KineticPathSample(
            x_0=x_0,
            x_1=x_1,
            v_0=v_0,
            x_t=x_t,
            v_t=v_t,
            dx_t=d if self.simplified else u_t_v,
            dv_t=u_t_v,
            t=t,
        )

    def get_acceleration_coeff(self, t: Tensor) -> Tensor:
        """Return the multiplier converting target displacement d into target acceleration u_{t,v}."""
        # Derivative weight for quintic path: (60*t - 180*t^2 + 120*t^3)
        return 60.0 * t - 180.0 * (t**2) + 120.0 * (t**3)

    def reconstruct_acceleration(self, pred_target: Tensor, t: Tensor) -> Tensor:
        """Convert network prediction into target acceleration u_{t,v}."""
        if self.simplified:
            multiplier = self.get_acceleration_coeff(t)
            return multiplier * pred_target
        return pred_target


class StochasticKineticTorusProbPath(ProbPath):
    r"""Stochastic second-order kinetic probability path on a flat torus.

    The path satisfies

        dx_t / dt = v_t
        dv_t / dt = a_t

    exactly.

    The deterministic component is a cubic Hermite bridge from
    (x_0, v_0) to (x_1, v_1=0).

    A zero-integral stochastic velocity bridge is then added:

        delta_v(t) = sigma(t) * b(t) * eps

    with

        b(t) = t (1 - t) (1 - 2t)

    and

        integral_0^1 b(t) dt = 0.

    Consequently the stochastic velocity perturbation changes the
    intermediate kinetic state without changing the final position:

        integral_0^1 delta_v(t) dt = 0.

    This is important because the velocity is no longer a deterministic
    encoding of the endpoint displacement d.

    The returned dv_t is the actual acceleration target:

        dv_t = d v_t / dt.

    Therefore the network can directly predict acceleration and no
    displacement -> acceleration reconstruction is required.
    """

    def __init__(
        self,
        manifold: UnitFlatTorus | None = None,
        *,
        sigma_v: float = 0.1,
        time_power: float = 1.0,
        simplified: bool = False,
    ) -> None:
        super().__init__()

        self.manifold = manifold if manifold is not None else UnitFlatTorus(scale=1.0)

        self.sigma_v = sigma_v
        self.time_power = time_power
        self.simplified = simplified

    # ------------------------------------------------------------------
    # Stochastic bridge
    # ------------------------------------------------------------------

    def _bridge_velocity_basis(self, t: Tensor) -> Tensor:
        r"""Zero-integral velocity bridge basis

            b(t) = t (1-t) (1-2t).

        Properties:

            b(0) = b(1) = 0

            integral_0^1 b(t) dt = 0

        Therefore adding sigma * b(t) * eps to velocity does not
        change the final displacement.
        """
        return t * (1.0 - t) * (1.0 - 2.0 * t)

    def _bridge_velocity_basis_dt(self, t: Tensor) -> Tensor:
        r"""Derivative of

            b(t) = t (1-t)(1-2t)

        which is

            b'(t) = 1 - 6t + 6t^2.
        """
        return 1.0 - 6.0 * t + 6.0 * t**2

    def _bridge_position_basis(self, t: Tensor) -> Tensor:
        r"""Integral of the velocity bridge:

            B(t) = integral_0^t b(s) ds

        giving

            B(t) = 1/2 t^2 - t^3 + 1/2 t^4.

        This satisfies

            B(0) = 0
            B(1) = 0

        and

            dB/dt = b(t).
        """
        t2 = t**2
        t3 = t**3
        t4 = t**4

        return 0.5 * t2 - t3 + 0.5 * t4

    # ------------------------------------------------------------------
    # Cubic deterministic kinetic bridge
    # ------------------------------------------------------------------

    def _cubic_position(
        self,
        d: Tensor,
        v_0: Tensor,
        t: Tensor,
    ) -> Tensor:
        r"""Cubic Hermite position path satisfying

            x(0) = x_0
            x(1) = x_1
            v(0) = v_0
            v(1) = 0.

        x_t = x_0
             + (3t^2 - 2t^3) d
             + (t - 2t^2 + t^3) v_0

        where d = logmap(x_0, x_1).
        """
        t2 = t**2
        t3 = t**3

        h01 = 3.0 * t2 - 2.0 * t3
        h10 = t - 2.0 * t2 + t3

        return h01 * d + h10 * v_0

    def _cubic_velocity(
        self,
        d: Tensor,
        v_0: Tensor,
        t: Tensor,
    ) -> Tensor:
        r"""Derivative of the cubic position path.

        v_t =
            (6t - 6t^2) d
            + (1 - 4t + 3t^2) v_0
        """
        t2 = t**2

        return (6.0 * t - 6.0 * t2) * d + (1.0 - 4.0 * t + 3.0 * t2) * v_0

    def _cubic_acceleration(
        self,
        d: Tensor,
        v_0: Tensor,
        t: Tensor,
    ) -> Tensor:
        r"""Derivative of the cubic velocity:

        a_t =
            (6 - 12t) d
            + (-4 + 6t) v_0
        """
        return (6.0 - 12.0 * t) * d + (-4.0 + 6.0 * t) * v_0

    # ------------------------------------------------------------------
    # Main path
    # ------------------------------------------------------------------

    def sample(
        self,
        x_0: Tensor,
        x_1: Tensor,
        v_0: Tensor,
        t: Tensor,
    ) -> KineticPathSample:
        r"""Sample a stochastic kinetic bridge.

        Args:
            x_0:
                Source positions, shape (..., 3).

            x_1:
                Target positions, shape (..., 3).

            v_0:
                Source velocities, shape (..., 3).

            t:
                Times in [0, 1], shape (batch,) or (...,).

        Returns:
            KineticPathSample containing:

                x_t:
                    Intermediate fractional position.

                v_t:
                    Intermediate auxiliary velocity.

                dx_t:
                    Wrapped displacement d.

                dv_t:
                    Actual acceleration target.

        """
        self.assert_sample_shape(x_0, x_1, t)

        t_exp = expand_tensor_like(
            input_tensor=t,
            expand_to=x_1,
        )

        # --------------------------------------------------------------
        # Endpoint displacement on the torus
        # --------------------------------------------------------------

        d = self.manifold.logmap(x_0, x_1)

        # --------------------------------------------------------------
        # Deterministic cubic kinetic bridge
        # --------------------------------------------------------------

        omega_t = self._cubic_position(
            d=d,
            v_0=v_0,
            t=t_exp,
        )

        v_cubic = self._cubic_velocity(
            d=d,
            v_0=v_0,
            t=t_exp,
        )

        a_cubic = self._cubic_acceleration(
            d=d,
            v_0=v_0,
            t=t_exp,
        )

        # --------------------------------------------------------------
        # Stochastic zero-integral velocity bridge
        # --------------------------------------------------------------

        # eps is independent of d.
        #
        # This is the important difference from the old path:
        #
        #     v_t != g(t) * d
        #
        # anymore.
        eps = torch.randn_like(v_cubic)

        # Optional time-dependent noise amplitude.
        #
        # sigma(t) = sigma_v * [t(1-t)]^time_power
        #
        # This forces the stochastic perturbation to vanish at both
        # endpoints.
        time_envelope = (t_exp * (1.0 - t_exp)).clamp_min(0.0).pow(self.time_power)

        sigma_t = self.sigma_v * time_envelope

        bridge_v = self._bridge_velocity_basis(t_exp)

        bridge_a = self._bridge_velocity_basis_dt(t_exp)

        bridge_x = self._bridge_position_basis(t_exp)

        stochastic_v = sigma_t * bridge_v * eps

        # For the simplest implementation we treat sigma_t as constant
        # over the bridge derivative. If time_power != 0, the exact
        # derivative includes d(sigma_t)/dt.
        #
        # Use the exact derivative below.
        if self.time_power == 0.0:
            stochastic_a = self.sigma_v * bridge_a * eps
        else:
            # sigma(t) = sigma_v * [t(1-t)]^p
            #
            # d sigma / dt
            base = (t_exp * (1.0 - t_exp)).clamp_min(1e-8)

            dbase_dt = 1.0 - 2.0 * t_exp

            dsigma_dt = self.sigma_v * self.time_power * base.pow(self.time_power - 1.0) * dbase_dt

            stochastic_a = dsigma_dt * bridge_v * eps + sigma_t * bridge_a * eps

        stochastic_x = sigma_t * bridge_x * eps

        # --------------------------------------------------------------
        # Combine deterministic + stochastic paths
        # --------------------------------------------------------------

        omega_total = omega_t + stochastic_x

        v_t = v_cubic + stochastic_v

        a_t = a_cubic + stochastic_a

        # --------------------------------------------------------------
        # Position update on the torus
        # --------------------------------------------------------------

        x_t = self.manifold.expmap(
            x=x_0,
            u=omega_total,
        )

        # --------------------------------------------------------------
        # Return actual kinetic target
        # --------------------------------------------------------------

        return KineticPathSample(
            x_0=x_0,
            x_1=x_1,
            v_0=v_0,
            x_t=x_t,
            v_t=v_t,
            # Keep d available for diagnostics / optional d parameterization.
            dx_t=d,
            # IMPORTANT:
            # This is the actual derivative dv/dt.
            dv_t=a_t,
            t=t,
        )

    # ------------------------------------------------------------------
    # Parameterization
    # ------------------------------------------------------------------

    def reconstruct_acceleration(
        self,
        pred_target: Tensor,
        t: Tensor,
    ) -> Tensor:
        r"""No reconstruction is required anymore.

        The network predicts acceleration directly.
        """
        return pred_target
