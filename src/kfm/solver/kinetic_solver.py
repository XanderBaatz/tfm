import math
from collections.abc import Callable, Sequence

import torch
from flow_matching.solver import Solver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Manifold
from torch import Tensor

from kfm.utils.manifolds.torus import UnitFlatTorus

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class KineticCrystalODESolver(Solver):
    """Kinetic Riemannian ODE Solver for crystal generation."""

    def __init__(
        self,
        manifold: Manifold | None = None,
        velocity_model: ModelWrapper | None = None,
    ) -> None:
        super().__init__()
        self.manifold = manifold if manifold is not None else UnitFlatTorus(scale=1.0)
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: dict[str, Tensor],
        step_size: float | None = None,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> dict[str, Tensor] | Sequence[dict[str, Tensor]]:
        """Integrates phase-space state from t=0 to t=1."""
        step_fns = {
            "euler": _kinetic_euler_step,
            "midpoint": _kinetic_midpoint_step,
            "rk4": _kinetic_rk4_step,
        }
        assert method in step_fns, f"Unknown method {method}"
        step_fn = step_fns[method]

        def velocity_func(x_dict: dict[str, Tensor], t: Tensor) -> dict[str, Tensor]:
            # Returns network predictions for velocity acceleration ("v") and lattice field ("l")
            return self.velocity_model(x=x_dict, t=t, **model_extras)

        device = x_init["x"].device
        time_grid = torch.sort(time_grid.to(device=device)).values

        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (t_final - t_init) > step_size, (
                f"Time interval [{t_init}, {t_final}] must be larger than step_size {step_size}."
            )
            n_steps = math.ceil((t_final - t_init) / step_size)
            t_discretization = torch.linspace(t_init, t_final, n_steps + 1, device=device)

        t0s = t_discretization[:-1]

        if verbose:
            if not TQDM_AVAILABLE:
                msg = "tqdm is required for verbose mode."
                raise ImportError(msg)
            t0s = tqdm(t0s)

        if return_intermediates:
            xts = []

        with torch.set_grad_enabled(enable_grad):
            xt = {k: v.clone() for k, v in x_init.items()}

            for t0, t1 in zip(t0s, t_discretization[1:]):
                dt = t1 - t0
                xt = step_fn(
                    velocity_func,
                    xt,
                    t0,
                    dt,
                    manifold=self.manifold,
                )
                if return_intermediates:
                    xts.append({k: v.clone() for k, v in xt.items()})

        if return_intermediates:
            return xts
        return xt


def _kinetic_euler_step(
    velocity_model: Callable,
    xt: dict[str, Tensor],
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
) -> dict[str, Tensor]:
    """Kinetic Euler Step: Position evolves via current velocity; velocity & lattice via model fields."""
    vf = velocity_model(xt, t0)  # vf contains {"v": dv/dt, "l": dl/dt}

    # Position update on Torus: x_{t+dt} = expmap(x_t, v_t * dt) = (x_t + v_t * dt) % 1.0
    x_next = manifold.expmap(xt["x"], xt["v"] * dt)
    v_next = xt["v"] + vf["v"] * dt
    l_next = xt["l"] + vf["l"] * dt

    return {"x": x_next, "v": v_next, "l": l_next}


def _kinetic_midpoint_step(
    velocity_model: Callable,
    xt: dict[str, Tensor],
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
) -> dict[str, Tensor]:
    """2nd-Order Midpoint Step for phase-space states."""
    half_dt = 0.5 * dt
    vf1 = velocity_model(xt, t0)

    # State at t + dt/2
    x_mid = manifold.expmap(xt["x"], xt["v"] * half_dt)
    v_mid = xt["v"] + vf1["v"] * half_dt
    l_mid = xt["l"] + vf1["l"] * half_dt
    mid_state = {"x": x_mid, "v": v_mid, "l": l_mid}

    # Field at midpoint
    vf_mid = velocity_model(mid_state, t0 + half_dt)

    # Full step update
    x_next = manifold.expmap(xt["x"], mid_state["v"] * dt)
    v_next = xt["v"] + vf_mid["v"] * dt
    l_next = xt["l"] + vf_mid["l"] * dt

    return {"x": x_next, "v": v_next, "l": l_next}


def _kinetic_rk4_step(
    velocity_model: Callable,
    xt: dict[str, Tensor],
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
) -> dict[str, Tensor]:
    """4th-Order Runge-Kutta Step adapted for Kinetic Phase-Space Dynamics."""
    # k1
    vf1 = velocity_model(xt, t0)
    k1_x, k1_v, k1_l = xt["v"], vf1["v"], vf1["l"]

    # k2
    state_k2 = {
        "x": manifold.expmap(xt["x"], 0.5 * dt * k1_x),
        "v": xt["v"] + 0.5 * dt * k1_v,
        "l": xt["l"] + 0.5 * dt * k1_l,
    }
    vf2 = velocity_model(state_k2, t0 + 0.5 * dt)
    k2_x, k2_v, k2_l = state_k2["v"], vf2["v"], vf2["l"]

    # k3
    state_k3 = {
        "x": manifold.expmap(xt["x"], 0.5 * dt * k2_x),
        "v": xt["v"] + 0.5 * dt * k2_v,
        "l": xt["l"] + 0.5 * dt * k2_l,
    }
    vf3 = velocity_model(state_k3, t0 + 0.5 * dt)
    k3_x, k3_v, k3_l = state_k3["v"], vf3["v"], vf3["l"]

    # k4
    state_k4 = {
        "x": manifold.expmap(xt["x"], dt * k3_x),
        "v": xt["v"] + dt * k3_v,
        "l": xt["l"] + dt * k3_l,
    }
    vf4 = velocity_model(state_k4, t0 + dt)
    k4_x, k4_v, k4_l = state_k4["v"], vf4["v"], vf4["l"]

    # Combine weighted stages
    eff_v_x = (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0
    eff_v_v = (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
    eff_v_l = (k1_l + 2 * k2_l + 2 * k3_l + k4_l) / 6.0

    return {
        "x": manifold.expmap(xt["x"], dt * eff_v_x),
        "v": xt["v"] + dt * eff_v_v,
        "l": xt["l"] + dt * eff_v_l,
    }
