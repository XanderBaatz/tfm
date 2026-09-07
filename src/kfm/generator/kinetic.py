from collections.abc import Sequence

import torch
from ase.data import chemical_symbols
from pymatgen.core import Lattice, Structure
from torch import Tensor
from torch_geometric.data import Batch

from kfm.data.transforms import ContinuousIntervalAngles, ContinuousIntervalLengths
from kfm.data_types import FlowState, GraphBatch
from kfm.generator.generator import FlowGenerator
from kfm.models.flow_module import FlowModule
from kfm.models.wrapper import CSPANetWrapper
from kfm.solver.kinetic_solver import KineticCrystalODESolver


def structures_from_tensors(
    tensors: dict[str, torch.Tensor],
    ptr: torch.Tensor,
    decoder: dict[int, str] | list[str],
    transform_lengths: ContinuousIntervalLengths,
    transform_angles: ContinuousIntervalAngles,
    pos_range: float = 1.0,
) -> list[Structure]:
    h = tensors["h"].to("cpu")
    if h.ndim > 1 and h.shape[1] > 1:
        h = torch.argmax(h, dim=1)

    h = h.numpy()
    pos = tensors["pos"].to("cpu").numpy() / pos_range
    l = tensors["l"].to("cpu").numpy()
    ptr = ptr.to("cpu").numpy()

    structures_list = []

    for i, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
        coords = pos[start_idx:end_idx, :]
        symbols = [decoder[idx.item()] for idx in h[start_idx:end_idx]]
        n = len(symbols)

        li = l[i]
        log_abc, tan_angles = li[:3], li[3:]  # FIXME: make more general
        a, b, c = transform_lengths.invert_one(log_abc, n)
        alpha, beta, gamma = transform_angles.invert_one(tan_angles, n)
        lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        coords = coords % 1.0
        struc = Structure(lattice=lattice, species=symbols, coords=coords, coords_are_cartesian=False)
        struc = struc.get_sorted_structure()

        structures_list.append(struc)

    return structures_list


def structures_from_batch(
    batch: Batch,
    decoder: dict[int, str] | list[str],
    transform_lengths: ContinuousIntervalLengths,
    transform_angles: ContinuousIntervalAngles,
    pos_range: float = 1.0,
) -> list[Structure]:
    tensors = {"h": batch.h, "pos": batch.pos, "l": batch.l}
    ptr = batch.ptr

    return structures_from_tensors(
        tensors,
        ptr,
        decoder,
        transform_lengths,
        transform_angles,
        pos_range=pos_range,
    )


class KineticCrystalFlowGenerator(FlowGenerator):
    """Generates crystal structures by integrating the kinetic phase-space ODE."""

    def __init__(
        self,
        solver: KineticCrystalODESolver,
        transform_lengths: ContinuousIntervalLengths,
        transform_angles: ContinuousIntervalAngles,
        flow_module: FlowModule | None = None,
        decoder: Sequence[str] | dict[int, str] = chemical_symbols,
        step_size: float | None = 0.01,
        ode_method: str = "midpoint",
    ) -> None:
        super().__init__(flow_module=flow_module, solver=solver)

        self.transform_lengths = transform_lengths
        self.transform_angles = transform_angles
        self.decoder = decoder
        self.step_size = step_size
        self.ode_method = ode_method

    @property
    def pos_range(self) -> float:
        return self.solver.manifold.scale

    def sample_state(
        self,
        batch: GraphBatch,
        step_size: float | None = None,
        ode_method: str | None = None,
        time_grid: Tensor | None = None,
        verbose: bool = False,
    ) -> FlowState:
        """Sample priors at t=0 and integrate the kinetic ODE up to t=1."""
        device = batch.pos.device
        num_graphs = getattr(batch, "num_graphs", 1)

        self.solver.velocity_model = CSPANetWrapper(model=self.flow_module.vector_field_model)

        t_zeros = torch.zeros(num_graphs, device=device)
        latents_0, _ = self.flow_module.multi_flow.sample_path(batch, t_zeros)

        x_init = {"x": latents_0["pos"], "v": latents_0["v"], "l": latents_0["l"]}

        return self.solver.sample(
            x_init=x_init,
            step_size=step_size if step_size is not None else self.step_size,
            method=ode_method or self.ode_method,
            time_grid=time_grid if time_grid is not None else torch.tensor([0.0, 1.0], device=device),
            verbose=verbose,
            batch=batch,
        )

    def state_to_structures(
        self,
        state: FlowState,
        batch: GraphBatch,
    ) -> list[Structure]:
        tensors = {"h": batch.h, "pos": state["x"], "l": state["l"]}

        return structures_from_tensors(
            tensors=tensors,
            ptr=batch.ptr,
            decoder=self.decoder,
            transform_lengths=self.transform_lengths,
            transform_angles=self.transform_angles,
            pos_range=self.pos_range,
        )

    def state_from_batch(self, batch: GraphBatch) -> FlowState:
        return {"x": batch.pos, "l": batch.l}
