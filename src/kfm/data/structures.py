# Taken from https://github.com/microsoft/mattergen/blob/main/mattergen/common/data/chemgraph.py
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from typing import Any

import torch_geometric.data as pyg_data
from torch import LongTensor, Tensor
from torch_geometric.typing import OptTensor


class KineticCrystalState(pyg_data.Data):
    """PyG Data object describing a crystal with positions x, lattice l, atomic numbers h, velocities l and accelerations a."""  # noqa: E501

    def __init__(
        self,
        pos: Tensor | None = None,
        l: Tensor | None = None,  # noqa: E741
        h: LongTensor | None = None,
        v: Tensor | None = None,
        edge_index: Tensor | None = None,
        **kwargs: dict[str, Tensor],
    ) -> None:
        """Initialize Kinetic Crystal State.

        Args:
            pos (Tensor | None, optional): fractional coordinates. Defaults to None.
            l (Tensor | None, optional): lattice. Defaults to None.
            h (LongTensor | None, optional): atomic numbers. Defaults to None.
            v (Tensor | None, optional): velocity. Defaults to None.
            edge_index (Tensor | None, optional): edge indexes. Defaults to None.
            **kwargs (optional): Additional attributes to be stored in the data object.

        """
        # Pass pos and edge_index into super() so PyG's Storage engine manages graph connectivity
        super().__init__(pos=pos, edge_index=edge_index, **kwargs)

        if l is not None:
            self.l = l.view(1, -1) if l.dim() == 1 else l
        if h is not None:
            self.h = h
        if v is not None:
            self.v = v

        self.__dict__["_frozen"] = True

    def __setattr__(self, attr: str, value: Any) -> None:  # noqa: ANN401
        """Set attribute."""
        if self.__dict__.get("_frozen", False) and attr not in (
            "_num_graphs",
            "_slice_dict",
            "_inc_dict",
            "_collate_structure",
        ):
            msg = f"Replacing KineticCrystalState.{attr} in-place. Use state.replace(...) to make a shallow copy."
            raise AttributeError(msg)

        return super().__setattr__(attr, value)

    def replace(self, **kwargs: OptTensor | str | float | list) -> KineticCrystalState:
        """Shallow copy updating specified fields."""
        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)

        for key, value in kwargs.items():
            out._store[key] = value  # noqa: SLF001
        out._store._parent = out  # noqa: SLF001

        return out

    def get_batch_idx(self, field_name: str) -> LongTensor | None:
        """Get batch indices for node-level vs graph-level fields."""
        if not isinstance(self, pyg_data.Batch):
            msg = "Should be Batch type."
            raise TypeError(msg)

        if field_name in ["l", "lengths", "angles"]:
            return None  # Graph-level feature [B, 6]
        if field_name in ["pos", "h", "v"]:
            return self.batch  # Node-level feature [sum(N_i)]

        return getattr(self, f"{field_name}_batch", None)


# Pointer to dynamic PyG batch class for type annotations
KineticCrystalStateBatch = pyg_data.Batch(_base_cls=KineticCrystalState).__class__
