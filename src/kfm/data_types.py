"""Shared type aliases for the flow-matching pipeline (priors, paths, flows, generation).

Centralized here so abstractions don't each repeat the same unions, and so a future,
data-type-agnostic version of this pipeline has a single place to widen these aliases.
"""

from typing import TypeVar

import torch
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

# A single training/validation/test batch of graphs (e.g. crystal structures). This is
# the concrete batch type used by KFM's PyG-based classes (KineticFlow, KFM, ...).
GraphBatch = Batch | Data

# Generic batch/data-container type, used by the abstract pipeline (BasePrior, Flow,
# MultiFlow, FlowModule, FlowGenerator, ...) so they aren't tied to `GraphBatch`
# specifically -- a non-graph domain can subclass them with its own batch type.
TBatch = TypeVar("TBatch")

# The state of one or more flow fields at a given time t, keyed by field name
# (e.g. {"pos": x_t, "v": v_t} or {"l": l_t}). Also used for the corresponding
# target vector field(s) u_t predicted by a `VectorFieldModel`.
FlowState = dict[str, Tensor]


def get_batch_size(batch: object, default: int = 1) -> int:
    """Best-effort, duck-typed batch size lookup (e.g. PyG's `Batch.num_graphs`)."""
    return getattr(batch, "num_graphs", default)


def get_batch_device(batch: object) -> torch.device:
    """Best-effort, duck-typed device lookup that doesn't assume a particular batch type."""
    device = getattr(batch, "device", None)
    if isinstance(device, torch.device):
        return device

    for attr in ("pos", "x", "l"):
        tensor = getattr(batch, attr, None)
        if isinstance(tensor, Tensor):
            return tensor.device

    msg = f"Could not determine device for batch of type {type(batch)!r}."
    raise TypeError(msg)
