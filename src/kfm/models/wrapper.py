import torch
from flow_matching.utils.model_wrapper import ModelWrapper
from torch import Tensor
from torch_geometric.data import Batch, Data

from kfm.models.flow import MultiFlow
from kfm.models.vector_field import VectorFieldModel


class CSPANetWrapper(ModelWrapper):
    def __init__(
        self,
        model: VectorFieldModel,
        multi_flow: MultiFlow | None = None,
    ) -> None:
        super().__init__(model=model)
        self.multi_flow = multi_flow

    def forward(
        self,
        x: dict[str, Tensor],
        t: Tensor,
        batch: Batch | Data = None,
        node_index: Tensor | None = None,
        edge_node_index: Tensor | None = None,
        h: Tensor | None = None,
        **extras,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x (dict[str, Tensor]): Current state dict containing:
                - "x": Toroidal fractional positions pos_t [num_nodes, 3]
                - "v": Particle velocities v_t [num_nodes, 3]
                - "L": Lattices l_t [batch_size, 6]
            t (Tensor): Continuous time scalar or 1D tensor [batch_size].
            batch (Batch | Data, optional): PyG data batch object.
            node_index (Tensor, optional): Batch index per node (defaults to batch.batch).
            edge_node_index (Tensor, optional): Edge connections [2, num_edges] (defaults to batch.edge_node_index).
            h (Tensor, optional): Atomic species features (defaults to batch.h).

        """
        # 1. Resolve PyG graph connectivity indices
        if batch is not None:
            if node_index is None:
                node_index = getattr(batch, "batch", None)
            if edge_node_index is None:
                edge_node_index = getattr(batch, "edge_node_index", None)
                if edge_node_index is None:
                    edge_node_index = getattr(batch, "edge_index", None)
            if h is None:
                h = getattr(batch, "h", None)

        if node_index is None or edge_node_index is None:
            msg = (
                "CSPANetModelWrapper requires valid PyG graph connectivity indices "
                "('node_index' and 'edge_node_index'), supplied either directly or via 'batch'."
            )
            raise ValueError(msg)

        # 2. Format 0D/1D continuous time tensor
        # if t.dim() == 0:
        #    num_graphs = getattr(batch, "num_graphs", int(node_index.max().item() + 1))
        #    t = t.expand(num_graphs)
        if t.ndim == 0:
            num_graphs = getattr(batch, "num_graphs", 1) if batch is not None else int(node_index.max().item() + 1)
            t = t.unsqueeze(0).expand(num_graphs)

        # 3. Unpack states from solver dictionary
        pos_t = x["x"]
        v_t = x["v"]
        l_t = x["l"]

        # 4. Evaluate network predictions
        net_out = self.model(
            t=t,
            pos=pos_t,
            v=v_t,
            h=h,
            l=l_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
            **extras,
        )

        dv_pred = net_out["dv"]

        # 1. Dynamically locate the Kinetic path module by class type

        # Correctly extract KineticFlow instance to call construct_prediction
        if self.multi_flow is not None:
            flows = getattr(self.multi_flow, "flows", [self.multi_flow])
            flow_list = flows.values() if isinstance(flows, (dict, torch.nn.ModuleDict)) else flows
            for flow in flow_list:
                if hasattr(flow, "construct_prediction"):
                    dv_pred = flow.construct_prediction(pred=dv_pred, t=t, node_index=node_index)
                    break

        # 5. Construct phase-space velocity field for ODE integration
        return {
            "x": v_t,  # dx/dt = v_t (Kinematic position update)
            "v": dv_pred,  # dv/dt = predicted acceleration a_t
            "l": net_out["dl"],  # dL/dt = predicted lattice velocity u_L_t
        }
