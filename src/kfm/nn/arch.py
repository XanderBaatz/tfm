# Originally from: https://github.com/frcnt/kldm/blob/main/src_kldm/nn/arch.py

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch_geometric.utils import scatter

from kfm.models.vector_field import VectorFieldModel
from kfm.nn.embedding import FourierEmbedding, SinEmbedding
from kfm.nn.utils import scatter_center
from kfm.utils.manifolds.torus import UnitFlatTorus


class CSPALayer(Module):
    """GNN Crystal Structure Prediction Acceleration (CSPA) layer.

    Adapted from DiffCSP with modifications to support auxiliary velocity.
    """

    def __init__(
        self,
        dis_emb: Module,
        hidden_dim: int = 128,
        *,
        act_fn: Module | None = None,
        ln: bool = False,
    ) -> None:
        """Initialize CSPALayer."""
        super().__init__()
        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim
        act_fn = nn.SiLU() if act_fn is None else act_fn

        # Input: h_i, h_j, l_edge (6), v_ij (dis_dim), pos_diff (dis_dim)
        input_dim = hidden_dim * 2 + 2 * self.dis_dim + 6  # hidden states + distance/velocity + lattice

        self.v_proj = nn.Linear(3, self.dis_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        pos_diff: Tensor,
        v: Tensor,
        node_features: Tensor,
        lattices: Tensor,
        edge_node_index: Tensor,
        edge_graph_index: Tensor,
    ) -> Tensor:
        hi, hj = node_features[edge_node_index[0]], node_features[edge_node_index[1]]
        vi, vj = v[edge_node_index[0]], v[edge_node_index[1]]

        # Linear projection of relative velocity \bm{v}_j - \bm{v}_i
        vij = self.v_proj(vj - vi)

        # Sine/Cosine Fourier embedding on torus-wrapped periodic distance \bm{d}
        pos_diff_emb = self.dis_emb(pos_diff)
        l_edge = lattices[edge_graph_index]

        edges_input = torch.cat([hi, hj, l_edge, vij, pos_diff_emb], dim=1)

        return self.edge_mlp(edges_input)  # edge features

    def node_model(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_node_index: Tensor,
    ) -> Tensor:
        agg = scatter(
            edge_features,
            edge_node_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        return self.node_mlp(agg)

    def forward(
        self,
        pos_diff: Tensor,
        v: Tensor,
        node_features: Tensor,
        l: Tensor,
        edge_node_index: Tensor,
        edge_graph_index: Tensor,
    ) -> Tensor:
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(
            pos_diff=pos_diff,
            v=v,
            node_features=node_features,
            lattices=l,
            edge_node_index=edge_node_index,
            edge_graph_index=edge_graph_index,
        )

        node_output = self.node_model(
            node_features=node_features,
            edge_features=edge_features,
            edge_node_index=edge_node_index,
        )
        return node_input + node_output


class CSPANet(VectorFieldModel):
    r"""Neural acceleration and lattice field network for Kinetic Flow Matching.

    Outputs:
        - "dv": Target acceleration field \hat{u}_\theta(x_t, v_t, t, L) \approx \frac{dv_t}{dt}
        - "dl": Lattice velocity \hat{v}_{L,\theta}(x_t, L, t) \approx \frac{dL}{dt}
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 128,
        num_layers: int = 4,
        h_dim: int = 100,
        num_freqs: int = 10,
        ln: bool = True,
        smooth: bool = False,
        pred_h: bool = False,
        pred_dv: bool = True,
        pred_dl: bool = True,
        zero_cog: bool = True,
        time_emb: Module = None,
        manifold: UnitFlatTorus = None,
    ):
        super().__init__()

        self.manifold = manifold if manifold is not None else UnitFlatTorus(scale=1.0)

        self.act_fn = nn.SiLU()

        if smooth:
            self.node_embedding = nn.Linear(h_dim, hidden_dim, bias=False)
        else:
            self.node_embedding = nn.Embedding(h_dim + 1, hidden_dim)

        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.dis_emb = SinEmbedding(n_frequencies=num_freqs)

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)
        self.time_emb = time_emb

        self.layers = nn.ModuleList(
            [CSPALayer(self.dis_emb, hidden_dim=hidden_dim, act_fn=self.act_fn, ln=ln) for _ in range(num_layers)]
        )

        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Output Head 1: Acceleration u_v = dv/dt \in \mathbb{R}^3
        self.pred_dv = pred_dv
        if pred_dv:
            self.out_dv = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act_fn,
                nn.Linear(hidden_dim, 3, bias=False),
            )

        # Output Head 2: Lattice time derivative dL/dt \in \mathbb{R}^6
        self.pred_dl = pred_dl
        if pred_dl:
            # self.out_dl = nn.Sequential(
            #    nn.Linear(hidden_dim, hidden_dim),
            #    self.act_fn,
            #    nn.Linear(hidden_dim, 6, bias=False),
            # )
            self.out_dl = nn.Linear(hidden_dim, 6, bias=False)

        self.ln = ln
        self.smooth = smooth
        self.zero_cog = zero_cog

    def forward(
        self,
        t: Tensor,
        pos: Tensor,
        v: Tensor,
        h: Tensor,
        l: Tensor,
        node_index: Tensor,
        edge_node_index: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            t (Tensor): 1D time tensor [batch_size].
            pos (Tensor): Toroidal positions x_t in [0, 1)^3 [num_nodes, 3].
            v (Tensor): Velocities v_t [num_nodes, 3].
            h (Tensor): Atomic species discrete IDs or soft vectors.
            l (Tensor): Lattices [batch_size, 6].
            node_index (Tensor): Batch graph indices for each node [num_nodes].
            edge_node_index (Tensor): Edge source/target node index [2, num_edges].

        """
        # 1. Embed time t (expects 1D scalar per graph)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_emb(t)
        t_per_atom = t_emb[node_index]

        # 2. Node feature initialization
        node_features = self.node_embedding(h)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # 3. Compute minimal periodic torus displacement d \in [-0.5, 0.5)^3
        # pos_diff = self.manifold.logmap(pos[edge_node_index[0]], pos[edge_node_index[1]])
        pos_diff = pos[edge_node_index[1]] - pos[edge_node_index[0]]
        edge_graph_index = node_index[edge_node_index[0]]

        # 4. Message Passing
        for layer in self.layers:
            node_features = layer(
                pos_diff=pos_diff,
                v=v,
                node_features=node_features,
                l=l,
                edge_node_index=edge_node_index,
                edge_graph_index=edge_graph_index,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        out = {}

        # 5. Acceleration prediction \hat{u}_{t,v}
        if self.pred_dv:
            out_dv = self.out_dv(node_features)
            if self.zero_cog:
                out_dv = scatter_center(out_dv, index=node_index)
            out["dv"] = out_dv  # dv_t / dt

        # 6. Lattice velocity prediction \hat{v}_{t,L}
        if self.pred_dl:
            graph_features = scatter(node_features, node_index, dim=0, reduce="mean")
            out_dl = self.out_dl(graph_features)
            out["dl"] = out_dl  # dl_t / dt

        return out
