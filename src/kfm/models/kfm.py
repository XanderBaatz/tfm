import torch
import torch.nn.functional as F

from kfm.data_types import FlowState, GraphBatch
from kfm.models.flow_module import FlowModule


class KFM(FlowModule):
    """Kinetic Flow Matching Module."""

    # Map target keys from sample_path to corresponding network output keys in predict_velocity
    TARGET_TO_PRED_KEY = {
        "d": "dv",  # Kinetic displacement target -> network 'dv' output
        "l": "dl",  # Target lattice matrix L_1 -> network 'dl' output
    }

    def calc_loss(self, batch: GraphBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = self.sample_timesteps(batch)
        latents, targets = self.multi_flow.sample_path(batch, t)
        preds = self.predict_velocity(batch=batch, latents=latents, t=t)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=batch.pos.device)

        # 1. Main MSE loss across fields
        for key, target_val in targets.items():
            # Resolve prediction key, falling back to original key if present in preds
            pred_key = self.TARGET_TO_PRED_KEY.get(key, key)
            if pred_key not in preds and key in preds:
                pred_key = key

            field_loss = F.mse_loss(preds[pred_key], target_val)
            loss_dict[f"loss_{key}"] = field_loss.detach()
            total_loss += field_loss

        # 2. Slice lattice sub-components for tracking ('dl' or 'l')
        lat_pred_key = next((k for k in ("dl", "l") if k in preds), None)
        lat_target_key = next((k for k in ("dl", "l") if k in targets), None)

        if lat_pred_key is not None and lat_target_key is not None:
            pred_l = preds[lat_pred_key]
            target_l = targets[lat_target_key]

            # Lengths (first 3 channels: log_abc)
            loss_len = F.mse_loss(pred_l[:, :3], target_l[:, :3])
            loss_dict["loss_len"] = loss_len.detach()

            # Angles (channels 3+: tan_angles)
            loss_ang = F.mse_loss(pred_l[:, 3:], target_l[:, 3:])
            loss_dict["loss_ang"] = loss_ang.detach()

        loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict

    def predict_velocity(
        self,
        batch: GraphBatch,
        latents: FlowState,
        t: torch.Tensor,
    ) -> FlowState:
        """Unpacks latents and graph indices into vector field forward arguments."""
        return self.vector_field_model(
            t=t,
            pos=latents["pos"],
            v=latents["v"],
            l=latents["l"],
            h=getattr(batch, "h", None),
            node_index=getattr(batch, "batch", None),
            edge_node_index=getattr(batch, "edge_index", None),
        )


class KFMOld(FlowModule):
    """Kinetic Flow Matching Module."""

    def calc_loss(self, batch: GraphBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = self.sample_timesteps(batch)
        latents, targets = self.multi_flow.sample_path(batch, t)
        preds = self.predict_velocity(batch=batch, latents=latents, t=t)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=batch.pos.device)

        # 1. Main MSE loss across fields (e.g., 'dv' / 'x', 'dl' / 'l')
        for key, target_val in targets.items():
            pred_key = "dv" if key == "d" else key  # FIXME: make this more general
            field_loss = F.mse_loss(preds[pred_key], target_val)
            loss_dict[f"loss_{key}"] = field_loss.detach()
            total_loss += field_loss

        # 2. Slice lattice sub-components for tracking ('dl' or 'l')
        lat_key = "dl" if "dl" in targets else ("l" if "l" in targets else None)
        if lat_key is not None:
            pred_l = preds[lat_key]
            target_l = targets[lat_key]

            # Lengths (first 3 channels: log_abc)
            loss_len = F.mse_loss(pred_l[:, :3], target_l[:, :3])
            loss_dict["loss_len"] = loss_len.detach()

            # Angles (channels 3+: tan_angles)
            loss_ang = F.mse_loss(pred_l[:, 3:], target_l[:, 3:])
            loss_dict["loss_ang"] = loss_ang.detach()

        loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict

    def predict_velocity(
        self,
        batch: GraphBatch,
        latents: FlowState,
        t: torch.Tensor,
    ) -> FlowState:
        """Unpacks latents and graph indices into CSPANet forward arguments."""
        return self.vector_field_model(
            t=t,
            pos=latents["pos"],
            v=latents["v"],
            l=latents["l"],
            h=getattr(batch, "h", None),
            node_index=getattr(batch, "batch", None),
            edge_node_index=getattr(batch, "edge_index", None),
        )
