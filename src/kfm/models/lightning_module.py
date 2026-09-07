from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torch_geometric.data import Batch
from torch_geometric.data.data import Data

from kfm.models.flow_module import FlowModule  # noqa: TC001


class OptimizerPartial(Protocol):
    """Callable protocol to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer: ...


class SchedulerPartial(Protocol):
    """Callable protocol to instantiate a learning rate scheduler."""

    def __call__(self, optimizer: Optimizer) -> Any: ...


def get_default_optimizer(params: Any) -> Optimizer:
    return AdamW(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


class FlowLightningModule(L.LightningModule):
    """LightningModule for instantiating and training a FlowModule."""

    def __init__(
        self,
        flow_module: FlowModule,
        optimizer_partial: OptimizerPartial | None = None,
        scheduler_partials: Sequence[dict[str, Any | SchedulerPartial]] | None = None,
    ) -> None:
        """Initialize FlowLightningModule.

        Args:
            flow_module: Core PyTorch module managing vector fields and flow sampling.
            optimizer_partial: Partial function/callable to instantiate the optimizer.
            scheduler_partials: Sequence of scheduler configs or partials.

        """
        super().__init__()
        self.save_hyperparameters(ignore=("optimizer_partial", "scheduler_partials", "flow_module"))

        self.flow_module = flow_module
        self._optimizer_partial = optimizer_partial or get_default_optimizer
        self._scheduler_partials = scheduler_partials or []

    @classmethod
    def load_from_checkpoint_and_config(
        cls,
        checkpoint_path: str,
        config: DictConfig,
        map_location: str | None = None,
        strict: bool = True,
    ) -> tuple[FlowLightningModule, torch.nn.modules.module._IncompatibleKeys]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        lightning_module = instantiate(config)
        assert isinstance(lightning_module, cls)

        result = lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)

        return lightning_module, result

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer_partial(params=self.flow_module.parameters())

        if not self._scheduler_partials:
            return optimizer

        lr_schedulers = [
            {
                **scheduler_dict,
                "scheduler": scheduler_dict["scheduler"](optimizer=optimizer),
            }
            for scheduler_dict in self._scheduler_partials
        ]

        return [optimizer], lr_schedulers

    def training_step(self, batch: Batch | Data, batch_idx: int) -> STEP_OUTPUT:
        return self._calc_loss(batch, train=True)

    def validation_step(self, batch: Batch | Data, batch_idx: int) -> STEP_OUTPUT | None:
        return self._calc_loss(batch, train=False)

    def test_step(self, batch: Batch | Data, batch_idx: int) -> STEP_OUTPUT | None:
        return self._calc_loss(batch, train=False)

    def _calc_loss(self, batch: Batch | Data, train: bool) -> STEP_OUTPUT | None:
        loss, loss_dict = self.flow_module.calc_loss(batch)

        step_type = "train" if train else "val"
        batch_size = getattr(batch, "num_graphs", 1)

        self.log(
            f"loss_{step_type}",
            loss,
            on_step=train,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        for key, value in loss_dict.items():
            if key == "total_loss":
                continue
            self.log(
                f"{key}_{step_type}",
                value,
                on_step=train,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )

        return loss
