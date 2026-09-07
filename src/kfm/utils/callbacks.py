import warnings
from collections.abc import Sequence
from typing import Any

import lightning as L
import torch
from lightning import Callback
from torch_geometric.data import Batch
from torchmetrics import Metric, MetricCollection, MetricTracker

from kfm.generator.generator import FlowGenerator
from kfm.generator.kinetic import structures_from_batch, structures_from_tensors

__all__ = [
    "StructureEvaluator",
    "structures_from_batch",
    "structures_from_tensors",
]


class StructureEvaluator(Callback):
    """Runs sampling-based structure generation and tracks torchmetrics during validation.

    Modular by design: any `FlowGenerator` may be plugged in, and `metrics` accepts a single
    `Metric`, a ready-made `MetricCollection`, or a sequence/dict of `Metric`s (e.g. CSP, DNG).
    """

    def __init__(
        self,
        generator: FlowGenerator,
        metrics: Metric | MetricCollection | Sequence[Metric] | dict[str, Metric] | None = None,
        eval_every_n_epochs: int = 25,
        log_prefix: str = "val",
    ) -> None:
        super().__init__()

        self.generator = generator
        self.eval_every_n_epochs = eval_every_n_epochs
        self.log_prefix = log_prefix

        # Standardize inputs into a MetricCollection tracked across validation epochs
        if metrics is None:
            metric_collection = MetricCollection([])
        elif isinstance(metrics, MetricCollection):
            metric_collection = metrics
        elif isinstance(metrics, Metric):
            metric_collection = MetricCollection([metrics])
        elif isinstance(metrics, (list, tuple, dict)):
            metric_collection = MetricCollection(metrics)
        else:
            msg = f"Unsupported metric container type: {type(metrics)}"
            raise TypeError(msg)

        self.metrics = MetricTracker(metric_collection, maximize=True)

    def _should_eval(self, trainer: L.Trainer) -> bool:
        """Determines whether sampling evaluation should run on current epoch."""
        if trainer.sanity_checking:
            return False
        return (trainer.current_epoch + 1) % self.eval_every_n_epochs == 0

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._should_eval(trainer):
            self.metrics.increment()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._should_eval(trainer):
            return

        # Wire the live (correctly-placed/trained) flow module into the generator
        self.generator.flow_module = pl_module.flow_module
        num_graphs = getattr(batch, "num_graphs", 1)

        try:
            with torch.no_grad():
                pred_structures = self.generator.generate(batch)
        except Exception as e:
            warnings.warn(f"Failed to generate structures: {e}")
            pred_structures = [None] * num_graphs

        target_structures = self.generator.target_structures(batch)

        self.metrics.update(preds=pred_structures, targets=target_structures)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._should_eval(trainer):
            return

        for metric_name, val in self.metrics.compute().items():
            pl_module.log(
                f"{self.log_prefix}/{metric_name}",
                val,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
