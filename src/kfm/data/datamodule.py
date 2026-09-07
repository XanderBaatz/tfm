# Originally from: https://github.com/frcnt/kldm/blob/main/src_kldm/data/dataset.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_geometric.transforms as T  # noqa: N812
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from kfm.data.dataset import Dataset


def worker_init_fn(id: int):
    """DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystalDataModule(LightningDataModule):
    """Crystal data module."""

    def __init__(  # noqa: PLR0913
        self,
        train_dataset: Dataset | Subset,
        num_workers: DictConfig | int,
        batch_size: DictConfig | int,
        val_dataset: Dataset | Subset | None = None,
        test_dataset: Dataset | Subset | None = None,
        num_train_subset: float | None = None,
        num_val_subset: float | None = None,
        num_test_subset: float | None = None,
        subset_seed: int = 42,
        **_: Any,  # noqa: ANN401
    ) -> None:
        """Initialize crystal dataset."""
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        if num_train_subset is not None and num_train_subset > 0:
            train_dataset = self.get_random_subset(train_dataset, num_train_subset, seed=subset_seed)
        self.train_dataset = train_dataset

        if val_dataset is not None and num_val_subset is not None and num_val_subset > 0:
            val_dataset = self.get_random_subset(val_dataset, num_val_subset, seed=subset_seed)
        self.val_dataset = val_dataset

        if test_dataset is not None and num_test_subset is not None and num_test_subset > 0:
            test_dataset = self.get_random_subset(test_dataset, num_test_subset, seed=subset_seed)
        self.test_dataset = test_dataset

        self.datasets = [train_dataset, val_dataset, test_dataset]

        self.save_hyperparameters(logger=False, ignore=["train_dataset", "val_dataset", "test_dataset"])

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Train dataloader for crystal dataset."""
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        """Val dataloader."""
        return (
            DataLoader(
                self.val_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            if self.val_dataset is not None
            else None
        )

    def test_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        """Test dataloader."""
        return (
            DataLoader(
                self.test_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            if self.test_dataset is not None
            else None
        )

    @staticmethod
    def get_random_subset(dataset: Dataset | Subset, subset_size: float, seed: int) -> Subset:
        """Return a random subset of dataset given a count or fraction.

        Fixes index selection bug by sampling from the full range of dataset indices.
        """
        total_len = len(dataset)
        if isinstance(subset_size, float) and 0.0 < subset_size <= 1.0:
            count = int(total_len * subset_size)
        elif isinstance(subset_size, int) and 0 < subset_size < total_len:
            count = subset_size
        else:
            return dataset

        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(total_len))[:count]
        return Subset(dataset, indices=indices)


class DataModule(LightningDataModule):
    def __init__(
        self,
        transform: T.BaseTransform,
        train_path: str | Path,
        val_path: str | Path,
        train_batch_size: int,
        val_batch_size: int,
        num_val_subset: int | None = -1,
        test_path: str | Path | None = None,
        test_batch_size: int | None = None,
        num_test_subset: int | None = 10000,
        num_workers: int = 0,
        pin_memory: bool = False,
        subset_seed: int = 42,
    ):
        super().__init__()

        self.train_dataset = Dataset(path=train_path, transform=transform)

        val_dataset = Dataset(path=val_path, transform=transform)
        if isinstance(num_val_subset, int) and num_val_subset > -1 and num_val_subset < len(val_dataset):
            val_dataset = self.get_random_subset(val_dataset, num_val_subset, seed=subset_seed)
        self.val_dataset = val_dataset

        if test_path:
            test_dataset = Dataset(path=test_path, transform=transform)
            if isinstance(num_test_subset, int) and num_test_subset > -1 and num_test_subset < len(test_dataset):
                test_dataset = self.get_random_subset(test_dataset, num_test_subset, seed=subset_seed)
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None

        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def get_random_subset(dataset, subset_size, seed):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(subset_size))
        return Subset(dataset, indices=indices)
