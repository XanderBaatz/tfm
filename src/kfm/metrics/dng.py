import numpy as np
import torch
from pymatgen.core import Structure
from scipy.spatial.distance import cdist
from torch import Tensor
from torchmetrics import Metric

from .common import (
    fp_composition,
    fp_structural,
    validity_composition,
    validity_structure,
    wdist_density,
    wdist_num_elements,
)
from .constants import CompScalerMeans, CompScalerStds

COV_Cutoffs = {
    "mp_20": {"struc": 0.4, "comp": 10.0},
    "alex_mp_20": {"struc": 0.4, "comp": 10.0},
    "carbon_24": {"struc": 0.2, "comp": 4.0},
    "perov_5": {"struc": 0.2, "comp": 4},
}


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """:param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.0,
)


def filter_fps(struct_fps, comp_fps):
    assert len(struct_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struct_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(input_fps: tuple, gt_fps: tuple, struc_cutoff: float, comp_cutoff: int):
    fp_struct, fp_comp = input_fps
    fp_struct_gt, fp_comp_gt = gt_fps
    n = len(fp_struct)

    assert len(fp_struct) == len(fp_comp)
    assert len(fp_struct_gt) == len(fp_comp_gt)

    fp_struct_filtered, fp_comp_filtered = filter_fps(fp_struct, fp_comp)

    fp_comp_filtered = CompScaler.transform(fp_comp_filtered)
    fp_comp_gt = CompScaler.transform(fp_comp_gt)

    fp_struct_filtered = np.array(fp_struct_filtered)
    fp_struct_gt = np.array(fp_struct_gt)

    fp_comp_filtered = np.array(fp_comp_filtered)
    fp_comp_gt = np.array(fp_comp_gt)

    struc_pdist = cdist(fp_struct_filtered, fp_struct_gt)
    comp_pdist = cdist(fp_comp_filtered, fp_comp_gt)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(struc_precision_dist <= struc_cutoff, comp_precision_dist <= comp_cutoff)) / n

    metrics_dict = {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        "struc_recall_dist": struc_recall_dist.tolist(),
        "struc_precision_dist": struc_precision_dist.tolist(),
        "comp_recall_dist": comp_recall_dist.tolist(),
        "comp_precision_dist": comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


class DNGMetric(Metric):
    """Evaluates de-novo generation quality (validity, coverage, density/#-elements match) via torchmetrics."""

    full_state_update: bool = False

    def __init__(self, data_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_name = data_name

        # structure/composition validity of generated structures
        self.add_state("valid_struct", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("valid_comp", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("valid", default=[], dist_reduce_fx=None, persistent=False)

        # fingerprints/structures of generated samples (valid ones only, else None)
        self.add_state("fp_struct", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("fp_comp", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("structures", default=[], dist_reduce_fx=None, persistent=False)

        # fingerprints/structures of the reference (ground-truth) dataset
        self.add_state("fp_struct_gt", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("fp_comp_gt", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("structures_gt", default=[], dist_reduce_fx=None, persistent=False)

    def update(self, preds: list[Structure], targets: list[Structure]) -> None:
        """Updater for metrics."""
        assert len(preds) == len(targets)

        for t in targets:
            self.fp_struct_gt.append(fp_structural(t))
            self.fp_comp_gt.append(fp_composition(t))
            self.structures_gt.append(t)

        for p in preds:
            vs = validity_structure(p) if p is not None else False
            vc = validity_composition(p) if p is not None else False

            fp_struct = fp_structural(p) if (p is not None and vs and vc) else None
            fp_comp = fp_composition(p) if (p is not None and vc) else None
            v = bool(vs and vc and fp_struct is not None)

            self.valid_struct.append(vs)
            self.valid_comp.append(vc)
            self.valid.append(v)

            self.fp_struct.append(fp_struct)
            self.fp_comp.append(fp_comp)
            self.structures.append(p)

    def compute(self) -> dict[str, Tensor]:
        total = len(self.valid)

        if total == 0:
            return {}

        result = {
            "valid_struct": torch.tensor(sum(self.valid_struct) / total, dtype=torch.float32),
            "valid_comp": torch.tensor(sum(self.valid_comp) / total, dtype=torch.float32),
            "valid": torch.tensor(sum(self.valid) / total, dtype=torch.float32),
        }

        cutoffs = COV_Cutoffs[self.data_name]
        cov_metrics, _ = compute_cov(
            (self.fp_struct, self.fp_comp),
            (self.fp_struct_gt, self.fp_comp_gt),
            struc_cutoff=cutoffs["struc"],
            comp_cutoff=cutoffs["comp"],
        )
        for key, val in cov_metrics.items():
            result[key] = torch.tensor(val, dtype=torch.float32)

        valid_structures = [s for s in self.structures if s is not None]
        if valid_structures and self.structures_gt:
            result["wdist_density"] = torch.tensor(
                wdist_density(valid_structures, self.structures_gt), dtype=torch.float32
            )
            result["wdist_num_elements"] = torch.tensor(
                wdist_num_elements(valid_structures, self.structures_gt), dtype=torch.float32
            )
        else:
            result["wdist_density"] = torch.tensor(float("nan"))
            result["wdist_num_elements"] = torch.tensor(float("nan"))

        return result


class DNGMetrics:
    def __init__(self, data_name: str):
        self.data_name = data_name

        self.valid_struct = ...
        self.valid_comp = ...
        self.valid = ...

        self.fp_comp_gt = ...
        self.fp_struct_gt = ...
        self.structures_gt = ...

        self.fp_comp = ...
        self.fp_struct = ...
        self.structures = ...

        self.reset()

    def __call__(self, input_s: list[Structure], target_s: list[Structure]):
        return self.update(input_s, target_s)

    def update(self, input_s: list[Structure], target_s: list[Structure]):
        assert len(input_s) == len(target_s)

        for st in target_s:
            fp_struct_gt, fp_comp_gt = fp_structural(st), fp_composition(st)
            self.fp_comp_gt.append(fp_comp_gt)
            self.fp_struct_gt.append(fp_struct_gt)
            self.structures_gt.append(st)

        for si in input_s:
            vs = validity_structure(si)
            vc = validity_composition(si)

            fp_struct, fp_comp = fp_structural(si) if (vs and vc) else None, (fp_composition(si) if vc else None)
            v = vs and vc and (fp_struct is not None)

            self.valid_struct.append(vs)
            self.valid_comp.append(vc)
            self.valid.append(v)

            self.fp_comp.append(fp_comp)
            self.fp_struct.append(fp_struct)
            self.structures.append(si)

    def summarize(self) -> dict:
        summary = {}

        summary["valid_struct"] = sum(self.valid_struct) / len(self.valid_struct)
        summary["valid_comp"] = sum(self.valid_comp) / len(self.valid_comp)
        summary["valid"] = sum(self.valid) / len(self.valid)

        summary_cov = self.summary_coverage()
        summary_density = self.summary_density()
        summary_num_elements = self.summary_num_elements()

        summary.update(**summary_cov, **summary_density, **summary_num_elements)

        return summary

    def reset(self):
        self.valid = []
        self.valid_struct = []
        self.valid_comp = []

        self.fp_comp_gt = []
        self.fp_struct_gt = []
        self.structures_gt = []

        self.fp_comp = []
        self.fp_struct = []
        self.structures = []

    def summary_coverage(self) -> dict:
        cutoff_dict = COV_Cutoffs[self.data_name]

        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            (self.fp_struct, self.fp_comp),
            (self.fp_struct_gt, self.fp_comp_gt),
            struc_cutoff=cutoff_dict["struc"],
            comp_cutoff=cutoff_dict["comp"],
        )
        return cov_metrics_dict

    def summary_density(self) -> dict:
        input_structures = [s for s in self.structures if s]
        return {"wdist_density": wdist_density(input_structures, self.structures_gt)}

    def summary_num_elements(self) -> dict:
        input_structures = [s for s in self.structures if s]
        return {"wdist_num_elements": wdist_density(input_structures, self.structures_gt)}
