import warnings

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from torch import Tensor
from torchmetrics import Metric

from kfm.metrics.common import validity_composition, validity_structure
from kfm.utils import safe_divide


class CSPMetric(Metric):
    """Evaluates spatial RMSE and match rate using PyMatGen's StructureMatcher."""

    full_state_update: bool = False

    def __init__(
        self,
        ltol: float = 0.3,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

        # structure validity
        self.add_state("valid_struct", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("valid_comp", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("valid", default=[], dist_reduce_fx=None, persistent=False)

        # match rate and rmse
        self.add_state("matches", default=[], dist_reduce_fx=None, persistent=False)
        self.add_state("rmses", default=[], dist_reduce_fx=None, persistent=False)

    def update(
        self,
        preds: list[Structure],
        targets: list[Structure],
    ) -> None:
        """Updater for metrics."""
        assert len(preds) == len(targets)

        for p, t in zip(preds, targets, strict=False):
            # evaluate structure and composition validity
            vs = validity_structure(p) if p is not None else False
            vc = validity_composition(p) if p is not None else False
            v = vs and vc

            self.valid_struct.append(vs)
            self.valid_comp.append(vc)
            self.valid.append(v)

            # match evaluation
            if not (v or p is None) or (t is None):
                self.matches.append(False)
                self.rmses.append(float("inf"))
                continue

            try:
                # numerically degenerate lattices (near-collinear vectors, extreme lengths)
                # can crash pymatgen/LAPACK's internal lattice reduction here
                rms_dist = self.matcher.get_rms_dist(p, t)
            except Exception as e:
                warnings.warn(f"StructureMatcher failed to compare structures: {e}", stacklevel=2)
                rms_dist = None

            if rms_dist is not None:
                self.matches.append(True)
                self.rmses.append(rms_dist[0])  # normalized rmse
            else:
                self.matches.append(False)
                self.rmses.append(float("inf"))

    def compute(self) -> dict[str, Tensor]:
        total = len(self.valid)

        if total == 0:
            return {}

        val_struct = torch.tensor(sum(self.valid_struct) / total, dtype=torch.float32)
        val_comp = torch.tensor(sum(self.valid_comp) / total, dtype=torch.float32)
        val_comb = torch.tensor(sum(self.valid) / total, dtype=torch.float32)

        match_rate = torch.tensor(sum(self.matches) / total, dtype=torch.float32)
        valid_rmses = [r for r in self.rmses if r != float("inf")]
        avg_rmse = (
            torch.tensor(sum(valid_rmses) / len(valid_rmses), dtype=torch.float32)
            if valid_rmses
            else torch.tensor(float("nan"))
        )

        return {
            "valid_struct": val_struct,
            "valid_comp": val_comp,
            "valid": val_comb,
            "match_rate": match_rate,
            "rms_distance": avg_rmse,
        }


class CSPMetrics:
    def __init__(self, stol: float = 0.5, angle_tol: float = 10.0, ltol: float = 0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        self.valid = ...
        self.match = ...
        self.rmse = ...

        self.reset()

    def __call__(self, input_s: list[Structure], target_s: list[Structure]):
        return self.update(input_s, target_s)

    def update(self, input_s: list[Structure], target_s: list[Structure]):

        assert len(input_s) == len(target_s)

        for si, st in zip(input_s, target_s):
            v, m = 0, 0
            if si is not None:
                v = validity_structure(si)
                if v:
                    rms = self.matcher.get_rms_dist(si, st)
                    m = int(rms is not None)

                    if rms is not None:
                        self.rmse.append(rms[0])  # NOTE: only for valid and matching

            self.match.append(m)
            self.valid.append(v)

    def summarize(self) -> dict:
        summary = {}

        summary["valid"] = safe_divide(sum(self.valid), len(self.valid))
        summary["match_rate"] = safe_divide(sum(self.match), len(self.match))
        summary["rmse"] = safe_divide(sum(self.rmse), len(self.rmse))

        return summary

    def reset(self):
        self.valid = []
        self.match = []
        self.rmse = []

    @property
    def details(self) -> dict:
        return {"valid": self.valid, "match": self.match, "rmse": self.rmse}
