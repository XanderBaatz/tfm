# Originally from: https://github.com/frcnt/kldm/blob/main/src_kldm/data/dataset.py

"""Preprocess mp_20 CSV splits into PyG .pt files for flow matching.

Usage:
    uv run python -m kfm_new.data.preprocess_csv --csv_folder data/mp_20
    uv run python -m kfm_new.data.preprocess_csv --csv_folder data/mp_20 --max_atoms 20
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fire
import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from tqdm.auto import tqdm

from kfm.data.utils import save_json

if TYPE_CHECKING:
    from collections.abc import Iterable


def lattice_params_to_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert lattice abc/angles to matrix (matches pymatgen convention)."""

    def abs_cap(val: float, max_abs_val: float = 1.0) -> float:
        return max(min(val, max_abs_val), -max_abs_val)

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, _ = np.sin(angles_r)

    val = abs_cap((cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta))
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def process_cif(crystal_str: str) -> dict[str, np.ndarray]:
    crystal = Structure.from_str(crystal_str, fmt="cif")
    crystal = crystal.get_reduced_structure()

    canonical = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )

    params = canonical.lattice.parameters
    lengths = np.array(params[:3])
    angles = np.array(params[3:])

    assert np.allclose(canonical.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))

    return {
        "pos": canonical.frac_coords,
        "h": np.array(canonical.atomic_numbers),
        "lengths": lengths,
        "angles": angles,
    }


def compute_len_distribution(num_atoms_list: list[int]) -> np.ndarray:
    """Empirical distribution over number of atoms (index = count)."""
    max_n = max(num_atoms_list)
    counts = np.zeros(max_n + 1)
    for n in num_atoms_list:
        counts[n] += 1
    counts /= counts.sum()
    return counts


def compute_angle_loc_scale(angles_list: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """Per-component (alpha, beta, gamma) loc/scale for tan(angle - pi/2),
    fit the same way lengths_loc_scale is -- trimmed mean/std over the
    inner 95%, per component. Unlike lengths, this is *not* binned by
    num_atoms: cell volume scales directly with atom count, but angle is
    governed by crystal symmetry class, which isn't tied to n the same way.
    """
    angles_rad = np.radians(np.array(angles_list))  # [N, 3]
    tan_angles = np.tan(angles_rad - np.pi / 2.0)  # [N, 3]
    tan_sorted = np.sort(tan_angles, axis=0)  # per-component order stats
    idx = int(len(tan_sorted) * 0.025)
    trimmed = tan_sorted[idx : max(idx + 1, len(tan_sorted) - idx)]
    loc = np.mean(trimmed, axis=0)
    scale = np.std(trimmed, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return loc.tolist(), scale.tolist()


def compute_angle_loc_scale_per_n(
    angles_lst: list[np.ndarray],
) -> tuple[list[float], list[float]]:
    """Per-component (alpha, beta, gamma) loc/scale for tan(angle - pi/2) for a given atom count bucket."""
    angles_rad = np.radians(np.array(angles_lst))
    tan_angles = np.tan(angles_rad - np.pi / 2.0)
    tan_sorted = np.sort(tan_angles, axis=0)

    idx = int(len(tan_sorted) * 0.025)
    trimmed = tan_sorted[idx : max(idx + 1, len(tan_sorted) - idx)]

    loc = np.mean(trimmed, axis=0)
    scale = np.std(trimmed, axis=0)
    scale = np.where(scale == 0, 1.0, scale)

    return loc.tolist(), scale.tolist()


def preprocess_csv(
    csv_folder: str | Path = "data/mp_20",
    splits: Iterable[str] = ("train", "val", "test"),
    fmt: Literal["pyg", "numpy"] = "pyg",
    max_atoms: int = -1,
) -> None:
    csv_folder = Path(csv_folder)

    for split in splits:
        csv_path = csv_folder / f"{split}.csv"
        if not csv_path.exists():
            print(f"Did not find {csv_path}, skipping...")
            continue

        df = pd.read_csv(csv_path)
        data_list: list = []
        loc_scale_dct: dict[int, list[np.ndarray]] = {}
        angles_dct: dict[int, list[np.ndarray]] = {}
        num_atoms_list: list[int] = []

        for i in tqdm(range(len(df)), desc=f"Preprocessing {split}"):
            cif = df.iloc[i]["cif"]
            try:
                data = process_cif(cif)
            except Exception as e:  # noqa: BLE001
                print(f"Skipping row {i}: {e}")
                continue

            n = data["pos"].shape[0]
            if 0 < max_atoms < n:
                continue

            num_atoms_list.append(n)

            if n not in loc_scale_dct:
                loc_scale_dct[n] = []
                angles_dct[n] = []
            loc_scale_dct[n].append(data["lengths"])
            angles_dct[n].append(data["angles"])

            if fmt == "pyg":
                item = Data(
                    pos=torch.tensor(data["pos"], dtype=torch.float32),
                    h=torch.tensor(data["h"], dtype=torch.long),
                    lengths=torch.tensor(data["lengths"], dtype=torch.float32).view(1, -1),
                    angles=torch.tensor(data["angles"], dtype=torch.float32).view(1, -1),
                )
            else:
                item = data
            data_list.append(item)

        # Save processed split
        pt_path = csv_folder / f"{split}.pt"
        torch.save(data_list, pt_path)
        print(f"Saved {len(data_list)} structures → {pt_path}")

        # Per-num-atoms length statistics (log-space, inner 95%)
        loc_scale: dict[int, tuple[list, list]] = {}
        for n, lengths_lst in loc_scale_dct.items():
            log_abc = np.log(np.sort(np.array(lengths_lst), axis=0))
            idx = int(len(log_abc) * 0.025)
            trimmed = log_abc[idx : max(idx + 1, len(log_abc) - idx)]
            loc = np.mean(trimmed, axis=0)
            scale = np.std(trimmed, axis=0)
            scale = np.where(scale == 0, 1.0, scale)  # avoid zero std
            loc_scale[n] = (loc.tolist(), scale.tolist())

        loc_scale_path = csv_folder / f"{split}_loc_scale.json"
        save_json(loc_scale, str(loc_scale_path), sort_keys=True)
        print(f"Saved length loc/scale → {loc_scale_path}")

        # Per-num-atoms angle statistics
        angle_loc_scale: dict[int, tuple[list, list]] = {}
        for n, angles_lst in angles_dct.items():
            angle_loc_scale[n] = compute_angle_loc_scale_per_n(angles_lst)

        angle_loc_scale_path = csv_folder / f"{split}_angle_loc_scale.json"
        save_json(angle_loc_scale, str(angle_loc_scale_path), sort_keys=True)
        print(f"Saved angle loc/scale → {angle_loc_scale_path}")

        # Empirical num-atoms distribution
        if num_atoms_list:
            dist = compute_len_distribution(num_atoms_list)
            dist_path = csv_folder / f"{split}_len_distribution.npy"
            np.save(dist_path, dist)
            print(f"Saved num-atoms distribution → {dist_path}")


def main() -> None:
    fire.Fire(preprocess_csv)


if __name__ == "__main__":
    main()
