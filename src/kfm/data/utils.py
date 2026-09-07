# Originally from: https://github.com/frcnt/kldm/blob/main/src_kldm/data/dataset.py

import json
from collections.abc import Sequence
from pathlib import Path

import ase.io
import numpy as np


def read_json(json_path: str | Path):
    with open(json_path, encoding="utf-8") as fp:
        return json.load(fp)


def save_json_old(json_dict: dict, json_path: str | Path, sort_keys: bool = False):
    def _fix_dict():
        for key in json_dict.items():
            if isinstance(json_dict[key], np.ndarray):
                json_dict[key] = json_dict[key].tolist()

    _fix_dict()
    with Path.open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp, sort_keys=sort_keys)


def save_json(json_dict: dict, json_path: str | Path, sort_keys: bool = False):
    def _fix_dict(d: dict):
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, dict):
                _fix_dict(v)

    _fix_dict(json_dict)
    with Path(json_path).open(encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp, sort_keys=sort_keys)


def save_images(images: Sequence[ase.Atoms], filename: str | Path, fmt: str = "extxyz"):
    ase.io.write(filename=filename, images=images, format=fmt)
