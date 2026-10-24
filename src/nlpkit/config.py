"""Загрузка config.yaml, пути проекта и фиксация seed."""
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # пути делаем абсолютными
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def results_dir(cfg: dict, step: str) -> Path:
    """results/pr1, results/pr2, ... (создаётся автоматически)."""
    d = cfg["paths"]["results"] / step
    d.mkdir(parents=True, exist_ok=True)
    return d