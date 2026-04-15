import json
import pickle
import random
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_pickle(path: Path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def plot_auc_comparison(metrics_by_model: dict, output_path: Path, title: str) -> None:
    names = list(metrics_by_model.keys())
    auc_values = [metrics_by_model[name]["auc"] for name in names]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, auc_values, color=["#2b6cb0", "#2f855a", "#c05621"][: len(names)])
    plt.ylim(0.5, 1.0)
    plt.ylabel("Validation ROC-AUC")
    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, auc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
