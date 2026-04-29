from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from model import BasicCNN, MNIST_MEAN, MNIST_STD


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
HISTORY_DIR = OUTPUT_DIR / "history"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

TRAIN_PATH = DATA_DIR / "train.csv"

RANDOM_SEED = 42
MAX_EPOCHS = 20
PATIENCE = 4


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    optimizer_name: str
    lr: float
    batch_size: int
    use_augmentation: bool
    early_stopping: bool


EXPERIMENTS = [
    ExperimentConfig("Exp1", "SGD", 0.01, 64, False, False),
    ExperimentConfig("Exp2", "Adam", 0.001, 64, False, False),
    ExperimentConfig("Exp3", "Adam", 0.001, 128, False, True),
    ExperimentConfig("Exp4", "Adam", 0.001, 64, True, True),
]


class DigitDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: v2.Compose | None = None,
    ) -> None:
        self.features = torch.from_numpy(features).float().view(-1, 1, 28, 28)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        image = self.features[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(use_augmentation: bool):
    transforms = []
    if use_augmentation:
        transforms.append(v2.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    transforms.append(v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]))
    train_transform = v2.Compose(transforms)
    eval_transform = v2.Compose([v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD])])
    return train_transform, eval_transform


def load_split():
    train_df = pd.read_csv(TRAIN_PATH)
    x = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y = train_df["label"].to_numpy(dtype=np.int64)
    return train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=y,
    )


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    transform: v2.Compose,
    shuffle: bool,
    device: torch.device,
):
    dataset = DigitDataset(features, labels, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_optimizer(config: ExperimentConfig, model: nn.Module):
    if config.optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    if config.optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def run_one_experiment(config: ExperimentConfig, split_data, device: torch.device):
    x_train, x_val, y_train, y_val = split_data
    train_transform, eval_transform = build_transforms(config.use_augmentation)
    train_loader = make_loader(
        x_train, y_train, config.batch_size, train_transform, True, device
    )
    train_eval_loader = make_loader(
        x_train, y_train, config.batch_size, eval_transform, False, device
    )
    val_loader = make_loader(x_val, y_val, config.batch_size, eval_transform, False, device)

    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config, model)

    history = []
    best = {
        "val_loss": float("inf"),
        "val_acc": 0.0,
        "train_acc": 0.0,
        "epoch": 0,
    }
    stale_epochs = 0

    train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history.append(
        {
            "experiment": config.name,
            "epoch": 0,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "experiment": config.name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"{config.name} epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4%}"
        )

        if val_loss < best["val_loss"] - 1e-4:
            best = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "epoch": epoch,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.early_stopping and stale_epochs >= PATIENCE:
            print(f"{config.name} early stopped at epoch {epoch}.")
            break

    result = {
        "experiment": config.name,
        "optimizer": config.optimizer_name,
        "learning_rate": config.lr,
        "batch_size": config.batch_size,
        "augmentation": "Yes" if config.use_augmentation else "No",
        "early_stopping": "Yes" if config.early_stopping else "No",
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": "N/A",
        "min_loss": best["val_loss"],
        "converged_epoch": best["epoch"],
    }
    return pd.DataFrame(history), result


def plot_comparison(history: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    for exp_name in history["experiment"].unique():
        exp_history = history[history["experiment"] == exp_name]
        plt.plot(
            exp_history["epoch"],
            exp_history["val_loss"],
            linewidth=2,
            label=f"{exp_name} Val Loss",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Comparison")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / "comparison_loss_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved loss comparison plot to: {path}")


def main() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_data = load_split()
    histories = []
    results = []

    for config in EXPERIMENTS:
        print(f"\nRunning {config.name}: {config}")
        history, result = run_one_experiment(config, split_data, device)
        histories.append(history)
        results.append(result)

    history_df = pd.concat(histories, ignore_index=True)
    results_df = pd.DataFrame(results)
    history_path = HISTORY_DIR / "comparison_history.csv"
    results_path = TABLE_DIR / "comparison_results.csv"
    history_df.to_csv(history_path, index=False)
    results_df.to_csv(results_path, index=False)
    plot_comparison(history_df)

    print(f"Saved history to: {history_path}")
    print(f"Saved results to: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
