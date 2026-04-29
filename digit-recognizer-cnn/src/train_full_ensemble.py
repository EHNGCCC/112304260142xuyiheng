from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import v2

from model import FinalCNN, MNIST_MEAN, MNIST_STD


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FULL_SUBMISSION_PATH = SUBMISSION_DIR / "submission_full_ensemble.csv"
DEFAULT_SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"
TRAIN_SUMMARY_PATH = TABLE_DIR / "full_ensemble_train_results.csv"

BATCH_SIZE = 256
EPOCHS = 36
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEEDS = [42, 2024, 3407]


@dataclass
class FullTrainResult:
    seed: int
    model_path: str
    epochs: int
    final_train_loss: float
    final_train_acc: float


class DigitDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        transform: v2.Compose | None = None,
    ) -> None:
        self.features = torch.from_numpy(features).float().view(-1, 1, 28, 28)
        self.labels = None if labels is None else torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        image = self.features[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_transform():
    return v2.Compose(
        [
            v2.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
            ),
            v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )


def load_train_data():
    train_df = pd.read_csv(TRAIN_PATH)
    x = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y = train_df["label"].to_numpy(dtype=np.int64)
    return x, y


def load_test_data():
    return pd.read_csv(TEST_PATH).to_numpy(dtype=np.float32) / 255.0


def make_train_loader(x_train: np.ndarray, y_train: np.ndarray, device: torch.device):
    return DataLoader(
        DigitDataset(x_train, y_train, transform=train_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def train_one_full_model(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
) -> FullTrainResult:
    seed_everything(seed)
    model_path = MODEL_DIR / f"full_ensemble_seed{seed}.pt"
    print(f"Training full-data model seed {seed} -> {model_path}", flush=True)

    loader = make_train_loader(x_train, y_train, device)
    model = FinalCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-5,
    )

    final_loss = 0.0
    final_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in loader:
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

        scheduler.step()
        final_loss = running_loss / total
        final_acc = correct / total
        print(
            f"seed {seed} epoch {epoch:02d}/{EPOCHS} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | "
            f"train_loss={final_loss:.4f} train_acc={final_acc:.4%}",
            flush=True,
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved full-data seed {seed}: {model_path}", flush=True)
    return FullTrainResult(seed, str(model_path), EPOCHS, final_loss, final_acc)


def load_model(path: Path, device: torch.device) -> FinalCNN:
    model = FinalCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def normalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch - MNIST_MEAN) / MNIST_STD


def predict_ensemble(models: list[FinalCNN], test_x: np.ndarray, device: torch.device):
    x_tensor = torch.from_numpy(test_x).float().view(-1, 1, 28, 28)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=BATCH_SIZE, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (features,) in loader:
            features = normalize(features.to(device))
            probs = torch.zeros((features.size(0), 10), device=device)
            for model in models:
                probs += torch.softmax(model(features), dim=1)
            probs /= len(models)
            predictions.extend(probs.argmax(dim=1).cpu().numpy().tolist())
    return predictions


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    x_train, y_train = load_train_data()

    results = [
        train_one_full_model(seed, x_train, y_train, device)
        for seed in SEEDS
    ]
    pd.DataFrame([result.__dict__ for result in results]).to_csv(
        TRAIN_SUMMARY_PATH,
        index=False,
    )
    print(f"Saved train summary to: {TRAIN_SUMMARY_PATH}", flush=True)

    model_paths = [MODEL_DIR / f"full_ensemble_seed{seed}.pt" for seed in SEEDS]
    models = [load_model(path, device) for path in model_paths]
    test_x = load_test_data()
    predictions = predict_ensemble(models, test_x, device)

    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1),
            "Label": predictions,
        }
    )
    submission.to_csv(FULL_SUBMISSION_PATH, index=False)
    shutil.copyfile(FULL_SUBMISSION_PATH, DEFAULT_SUBMISSION_PATH)
    print(f"Saved full ensemble submission to: {FULL_SUBMISSION_PATH}", flush=True)
    print(f"Also updated default submission: {DEFAULT_SUBMISSION_PATH}", flush=True)


if __name__ == "__main__":
    main()
