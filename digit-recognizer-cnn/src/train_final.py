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


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
HISTORY_DIR = OUTPUT_DIR / "history"
PLOT_DIR = OUTPUT_DIR / "plots"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "final_cnn.pt"
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"
HISTORY_PATH = HISTORY_DIR / "final_training_history.csv"
LOSS_PLOT_PATH = PLOT_DIR / "final_loss_curve_epoch0_100.png"

RANDOM_SEED = 42
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_transforms():
    train_transform = v2.Compose(
        [
            v2.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )
    return train_transform, eval_transform


def build_loaders():
    train_df = pd.read_csv(TRAIN_PATH)
    x = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y = train_df["label"].to_numpy(dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_transform, eval_transform = build_transforms()
    train_dataset = DigitDataset(x_train, y_train, transform=train_transform)
    train_eval_dataset = DigitDataset(x_train, y_train, transform=eval_transform)
    val_dataset = DigitDataset(x_val, y_val, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, train_eval_loader, val_loader


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

            total_loss += loss.item() * features.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_loss_curves(history: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss (Epoch 0-100)")
    plt.xticks(range(0, EPOCHS + 1, 10))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=180)
    plt.close()
    print(f"Saved loss plot to: {LOSS_PLOT_PATH}")


def train_model():
    seed_everything(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, train_eval_loader, val_loader = build_loaders()
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_val_acc = 0.0
    history_rows = []

    initial_train_loss, initial_train_acc = evaluate(model, train_eval_loader, criterion, device)
    initial_val_loss, initial_val_acc = evaluate(model, val_loader, criterion, device)
    history_rows.append(
        {
            "epoch": 0,
            "train_loss": initial_train_loss,
            "train_acc": initial_train_acc,
            "val_loss": initial_val_loss,
            "val_acc": initial_val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )
    print(
        f"Epoch 00/{EPOCHS} | "
        f"lr={optimizer.param_groups[0]['lr']:.6f} | "
        f"train_loss={initial_train_loss:.4f} | "
        f"train_acc={initial_train_acc:.4%} | "
        f"val_loss={initial_val_loss:.4f} | "
        f"val_acc={initial_val_acc:.4%}"
    )

    for epoch in range(1, EPOCHS + 1):
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

            running_loss += loss.item() * features.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    torch.save(best_state, MODEL_PATH)
    print(f"Best validation accuracy: {best_val_acc:.4%}")
    print(f"Saved model to: {MODEL_PATH}")

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    history.to_csv(HISTORY_PATH, index=False)
    print(f"Saved history to: {HISTORY_PATH}")
    plot_loss_curves(history)
    return model, device


def create_submission(model: nn.Module, device: torch.device) -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    _, eval_transform = build_transforms()
    test_df = pd.read_csv(TEST_PATH)
    x_test = test_df.to_numpy(dtype=np.float32) / 255.0
    test_dataset = DigitDataset(x_test, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        for features in test_loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1),
            "Label": predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


def main() -> None:
    model, device = train_model()
    create_submission(model, device)


if __name__ == "__main__":
    main()
