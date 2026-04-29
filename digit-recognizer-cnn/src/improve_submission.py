from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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

BASELINE_MODEL_PATH = MODEL_DIR / "final_cnn.pt"
BEST_SUBMISSION_PATH = SUBMISSION_DIR / "submission_best_local.csv"
DEFAULT_SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"
RESULTS_PATH = TABLE_DIR / "improvement_results.csv"

SEEDS_TO_TRAIN = [2024, 3407]
ALL_MODEL_SPECS = [
    ("seed42_existing", BASELINE_MODEL_PATH),
    ("seed2024", MODEL_DIR / "ensemble_seed2024.pt"),
    ("seed3407", MODEL_DIR / "ensemble_seed3407.pt"),
]

BATCH_SIZE = 256
MAX_EPOCHS = 45
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SPLIT_SEED = 42
TTA_SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class TrainResult:
    name: str
    model_path: Path
    best_epoch: int
    best_val_acc: float
    best_val_loss: float


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


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    x = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y = train_df["label"].to_numpy(dtype=np.int64)
    split = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=RANDOM_SPLIT_SEED,
        stratify=y,
    )
    test_x = pd.read_csv(TEST_PATH).to_numpy(dtype=np.float32) / 255.0
    return split, test_x


def build_transforms():
    train_transform = v2.Compose(
        [
            v2.RandomAffine(
                degrees=12,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
            ),
            v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )
    eval_transform = v2.Compose([v2.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD])])
    return train_transform, eval_transform


def make_train_loaders(x_train, y_train, x_val, y_val, device: torch.device):
    train_transform, eval_transform = build_transforms()
    train_loader = DataLoader(
        DigitDataset(x_train, y_train, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DigitDataset(x_val, y_val, transform=eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def evaluate_loss_acc(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
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


def train_seed_model(
    seed: int,
    model_path: Path,
    split_data,
    device: torch.device,
) -> TrainResult:
    if model_path.exists():
        print(f"Skip training seed {seed}; found existing model: {model_path}", flush=True)
        return TrainResult(f"seed{seed}", model_path, -1, float("nan"), float("nan"))

    print(f"Training seed {seed} -> {model_path}", flush=True)
    seed_everything(seed)
    x_train, x_val, y_train, y_val = split_data
    train_loader, val_loader = make_train_loaders(x_train, y_train, x_val, y_val, device)
    model = FinalCNN().to(device)
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
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    stale_epochs = 0

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
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(
            f"seed {seed} epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4%}",
            flush=True,
        )

        improved = val_acc > best_val_acc or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch >= 18 and stale_epochs >= PATIENCE:
            print(f"seed {seed} early stopped at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"Seed {seed} did not produce a model.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path)
    print(
        f"Saved seed {seed}: best_epoch={best_epoch}, "
        f"best_val_acc={best_val_acc:.4%}, best_val_loss={best_val_loss:.4f}",
        flush=True,
    )
    return TrainResult(f"seed{seed}", model_path, best_epoch, best_val_acc, best_val_loss)


def load_model(model_path: Path, device: torch.device) -> FinalCNN:
    model = FinalCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def raw_loader(features: np.ndarray, labels: np.ndarray | None = None):
    x_tensor = torch.from_numpy(features).float().view(-1, 1, 28, 28)
    if labels is None:
        dataset = TensorDataset(x_tensor)
    else:
        dataset = TensorDataset(x_tensor, torch.from_numpy(labels).long())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def normalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch - MNIST_MEAN) / MNIST_STD


def shift_zero(batch: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return batch
    shifted = torch.zeros_like(batch)
    src_y1 = max(0, -dy)
    src_y2 = min(batch.size(2), batch.size(2) - dy)
    dst_y1 = max(0, dy)
    dst_y2 = min(batch.size(2), batch.size(2) + dy)
    src_x1 = max(0, -dx)
    src_x2 = min(batch.size(3), batch.size(3) - dx)
    dst_x1 = max(0, dx)
    dst_x2 = min(batch.size(3), batch.size(3) + dx)
    shifted[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = batch[
        :, :, src_y1:src_y2, src_x1:src_x2
    ]
    return shifted


def predict_probabilities(
    models: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    all_probs = []
    shifts = TTA_SHIFTS if use_tta else [(0, 0)]
    with torch.no_grad():
        for batch in loader:
            raw_features = batch[0].to(device)
            probs = torch.zeros((raw_features.size(0), 10), device=device)
            for model in models:
                model.eval()
                for dy, dx in shifts:
                    features = normalize(shift_zero(raw_features, dy, dx))
                    logits = model(features)
                    probs += torch.softmax(logits, dim=1)
            probs /= len(models) * len(shifts)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate_candidate(
    name: str,
    models: list[nn.Module],
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    use_tta: bool,
):
    probs = predict_probabilities(models, raw_loader(x_val, y_val), device, use_tta)
    predictions = probs.argmax(axis=1)
    acc = float((predictions == y_val).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, 1.0)).mean())
    return {
        "candidate": name,
        "model_count": len(models),
        "tta": "Yes" if use_tta else "No",
        "val_acc": acc,
        "val_nll": nll,
    }


def save_submission(
    candidate_name: str,
    models: list[nn.Module],
    test_x: np.ndarray,
    device: torch.device,
    use_tta: bool,
) -> None:
    probs = predict_probabilities(models, raw_loader(test_x), device, use_tta)
    predictions = probs.argmax(axis=1)
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1),
            "Label": predictions,
        }
    )
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(BEST_SUBMISSION_PATH, index=False)
    shutil.copyfile(BEST_SUBMISSION_PATH, DEFAULT_SUBMISSION_PATH)
    print(f"Best local candidate: {candidate_name}", flush=True)
    print(f"Saved best submission to: {BEST_SUBMISSION_PATH}", flush=True)
    print(f"Also updated: {DEFAULT_SUBMISSION_PATH}", flush=True)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    split_data, test_x = load_data()
    x_train, x_val, y_train, y_val = split_data

    train_results = []
    seed_to_path = {2024: MODEL_DIR / "ensemble_seed2024.pt", 3407: MODEL_DIR / "ensemble_seed3407.pt"}
    for seed in SEEDS_TO_TRAIN:
        train_results.append(train_seed_model(seed, seed_to_path[seed], split_data, device))

    models_by_name = {
        name: load_model(path, device)
        for name, path in ALL_MODEL_SPECS
        if path.exists()
    }
    single = [models_by_name["seed42_existing"]]
    ensemble = [models_by_name[name] for name, _ in ALL_MODEL_SPECS if name in models_by_name]

    candidates = [
        ("single_existing", single, False),
        ("single_existing_tta", single, True),
        ("ensemble_3seed", ensemble, False),
        ("ensemble_3seed_tta", ensemble, True),
    ]

    results = [
        evaluate_candidate(name, models, x_val, y_val, device, use_tta)
        for name, models, use_tta in candidates
    ]
    results_df = pd.DataFrame(results).sort_values(
        by=["val_acc", "val_nll"],
        ascending=[False, True],
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print("\nValidation comparison:", flush=True)
    print(results_df.to_string(index=False), flush=True)
    print(f"Saved comparison to: {RESULTS_PATH}", flush=True)

    best = results_df.iloc[0]
    best_name = str(best["candidate"])
    best_models = dict((name, models) for name, models, _ in candidates)[best_name]
    best_tta = bool(best["tta"] == "Yes")
    save_submission(best_name, best_models, test_x, device, best_tta)

    if train_results:
        train_df = pd.DataFrame([result.__dict__ for result in train_results])
        train_path = TABLE_DIR / "improvement_train_results.csv"
        train_df.to_csv(train_path, index=False)
        print(f"Saved training summary to: {train_path}", flush=True)


if __name__ == "__main__":
    main()
