from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import FinalCNN, MNIST_MEAN, MNIST_STD


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "final_cnn.pt"
SUBMISSION_PATH = PROJECT_DIR / "outputs" / "submissions" / "submission.csv"
BATCH_SIZE = 256


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    x_test = test_df.to_numpy(dtype=np.float32) / 255.0
    x_test = (x_test - MNIST_MEAN) / MNIST_STD
    test_tensor = torch.from_numpy(x_test).float().view(-1, 1, 28, 28)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (features,) in test_loader:
            logits = model(features.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(predictions) + 1),
            "Label": predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
