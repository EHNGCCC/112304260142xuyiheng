from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .utils import save_json


class ReviewDataset(Dataset):
    def __init__(self, token_lists: list[list[str]], labels: list[int] | None, vocab: dict[str, int], max_len: int):
        self.token_lists = token_lists
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.token_lists)

    def __getitem__(self, index: int):
        tokens = self.token_lists[index][: self.max_len]
        ids = [self.vocab.get(token, 1) for token in tokens]
        length = max(1, len(ids))
        ids = ids + [0] * (self.max_len - len(ids))
        features = torch.tensor(ids, dtype=torch.long)
        length_tensor = torch.tensor(length, dtype=torch.long)

        if self.labels is None:
            return features, length_tensor

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return features, length_tensor, label


class Word2VecBiLSTM(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_size: int, dropout: float):
        super().__init__()
        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.size(1))
        mask = (input_ids != 0).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.classifier(self.dropout(pooled)).squeeze(1)
        return logits


def _build_vocab_and_embeddings(word2vec_model, dim: int) -> tuple[dict[str, int], np.ndarray]:
    vocab = {"<pad>": 0, "<unk>": 1}
    embedding_matrix = np.random.normal(0, 0.05, (len(word2vec_model.wv.index_to_key) + 2, dim)).astype(np.float32)
    embedding_matrix[0] = np.zeros(dim, dtype=np.float32)

    for index, token in enumerate(word2vec_model.wv.index_to_key, start=2):
        vocab[token] = index
        embedding_matrix[index] = word2vec_model.wv[token]

    return vocab, embedding_matrix


def _predict_probabilities(model: nn.Module, data_loader: DataLoader, device: str, has_labels: bool) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if has_labels:
                input_ids, lengths, labels = batch
                all_labels.extend(labels.numpy().tolist())
            else:
                input_ids, lengths = batch

            logits = model(input_ids.to(device), lengths.to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.extend(probabilities.tolist())

    labels_array = np.array(all_labels) if has_labels else None
    return np.array(all_probabilities), labels_array


def run_bilstm_pipeline(labeled_df: pd.DataFrame, test_df: pd.DataFrame, word2vec_model, config) -> dict:
    train_df, val_df = train_test_split(
        labeled_df,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labeled_df["sentiment"],
    )

    vocab, embedding_matrix = _build_vocab_and_embeddings(word2vec_model, config.word2vec_dim)
    train_dataset = ReviewDataset(
        train_df["tokens"].tolist(),
        train_df["sentiment"].astype(int).tolist(),
        vocab,
        config.max_len,
    )
    val_dataset = ReviewDataset(
        val_df["tokens"].tolist(),
        val_df["sentiment"].astype(int).tolist(),
        vocab,
        config.max_len,
    )
    test_dataset = ReviewDataset(test_df["tokens"].tolist(), None, vocab, config.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = Word2VecBiLSTM(embedding_matrix, config.bilstm_hidden_size, config.bilstm_dropout).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_losses = []
    val_aucs = []

    for epoch in range(config.bilstm_epochs):
        model.train()
        running_loss = 0.0

        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(config.device)
            lengths = lengths.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

        average_train_loss = running_loss / len(train_dataset)
        train_losses.append(float(average_train_loss))

        val_prob, val_labels = _predict_probabilities(model, val_loader, config.device, has_labels=True)
        val_auc = float(roc_auc_score(val_labels, val_prob))
        val_aucs.append(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_prob, val_labels = _predict_probabilities(model, val_loader, config.device, has_labels=True)
    val_pred = (val_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(val_labels, val_prob)),
        "accuracy": float(accuracy_score(val_labels, val_pred)),
        "f1": float(f1_score(val_labels, val_pred)),
    }

    test_prob, _ = _predict_probabilities(model, test_loader, config.device, has_labels=False)
    submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_prob})
    submission_path = config.submissions_dir / "submission_bilstm_auc.csv"
    submission.to_csv(submission_path, index=False)

    save_json(
        config.reports_dir / "bilstm_validation_metrics.json",
        {"metric_priority": "roc_auc", "results": metrics},
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(train_losses, marker="o", label="Train loss")
    plt.plot(val_aucs, marker="s", label="Validation ROC-AUC")
    plt.title("BiLSTM Training Summary")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.plots_dir / "bilstm_training_curve.png", dpi=160)
    plt.close()

    return {
        "validation_metrics": metrics,
        "validation_labels": val_labels,
        "validation_probabilities": val_prob,
        "test_probabilities": test_prob,
        "submission_path": str(submission_path),
    }
