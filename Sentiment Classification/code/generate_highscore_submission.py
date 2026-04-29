import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = ROOT / "artifacts" / "cache" / "preprocessed"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"


def soft_clean(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NbSvm:
    def __init__(self, c: float = 4.0, alpha: float = 1.0):
        self.c = c
        self.alpha = alpha
        self.model = LogisticRegression(max_iter=3000, C=c, solver="liblinear")
        self.r = None

    def fit(self, x, y):
        positive = x[y == 1].sum(0) + self.alpha
        negative = x[y == 0].sum(0) + self.alpha
        self.r = np.asarray(np.log(positive / positive.sum()) - np.log(negative / negative.sum())).ravel()
        self.model.fit(x.multiply(self.r), y)
        return self

    def decision_function(self, x):
        return self.model.decision_function(x.multiply(self.r))


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


def main() -> None:
    labeled = pd.read_pickle(PREPROCESSED_DIR / "labeled_preprocess_v2.pkl")
    test = pd.read_pickle(PREPROCESSED_DIR / "test_preprocess_v2.pkl")

    train_df, val_df = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=labeled["sentiment"],
    )

    y_train = train_df["sentiment"].astype(int).to_numpy()
    y_val = val_df["sentiment"].astype(int).to_numpy()
    y_full = labeled["sentiment"].astype(int).to_numpy()

    train_clean = train_df["clean_review"].tolist()
    val_clean = val_df["clean_review"].tolist()
    full_clean = labeled["clean_review"].tolist()
    test_clean = test["clean_review"].tolist()

    train_soft = train_df["review"].map(soft_clean).tolist()
    val_soft = val_df["review"].map(soft_clean).tolist()
    full_soft = labeled["review"].map(soft_clean).tolist()
    test_soft = test["review"].map(soft_clean).tolist()

    model_specs = [
        {
            "name": "word_clean_linsvc",
            "weight": 0.25,
            "vectorizer": TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=200000,
            ),
            "builder": lambda: LinearSVC(C=0.6),
            "train_text": train_clean,
            "val_text": val_clean,
            "full_text": full_clean,
            "test_text": test_clean,
        },
        {
            "name": "word_soft_linsvc",
            "weight": 0.23,
            "vectorizer": TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=250000,
            ),
            "builder": lambda: LinearSVC(C=0.5),
            "train_text": train_soft,
            "val_text": val_soft,
            "full_text": full_soft,
            "test_text": test_soft,
        },
        {
            "name": "nbsvm_word_soft",
            "weight": 0.31,
            "vectorizer": CountVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                strip_accents="unicode",
                max_features=250000,
                binary=True,
            ),
            "builder": lambda: NbSvm(c=4.0, alpha=1.0),
            "train_text": train_soft,
            "val_text": val_soft,
            "full_text": full_soft,
            "test_text": test_soft,
        },
        {
            "name": "nbsvm_char_soft",
            "weight": 0.21,
            "vectorizer": CountVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.99,
                strip_accents="unicode",
                max_features=250000,
                binary=True,
            ),
            "builder": lambda: NbSvm(c=4.0, alpha=1.0),
            "train_text": train_soft,
            "val_text": val_soft,
            "full_text": full_soft,
            "test_text": test_soft,
        },
    ]

    validation_scores = {}
    validation_blend = np.zeros(len(val_df), dtype=np.float64)

    for spec in model_specs:
        vectorizer = spec["vectorizer"]
        x_train = vectorizer.fit_transform(spec["train_text"])
        x_val = vectorizer.transform(spec["val_text"])
        model = spec["builder"]()
        model.fit(x_train, y_train)
        val_score = model.decision_function(x_val)
        val_rank = rank_normalize(val_score)
        validation_scores[spec["name"]] = float(roc_auc_score(y_val, val_rank))
        validation_blend += spec["weight"] * val_rank

    validation_auc = float(roc_auc_score(y_val, validation_blend))

    test_blend = np.zeros(len(test), dtype=np.float64)
    for spec in model_specs:
        vectorizer = spec["vectorizer"]
        x_full = vectorizer.fit_transform(spec["full_text"])
        x_test = vectorizer.transform(spec["test_text"])
        model = spec["builder"]()
        model.fit(x_full, y_full)
        test_score = model.decision_function(x_test)
        test_rank = rank_normalize(test_score)
        test_blend += spec["weight"] * test_rank

    submission = pd.DataFrame({"id": test["id"], "sentiment": test_blend})
    submission_path = SUBMISSIONS_DIR / "submission_sparse_rank_blend_auc.csv"
    submission.to_csv(submission_path, index=False)

    report = {
        "metric_priority": "roc_auc",
        "model_family": "sparse_text_rank_blend",
        "validation_auc": validation_auc,
        "component_validation_auc": validation_scores,
        "weights": {spec["name"]: spec["weight"] for spec in model_specs},
        "submission_path": str(submission_path),
    }
    report_path = REPORTS_DIR / "sparse_rank_blend_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation ROC-AUC: {validation_auc:.8f}")
    print(f"Saved submission to: {submission_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
