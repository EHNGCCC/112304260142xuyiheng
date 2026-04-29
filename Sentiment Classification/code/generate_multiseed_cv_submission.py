import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = ROOT / "artifacts" / "cache" / "preprocessed"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"

SEEDS = [42, 2024, 3407]
N_FOLDS = 3


def soft_clean(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NbSvm:
    def __init__(self, c: float = 4.0, alpha: float = 1.0):
        self.model = LogisticRegression(max_iter=3000, C=c, solver="liblinear")
        self.alpha = alpha
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


def build_model_specs(clean_texts: list[str], soft_texts: list[str], clean_test: list[str], soft_test: list[str]) -> dict:
    return {
        "word_clean": {
            "vectorizer": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=200000,
            ),
            "builder": lambda: LinearSVC(C=0.6),
            "train_texts": clean_texts,
            "test_texts": clean_test,
        },
        "word_soft": {
            "vectorizer": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=250000,
            ),
            "builder": lambda: LinearSVC(C=0.5),
            "train_texts": soft_texts,
            "test_texts": soft_test,
        },
        "nb_word": {
            "vectorizer": lambda: CountVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                strip_accents="unicode",
                max_features=250000,
                binary=True,
            ),
            "builder": lambda: NbSvm(c=4.0, alpha=1.0),
            "train_texts": soft_texts,
            "test_texts": soft_test,
        },
        "nb_char": {
            "vectorizer": lambda: CountVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.99,
                strip_accents="unicode",
                max_features=250000,
                binary=True,
            ),
            "builder": lambda: NbSvm(c=4.0, alpha=1.0),
            "train_texts": soft_texts,
            "test_texts": soft_test,
        },
    }


def search_best_weights(ranked_oof: dict[str, np.ndarray], y_true: np.ndarray) -> tuple[dict[str, float], float]:
    names = list(ranked_oof.keys())
    best_auc = -1.0
    best_weights = None

    for w1 in np.arange(0.15, 0.51, 0.05):
        for w2 in np.arange(0.05, 0.31, 0.05):
            for w3 in np.arange(0.20, 0.46, 0.05):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0.05 or w4 > 0.40:
                    continue
                weights = {
                    names[0]: float(w1),
                    names[1]: float(w2),
                    names[2]: float(w3),
                    names[3]: float(w4),
                }
                blend = sum(weights[name] * ranked_oof[name] for name in names)
                auc = float(roc_auc_score(y_true, blend))
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights

    if best_weights is None:
        raise RuntimeError("Weight search failed to find a valid blend.")

    fine_names = names
    start = best_weights
    for w1 in np.arange(max(0.0, start[fine_names[0]] - 0.06), min(1.0, start[fine_names[0]] + 0.061), 0.01):
        for w2 in np.arange(max(0.0, start[fine_names[1]] - 0.06), min(1.0, start[fine_names[1]] + 0.061), 0.01):
            for w3 in np.arange(max(0.0, start[fine_names[2]] - 0.06), min(1.0, start[fine_names[2]] + 0.061), 0.01):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0.02 or w4 > 0.50:
                    continue
                weights = {
                    fine_names[0]: round(float(w1), 4),
                    fine_names[1]: round(float(w2), 4),
                    fine_names[2]: round(float(w3), 4),
                    fine_names[3]: round(float(w4), 4),
                }
                blend = sum(weights[name] * ranked_oof[name] for name in fine_names)
                auc = float(roc_auc_score(y_true, blend))
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights

    return best_weights, best_auc


def main() -> None:
    labeled = pd.read_pickle(PREPROCESSED_DIR / "labeled_preprocess_v2.pkl")
    test = pd.read_pickle(PREPROCESSED_DIR / "test_preprocess_v2.pkl")
    y = labeled["sentiment"].astype(int).to_numpy()

    clean_texts = labeled["clean_review"].tolist()
    clean_test = test["clean_review"].tolist()
    soft_texts = labeled["review"].map(soft_clean).tolist()
    soft_test = test["review"].map(soft_clean).tolist()

    model_specs = build_model_specs(clean_texts, soft_texts, clean_test, soft_test)

    oof_by_model_seed = {}
    test_rank_by_model_seed = {}
    fold_auc_summary = {}

    for seed in SEEDS:
        print(f"Running seed {seed} with {N_FOLDS}-fold CV...")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for model_name, spec in model_specs.items():
            oof_scores = np.zeros(len(labeled), dtype=np.float64)
            test_fold_ranks = []
            fold_aucs = []

            for fold_index, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labeled)), y), start=1):
                vectorizer = spec["vectorizer"]()
                train_text = [spec["train_texts"][i] for i in train_idx]
                val_text = [spec["train_texts"][i] for i in val_idx]

                x_train = vectorizer.fit_transform(train_text)
                x_val = vectorizer.transform(val_text)
                x_test = vectorizer.transform(spec["test_texts"])

                model = spec["builder"]()
                model.fit(x_train, y[train_idx])
                val_score = model.decision_function(x_val)
                test_score = model.decision_function(x_test)

                oof_scores[val_idx] = val_score
                test_fold_ranks.append(rank_normalize(test_score))
                fold_auc = float(roc_auc_score(y[val_idx], val_score))
                fold_aucs.append(fold_auc)
                print(f"  {model_name} fold {fold_index}: AUC={fold_auc:.6f}")

            key = f"{model_name}_seed{seed}"
            oof_by_model_seed[key] = rank_normalize(oof_scores)
            test_rank_by_model_seed[key] = np.mean(test_fold_ranks, axis=0)
            fold_auc_summary[key] = fold_aucs

    model_names = list(model_specs.keys())
    aggregated_oof = {}
    aggregated_test = {}
    component_oof_auc = {}

    for model_name in model_names:
        model_seed_keys = [f"{model_name}_seed{seed}" for seed in SEEDS]
        aggregated_oof[model_name] = np.mean([oof_by_model_seed[key] for key in model_seed_keys], axis=0)
        aggregated_test[model_name] = np.mean([test_rank_by_model_seed[key] for key in model_seed_keys], axis=0)
        component_oof_auc[model_name] = float(roc_auc_score(y, aggregated_oof[model_name]))

    best_weights, blend_oof_auc = search_best_weights(aggregated_oof, y)
    final_test_blend = sum(best_weights[name] * aggregated_test[name] for name in model_names)

    submission = pd.DataFrame({"id": test["id"], "sentiment": final_test_blend})
    submission_path = SUBMISSIONS_DIR / "submission_multiseed_cv_rank_blend_auc.csv"
    submission.to_csv(submission_path, index=False)

    report = {
        "metric_priority": "roc_auc",
        "seeds": SEEDS,
        "cv_folds": N_FOLDS,
        "component_oof_auc": component_oof_auc,
        "best_weights": best_weights,
        "blend_oof_auc": blend_oof_auc,
        "fold_auc_summary": fold_auc_summary,
        "submission_path": str(submission_path),
    }
    report_path = REPORTS_DIR / "multiseed_cv_rank_blend_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Best multi-seed blend OOF ROC-AUC: {blend_oof_auc:.8f}")
    print(f"Best weights: {best_weights}")
    print(f"Saved submission to: {submission_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
