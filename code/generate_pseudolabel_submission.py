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


def build_model_specs():
    return [
        {
            "name": "word_clean",
            "weight": 0.35,
            "text_column": "clean_review",
            "vectorizer": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=200000,
            ),
            "builder": lambda: LinearSVC(C=0.6),
        },
        {
            "name": "word_soft",
            "weight": 0.10,
            "text_column": "soft_review",
            "vectorizer": lambda: TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=250000,
            ),
            "builder": lambda: LinearSVC(C=0.5),
        },
        {
            "name": "nb_word",
            "weight": 0.35,
            "text_column": "soft_review",
            "vectorizer": lambda: CountVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                strip_accents="unicode",
                max_features=250000,
                binary=True,
            ),
            "builder": lambda: NbSvm(c=4.0, alpha=1.0),
        },
        {
            "name": "nb_char",
            "weight": 0.20,
            "text_column": "soft_review",
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
        },
    ]


def fit_sparse_blend(train_df: pd.DataFrame, predict_dfs: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    scores_by_name = {name: np.zeros(len(df), dtype=np.float64) for name, df in predict_dfs.items()}
    model_specs = build_model_specs()

    for spec in model_specs:
        vectorizer = spec["vectorizer"]()
        x_train = vectorizer.fit_transform(train_df[spec["text_column"]].tolist())
        model = spec["builder"]()
        model.fit(x_train, train_df["sentiment"].astype(int).to_numpy())

        for name, predict_df in predict_dfs.items():
            x_pred = vectorizer.transform(predict_df[spec["text_column"]].tolist())
            pred_rank = rank_normalize(model.decision_function(x_pred))
            scores_by_name[name] += spec["weight"] * pred_rank

    return scores_by_name


def build_pseudo_dataset(test_df: pd.DataFrame, teacher_scores: np.ndarray, frac: float) -> pd.DataFrame:
    selected_mask = (teacher_scores <= frac) | (teacher_scores >= 1.0 - frac)
    pseudo = test_df.loc[selected_mask, ["id", "review", "clean_review", "soft_review"]].copy()
    pseudo["sentiment"] = (teacher_scores[selected_mask] >= 0.5).astype(int)
    return pseudo


def main() -> None:
    labeled = pd.read_pickle(PREPROCESSED_DIR / "labeled_preprocess_v2.pkl")
    test = pd.read_pickle(PREPROCESSED_DIR / "test_preprocess_v2.pkl")
    labeled = labeled.copy()
    test = test.copy()
    labeled["soft_review"] = labeled["review"].map(soft_clean)
    test["soft_review"] = test["review"].map(soft_clean)

    train_df, val_df = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=labeled["sentiment"],
    )
    y_val = val_df["sentiment"].astype(int).to_numpy()

    print("Training teacher model on train split...")
    teacher_scores = fit_sparse_blend(train_df, {"val": val_df, "test": test})
    teacher_val = teacher_scores["val"]
    teacher_test = teacher_scores["test"]
    teacher_auc = float(roc_auc_score(y_val, teacher_val))
    print(f"Teacher validation ROC-AUC: {teacher_auc:.8f}")

    best_result = {
        "validation_auc": teacher_auc,
        "pseudo_frac": 0.0,
        "alpha": 0.0,
        "pseudo_count": 0,
        "student_val_scores": teacher_val,
    }
    search_log = []

    pseudo_fracs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    alphas = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

    for frac in pseudo_fracs:
        pseudo_df = build_pseudo_dataset(test, teacher_test, frac)
        train_plus_pseudo = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"Training student with pseudo fraction {frac:.3f} ({len(pseudo_df)} samples)...")
        student_scores = fit_sparse_blend(train_plus_pseudo, {"val": val_df})
        student_val = student_scores["val"]

        for alpha in alphas:
            blended_val = (1.0 - alpha) * teacher_val + alpha * student_val
            auc = float(roc_auc_score(y_val, blended_val))
            result = {
                "pseudo_frac": float(frac),
                "alpha": float(alpha),
                "pseudo_count": int(len(pseudo_df)),
                "validation_auc": auc,
            }
            search_log.append(result)
            print(
                f"  pseudo_frac={frac:.3f} alpha={alpha:.2f} "
                f"validation_auc={auc:.8f}"
            )
            if auc > best_result["validation_auc"]:
                best_result = {
                    "validation_auc": auc,
                    "pseudo_frac": float(frac),
                    "alpha": float(alpha),
                    "pseudo_count": int(len(pseudo_df)),
                    "student_val_scores": student_val,
                }

    print("Best pseudo-label config:", {k: v for k, v in best_result.items() if k != "student_val_scores"})

    print("Training final teacher on full labeled data...")
    final_teacher_scores = fit_sparse_blend(labeled, {"test": test})
    final_teacher_test = final_teacher_scores["test"]

    if best_result["pseudo_frac"] > 0.0:
        final_pseudo_df = build_pseudo_dataset(test, final_teacher_test, best_result["pseudo_frac"])
        final_train = pd.concat([labeled, final_pseudo_df], ignore_index=True)
        print(f"Training final student with {len(final_pseudo_df)} pseudo-labeled test samples...")
        final_student_scores = fit_sparse_blend(final_train, {"test": test})
        final_student_test = final_student_scores["test"]
        final_test_scores = (1.0 - best_result["alpha"]) * final_teacher_test + best_result["alpha"] * final_student_test
    else:
        final_pseudo_df = pd.DataFrame()
        final_test_scores = final_teacher_test

    submission = pd.DataFrame({"id": test["id"], "sentiment": final_test_scores})
    submission_path = SUBMISSIONS_DIR / "submission_pseudolabel_rank_blend_auc.csv"
    submission.to_csv(submission_path, index=False)

    report = {
        "metric_priority": "roc_auc",
        "teacher_validation_auc": teacher_auc,
        "best_validation_auc": best_result["validation_auc"],
        "best_pseudo_frac": best_result["pseudo_frac"],
        "best_alpha": best_result["alpha"],
        "best_pseudo_count": best_result["pseudo_count"],
        "search_log": search_log,
        "submission_path": str(submission_path),
    }
    report_path = REPORTS_DIR / "pseudolabel_rank_blend_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved submission to: {submission_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
