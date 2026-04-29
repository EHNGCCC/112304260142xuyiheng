import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .feature_engineering import build_feature_matrix, build_idf_lookup, fit_tfidf_vectorizer
from .utils import ensure_directories, plot_auc_comparison, save_json


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def _build_candidate_models(config) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=2.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=config.seed,
            n_jobs=1,
        ),
    }


def _feature_cache_paths(config) -> dict[str, Path]:
    cache_dir = config.feature_cache_dir / config.classical_feature_cache_key
    ensure_directories([cache_dir])
    return {
        "cache_dir": cache_dir,
        "split_npz": cache_dir / "split_features.npz",
        "full_npz": cache_dir / "full_features.npz",
        "full_vectorizer": cache_dir / "full_vectorizer.pkl",
        "metadata": cache_dir / "metadata.json",
    }


def _build_or_load_feature_cache(
    labeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    word2vec_model,
    config,
    rebuild_cache: bool = False,
) -> dict:
    paths = _feature_cache_paths(config)
    cache_exists = all(
        path.exists()
        for key, path in paths.items()
        if key != "cache_dir"
    )

    if cache_exists and not rebuild_cache:
        print(f"Loading cached classical features from {paths['cache_dir']}")
        split_arrays = np.load(paths["split_npz"])
        full_arrays = np.load(paths["full_npz"])
        with paths["full_vectorizer"].open("rb") as handle:
            full_vectorizer = pickle.load(handle)
        return {
            "x_train": split_arrays["x_train"],
            "x_val": split_arrays["x_val"],
            "y_train": split_arrays["y_train"],
            "y_val": split_arrays["y_val"],
            "x_full": full_arrays["x_full"],
            "y_full": full_arrays["y_full"],
            "x_test": full_arrays["x_test"],
            "full_vectorizer": full_vectorizer,
        }

    print("Building classical feature cache from Word2Vec embeddings...")
    train_df, val_df = train_test_split(
        labeled_df,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labeled_df["sentiment"],
    )

    split_vectorizer = fit_tfidf_vectorizer(train_df["joined_tokens"], max_features=config.tfidf_max_features)
    split_idf_lookup = build_idf_lookup(split_vectorizer)
    x_train = build_feature_matrix(train_df, word2vec_model, split_idf_lookup, config.word2vec_dim)
    x_val = build_feature_matrix(val_df, word2vec_model, split_idf_lookup, config.word2vec_dim)
    y_train = train_df["sentiment"].astype(int).to_numpy()
    y_val = val_df["sentiment"].astype(int).to_numpy()

    full_vectorizer = fit_tfidf_vectorizer(labeled_df["joined_tokens"], max_features=config.tfidf_max_features)
    full_idf_lookup = build_idf_lookup(full_vectorizer)
    x_full = build_feature_matrix(labeled_df, word2vec_model, full_idf_lookup, config.word2vec_dim)
    x_test = build_feature_matrix(test_df, word2vec_model, full_idf_lookup, config.word2vec_dim)
    y_full = labeled_df["sentiment"].astype(int).to_numpy()

    np.savez_compressed(
        paths["split_npz"],
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
    )
    np.savez_compressed(
        paths["full_npz"],
        x_full=x_full,
        y_full=y_full,
        x_test=x_test,
    )
    with paths["full_vectorizer"].open("wb") as handle:
        pickle.dump(full_vectorizer, handle)
    save_json(
        paths["metadata"],
        {
            "feature_type": "tfidf_weighted_word2vec_plus_manual_features",
            "metric_priority": "roc_auc",
            "cache_key": config.classical_feature_cache_key,
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "full_rows": int(len(y_full)),
            "test_rows": int(len(test_df)),
            "feature_dim": int(x_full.shape[1]),
        },
    )
    print(f"Saved classical feature cache to {paths['cache_dir']}")

    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "x_full": x_full,
        "y_full": y_full,
        "x_test": x_test,
        "full_vectorizer": full_vectorizer,
    }


def run_classical_pipeline(
    labeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    word2vec_model,
    config,
    rebuild_cache: bool = False,
) -> dict:
    feature_cache = _build_or_load_feature_cache(
        labeled_df,
        test_df,
        word2vec_model,
        config,
        rebuild_cache=rebuild_cache,
    )

    x_train = feature_cache["x_train"]
    x_val = feature_cache["x_val"]
    y_train = feature_cache["y_train"]
    y_val = feature_cache["y_val"]

    candidate_models = _build_candidate_models(config)
    metrics_by_model = {}
    best_name = None
    best_auc = -1.0
    best_val_prob = None

    for name, estimator in candidate_models.items():
        estimator.fit(x_train, y_train)
        val_prob = estimator.predict_proba(x_val)[:, 1]
        metrics = _compute_metrics(y_val, val_prob)
        metrics_by_model[name] = metrics
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_name = name
            best_val_prob = val_prob

    save_json(
        config.reports_dir / "classical_validation_metrics.json",
        {"metric_priority": "roc_auc", "results": metrics_by_model, "best_model": best_name},
    )
    plot_auc_comparison(
        metrics_by_model,
        config.plots_dir / "classical_model_auc.png",
        title="Classical Models Validation ROC-AUC",
    )

    x_full = feature_cache["x_full"]
    x_test = feature_cache["x_test"]
    y_full = feature_cache["y_full"]
    full_vectorizer = feature_cache["full_vectorizer"]

    final_model = clone(candidate_models[best_name])
    final_model.fit(x_full, y_full)
    test_prob = final_model.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_prob})
    submission_path = config.submissions_dir / "submission_classical_auc.csv"
    submission.to_csv(submission_path, index=False)

    artifact_path = config.artifacts_dir / "best_classical_model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(
            {
                "model_name": best_name,
                "classifier": final_model,
                "tfidf_vectorizer": full_vectorizer,
                "word2vec_dim": config.word2vec_dim,
            },
            handle,
        )

    return {
        "best_model": best_name,
        "validation_metrics": metrics_by_model,
        "validation_labels": y_val,
        "validation_probabilities": best_val_prob,
        "test_probabilities": test_prob,
        "submission_path": str(submission_path),
        "artifact_path": str(artifact_path),
    }
