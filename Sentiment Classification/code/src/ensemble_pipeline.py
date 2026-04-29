import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .utils import save_json


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def run_blend_pipeline(classical_result: dict, bilstm_result: dict, test_ids: pd.Series, config) -> dict:
    classical_labels = np.asarray(classical_result["validation_labels"])
    bilstm_labels = np.asarray(bilstm_result["validation_labels"])
    if classical_labels.shape != bilstm_labels.shape or not np.array_equal(classical_labels, bilstm_labels):
        raise ValueError("Validation labels do not match between classical and BiLSTM pipelines.")

    classical_val_prob = np.asarray(classical_result["validation_probabilities"])
    bilstm_val_prob = np.asarray(bilstm_result["validation_probabilities"])
    classical_test_prob = np.asarray(classical_result["test_probabilities"])
    bilstm_test_prob = np.asarray(bilstm_result["test_probabilities"])

    best_weight = None
    best_metrics = None
    search_history = []

    for bilstm_weight in np.linspace(0.0, 1.0, 41):
        blend_val_prob = bilstm_weight * bilstm_val_prob + (1.0 - bilstm_weight) * classical_val_prob
        metrics = _compute_metrics(classical_labels, blend_val_prob)
        search_history.append(
            {
                "bilstm_weight": float(bilstm_weight),
                "classical_weight": float(1.0 - bilstm_weight),
                "auc": metrics["auc"],
            }
        )
        if best_metrics is None or metrics["auc"] > best_metrics["auc"]:
            best_weight = float(bilstm_weight)
            best_metrics = metrics

    blend_test_prob = best_weight * bilstm_test_prob + (1.0 - best_weight) * classical_test_prob
    submission = pd.DataFrame({"id": test_ids, "sentiment": blend_test_prob})
    submission_path = config.submissions_dir / "submission_blend_auc.csv"
    submission.to_csv(submission_path, index=False)

    report = {
        "metric_priority": "roc_auc",
        "best_bilstm_weight": best_weight,
        "best_classical_weight": float(1.0 - best_weight),
        "best_validation_metrics": best_metrics,
        "classical_model": classical_result["best_model"],
        "search_history": search_history,
    }
    save_json(config.reports_dir / "blend_validation_metrics.json", report)

    return {
        "validation_metrics": best_metrics,
        "best_bilstm_weight": best_weight,
        "best_classical_weight": float(1.0 - best_weight),
        "submission_path": str(submission_path),
    }
