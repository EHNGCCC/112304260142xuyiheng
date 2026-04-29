from pathlib import Path

import pandas as pd

from .preprocess import preprocess_dataframe
from .utils import ensure_directories


def _preprocessed_cache_paths(config) -> dict[str, Path]:
    ensure_directories([config.preprocessed_dir])
    key = config.preprocess_cache_key
    return {
        "labeled": config.preprocessed_dir / f"labeled_{key}.pkl",
        "unlabeled": config.preprocessed_dir / f"unlabeled_{key}.pkl",
        "test": config.preprocessed_dir / f"test_{key}.pkl",
    }


def load_or_preprocess_datasets(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config,
    rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_paths = _preprocessed_cache_paths(config)

    if not rebuild_cache and all(path.exists() for path in cache_paths.values()):
        print(f"Loading cached preprocessed datasets from {config.preprocessed_dir}")
        return (
            pd.read_pickle(cache_paths["labeled"]),
            pd.read_pickle(cache_paths["unlabeled"]),
            pd.read_pickle(cache_paths["test"]),
        )

    print("Preprocessing labeled, unlabeled, and test reviews...")
    labeled_prepared = preprocess_dataframe(labeled_df)
    unlabeled_prepared = preprocess_dataframe(unlabeled_df)
    test_prepared = preprocess_dataframe(test_df)

    labeled_prepared.to_pickle(cache_paths["labeled"])
    unlabeled_prepared.to_pickle(cache_paths["unlabeled"])
    test_prepared.to_pickle(cache_paths["test"])
    print(f"Saved preprocessed dataset cache to {config.preprocessed_dir}")

    return labeled_prepared, unlabeled_prepared, test_prepared
