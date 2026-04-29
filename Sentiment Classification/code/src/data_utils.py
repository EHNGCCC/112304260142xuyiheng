import csv
from pathlib import Path
from typing import Tuple

import pandas as pd


def _read_competition_tsv(path: Path) -> pd.DataFrame:
    kwargs = {"sep": "\t"}
    if path.name == "unlabeledTrainData.tsv":
        kwargs["quoting"] = csv.QUOTE_NONE
    return pd.read_csv(path, **kwargs)


def load_competition_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = _read_competition_tsv(data_dir / "labeledTrainData.tsv")
    unlabeled = _read_competition_tsv(data_dir / "unlabeledTrainData.tsv")
    test = _read_competition_tsv(data_dir / "testData.tsv")
    return labeled, unlabeled, test


def describe_datasets(labeled: pd.DataFrame, unlabeled: pd.DataFrame, test: pd.DataFrame) -> dict:
    sentiment_counts = labeled["sentiment"].value_counts().sort_index().to_dict()
    return {
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(len(unlabeled)),
        "test_rows": int(len(test)),
        "sentiment_distribution": {int(k): int(v) for k, v in sentiment_counts.items()},
    }

