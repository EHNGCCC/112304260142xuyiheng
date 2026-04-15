from dataclasses import dataclass, field
from pathlib import Path
import os

import torch


@dataclass
class ProjectConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    cache_version: str = "v2"
    seed: int = 42
    test_size: float = 0.2
    word2vec_dim: int = 200
    word2vec_window: int = 5
    word2vec_min_count: int = 3
    word2vec_epochs: int = 12
    word2vec_workers: int = field(default_factory=lambda: max(1, min(4, os.cpu_count() or 1)))
    tfidf_max_features: int = 30000
    batch_size: int = 64
    max_len: int = 220
    bilstm_hidden_size: int = 96
    bilstm_dropout: float = 0.3
    bilstm_epochs: int = 6
    learning_rate: float = 1e-3
    patience: int = 2
    num_workers: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        self.data_dir = self.project_root.parent / "word2vec-nlp-tutorial"
        self.artifacts_dir = self.project_root / "artifacts"
        self.cache_dir = self.artifacts_dir / "cache"
        self.preprocessed_dir = self.cache_dir / "preprocessed"
        self.feature_cache_dir = self.cache_dir / "features"
        self.reports_dir = self.project_root / "reports"
        self.plots_dir = self.reports_dir / "plots"
        self.submissions_dir = self.project_root / "submissions"

    @property
    def preprocess_cache_key(self) -> str:
        return f"preprocess_{self.cache_version}"

    @property
    def word2vec_cache_key(self) -> str:
        return (
            f"w2v_{self.cache_version}_dim{self.word2vec_dim}_win{self.word2vec_window}"
            f"_min{self.word2vec_min_count}_ep{self.word2vec_epochs}_seed{self.seed}"
        )

    @property
    def word2vec_path(self) -> Path:
        return self.artifacts_dir / f"{self.word2vec_cache_key}.model"

    @property
    def classical_feature_cache_key(self) -> str:
        split_pct = int(self.test_size * 100)
        return (
            f"{self.word2vec_cache_key}_tfidf{self.tfidf_max_features}"
            f"_split{split_pct}_seed{self.seed}"
        )
