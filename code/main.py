import argparse

from gensim.models import Word2Vec

from src.bilstm_pipeline import run_bilstm_pipeline
from src.cache_utils import load_or_preprocess_datasets
from src.classical_pipeline import run_classical_pipeline
from src.config import ProjectConfig
from src.data_utils import describe_datasets, load_competition_data
from src.ensemble_pipeline import run_blend_pipeline
from src.feature_engineering import train_word2vec
from src.utils import ensure_directories, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Word2Vec NLP Tutorial project runner")
    parser.add_argument(
        "--include-bilstm",
        action="store_true",
        help="Also train the optional Word2Vec + BiLSTM pipeline.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore saved preprocessing and feature caches and rebuild them for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    ensure_directories(
        [
            config.artifacts_dir,
            config.cache_dir,
            config.preprocessed_dir,
            config.feature_cache_dir,
            config.reports_dir,
            config.plots_dir,
            config.submissions_dir,
        ]
    )
    set_seed(config.seed)

    labeled_df, unlabeled_df, test_df = load_competition_data(config.data_dir)
    summary = describe_datasets(labeled_df, unlabeled_df, test_df)
    save_json(config.reports_dir / "dataset_summary.json", summary)

    print("Dataset summary:", summary)
    labeled_df, unlabeled_df, test_df = load_or_preprocess_datasets(
        labeled_df,
        unlabeled_df,
        test_df,
        config,
        rebuild_cache=args.rebuild_cache,
    )

    word2vec_path = config.word2vec_path
    legacy_word2vec_path = config.artifacts_dir / "word2vec.model"
    legacy_compatible = (
        config.word2vec_dim == 200
        and config.word2vec_window == 5
        and config.word2vec_min_count == 3
        and config.word2vec_epochs == 12
        and config.seed == 42
    )
    if word2vec_path.exists():
        print(f"Loading cached Word2Vec model from {word2vec_path}")
        word2vec_model = Word2Vec.load(str(word2vec_path))
    elif not args.rebuild_cache and legacy_compatible and legacy_word2vec_path.exists():
        print(f"Loading legacy cached Word2Vec model from {legacy_word2vec_path}")
        word2vec_model = Word2Vec.load(str(legacy_word2vec_path))
    else:
        print("Training Word2Vec on labeled + unlabeled reviews...")
        unsupervised_corpus = labeled_df["tokens"].tolist() + unlabeled_df["tokens"].tolist()
        word2vec_model = train_word2vec(unsupervised_corpus, config)
        word2vec_model.save(str(word2vec_path))
        print(f"Word2Vec model saved to {word2vec_path}")

    classical_result = run_classical_pipeline(
        labeled_df,
        test_df,
        word2vec_model,
        config,
        rebuild_cache=args.rebuild_cache,
    )
    print("Classical pipeline finished.")
    print("Best model:", classical_result["best_model"])
    print("Validation metrics:", classical_result["validation_metrics"])
    print("Submission:", classical_result["submission_path"])

    if args.include_bilstm:
        print("Training optional BiLSTM pipeline...")
        bilstm_result = run_bilstm_pipeline(labeled_df, test_df, word2vec_model, config)
        print("BiLSTM validation metrics:", bilstm_result["validation_metrics"])
        print("BiLSTM submission:", bilstm_result["submission_path"])

        print("Searching for the best validation AUC blend between classical and BiLSTM...")
        blend_result = run_blend_pipeline(classical_result, bilstm_result, test_df["id"], config)
        print("Blend validation metrics:", blend_result["validation_metrics"])
        print(
            "Blend weights:",
            {
                "bilstm": blend_result["best_bilstm_weight"],
                "classical": blend_result["best_classical_weight"],
            },
        )
        print("Blend submission:", blend_result["submission_path"])


if __name__ == "__main__":
    main()
