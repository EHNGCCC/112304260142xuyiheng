from typing import Iterable

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


POSITIVE_WORDS = {"good", "great", "excellent", "best", "love", "amazing", "wonderful", "favorite"}
NEGATIVE_WORDS = {"bad", "worst", "awful", "boring", "waste", "terrible", "poor", "hate"}
NEGATION_WORDS = {"no", "nor", "not", "never"}


def train_word2vec(token_sequences: list[list[str]], config) -> Word2Vec:
    return Word2Vec(
        sentences=token_sequences,
        vector_size=config.word2vec_dim,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        workers=config.word2vec_workers,
        sg=1,
        hs=0,
        negative=10,
        epochs=config.word2vec_epochs,
        seed=config.seed,
    )


def fit_tfidf_vectorizer(joined_tokens: Iterable[str], max_features: int) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        sublinear_tf=True,
    )
    vectorizer.fit(list(joined_tokens))
    return vectorizer


def build_idf_lookup(vectorizer: TfidfVectorizer) -> dict[str, float]:
    names = vectorizer.get_feature_names_out()
    return dict(zip(names, vectorizer.idf_))


def _weighted_average_embedding(tokens: list[str], word2vec_model: Word2Vec, idf_lookup: dict[str, float], dim: int) -> np.ndarray:
    weighted_sum = np.zeros(dim, dtype=np.float32)
    total_weight = 0.0

    for token in tokens:
        if token not in word2vec_model.wv:
            continue
        weight = float(idf_lookup.get(token, 1.0))
        weighted_sum += word2vec_model.wv[token] * weight
        total_weight += weight

    if total_weight == 0.0:
        return weighted_sum
    return weighted_sum / total_weight


def _handcrafted_features(review: str, tokens: list[str]) -> np.ndarray:
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(token_count, 1)
    avg_token_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
    positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
    negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
    negation_hits = sum(token in NEGATION_WORDS for token in tokens)
    exclamation_count = review.count("!")
    question_count = review.count("?")

    return np.array(
        [
            token_count / 300.0,
            unique_ratio,
            avg_token_length / 10.0,
            positive_hits / max(token_count, 1),
            negative_hits / max(token_count, 1),
            negation_hits / max(token_count, 1),
            exclamation_count / 20.0,
            question_count / 20.0,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(df: pd.DataFrame, word2vec_model: Word2Vec, idf_lookup: dict[str, float], dim: int) -> np.ndarray:
    vectors = [
        _weighted_average_embedding(tokens, word2vec_model, idf_lookup, dim)
        for tokens in df["tokens"].tolist()
    ]
    handcrafted = [
        _handcrafted_features(review, tokens)
        for review, tokens in zip(df["clean_review"].tolist(), df["tokens"].tolist())
    ]
    return np.hstack([np.vstack(vectors), np.vstack(handcrafted)])

