import html
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}

NEGATION_WORDS = {"no", "nor", "not", "never"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def normalize_review(review: str) -> str:
    if pd.isna(review):
        return ""

    text = html.unescape(str(review))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()

    for short, long_form in CONTRACTIONS.items():
        text = text.replace(short, long_form)

    text = re.sub(r"[^a-z!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_review(clean_review: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(clean_review)
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "review") -> pd.DataFrame:
    prepared = df.copy()
    prepared["clean_review"] = prepared[text_column].map(normalize_review)
    prepared["tokens"] = prepared["clean_review"].map(tokenize_review)
    prepared["joined_tokens"] = prepared["tokens"].map(" ".join)
    return prepared

