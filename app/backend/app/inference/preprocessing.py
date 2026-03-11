from __future__ import annotations

import re
import unicodedata

import numpy as np


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = re.sub(r"http\S+|www\S+", " ", normalized)
    normalized = re.sub(r"@\w+|#\w+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s!?.,'\"-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def preprocess_sequence_text(text: str) -> str:
    tokens = normalize_text(text).split()
    filtered = [token for token in tokens if token not in STOPWORDS]
    return " ".join(filtered or tokens)


def texts_to_sequences(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    unk_idx = vocab.get("<UNK>", 1)
    sequences = np.zeros((len(texts), max_len), dtype=np.int32)
    for row_idx, text in enumerate(texts):
        for col_idx, token in enumerate(str(text).split()[:max_len]):
            sequences[row_idx, col_idx] = vocab.get(token, unk_idx)
    return sequences

