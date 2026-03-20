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


def preprocess_from_normalized(text: str) -> str:
    tokens = str(text).lower().split()
    filtered = [token for token in tokens if token not in STOPWORDS]
    return " ".join(filtered or tokens)


def preprocess_sequence_text(text: str) -> str:
    normalized = normalize_text(text)
    processed = preprocess_from_normalized(normalized)
    return processed or normalized


def parse_preprocessing_spec(spec: str | None) -> list[str]:
    if spec is None:
        return []

    parts = re.split(r"\s*(?:\+|,|->)\s*", spec.strip())
    return [
        part.strip().lower().replace("-", "_")
        for part in parts
        if part.strip()
    ]


def apply_text_preprocessing(
    text: str,
    spec: str | None,
    *,
    default_steps: tuple[str, ...] = ("normalize_text",),
) -> str:
    steps = parse_preprocessing_spec(spec) or list(default_steps)
    value = str(text)

    for step in steps:
        if step == "texts_to_sequences":
            break
        if step == "normalize_text":
            value = normalize_text(value)
            continue
        if step == "preprocess_from_normalized":
            value = preprocess_from_normalized(value)
            continue
        if step == "preprocess_sequence_text":
            value = preprocess_sequence_text(value)
            continue
        raise ValueError(f"Unsupported preprocessing step '{step}'.")

    return value


def texts_to_sequences(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    unk_idx = vocab.get("<UNK>", 1)
    sequences = np.zeros((len(texts), max_len), dtype=np.int32)
    for row_idx, text in enumerate(texts):
        for col_idx, token in enumerate(str(text).split()[:max_len]):
            sequences[row_idx, col_idx] = vocab.get(token, unk_idx)
    return sequences
