
from __future__ import annotations

import re
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

nltk.download(["stopwords", "wordnet", "punkt"], quiet=True)

STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_from_normalized(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    text = text.lower()
    tokens = text.split()
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP]
    return " ".join(tokens)


def clean_text_fallback(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_words_tfidf(
    df: pd.DataFrame,
    cls,
    n: int = 15,
    classes=None,
    label_col: str | None = None,
    text_col: str = "text",
    extra_stopwords=None,
):
    if label_col is None:
        for candidate in ["label", "complexity", "age"]:
            if candidate in df.columns:
                label_col = candidate
                break
        else:
            raise ValueError("No label column found. Expected one of: 'label', 'complexity', 'age'.")

    if classes is None:
        classes = df[label_col].dropna().astype(str).sort_values().unique().tolist()

    extra_stopwords = set(extra_stopwords or set())
    class_texts = {}
    for class_name in classes:
        texts = (
            df[df[label_col] == class_name][text_col]
            .dropna()
            .astype(str)
            .apply(clean_text_fallback)
            .tolist()
        )
        class_texts[class_name] = " ".join(texts)

    docs = list(class_texts.values())
    labels = list(class_texts.keys())
    vectorizer = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(extra_stopwords)),
        max_features=10_000,
        ngram_range=(1, 1),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(docs)
    words = vectorizer.get_feature_names_out()

    cls_idx = labels.index(cls)
    scores = matrix[cls_idx].toarray().flatten()
    top = sorted(zip(words, scores), key=lambda item: item[1], reverse=True)[:n]
    return pd.DataFrame(top, columns=["word", "score"])
