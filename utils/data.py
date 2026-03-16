
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import gensim.downloader as api
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .text import normalize_text, preprocess_from_normalized

CFG: dict = {}
class2id: dict = {}
splits: dict = {}

def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def size_tag_from_split(split_key: str) -> str:
    m = re.search(r"(\d+[km]?)$", split_key.lower())
    return m.group(1) if m else slugify(split_key)


def first_existing_path(*paths):
    for p in paths:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def load_pickle_with_fallback(primary: Path, *fallbacks: Path):
    path = first_existing_path(primary, *fallbacks)
    if path is None:
        raise FileNotFoundError(f"Could not find pickle file. Tried: {[str(primary), *map(str, fallbacks)]}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded pickle: {path}")
    return obj


def load_csv_with_fallback(primary: Path, *fallbacks: Path) -> pd.DataFrame:
    path = first_existing_path(primary, *fallbacks)
    if path is None:
        raise FileNotFoundError(f"Could not find CSV file. Tried: {[str(primary), *map(str, fallbacks)]}")
    print(f"Loaded csv: {path}")
    return pd.read_csv(path)


def load_optional_dataframe(csv_path: Path, pkl_path: Path = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    pkl_path = Path(pkl_path) if pkl_path is not None else None

    if pkl_path is not None and pkl_path.exists():
        try:
            obj = joblib.load(pkl_path)
            if isinstance(obj, pd.DataFrame):
                return obj
        except Exception as e:
            print(f"Could not load {pkl_path}: {e}")

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not load {csv_path}: {e}")

    return pd.DataFrame()


def _load_latest_run_array(runs_dir, stem: str):
    if runs_dir is None or not Path(runs_dir).exists():
        return np.asarray([], dtype=np.int64), None

    runs_dir = Path(runs_dir)
    latest_file = runs_dir / f"{stem}_latest.npy"
    if latest_file.exists():
        try:
            return np.load(latest_file), latest_file
        except Exception as e:
            print(f"Could not load {latest_file}: {e}")

    candidates = sorted(runs_dir.glob(f"{stem}_*.npy"))
    for candidate in reversed(candidates):
        try:
            return np.load(candidate), candidate
        except Exception as e:
            print(f"Could not load {candidate}: {e}")

    return np.asarray([], dtype=np.int64), None


def load_cached_internal_run_artifacts(runs_dir: Path):
    runs_dir = Path(runs_dir)
    metrics_path = runs_dir / "metrics_latest.json"

    if not metrics_path.exists():
        return None, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), None

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Could not load cached metrics from {metrics_path}: {e}")
        return None, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), None

    if not isinstance(payload, dict) or "val" not in payload or "test" not in payload:
        return None, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), None

    val_preds, _ = _load_latest_run_array(runs_dir, "val_preds")
    test_preds, _ = _load_latest_run_array(runs_dir, "test_preds")
    return payload, np.asarray(val_preds), np.asarray(test_preds), metrics_path


def _link_or_copy_file(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def _link_or_copy_dir_contents(src_dir: Path, dst_dir: Path, include_pattern: str = None):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    if include_pattern is None:
        items = [p for p in src_dir.iterdir() if p.name != ".DS_Store"]
    else:
        items = [p for p in src_dir.glob(include_pattern)]

    copied_any = False
    for item in items:
        target = dst_dir / item.name
        if target.exists():
            copied_any = True
            continue
        try:
            if item.is_dir():
                os.symlink(item.resolve(), target, target_is_directory=True)
            else:
                os.symlink(item.resolve(), target)
        except Exception:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        copied_any = True

    return copied_any


def find_latest_experiment_dir(prefix: str, model_root: Path = None, require_best: bool = False):
    model_root = model_root or CFG["model_dir"]
    patt = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}$")
    candidates = [p for p in model_root.iterdir() if p.is_dir() and patt.match(p.name)]
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.name)
    if not require_best:
        return candidates[-1]

    for cand in reversed(candidates):
        best_dir = cand / "best_model"
        if best_dir.exists() and any(best_dir.iterdir()):
            return cand
    return None


def resolve_experiment_paths(model_family: str, dataset: str, size_tag: str, model_root: Path = None) -> dict:
    model_root = model_root or CFG["model_dir"]
    prefix = f"{slugify(model_family)}_{slugify(dataset)}_{slugify(size_tag)}"

    exp_root = find_latest_experiment_dir(prefix, model_root=model_root, require_best=True)
    if exp_root is None:
        exp_root = find_latest_experiment_dir(prefix, model_root=model_root, require_best=False)
    if exp_root is None:
        exp_root = model_root / f"{prefix}_{now_ts()}"

    exp_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "prefix": prefix,
        "root": exp_root,
        "best_model": exp_root / "best_model",
        "checkpoints": exp_root / "checkpoints",
        "runs": exp_root / "runs",
    }
    for key in ["best_model", "checkpoints", "runs"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def bootstrap_file_from_legacy(dst_path: Path, legacy_candidates=()):
    dst_path = Path(dst_path)
    if dst_path.exists():
        return dst_path

    for cand in legacy_candidates:
        p = Path(cand)
        if p.exists() and p.is_file():
            _link_or_copy_file(p, dst_path)
            if dst_path.exists():
                print(f"Linked/copied legacy file: {p} -> {dst_path}")
                break

    return dst_path


def bootstrap_dir_from_legacy(dst_dir: Path, legacy_candidates=(), include_pattern: str = None):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if any(dst_dir.iterdir()):
        return dst_dir

    for cand in legacy_candidates:
        p = Path(cand)
        if p.exists() and p.is_dir():
            ok = _link_or_copy_dir_contents(p, dst_dir, include_pattern=include_pattern)
            if ok:
                print(f"Bootstrapped legacy dir: {p} -> {dst_dir}")
                break

    return dst_dir


def resolve_target_col(df: pd.DataFrame, cfg: dict | None = None) -> str:
    cfg = cfg or CFG
    candidates = []
    if cfg.get("encoded_label_col"):
        candidates.append(cfg["encoded_label_col"])
    candidates.extend(["label_enc", "age_enc"])

    for col in candidates:
        if col and col in df.columns:
            return col

    raise ValueError(
        f"Missing target column: expected one of {sorted(set(candidates))}."
    )


def prepare_encoded_text_frame(
    df: pd.DataFrame,
    cfg: dict | None = None,
    label_mapping: dict | None = None,
    drop_duplicates: bool = False,
    clean_text: str = "raw",
) -> pd.DataFrame:
    cfg = cfg or CFG
    label_mapping = label_mapping or globals().get("class2id")
    if not label_mapping:
        raise ValueError("label_mapping is required to encode labels.")
    if clean_text == "clean":
        clean_text = "processed"
    if clean_text not in {"raw", "processed"}:
        raise ValueError("clean_text must be 'raw', 'clean', or 'processed'.")

    target_col = cfg.get("encoded_label_col", "label_enc")
    frame = (
        df[[cfg["text_col"], cfg["label_col"]]]
        .dropna(subset=[cfg["text_col"], cfg["label_col"]])
        .copy()
    )
    if drop_duplicates:
        frame = frame.drop_duplicates(subset=[cfg["text_col"]])

    frame = frame[frame[cfg["label_col"]].isin(label_mapping)].reset_index(drop=True)
    frame["text_raw"] = frame[cfg["text_col"]].map(normalize_text)
    if clean_text == "processed":
        frame["text_clean"] = frame["text_raw"].map(preprocess_from_normalized)
    else:
        frame["text_clean"] = frame["text_raw"]
    frame[target_col] = frame[cfg["label_col"]].map(label_mapping).astype(int)
    return frame.reset_index(drop=True)


def make_splits_and_arrays(df, CFG):
    target_col = resolve_target_col(df, CFG)

    df_train_full, df_test = train_test_split(
        df,
        test_size=CFG["test_size"],
        random_state=CFG["seed"],
        stratify=df[target_col],
    )

    df_train, df_val = train_test_split(
        df_train_full,
        test_size=CFG["val_size"] / (1 - CFG["test_size"]),
        random_state=CFG["seed"],
        stratify=df_train_full[target_col],
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    X_train      = df_train["text_clean"].values
    X_val        = df_val["text_clean"].values
    X_test       = df_test["text_clean"].values

    X_train_raw  = df_train["text_raw"].values
    X_val_raw    = df_val["text_raw"].values
    X_test_raw   = df_test["text_raw"].values

    y_train      = df_train[target_col].values
    y_val        = df_val[target_col].values
    y_test       = df_test[target_col].values

    return {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
        "y_train": y_train, "y_val": y_val, "y_test": y_test
    }


def build_vocab(
    texts,
    max_vocab: int = 50_000,
    min_freq: int = 2,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> dict:
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    vocab = {pad_token: 0, unk_token: 1}

    filtered = [(w, c) for w, c in counter.items() if c >= min_freq]
    filtered.sort(key=lambda x: x[1], reverse=True)

    for word, _ in filtered[: max_vocab - 2]:
        vocab[word] = len(vocab)

    return vocab


def load_embedding_matrix(vocab: dict, model_name: str, embed_dim: int) -> np.ndarray:
    """
    Downloads a pre-trained gensim embedding model and builds a numpy matrix
    aligned with `vocab`. Unknown words get random uniform initialisations.

    Parameters
    ----------
    vocab      : {word: index} mapping
    model_name : gensim key (e.g. "glove-wiki-gigaword-100")
    embed_dim  : embedding dimensionality

    Returns
    -------
    np.ndarray of shape (vocab_size, embed_dim)
    """
    print(f"Loading {model_name} ...")
    wv = api.load(model_name)

    matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    hits = misses = 0
    for word, idx in vocab.items():
        if word in wv:
            matrix[idx] = wv[word]
            hits += 1
        else:
            matrix[idx] = np.random.uniform(-0.25, 0.25, embed_dim)
            misses += 1

    matrix[0] = 0.0  # PAD stays all-zeros
    print(f"Coverage: {hits/(hits+misses)*100:.1f}%  (hits={hits}, misses={misses})")
    return matrix


def texts_to_sequences(texts, vocab: dict, max_len: int,
                       unk_token: str = "<UNK>") -> np.ndarray:
    """
    Converts a list of strings to a (N, max_len) int32 numpy array.
    Truncates long sequences; pads short ones with 0 (PAD index).
    """
    unk_idx = vocab.get(unk_token, 1)
    seqs = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, text in enumerate(texts):
        tokens = str(text).split()[:max_len]
        for j, tok in enumerate(tokens):
            seqs[i, j] = vocab.get(tok, unk_idx)
    return seqs


class TextSequenceDataset(Dataset):
    """
    Wraps padded token-ID sequences and integer class labels for classification.

    Parameters
    ----------
    sequences : np.ndarray shape (N, max_len)
    labels    : np.ndarray shape (N,) — integer class indices
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels    = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def make_loaders(seq_tr, y_tr, seq_va, y_va, seq_te, y_te,
                 batch_size: int = 64):
    """Creates train / val / test DataLoaders."""
    loader_train = DataLoader(TextSequenceDataset(seq_tr, y_tr),
                              batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    loader_val   = DataLoader(TextSequenceDataset(seq_va, y_va),
                              batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    loader_test  = DataLoader(TextSequenceDataset(seq_te, y_te),
                              batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader_train, loader_val, loader_test


class HFTextDataset(Dataset):
    """
    PyTorch Dataset wrapping a HuggingFace tokenizer for classification.
    Tokenization is done on-the-fly to keep memory usage stable on large corpora.
    """
    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.texts = np.asarray(texts, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_transformer_bundle_from_split(split_key: str) -> dict:
    s = splits[split_key]
    return {
        "name": split_key,
        "df_test": s["df_test"].copy(),
        "X_train_raw": np.asarray(s["X_train_raw"], dtype=object),
        "X_val_raw": np.asarray(s["X_val_raw"], dtype=object),
        "X_test_raw": np.asarray(s["X_test_raw"], dtype=object),
        "y_train": np.asarray(s["y_train"], dtype=np.int64),
        "y_val": np.asarray(s["y_val"], dtype=np.int64),
        "y_test": np.asarray(s["y_test"], dtype=np.int64),
    }


def build_transformer_bundle_from_parquet(data_path: Path, drop_duplicates: bool = False) -> dict:
    df = prepare_encoded_text_frame(
        pd.read_parquet(data_path),
        cfg=CFG,
        label_mapping=class2id,
        drop_duplicates=drop_duplicates,
        clean_text="raw",
    )

    split = make_splits_and_arrays(df, CFG)
    print(
        f"{data_path.name}: train={len(split['X_train']):,} | "
        f"val={len(split['X_val']):,} | test={len(split['X_test']):,}"
    )

    return {
        "name": data_path.stem,
        "df_test": split["df_test"].copy(),
        "X_train_raw": np.asarray(split["X_train_raw"], dtype=object),
        "X_val_raw": np.asarray(split["X_val_raw"], dtype=object),
        "X_test_raw": np.asarray(split["X_test_raw"], dtype=object),
        "y_train": np.asarray(split["y_train"], dtype=np.int64),
        "y_val": np.asarray(split["y_val"], dtype=np.int64),
        "y_test": np.asarray(split["y_test"], dtype=np.int64),
    }


def build_transformer_eval_bundle_from_df(df: pd.DataFrame, dataset_name: str, drop_duplicates: bool = False) -> dict:
    df = prepare_encoded_text_frame(
        df,
        cfg=CFG,
        label_mapping=class2id,
        drop_duplicates=drop_duplicates,
        clean_text="raw",
    )
    target_col = resolve_target_col(df, CFG)

    print(f"{dataset_name}: eval={len(df):,}")

    return {
        "name": dataset_name,
        "df_eval": df.copy(),
        "X_eval_raw": np.asarray(df["text_raw"], dtype=object),
        "y_eval": np.asarray(df[target_col], dtype=np.int64),
    }


def build_transformer_eval_bundle_from_parquet(data_path: Path, drop_duplicates: bool = False) -> dict:
    df = pd.read_parquet(data_path)
    return build_transformer_eval_bundle_from_df(df, dataset_name=data_path.stem, drop_duplicates=drop_duplicates)


def get_small_bundle(data_bundle: dict, n: int = 1000, seed: int = 42) -> dict:
    """
    Returns a copy of `data_bundle` with the training arrays subsampled to at most `n` examples.

    Parameters
    ----------
    data_bundle : dict with keys "X_train_raw", "y_train", and optionally "name"
    n           : maximum training-set size to keep
    seed        : random seed for reproducibility

    Returns
    -------
    dict — same structure as `data_bundle` with training arrays replaced by the subsample
    """
    import random as _random
    _random.seed(seed)
    train_size = len(data_bundle["X_train_raw"])
    indices = _random.sample(range(train_size), min(n, train_size))
    return {
        **data_bundle,
        "X_train_raw": [data_bundle["X_train_raw"][i] for i in indices],
        "y_train": [data_bundle["y_train"][i] for i in indices],
        "name": data_bundle.get("name", "") + f"_small{n}",
    }
