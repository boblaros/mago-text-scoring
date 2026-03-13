
from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from xgboost import XGBClassifier

from .data import HFTextDataset, bootstrap_dir_from_legacy, bootstrap_file_from_legacy, load_cached_internal_run_artifacts, load_optional_dataframe, now_ts, resolve_experiment_paths, size_tag_from_split, slugify
from .metrics import RESULTS, compute_metrics, evaluate_clf
from .plots import (
    display_saved_plot,
    plot_confusion_matrix_clf,
    plot_history,
    plot_learning_curve_from_metrics,
)

CFG: dict = {}
CLASSES: list = []
le = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_baseline_pipeline_models(cfg: dict = None) -> dict:
    cfg = cfg or CFG

    tfidf_kwargs = dict(
        analyzer="word",
        ngram_range=cfg.get("tfidf_word_ngrams", (1, 2)),
        max_features=cfg.get("tfidf_max_features", 100_000),
        sublinear_tf=True,
        min_df=2,
    )

    return {
        "Majority Class": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent", random_state=cfg["seed"]))
        ]),
        "TF-IDF + LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight=cfg.get("class_weight"),
                solver="saga",
                random_state=cfg["seed"],
            )),
        ]),
        "TF-IDF + SVC": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", LinearSVC(C=1.0, class_weight=cfg.get("class_weight"), random_state=cfg["seed"]))
        ]),
        "TF-IDF + XGBoost": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=cfg["seed"],
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
    }


def tune_baselines_on_100k(
    models: dict,
    split_100k: dict,
    seed: int = 42,
    n_iter: int = 4,
):
    x_tr, y_tr = split_100k["X_train"], split_100k["y_train"]
    x_va, y_va = split_100k["X_val"], split_100k["y_val"]

    search_spaces = {
        "TF-IDF + LogReg": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [50_000, 100_000],
            "clf__C": [0.5, 1.0, 2.0, 4.0],
        },
        "TF-IDF + SVC": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [50_000, 100_000],
            "clf__C": [0.5, 1.0, 2.0],
        },
    }

    best_models = {}
    rows = []

    fixed_models = sum(1 for name in models if name in {"Majority Class", "TF-IDF + XGBoost"})
    tuned_models = [name for name in models if name in search_spaces]
    total_steps = fixed_models + len(tuned_models) * n_iter

    with tqdm(total=total_steps, desc="Tuning baselines (reference split)") as pbar:
        for model_name, base_model in models.items():
            if model_name in {"Majority Class", "TF-IDF + XGBoost"}:
                best_models[model_name] = clone(base_model)
                rows.append({
                    "model": model_name,
                    "params": "fixed",
                    "val_f1_macro": np.nan,
                    "selected": True,
                    "note": "kept fixed",
                })
                pbar.set_description(f"{model_name} | fixed")
                pbar.update(1)
                continue

            candidates = list(ParameterSampler(search_spaces[model_name], n_iter=n_iter, random_state=seed))

            best_f1 = -1.0
            best_model = None
            best_params = None

            for params in candidates:
                pbar.set_description(f"{model_name} | tuning")
                est = clone(base_model)
                est.set_params(**params)
                est.fit(x_tr, y_tr)
                preds = est.predict(x_va)
                f1 = f1_score(y_va, preds, average="macro", zero_division=0)

                rows.append({
                    "model": model_name,
                    "params": str(params),
                    "val_f1_macro": f1,
                    "selected": False,
                    "note": "candidate",
                })

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = est
                    best_params = params

                pbar.set_postfix({"best_f1": f"{best_f1:.4f}"})
                pbar.update(1)

            rows.append({
                "model": model_name,
                "params": str(best_params),
                "val_f1_macro": best_f1,
                "selected": True,
                "note": "best_on_reference_val",
            })
            best_models[model_name] = best_model

    tuning_df = pd.DataFrame(rows)
    return best_models, tuning_df


def train_baselines_across_splits(
    models: dict,
    splits: dict,
    split_keys: list,
    label_encoder,
    model_root: Path,
    primary_curve_split: str = "val",
    single_point_models=("Majority Class",),
    single_point_key=None,
):
    if primary_curve_split not in {"val", "test"}:
        raise ValueError("primary_curve_split must be 'val' or 'test'")
    if single_point_key is None and split_keys:
        single_point_key = split_keys[-1]

    model_root.mkdir(parents=True, exist_ok=True)

    curve_rows = []
    eval_rows = []
    model_registry = {}
    cached_summary_rows = []

    use_cached_metrics = CFG.get("prefer_cached_internal_metrics", True)
    metrics_dir = CFG["output_paths"]["metrics"]
    baseline_eval_cache_df = (
        load_optional_dataframe(
            metrics_dir / "baseline_eval_metrics.csv",
            metrics_dir / "baseline_eval_metrics.pkl",
        )
        if use_cached_metrics else pd.DataFrame()
    )

    def _clean_metrics(metrics: dict) -> dict:
        clean = {}
        if not isinstance(metrics, dict):
            return clean
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                clean[key] = float(value)
        return clean

    def _lookup_cached_metrics(model_name: str, split_name: str, split_key_local: str):
        if baseline_eval_cache_df.empty:
            return None
        required_cols = {"model", "split", "split_key"}
        if not required_cols.issubset(baseline_eval_cache_df.columns):
            return None

        rows = baseline_eval_cache_df.loc[
            (baseline_eval_cache_df["model"] == model_name)
            & (baseline_eval_cache_df["split"] == split_name)
            & (baseline_eval_cache_df["split_key"] == split_key_local)
        ]
        if rows.empty:
            return None

        row = rows.iloc[-1]
        metrics = {}
        for col in ["accuracy", "f1_macro", "f1_weighted", "roc_auc"]:
            if col in row.index and pd.notna(row[col]):
                metrics[col] = float(row[col])
        return metrics or None

    total_steps = sum(
        1
        for split_key in split_keys
        for model_name in models
        if not (model_name in set(single_point_models) and split_key != single_point_key)
    )

    with tqdm(total=total_steps, desc="Baseline train/eval") as pbar:
        for split_key in split_keys:
            split = splits[split_key]
            x_tr, y_tr = split["X_train"], split["y_train"]
            x_val, y_val = split["X_val"], split["y_val"]
            x_test, y_test = split["X_test"], split["y_test"]
            train_size = len(x_tr)

            for model_name, base_model in models.items():
                if model_name in set(single_point_models) and split_key != single_point_key:
                    continue

                pbar.set_description(f"{model_name} | {split_key}")

                exp_paths = resolve_experiment_paths(
                    model_family=model_name,
                    dataset=CFG.get("experiment_dataset_slug", slugify(CFG.get("task", "dataset"))),
                    size_tag=size_tag_from_split(split_key),
                    model_root=model_root,
                )

                legacy_name_old = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name).strip("_").lower()
                legacy_candidates = [
                    CFG["model_dir"] / "baselines" / f"{legacy_name_old}__{split_key}.pkl",
                    CFG["model_dir"] / "baselines" / f"{slugify(model_name)}__{split_key}.pkl",
                ]
                best_model_file = exp_paths["best_model"] / "model.pkl"
                best_model_file = bootstrap_file_from_legacy(best_model_file, legacy_candidates)

                tag = f"{model_name} [{split_key}]"
                val_metrics = None
                test_metrics = None
                cache_source = None

                if use_cached_metrics:
                    val_metrics = _lookup_cached_metrics(model_name, "val", split_key)
                    test_metrics = _lookup_cached_metrics(model_name, "test", split_key)
                    if val_metrics is not None and test_metrics is not None:
                        cache_source = metrics_dir / "baseline_eval_metrics.csv"
                    else:
                        cached_payload, _, _, cached_path = load_cached_internal_run_artifacts(exp_paths["runs"])
                        if cached_payload is not None:
                            cached_val_metrics = _clean_metrics(cached_payload.get("val", {}))
                            cached_test_metrics = _clean_metrics(cached_payload.get("test", {}))
                            if cached_val_metrics and cached_test_metrics:
                                val_metrics = cached_val_metrics
                                test_metrics = cached_test_metrics
                                cache_source = cached_path

                if cache_source is not None:
                    status = "cached"
                    RESULTS[f"{tag} | val"] = val_metrics
                    RESULTS[f"{tag} | test"] = test_metrics
                    cached_summary_rows.append({
                        "model": model_name,
                        "split_key": split_key,
                        "train_size": int(train_size),
                        "status": status,
                        "source": Path(cache_source).name,
                        "val_f1_macro": float(val_metrics.get("f1_macro", np.nan)),
                        "test_f1_macro": float(test_metrics.get("f1_macro", np.nan)),
                    })
                else:
                    if best_model_file.exists():
                        est = joblib.load(best_model_file)
                        status = "loaded"
                    else:
                        est = clone(base_model)
                        est.fit(x_tr, y_tr)
                        joblib.dump(est, best_model_file)
                        status = "trained"

                    val_preds = est.predict(x_val)
                    test_preds = est.predict(x_test)

                    val_metrics = evaluate_clf(tag, y_val, val_preds, label_encoder=label_encoder, split="val")
                    test_metrics = evaluate_clf(tag, y_test, test_preds, label_encoder=label_encoder, split="test")

                    run_metrics = {
                        "val": val_metrics,
                        "test": test_metrics,
                        "experiment_dir": str(exp_paths["root"]),
                        "best_model_file": str(best_model_file),
                    }
                    ts = now_ts()
                    with open(exp_paths["runs"] / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
                        json.dump(run_metrics, f, indent=2)
                    with open(exp_paths["runs"] / "metrics_latest.json", "w", encoding="utf-8") as f:
                        json.dump(run_metrics, f, indent=2)

                    np.save(exp_paths["runs"] / f"val_preds_{ts}.npy", val_preds)
                    np.save(exp_paths["runs"] / f"test_preds_{ts}.npy", test_preds)
                    np.save(exp_paths["runs"] / "val_preds_latest.npy", val_preds)
                    np.save(exp_paths["runs"] / "test_preds_latest.npy", test_preds)

                    pd.DataFrame([
                        {"split": "val", **val_metrics},
                        {"split": "test", **test_metrics},
                    ]).to_csv(exp_paths["runs"] / "summary_latest.csv", index=False)

                    print(
                        f"Final validation F1 (macro) [{tag}]: "
                        f"{float(val_metrics.get('f1_macro', np.nan)):.4f} ({status})"
                    )

                primary_metrics = val_metrics if primary_curve_split == "val" else test_metrics
                curve_rows.append({
                    "model": model_name,
                    "split_key": split_key,
                    "train_size": train_size,
                    "f1_macro": float(primary_metrics["f1_macro"]),
                })

                eval_rows.append({
                    "model": model_name,
                    "split_key": split_key,
                    "train_size": train_size,
                    "split": "val",
                    "accuracy": float(val_metrics.get("accuracy", np.nan)),
                    "f1_macro": float(val_metrics.get("f1_macro", np.nan)),
                    "f1_weighted": float(val_metrics.get("f1_weighted", np.nan)),
                    "roc_auc": float(val_metrics.get("roc_auc", np.nan)) if "roc_auc" in val_metrics else np.nan,
                    "model_tag": tag,
                    "model_path": str(best_model_file),
                    "status": status,
                })

                eval_rows.append({
                    "model": model_name,
                    "split_key": split_key,
                    "train_size": train_size,
                    "split": "test",
                    "accuracy": float(test_metrics.get("accuracy", np.nan)),
                    "f1_macro": float(test_metrics.get("f1_macro", np.nan)),
                    "f1_weighted": float(test_metrics.get("f1_weighted", np.nan)),
                    "roc_auc": float(test_metrics.get("roc_auc", np.nan)) if "roc_auc" in test_metrics else np.nan,
                    "model_tag": tag,
                    "model_path": str(best_model_file),
                    "status": status,
                })

                model_registry[f"{model_name}|{split_key}"] = str(best_model_file)

                pbar.set_postfix({"val_f1": f"{float(val_metrics.get('f1_macro', np.nan)):.4f}"})
                pbar.update(1)

    if cached_summary_rows:
        cached_summary_df = (
            pd.DataFrame(cached_summary_rows)
            .sort_values(["split_key", "model"])
            .reset_index(drop=True)
        )
        cached_summary_df["train_size"] = cached_summary_df["train_size"].map(lambda n: f"{int(n):,}")
        if len(cached_summary_df) == total_steps:
            print("All baseline runs were restored from cache. Detailed per-model metric logs were suppressed.")
        else:
            print(
                f"Restored {len(cached_summary_df)} baseline run(s) from cache; "
                "uncached runs were evaluated normally."
            )
        display(cached_summary_df)

    curve_df = pd.DataFrame(curve_rows).sort_values(["model", "train_size"]).reset_index(drop=True)
    eval_df = pd.DataFrame(eval_rows).sort_values(["model", "split", "train_size"]).reset_index(drop=True)
    return curve_df, eval_df, model_registry


def collect_deep_eval_from_results(
    results: dict,
    splits: dict,
    models=("GloVe MLP", "GloVe CNN", "GloVe BiLSTM"),
    split: str = "val",
) -> pd.DataFrame:
    model_alt = "|".join(re.escape(m) for m in models)
    key_pat = re.compile(rf"^({model_alt}) \[(.+?)\] \| {re.escape(split)}$")

    rows = []
    for key, metrics in results.items():
        m = key_pat.match(key)
        if not m:
            continue

        model, split_key = m.groups()
        if split_key not in splits:
            continue

        rows.append({
            "model": model,
            "split_key": split_key,
            "train_size": len(splits[split_key]["X_train"]),
            "split": split,
            "accuracy": float(metrics.get("accuracy", np.nan)),
            "f1_macro": float(metrics.get("f1_macro", np.nan)),
            "f1_weighted": float(metrics.get("f1_weighted", np.nan)),
            "roc_auc": float(metrics.get("roc_auc", np.nan)) if "roc_auc" in metrics else np.nan,
            "results_key": key,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "model", "split_key", "train_size", "split",
            "accuracy", "f1_macro", "f1_weighted", "roc_auc", "results_key"
        ])

    eval_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["model", "split_key", "split"], keep="last")
        .sort_values(["model", "train_size"])
        .reset_index(drop=True)
    )
    return eval_df


def build_deep_curve_from_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(columns=["model", "split_key", "train_size", "f1_macro"])

    return (
        eval_df[["model", "split_key", "train_size", "f1_macro"]]
        .copy()
        .sort_values(["model", "train_size"])
        .reset_index(drop=True)
    )


def render_deep_learning_curves_from_results(
    results: dict,
    splits: dict,
    cfg: dict,
    models=("GloVe MLP", "GloVe CNN", "GloVe BiLSTM"),
    split: str = "val",
    title: str = "Deep Model Learning Curves (GloVe MLP/CNN/BiLSTM)",
    save_name: str = "deep_learning_curves_glove.png",
):
    deep_eval_df = collect_deep_eval_from_results(results, splits, models=models, split=split)
    deep_curve_df = build_deep_curve_from_eval(deep_eval_df)

    if deep_curve_df.empty:
        print("No deep model entries found in RESULTS for the requested models/split.")
        return deep_curve_df, deep_eval_df

    for row in deep_curve_df.itertuples(index=False):
        results[f"LC | {row.model} | {row.split_key}"] = {"f1_macro": float(row.f1_macro)}

    display(deep_curve_df)
    display(deep_eval_df)

    plot_learning_curve_from_metrics(
        deep_curve_df,
        title=title,
        save_path=cfg["output_paths"]["plots_learning"] / save_name,
    )

    return deep_curve_df, deep_eval_df


def tune_baselines_on_reference_split(
    models: dict,
    split: dict,
    seed: int = 42,
    n_iter: int = 4,
):
    return tune_baselines_on_100k(
        models=models,
        split_100k=split,
        seed=seed,
        n_iter=n_iter,
    )


def get_logger(model_name: str) -> logging.Logger:
    log_path = CFG["log_dir"] / f"{CFG['task']}_{model_name.replace(' ', '_')}.log"

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _train_epoch(model, loader, optimizer, criterion, device):
    """One training pass. Returns average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(seqs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(loader.dataset)


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        total_loss += criterion(logits, labels).item() * seqs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1_macro, np.array(all_preds)


def load_torch_checkpoint(model, ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def train_model(model, loader_tr, loader_va, model_name: str,
                epochs: int = 20, lr: float = 1e-3,
                patience: int = 3, device: str = "cpu",
                resume: bool = True,
                ckpt_dir: Path = None,
                best_ckpt_path: Path = None,
                last_ckpt_path: Path = None,
                tensorboard_dir: Path = None) -> dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    logger = get_logger(model_name)

    safe_name = slugify(model_name)
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else (CFG["model_dir"] / safe_name / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = Path(best_ckpt_path) if best_ckpt_path is not None else (ckpt_dir / "best.pt")
    last_ckpt_path = Path(last_ckpt_path) if last_ckpt_path is not None else (ckpt_dir / "last.pt")
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if tensorboard_dir is None:
        tensorboard_dir = CFG["log_dir"] / "tensorboard" / f"{CFG['task']}_{model_name.replace(' ', '_')}"
    tensorboard_dir = Path(tensorboard_dir)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    best_f1 = -np.inf
    no_improve = 0
    start_epoch = 1

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt.get("history", history)
        best_f1 = ckpt.get("best_f1", best_f1)
        no_improve = ckpt.get("no_improve", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {last_ckpt_path} (epoch={start_epoch-1}, best_f1={best_f1:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss = _train_epoch(model, loader_tr, optimizer, criterion, device)
        va_loss, va_f1, _ = _eval_epoch(model, loader_va, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_f1_macro"].append(va_f1)
        scheduler.step(va_f1)

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalar("val/f1_macro", va_f1, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        last_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "no_improve": no_improve,
            "history": history,
            "model_name": model_name,
            "task": CFG["task"],
        }
        torch.save(last_payload, last_ckpt_path)

        if va_f1 > best_f1:
            best_f1 = va_f1
            no_improve = 0
            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_f1": best_f1,
                "history": history,
                "model_name": model_name,
                "task": CFG["task"],
            }
            torch.save(best_payload, best_ckpt_path)
            tag = " ✓  <-- new best"
        else:
            no_improve += 1
            tag = ""

        logger.info(
            f"Epoch {epoch:>3}/{epochs} | tr_loss={tr_loss:.4f} | "
            f"va_loss={va_loss:.4f} | val_f1={va_f1:.4f} | {elapsed:.1f}s{tag}"
        )

        if no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        logger.info(f"Restored best checkpoint | val_f1={best_ckpt['best_f1']:.4f} | path={best_ckpt_path}")

    writer.close()
    logger.info("Training complete.")
    return history


def predict(model, loader, device: str = "cpu") -> np.ndarray:
    model.eval()
    all_preds = []
    for seqs, _ in loader:
        all_preds.extend(model(seqs.to(device)).argmax(dim=1).cpu().numpy())
    return np.array(all_preds)


def run_torch_experiment(
    model_label: str,
    family_slug: str,
    split_key: str,
    split_data: dict,
    model_builder,
    label_encoder,
):
    exp_paths = resolve_experiment_paths(
        model_family=family_slug,
        dataset=CFG.get("experiment_dataset_slug", slugify(CFG.get("task", "dataset"))),
        size_tag=size_tag_from_split(split_key),
        model_root=CFG["model_dir"],
    )

    legacy_dir = CFG["model_dir"] / f"{model_label.replace(' ', '_')}_[{split_key}]"
    best_ckpt_path = bootstrap_file_from_legacy(
        exp_paths["best_model"] / "best.pt",
        [legacy_dir / "best.pt"],
    )
    last_ckpt_path = bootstrap_file_from_legacy(
        exp_paths["checkpoints"] / "last.pt",
        [legacy_dir / "last.pt"],
    )

    run_name = f"{model_label} [{split_key}]"
    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    use_cached_metrics = CFG.get("prefer_cached_internal_metrics", True)

    def _clean_metrics(metrics: dict) -> dict:
        clean = {}
        if not isinstance(metrics, dict):
            return clean
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                clean[key] = float(value)
        return clean

    if use_cached_metrics:
        cached_payload, val_preds, test_preds, cached_path = load_cached_internal_run_artifacts(exp_paths["runs"])
        if cached_payload is not None:
            val_metrics = _clean_metrics(cached_payload.get("val", {}))
            test_metrics = _clean_metrics(cached_payload.get("test", {}))
            if val_metrics and test_metrics:
                plot_history(history, run_name)
                RESULTS[f"{run_name} | val"] = val_metrics
                RESULTS[f"{run_name} | test"] = test_metrics

                if len(val_preds) == len(split_data["y_val"]) and len(test_preds) == len(split_data["y_test"]):
                    print(f"{run_name}: loaded cached predictions -> val_preds_latest.npy, test_preds_latest.npy")
                else:
                    print(f"{run_name}: cached predictions not found -> metrics-only return")

                print(f"{run_name}: loaded cached metrics -> skip evaluation ({cached_path})")
                print(f"{run_name} | VAL metrics: {val_metrics}")
                print(f"{run_name} | TEST metrics: {test_metrics}")
                print(
                    f"Final validation F1 (macro) [{run_name}]: "
                    f"{float(val_metrics.get('f1_macro', np.nan)):.4f} (cached)"
                )

                return None, history, val_metrics, test_metrics, np.asarray(test_preds)

    set_seed(CFG["seed"])
    train_mode = CFG.get("dl_train_mode", True)
    model = model_builder()

    if best_ckpt_path.exists():
        ckpt = load_torch_checkpoint(model, best_ckpt_path, CFG["device"])
        history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_f1_macro": []})
        print(f"{run_name}: best checkpoint found -> skip retraining ({best_ckpt_path})")
    elif train_mode:
        history = train_model(
            model,
            split_data["loader_train"],
            split_data["loader_val"],
            model_name=run_name,
            epochs=CFG["dl_epochs"],
            lr=CFG["dl_lr"],
            patience=CFG["dl_patience"],
            device=CFG["device"],
            resume=True,
            ckpt_dir=exp_paths["checkpoints"],
            best_ckpt_path=best_ckpt_path,
            last_ckpt_path=last_ckpt_path,
            tensorboard_dir=exp_paths["runs"] / "tensorboard",
        )
    elif last_ckpt_path.exists():
        ckpt = load_torch_checkpoint(model, last_ckpt_path, CFG["device"])
        history = ckpt.get("history", {"train_loss": [], "val_loss": [], "val_f1_macro": []})
        print(f"{run_name}: last checkpoint found -> evaluation only ({last_ckpt_path})")
    else:
        raise FileNotFoundError(
            f"{run_name}: no checkpoint found while CFG['dl_train_mode']=False. "
            "Enable training mode or provide a saved checkpoint."
        )

    model.to(CFG["device"])

    plot_history(history, run_name)

    val_preds = predict(model, split_data["loader_val"], CFG["device"])
    val_metrics = evaluate_clf(run_name, split_data["y_val"], val_preds, label_encoder=label_encoder, split="val")
    print(f"Final validation F1 (macro) [{run_name}]: {val_metrics['f1_macro']:.4f}")

    test_preds = predict(model, split_data["loader_test"], CFG["device"])
    test_metrics = evaluate_clf(run_name, split_data["y_test"], test_preds, label_encoder=label_encoder, split="test")

    ts = now_ts()
    run_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "experiment_dir": str(exp_paths["root"]),
        "best_checkpoint": str(best_ckpt_path),
    }
    with open(exp_paths["runs"] / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)
    with open(exp_paths["runs"] / "metrics_latest.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    np.save(exp_paths["runs"] / f"val_preds_{ts}.npy", val_preds)
    np.save(exp_paths["runs"] / f"test_preds_{ts}.npy", test_preds)
    np.save(exp_paths["runs"] / "val_preds_latest.npy", val_preds)
    np.save(exp_paths["runs"] / "test_preds_latest.npy", test_preds)

    pd.DataFrame([
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(exp_paths["runs"] / "summary_latest.csv", index=False)

    return model, history, val_metrics, test_metrics, test_preds


def run_cross_dataset_transformer_eval(
    display_name: str,
    data_bundle: dict,
    output_dir: Path,
    best_model_dir: Path,
    data_paths=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cached_metric_paths = [
        output_dir / "metrics_latest.json",
    ]

    def _load_cached_metrics():
        for p in cached_metric_paths:
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and "eval" in payload:
                    return payload, p
            except Exception as e:
                print(f"{display_name}: could not load cached metrics from {p}: {e}")
        return None, None

    def _load_latest_eval_preds():
        latest_file = output_dir / "eval_preds_latest.npy"
        if latest_file.exists():
            return np.load(latest_file), latest_file

        candidates = sorted(output_dir.glob("eval_preds_*.npy"))
        if candidates:
            return np.load(candidates[-1]), candidates[-1]

        return np.asarray([], dtype=np.int64), None

    def _register_eval_metrics(metrics: dict):
        if not isinstance(metrics, dict):
            return {}
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v
        RESULTS[f"{display_name} | eval"] = clean_metrics
        return clean_metrics

    skip_eval_with_cache = not CFG.get("run_transformer_eval", True)
    if skip_eval_with_cache:
        cached_metrics, cached_path = _load_cached_metrics()
        if cached_metrics is not None:
            eval_metrics = _register_eval_metrics(cached_metrics.get("eval", {}))
            eval_preds, eval_pred_path = _load_latest_eval_preds()

            if eval_pred_path is not None:
                print(f"{display_name}: loaded cached predictions -> {eval_pred_path.name}")
                plot_confusion_matrix_clf(data_bundle["y_eval"], eval_preds, display_name, le)
            else:
                print(f"{display_name}: cached predictions not found -> metrics-only return")
                cm_path = CFG["output_paths"]["plots_confusion"] / f"cm_{slugify(display_name)}.png"
                if display_saved_plot(cm_path, title=f"Confusion Matrix — {display_name}", figsize=(13, 5)):
                    print(f"Displayed existing confusion matrix: {cm_path}")

            print(f"{display_name}: loaded cached metrics -> skip evaluation ({cached_path})")
            print(f"{display_name} | EVAL metrics: {eval_metrics}")
            eval_f1 = float(eval_metrics.get("f1_macro", np.nan)) if isinstance(eval_metrics, dict) else np.nan
            print(f"Final evaluation F1 (macro) [{display_name}]: {eval_f1:.4f} (cached)")

            return {
                "model": display_name,
                "metrics": cached_metrics,
                "preds": {"eval": np.asarray(eval_preds)},
                "bundle_name": data_bundle["name"],
            }

        print(f"{display_name}: run_transformer_eval=False but cached metrics not found -> running evaluation")

    eval_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir), use_fast=True)
    eval_ds = HFTextDataset(
        data_bundle["X_eval_raw"], data_bundle["y_eval"], eval_tokenizer, CFG["max_len"]
    )
    eval_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))

    eval_args_dict = dict(
        output_dir=str(output_dir / "eval_tmp"),
        per_device_eval_batch_size=CFG["batch_size"],
        dataloader_num_workers=0,
        report_to="none",
        seed=CFG["seed"],
        fp16=False,
        bf16=False,
    )
    if "eval_strategy" in TrainingArguments.__dataclass_fields__:
        eval_args_dict["eval_strategy"] = "no"
    else:
        eval_args_dict["evaluation_strategy"] = "no"

    eval_trainer = Trainer(
        model=eval_model,
        args=TrainingArguments(**eval_args_dict),
        compute_metrics=compute_metrics,
    )

    eval_output = eval_trainer.predict(eval_ds)
    eval_preds = np.argmax(eval_output.predictions, axis=1)
    eval_metrics = evaluate_clf(
        display_name, data_bundle["y_eval"], eval_preds, label_encoder=le, split="eval"
    )
    print(f"Final evaluation F1 (macro) [{display_name}]: {eval_metrics['f1_macro']:.4f}")
    print(f"{display_name} | EVAL metrics: {eval_metrics}")

    plot_confusion_matrix_clf(data_bundle["y_eval"], eval_preds, display_name, le)

    run_metrics = {
        "eval": eval_metrics,
        "best_model_dir": str(best_model_dir),
        "dataset": data_bundle["name"],
        "evaluation_dir": str(output_dir),
    }
    if isinstance(data_paths, (list, tuple)):
        run_metrics["data_paths"] = [str(p) for p in data_paths]
    elif data_paths is not None:
        run_metrics["data_path"] = str(data_paths)

    ts = now_ts()
    with open(output_dir / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)
    with open(output_dir / "metrics_latest.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    np.save(output_dir / f"eval_preds_{ts}.npy", eval_preds)
    np.save(output_dir / "eval_preds_latest.npy", eval_preds)

    pd.DataFrame([
        {"split": "eval", **eval_metrics},
    ]).to_csv(output_dir / "summary_latest.csv", index=False)

    return {
        "model": display_name,
        "metrics": run_metrics,
        "preds": {"eval": eval_preds},
        "bundle_name": data_bundle["name"],
    }


def _make_training_args(common_args: dict, eval_mode: str, eval_steps=None, save_steps=None):
    args = dict(common_args)

    if eval_mode == "steps":
        args["save_strategy"] = "steps"
        args["save_steps"] = int(save_steps or eval_steps or 1000)
        if "eval_strategy" in TrainingArguments.__dataclass_fields__:
            args["eval_strategy"] = "steps"
            args["eval_steps"] = int(eval_steps or save_steps or 1000)
        else:
            args["evaluation_strategy"] = "steps"
            args["eval_steps"] = int(eval_steps or save_steps or 1000)
    else:
        args["save_strategy"] = "epoch"
        if "eval_strategy" in TrainingArguments.__dataclass_fields__:
            args["eval_strategy"] = "epoch"
        else:
            args["evaluation_strategy"] = "epoch"

    return TrainingArguments(**args)


def bootstrap_transformer_checkpoints(dst_dir: Path, legacy_candidates=()):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.glob("checkpoint-*")):
        return

    for cand in legacy_candidates:
        cand = Path(cand)
        if not cand.exists() or not cand.is_dir():
            continue

        ckpts = sorted(cand.glob("checkpoint-*"))
        if not ckpts and cand.name.startswith("checkpoint-"):
            ckpts = [cand]
        if not ckpts:
            continue

        for ckpt in ckpts:
            target = dst_dir / ckpt.name
            if target.exists():
                continue
            try:
                os.symlink(ckpt.resolve(), target, target_is_directory=True)
            except Exception:
                shutil.copytree(ckpt, target)

        print(f"Bootstrapped transformer checkpoints from legacy: {cand}")
        break


def run_transformer_experiment(
    display_name: str,
    family_slug: str,
    size_tag: str,
    model_name: str,
    data_bundle: dict,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 2,
    eval_mode: str = "epoch",
    eval_steps=None,
    save_steps=None,
    legacy_best_dirs=(),
    legacy_checkpoint_dirs=(),
):
    exp_paths = resolve_experiment_paths(
        model_family=family_slug,
        dataset=CFG.get("experiment_dataset_slug", slugify(CFG.get("task", "dataset"))),
        size_tag=size_tag,
        model_root=CFG["model_dir"],
    )

    bootstrap_dir_from_legacy(exp_paths["best_model"], legacy_best_dirs)
    bootstrap_transformer_checkpoints(exp_paths["checkpoints"], legacy_checkpoint_dirs)

    best_model_dir = exp_paths["best_model"]
    has_best_model = (best_model_dir / "config.json").exists()

    runs_dir = exp_paths["runs"]
    transformer_metrics_dir = CFG["output_paths"]["metrics"] / "transformers"
    transformer_metrics_dir.mkdir(parents=True, exist_ok=True)

    cached_metric_paths = [
        runs_dir / "metrics_latest.json",
        transformer_metrics_dir / f"{slugify(display_name)}_latest.json",
    ]

    def _load_cached_metrics():
        for p in cached_metric_paths:
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and "val" in payload and "test" in payload:
                    return payload, p
            except Exception as e:
                print(f"{display_name}: could not load cached metrics from {p}: {e}")
        return None, None

    def _load_latest_preds(split_name: str):
        latest_file = runs_dir / f"{split_name}_preds_latest.npy"
        if latest_file.exists():
            return np.load(latest_file), latest_file

        candidates = sorted(runs_dir.glob(f"{split_name}_preds_*.npy"))
        if candidates:
            return np.load(candidates[-1]), candidates[-1]

        return np.asarray([], dtype=np.int64), None

    def _register_split_metrics(split_name: str, metrics: dict):
        if not isinstance(metrics, dict):
            return
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v
        RESULTS[f"{display_name} | {split_name}"] = clean_metrics

    skip_eval_with_cache = has_best_model and (not CFG.get("run_transformer_eval", True))
    if skip_eval_with_cache:
        cached_metrics, cached_path = _load_cached_metrics()
        if cached_metrics is not None:
            val_metrics = cached_metrics.get("val", {})
            test_metrics = cached_metrics.get("test", {})

            _register_split_metrics("val", val_metrics)
            _register_split_metrics("test", test_metrics)

            val_preds, val_pred_path = _load_latest_preds("val")
            test_preds, test_pred_path = _load_latest_preds("test")

            if val_pred_path is not None and test_pred_path is not None:
                print(
                    f"{display_name}: loaded cached predictions -> "
                    f"{val_pred_path.name}, {test_pred_path.name}"
                )
            else:
                print(f"{display_name}: cached predictions not found -> metrics-only return")

            val_f1 = float(val_metrics.get("f1_macro", np.nan)) if isinstance(val_metrics, dict) else np.nan
            print(f"{display_name}: loaded cached metrics -> skip evaluation ({cached_path})")
            print(f"{display_name} | VAL metrics: {val_metrics}")
            print(f"{display_name} | TEST metrics: {test_metrics}")
            print(f"Final validation F1 (macro) [{display_name}]: {val_f1:.4f} (cached)")

            return {
                "name": display_name,
                "val_preds": np.asarray(val_preds),
                "test_preds": np.asarray(test_preds),
                "y_val": data_bundle["y_val"],
                "y_test": data_bundle["y_test"],
                "df_test": data_bundle["df_test"].copy().reset_index(drop=True),
                "best_dir": best_model_dir,
                "paths": exp_paths,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }

        print(f"{display_name}: run_transformer_eval=False but cached metrics not found -> running evaluation")

    tokenizer_src = str(best_model_dir) if has_best_model else model_name
    trf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)

    trf_ds_train = HFTextDataset(data_bundle["X_train_raw"], data_bundle["y_train"], trf_tokenizer, CFG["max_len"])
    trf_ds_val = HFTextDataset(data_bundle["X_val_raw"], data_bundle["y_val"], trf_tokenizer, CFG["max_len"])
    trf_ds_test = HFTextDataset(data_bundle["X_test_raw"], data_bundle["y_test"], trf_tokenizer, CFG["max_len"])

    label_names = list(getattr(le, "classes_", CLASSES))
    id2label = {i: label_names[i] for i in range(len(label_names))}
    label2id = {label: i for i, label in id2label.items()}

    if not has_best_model:
        trf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=CFG["num_labels"],
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            from_tf=False,
        )

        common_args = dict(
            output_dir=str(exp_paths["checkpoints"]),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_dir=str(exp_paths["runs"] / "tensorboard"),
            logging_strategy="steps",
            logging_steps=100,
            report_to="tensorboard",
            dataloader_num_workers=0,
            fp16=(CFG["device"] == "cuda"),
            bf16=False,
            seed=CFG["seed"],
        )

        training_args = _make_training_args(common_args, eval_mode=eval_mode, eval_steps=eval_steps, save_steps=save_steps)

        trainer = Trainer(
            model=trf_model,
            args=training_args,
            train_dataset=trf_ds_train,
            eval_dataset=trf_ds_val,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        last_ckpt = get_last_checkpoint(str(exp_paths["checkpoints"]))
        print(f"{display_name}: last checkpoint -> {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt is not None else None)

        trainer.save_model(str(best_model_dir))
        trf_tokenizer.save_pretrained(str(best_model_dir))

        hist = trainer.state.log_history
        ts = now_ts()
        with open(exp_paths["runs"] / f"trainer_log_history_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        pd.DataFrame(hist).to_csv(exp_paths["runs"] / f"trainer_log_history_{ts}.csv", index=False)

        print(f"{display_name}: training complete and best model saved -> {best_model_dir}")
    else:
        print(f"{display_name}: best model already exists -> skip retraining ({best_model_dir})")

    eval_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))

    eval_args_dict = dict(
        output_dir=str(exp_paths["runs"] / "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        report_to="none",
        seed=CFG["seed"],
        fp16=False,
        bf16=False,
    )
    if "eval_strategy" in TrainingArguments.__dataclass_fields__:
        eval_args_dict["eval_strategy"] = "no"
    else:
        eval_args_dict["evaluation_strategy"] = "no"

    eval_trainer = Trainer(
        model=eval_model,
        args=TrainingArguments(**eval_args_dict),
        compute_metrics=compute_metrics,
    )

    val_output = eval_trainer.predict(trf_ds_val)
    test_output = eval_trainer.predict(trf_ds_test)

    val_preds = np.argmax(val_output.predictions, axis=1)
    test_preds = np.argmax(test_output.predictions, axis=1)

    val_metrics = evaluate_clf(display_name, data_bundle["y_val"], val_preds, label_encoder=le, split="val")
    test_metrics = evaluate_clf(display_name, data_bundle["y_test"], test_preds, label_encoder=le, split="test")
    print(f"Final validation F1 (macro) [{display_name}]: {val_metrics['f1_macro']:.4f}")

    plot_confusion_matrix_clf(data_bundle["y_test"], test_preds, display_name, le)

    run_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_model_dir": str(best_model_dir),
        "checkpoints_dir": str(exp_paths["checkpoints"]),
        "experiment_dir": str(exp_paths["root"]),
    }
    ts = now_ts()
    with open(exp_paths["runs"] / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)
    with open(exp_paths["runs"] / "metrics_latest.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    with open(transformer_metrics_dir / f"{slugify(display_name)}_latest.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    np.save(exp_paths["runs"] / f"val_preds_{ts}.npy", val_preds)
    np.save(exp_paths["runs"] / f"test_preds_{ts}.npy", test_preds)
    np.save(exp_paths["runs"] / "val_preds_latest.npy", val_preds)
    np.save(exp_paths["runs"] / "test_preds_latest.npy", test_preds)

    return {
        "name": display_name,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "y_val": data_bundle["y_val"],
        "y_test": data_bundle["y_test"],
        "df_test": data_bundle["df_test"].copy().reset_index(drop=True),
        "best_dir": best_model_dir,
        "paths": exp_paths,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
