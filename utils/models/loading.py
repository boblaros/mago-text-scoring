
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from ..data import bootstrap_file_from_legacy, load_pickle_with_fallback, resolve_experiment_paths, size_tag_from_split, slugify
from ..metrics import DictLabelEncoder
from ..training import load_torch_checkpoint

CFG: dict = {}
class2id: dict = {}

TRANSFORMER_SUMMARY_COLUMNS = [
    'experiment', 'val_f1_macro', 'test_f1_macro', 'val_accuracy', 'test_accuracy', 'best_model', 'experiment_dir'
]

def load_saved_results_df(cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or CFG
    metrics_dir = cfg["output_paths"]["metrics"]
    pkl_path = metrics_dir / f"results_{cfg['task']}.pkl"
    csv_path = metrics_dir / f"results_{cfg['task']}.csv"
    if pkl_path.exists():
        try:
            obj = joblib.load(pkl_path)
            if isinstance(obj, pd.DataFrame):
                return obj
        except Exception as e:
            print(f"Could not load saved results from {pkl_path}: {e}")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_saved_label_encoder(shared_model_dir: Path | None = None, class_mapping: dict | None = None):
    shared_model_dir = Path(shared_model_dir) if shared_model_dir is not None else CFG["model_dir"] / "shared"
    class_mapping = class_mapping or globals().get("class2id")

    pkl_path = shared_model_dir / "label_encoder.pkl"
    classes_path = shared_model_dir / "label_classes.json"
    mapping_path = shared_model_dir / "class2id.json"

    if pkl_path.exists():
        try:
            return joblib.load(pkl_path)
        except Exception as e:
            print(f"Could not load saved label encoder from {pkl_path}: {e}")

    if classes_path.exists():
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = json.load(f)
            if isinstance(classes, list) and classes:
                return DictLabelEncoder({label: idx for idx, label in enumerate(classes)})
        except Exception as e:
            print(f"Could not load saved label classes from {classes_path}: {e}")

    if class_mapping is None and mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                class_mapping = {str(k): int(v) for k, v in json.load(f).items()}
        except Exception as e:
            print(f"Could not load saved class mapping from {mapping_path}: {e}")

    if class_mapping is not None:
        return DictLabelEncoder(class_mapping)

    raise FileNotFoundError(
        f"No reusable label encoder metadata found under {shared_model_dir}."
    )


def transformer_summary_from_runs(transformer_runs: dict | None) -> pd.DataFrame:
    rows = []
    for experiment, payload in (transformer_runs or {}).items():
        if not isinstance(payload, dict):
            continue
        rows.append({
            "experiment": str(experiment),
            "val_f1_macro": float(payload.get("val_metrics", {}).get("f1_macro", np.nan)),
            "test_f1_macro": float(payload.get("test_metrics", {}).get("f1_macro", np.nan)),
            "val_accuracy": float(payload.get("val_metrics", {}).get("accuracy", np.nan)),
            "test_accuracy": float(payload.get("test_metrics", {}).get("accuracy", np.nan)),
            "best_model": str(payload.get("best_dir", "") or ""),
            "experiment_dir": str(payload.get("paths", {}).get("root", "") or ""),
        })
    return pd.DataFrame(rows, columns=TRANSFORMER_SUMMARY_COLUMNS)


def merge_transformer_summary_frames(*frames) -> pd.DataFrame:
    prepared = []
    for priority, frame in enumerate(frames):
        if not isinstance(frame, pd.DataFrame) or frame.empty or "experiment" not in frame.columns:
            continue
        df = frame.copy()
        for col in TRANSFORMER_SUMMARY_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan if col not in {"experiment", "best_model", "experiment_dir"} else ""
        df = df[TRANSFORMER_SUMMARY_COLUMNS]
        for col in ["val_f1_macro", "test_f1_macro", "val_accuracy", "test_accuracy"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["experiment", "best_model", "experiment_dir"]:
            df[col] = df[col].fillna("").astype(str)
        df = df[df["experiment"].str.len() > 0].copy()
        if df.empty:
            continue
        df["_source_priority"] = priority
        df["_non_nulls"] = df.notna().sum(axis=1)
        prepared.append(df)

    if not prepared:
        return pd.DataFrame(columns=TRANSFORMER_SUMMARY_COLUMNS)

    combined = pd.concat(prepared, ignore_index=True)
    combined = (
        combined
        .sort_values(["experiment", "_source_priority", "_non_nulls"], ascending=[True, True, False])
        .drop_duplicates(subset=["experiment"], keep="first")
        .drop(columns=["_source_priority", "_non_nulls"])
        .sort_values(["val_f1_macro", "test_f1_macro", "experiment"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return combined


def load_saved_transformer_summary(cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or CFG
    summary_path = cfg["output_paths"]["metrics"] / "transformer_summary_metrics.csv"
    frames = []

    if summary_path.exists():
        try:
            frames.append(pd.read_csv(summary_path))
        except Exception as e:
            print(f"Could not load transformer summary from {summary_path}: {e}")

    transformer_metrics_dir = cfg["output_paths"]["metrics"] / "transformers"
    rows = []
    name_map = {
        "distilbert_train_pool": "DistilBERT_train_pool",
        "roberta_train_pool": "RoBERTa_train_pool",
        "deberta_v3_base_train_pool": "DeBERTaV3_train_pool",
    }
    if transformer_metrics_dir.exists():
        for p in sorted(transformer_metrics_dir.glob("*_latest.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rows.append({
                    "experiment": name_map.get(p.stem.replace("_latest", ""), p.stem.replace("_latest", "")),
                    "val_f1_macro": float(payload.get("val", {}).get("f1_macro", np.nan)),
                    "test_f1_macro": float(payload.get("test", {}).get("f1_macro", np.nan)),
                    "val_accuracy": float(payload.get("val", {}).get("accuracy", np.nan)),
                    "test_accuracy": float(payload.get("test", {}).get("accuracy", np.nan)),
                    "best_model": payload.get("best_model_dir", ""),
                    "experiment_dir": payload.get("experiment_dir", ""),
                })
            except Exception as e:
                print(f"Could not read transformer metrics from {p}: {e}")
    if rows:
        frames.append(pd.DataFrame(rows))

    return merge_transformer_summary_frames(*frames)


def _load_latest_run_array(runs_dir: Path | None, stem: str):
    if runs_dir is None or not Path(runs_dir).exists():
        return np.asarray([], dtype=np.int64), None
    runs_dir = Path(runs_dir)
    latest_file = runs_dir / f"{stem}_latest.npy"
    if latest_file.exists():
        return np.load(latest_file), latest_file
    candidates = sorted(runs_dir.glob(f"{stem}_*.npy"))
    if candidates:
        return np.load(candidates[-1]), candidates[-1]
    return np.asarray([], dtype=np.int64), None


def load_saved_transformer_run(experiment: str, data_bundle: dict | None = None, cfg: dict = None) -> dict | None:
    cfg = cfg or CFG
    summary_df = load_saved_transformer_summary(cfg)
    row = summary_df.loc[summary_df["experiment"] == experiment]
    row_dict = row.iloc[0].to_dict() if not row.empty else {}

    candidate_metric_paths = []
    transformer_metrics_dir = cfg["output_paths"]["metrics"] / "transformers"
    candidate_metric_paths.append(transformer_metrics_dir / f"{slugify(experiment)}_latest.json")

    experiment_dir = row_dict.get("experiment_dir", "")
    if experiment_dir:
        candidate_metric_paths.append(Path(experiment_dir) / "runs" / "metrics_latest.json")

    best_model_dir = Path(row_dict["best_model"]) if row_dict.get("best_model") else None
    if best_model_dir is not None:
        candidate_metric_paths.append(best_model_dir.parent / "runs" / "metrics_latest.json")

    metrics_payload = None
    seen = set()
    for candidate in candidate_metric_paths:
        candidate = Path(candidate)
        if str(candidate) in seen or not candidate.exists():
            continue
        seen.add(str(candidate))
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "val" in payload and "test" in payload:
                metrics_payload = payload
                break
        except Exception as e:
            print(f"Could not load saved transformer run from {candidate}: {e}")

    if metrics_payload is None and not row_dict:
        return None

    exp_root = None
    if row_dict.get("experiment_dir"):
        exp_root = Path(row_dict["experiment_dir"])
    elif metrics_payload is not None and metrics_payload.get("experiment_dir"):
        exp_root = Path(metrics_payload["experiment_dir"])
    elif best_model_dir is not None:
        exp_root = best_model_dir.parent

    runs_dir = exp_root / "runs" if exp_root is not None else None
    checkpoints_dir = exp_root / "checkpoints" if exp_root is not None else None

    if metrics_payload is None:
        metrics_payload = {
            "val": {
                "f1_macro": float(row_dict.get("val_f1_macro", np.nan)),
                "accuracy": float(row_dict.get("val_accuracy", np.nan)),
            },
            "test": {
                "f1_macro": float(row_dict.get("test_f1_macro", np.nan)),
                "accuracy": float(row_dict.get("test_accuracy", np.nan)),
            },
            "best_model_dir": row_dict.get("best_model", ""),
            "experiment_dir": row_dict.get("experiment_dir", ""),
        }

    val_preds, _ = _load_latest_run_array(runs_dir, "val_preds")
    test_preds, _ = _load_latest_run_array(runs_dir, "test_preds")

    history = {}
    history_plot = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    if exp_root is not None and (exp_root / "training_history.json").exists():
        try:
            history = json.loads((exp_root / "training_history.json").read_text())
            history_plot = {
                "train_loss": list(history.get("train_loss", [])),
                "val_loss": list(history.get("val_loss", [])),
                "val_f1_macro": list(history.get("val_f1", [])),
            }
        except Exception as e:
            print(f"Could not load saved DeBERTa history from {exp_root / 'training_history.json'}: {e}")

    return {
        "name": experiment,
        "val_metrics": metrics_payload.get("val", {}),
        "test_metrics": metrics_payload.get("test", {}),
        "val_preds": np.asarray(val_preds),
        "test_preds": np.asarray(test_preds),
        "y_val": np.asarray(data_bundle["y_val"], dtype=np.int64) if data_bundle is not None else np.asarray([], dtype=np.int64),
        "y_test": np.asarray(data_bundle["y_test"], dtype=np.int64) if data_bundle is not None else np.asarray([], dtype=np.int64),
        "df_test": data_bundle["df_test"].copy().reset_index(drop=True) if data_bundle is not None and "df_test" in data_bundle else pd.DataFrame(),
        "best_dir": best_model_dir,
        "paths": {
            "root": exp_root,
            "best_model": best_model_dir,
            "checkpoints": checkpoints_dir,
            "runs": runs_dir,
        },
        "history": history,
        "history_plot": history_plot,
        "resume_info": metrics_payload.get("resume_info", {}),
        "best_epoch": metrics_payload.get("best_epoch"),
        "best_val_f1": metrics_payload.get("best_val_f1"),
    }


def load_saved_cross_dataset_summaries(cfg: dict = None):
    cfg = cfg or CFG
    cross_dataset_root = cfg["output_dir"] / "cross_dataset_eval"
    summary_path = cross_dataset_root / "summary_latest.csv"
    family_path = cross_dataset_root / "family_summary_latest.csv"

    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    family_df = pd.read_csv(family_path) if family_path.exists() else pd.DataFrame()
    return summary_df, family_df


def load_best_torch_model(
    model_builder,
    family_slug: str,
    split_key: str,
    cfg: dict = None,
):
    cfg = cfg or CFG
    exp_paths = resolve_experiment_paths(
        model_family=family_slug,
        dataset=cfg.get("experiment_dataset_slug", slugify(cfg.get("task", "dataset"))),
        size_tag=size_tag_from_split(split_key),
        model_root=cfg["model_dir"],
    )
    best_ckpt_path = exp_paths["best_model"] / "best.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found for family='{family_slug}', split='{split_key}': {best_ckpt_path}"
        )

    model = model_builder()
    load_torch_checkpoint(model, best_ckpt_path, cfg["device"])
    model.to(cfg["device"])
    model.eval()
    return model, best_ckpt_path, exp_paths


def resolve_final_transformer_selection(cfg: dict = None):
    cfg = cfg or CFG

    current_name = globals().get("FINAL_TRANSFORMER_NAME")
    current_path = globals().get("TRANSFORMER_FINAL_BEST_PATH")
    if current_name and current_path and Path(current_path).exists():
        return current_name, Path(current_path)

    transformer_metrics_dir = cfg["output_paths"]["metrics"] / "transformers"
    metric_rows = []
    if transformer_metrics_dir.exists():
        for p in sorted(transformer_metrics_dir.glob("*_latest.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                experiment_name = p.stem.replace("_latest", "")
                experiment_name = experiment_name.replace("distilbert_train_pool", "DistilBERT_train_pool")
                experiment_name = experiment_name.replace("roberta_train_pool", "RoBERTa_train_pool")
                experiment_name = experiment_name.replace("deberta_v3_base_train_pool", "DeBERTaV3_train_pool")
                metric_rows.append({
                    "experiment": experiment_name,
                    "val_f1_macro": float(payload.get("val", {}).get("f1_macro", np.nan)),
                    "test_f1_macro": float(payload.get("test", {}).get("f1_macro", np.nan)),
                    "best_model": payload.get("best_model_dir"),
                })
            except Exception as e:
                print(f"Could not read transformer metrics from {p}: {e}")

    if metric_rows:
        metric_df = (
            pd.DataFrame(metric_rows)
            .sort_values(["val_f1_macro", "test_f1_macro", "experiment"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        best_row = metric_df.iloc[0]
        best_name = str(best_row["experiment"])
        best_path = Path(best_row["best_model"])
        if best_path.exists():
            globals()["FINAL_TRANSFORMER_NAME"] = best_name
            globals()["TRANSFORMER_FINAL_BEST_PATH"] = best_path
            return best_name, best_path

    summary_path = cfg["output_paths"]["metrics"] / "transformer_summary_metrics.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty and {"experiment", "val_f1_macro", "test_f1_macro", "best_model"}.issubset(summary_df.columns):
                best_row = (
                    summary_df
                    .sort_values(["val_f1_macro", "test_f1_macro", "experiment"], ascending=[False, False, True])
                    .iloc[0]
                )
                best_name = str(best_row["experiment"])
                best_path = Path(best_row["best_model"])
                if best_path.exists():
                    globals()["FINAL_TRANSFORMER_NAME"] = best_name
                    globals()["TRANSFORMER_FINAL_BEST_PATH"] = best_path
                    return best_name, best_path
        except Exception as e:
            print(f"Could not resolve final transformer from {summary_path}: {e}")

    raise FileNotFoundError(
        "Could not resolve a final transformer best-model directory from notebook state or saved metrics. "
        "Run Section 7 first."
    )
