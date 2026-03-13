
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from .data import now_ts

CFG: dict = {}
le = None
shared_vocab: dict = {}


RESULTS: dict[str, dict] = {}


class DictLabelEncoder:
    def __init__(self, mapping: dict):
        first_key = next(iter(mapping))
        if isinstance(first_key, int):
            id2class = mapping
        else:
            id2class = {value: key for key, value in mapping.items()}
        self.classes_ = np.array([id2class[index] for index in sorted(id2class)])

    def inverse_transform(self, y):
        y_arr = np.asarray(y, dtype=np.int64)
        return np.array([self.classes_[i] if 0 <= i < len(self.classes_) else str(i) for i in y_arr])

    def transform(self, labels):
        class_mapping = {label: index for index, label in enumerate(self.classes_)}
        labels_arr = np.asarray(labels)
        return np.array([class_mapping[label] for label in labels_arr], dtype=np.int64)


def get_top_models_df(
    results: dict,
    split: str = 'test',
    metric: str = 'f1_macro',
    top_n: int = 10,
) -> pd.DataFrame:
    suffix = f' | {split}'
    rows = []
    for full_name, metrics in results.items():
        if not full_name.endswith(suffix):
            continue
        model_name = full_name[:-len(suffix)]
        if model_name.startswith('LC |'):
            continue
        rows.append({
            'model': model_name,
            'accuracy': float(metrics.get('accuracy', np.nan)),
            'f1_macro': float(metrics.get('f1_macro', np.nan)),
            'f1_weighted': float(metrics.get('f1_weighted', np.nan)),
        })
    if not rows:
        return pd.DataFrame(columns=['model', 'accuracy', 'f1_macro', 'f1_weighted'])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=['model'], keep='last')
        .sort_values([metric, 'f1_weighted', 'accuracy', 'model'], ascending=[False, False, False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def print_top_models(results: dict, split: str = 'test', metric: str = 'f1_macro', top_n: int = 10) -> pd.DataFrame:
    top_df = get_top_models_df(results, split=split, metric=metric, top_n=top_n)
    if top_df.empty:
        print(f"No models found for split='{split}' and metric='{metric}'.")
        return top_df
    print(f"\nTop-{top_n} models | split={split} | sorted by {metric}")
    display(top_df.round(4))
    return top_df


def print_results_table(results: dict | None = None):
    results = RESULTS if results is None else results
    if not results:
        print('RESULTS is empty — run evaluate_clf first.')
        return
    all_cols = sorted({key for metrics in results.values() for key in metrics.keys()})
    header = f"{'Model':<45}" + ''.join(f"{col:>14}" for col in all_cols)
    sep = '=' * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, metrics in results.items():
        row = f"{name:<45}" + ''.join(f"{metrics.get(col, float('nan')):>14.4f}" for col in all_cols)
        print(row)
    print(sep)


def save_artefacts(cfg: dict | None = None, label_encoder=None, vocab=None):
    cfg = cfg or CFG
    label_encoder = label_encoder if label_encoder is not None else le
    vocab = vocab if vocab is not None else shared_vocab

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    metrics_dir = cfg['output_paths']['metrics']
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg_serializable = {key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()}
    with open(metrics_dir / f"cfg_{timestamp}.json", 'w', encoding='utf-8') as handle:
        json.dump(cfg_serializable, handle, indent=2, default=str)

    with open(metrics_dir / f"results_{timestamp}.json", 'w', encoding='utf-8') as handle:
        json.dump(RESULTS, handle, indent=2)

    if RESULTS:
        df_results = pd.DataFrame(RESULTS).T.rename_axis('Model | Split')
        if 'f1_macro' in df_results.columns:
            df_results = df_results.sort_values('f1_macro', ascending=False)
        df_results.to_csv(metrics_dir / f"results_{cfg['task']}.csv")

    shared_model_dir = cfg['model_dir'] / 'shared'
    shared_model_dir.mkdir(parents=True, exist_ok=True)
    if label_encoder not in (None, False):
        joblib.dump(label_encoder, shared_model_dir / 'label_encoder.pkl')
    if vocab not in (None, False):
        joblib.dump(vocab, shared_model_dir / 'vocab.pkl')

    print(f"Artefacts saved  ->  {metrics_dir}  |  {shared_model_dir}")


def evaluate_clf(model_name: str, y_true, y_pred,
                 y_prob=None, label_encoder=None, split: str = "test") -> dict:

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = (roc_auc_score(y_true, y_prob[:, 1]) if (y_prob is not None and y_prob.shape[1] == 2) else None)

    print(f"\n{'='*55}")
    print(f"  {model_name} | {split.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weight:.4f}")
    if auc is not None:
        print(f"  ROC-AUC       : {auc:.4f}")

    if label_encoder is not None:
        class_names = list(label_encoder.classes_)
        labels = np.arange(len(class_names))
        report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    else:
        report = classification_report(y_true, y_pred, zero_division=0)

    print()
    print(report)

    metrics = {"accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weight)}
    if auc is not None:
        metrics["roc_auc"] = float(auc)

    RESULTS[f"{model_name} | {split}"] = metrics
    return metrics


def compute_metrics(eval_pred) -> dict:
    """
    Passed to HuggingFace Trainer. Returns F1-macro and accuracy.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def error_analysis(df_split: pd.DataFrame, y_true, y_pred,
                   label_encoder=None, n: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame of the top-n misclassified samples.

    Parameters
    ----------
    df_split      : DataFrame for the relevant split (e.g. df_test)
    y_true, y_pred: encoded labels
    label_encoder : to decode label names
    n             : number of examples to display
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred length mismatch: {len(y_true)} vs {len(y_pred)}")

    err_mask = y_true != y_pred
    df_err = df_split[err_mask].copy()
    df_err["y_true"] = y_true[err_mask]
    df_err["y_pred"] = y_pred[err_mask]

    if label_encoder is not None:
        if hasattr(label_encoder, "inverse_transform"):
            df_err["true_label"] = label_encoder.inverse_transform(df_err["y_true"])
            df_err["pred_label"] = label_encoder.inverse_transform(df_err["y_pred"])
        elif hasattr(label_encoder, "classes_"):
            classes = np.asarray(label_encoder.classes_)

            def decode_ids(arr):
                arr = np.asarray(arr, dtype=np.int64)
                return np.array([classes[i] if 0 <= i < len(classes) else str(i) for i in arr])

            df_err["true_label"] = decode_ids(df_err["y_true"])
            df_err["pred_label"] = decode_ids(df_err["y_pred"])
        else:
            df_err["true_label"] = df_err["y_true"]
            df_err["pred_label"] = df_err["y_pred"]

        cols = [CFG["text_col"], "true_label", "pred_label"]
    else:
        cols = [CFG["text_col"], "y_true", "y_pred"]

    return df_err[cols].head(n)


def _coerce_metric_dict(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}

    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def register_results_metrics(display_name: str, split: str, metrics: dict) -> dict:
    clean = _coerce_metric_dict(metrics)
    RESULTS[f"{display_name} | {split}"] = clean
    return clean


def load_cached_eval_artifacts(output_dir: Path):
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics_latest.json"
    preds_path = output_dir / "eval_preds_latest.npy"

    if not metrics_path.exists():
        return None, np.asarray([], dtype=np.int64), None

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Could not load cached evaluation metrics from {metrics_path}: {e}")
        return None, np.asarray([], dtype=np.int64), None

    preds = np.asarray([], dtype=np.int64)
    if preds_path.exists():
        try:
            preds = np.load(preds_path)
        except Exception as e:
            print(f"Could not load cached evaluation predictions from {preds_path}: {e}")

    return payload, preds, metrics_path


def persist_eval_artifacts(output_dir: Path, metrics: dict, preds, meta=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_metrics = _coerce_metric_dict(metrics)
    payload = {"eval": clean_metrics}
    if meta:
        payload.update(meta)

    ts = now_ts()
    with open(output_dir / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(output_dir / "metrics_latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    preds_arr = np.asarray(preds, dtype=np.int64)
    np.save(output_dir / f"eval_preds_{ts}.npy", preds_arr)
    np.save(output_dir / "eval_preds_latest.npy", preds_arr)

    pd.DataFrame([
        {"split": "eval", **clean_metrics},
    ]).to_csv(output_dir / "summary_latest.csv", index=False)

    return payload


def maybe_predict_proba(estimator, X):
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        return estimator.predict_proba(X)
    except Exception:
        return None
