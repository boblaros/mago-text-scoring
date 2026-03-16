
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data import TextSequenceDataset, prepare_encoded_text_frame, resolve_target_col, texts_to_sequences
from ..metrics import evaluate_clf, load_cached_eval_artifacts, maybe_predict_proba, persist_eval_artifacts, register_results_metrics
from ..plots import plot_confusion_matrix_clf
from ..text import normalize_text, preprocess_from_normalized
from ..training import predict

CFG: dict = {}
class2id: dict = {}

def _decode_label(pred_id: int, label_encoder=None, model_obj=None) -> str:
    if label_encoder is not None:
        if hasattr(label_encoder, "inverse_transform"):
            return str(label_encoder.inverse_transform([pred_id])[0])
        if hasattr(label_encoder, "classes_"):
            classes = np.asarray(label_encoder.classes_)
            if 0 <= pred_id < len(classes):
                return str(classes[pred_id])

    if model_obj is not None and hasattr(model_obj, "config"):
        id2label = getattr(model_obj.config, "id2label", None)
        if isinstance(id2label, dict):
            key = pred_id if pred_id in id2label else str(pred_id)
            if key in id2label:
                return str(id2label[key])

    return str(pred_id)


def predict_with_transformer_components(
    text: str,
    tokenizer_obj,
    model_obj,
    device_obj,
    cfg: dict | None = None,
    label_encoder=None,
):
    cfg = cfg or CFG
    inputs = tokenizer_obj(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.get("max_len", 256),
    ).to(device_obj)

    outputs = model_obj(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    conf, pred_id = torch.max(probs, dim=-1)

    pred_id = int(pred_id.item())
    confidence = float(conf.item())
    pred_label = _decode_label(pred_id, label_encoder, model_obj)
    return pred_label, confidence


def make_transformer_predictor(
    tokenizer_obj,
    model_obj,
    device_obj,
    cfg: dict | None = None,
    label_encoder=None,
):
    cfg = cfg or CFG

    @torch.no_grad()
    def _predict_one(text: str):
        return predict_with_transformer_components(
            text,
            tokenizer_obj,
            model_obj,
            device_obj,
            cfg=cfg,
            label_encoder=label_encoder,
        )

    return _predict_one


def predict_with_deep_components(
    text: str,
    model_obj,
    vocab: dict,
    device_obj,
    cfg: dict | None = None,
    label_encoder=None,
):
    cfg = cfg or CFG
    processed_text = preprocess_from_normalized(normalize_text(text))
    seq = texts_to_sequences([processed_text], vocab, cfg.get("max_seq_len", 256))
    seq_tensor = torch.tensor(seq, dtype=torch.long, device=device_obj)

    logits = model_obj(seq_tensor)
    probs = torch.softmax(logits, dim=-1)
    conf, pred_id = torch.max(probs, dim=-1)

    pred_id = int(pred_id.item())
    confidence = float(conf.item())
    pred_label = _decode_label(pred_id, label_encoder, model_obj)
    return pred_label, confidence


def evaluate_classical_model_on_df(
    display_name: str,
    estimator,
    df: pd.DataFrame,
    label_encoder=None,
    output_dir: Path = None,
    data_paths=None,
    model_path: Path = None,
    cfg: dict = None,
    force_recompute: bool = False,
    plot_confusion: bool = False,
    quiet_cache: bool = False,
):
    cfg = cfg or CFG
    frame = prepare_encoded_text_frame(
        df,
        cfg=cfg,
        label_mapping=class2id,
        drop_duplicates=False,
        clean_text="clean",
    )
    target_col = resolve_target_col(frame, cfg)
    X_eval = np.asarray(frame["text_clean"], dtype=object)
    y_eval = np.asarray(frame[target_col], dtype=np.int64)

    if output_dir is not None and not force_recompute:
        cached_payload, cached_preds, cached_path = load_cached_eval_artifacts(output_dir)
        if cached_payload is not None:
            eval_metrics = register_results_metrics(display_name, "eval", cached_payload.get("eval", {}))
            if plot_confusion and len(cached_preds) == len(y_eval):
                plot_confusion_matrix_clf(y_eval, cached_preds, display_name, label_encoder)
            if not quiet_cache:
                print(f"{display_name}: loaded cached metrics -> {cached_path}")
            return {
                "model": display_name,
                "metrics": {"eval": eval_metrics},
                "preds": {"eval": np.asarray(cached_preds)},
                "df_eval": frame.copy().reset_index(drop=True),
                "y_eval": y_eval,
                "bundle_name": frame.attrs.get("dataset_name", display_name),
                "cache_status": "cached",
                "cache_source": str(cached_path) if cached_path is not None else None,
            }

    if estimator is None:
        raise ValueError(f"Estimator is required to evaluate {display_name} when no cache is available.")

    eval_preds = estimator.predict(X_eval)
    eval_prob = maybe_predict_proba(estimator, X_eval)
    eval_metrics = evaluate_clf(
        display_name,
        y_eval,
        eval_preds,
        y_prob=eval_prob,
        label_encoder=label_encoder,
        split="eval",
    )

    if plot_confusion:
        plot_confusion_matrix_clf(y_eval, eval_preds, display_name, label_encoder)

    meta = {
        "dataset": frame.attrs.get("dataset_name", display_name),
    }
    if model_path is not None:
        meta["model_path"] = str(model_path)
    if isinstance(data_paths, (list, tuple)):
        meta["data_paths"] = [str(p) for p in data_paths]
    elif data_paths is not None:
        meta["data_path"] = str(data_paths)

    payload = {"eval": _coerce_metric_dict(eval_metrics)}
    if output_dir is not None:
        payload = persist_eval_artifacts(output_dir, eval_metrics, eval_preds, meta=meta)

    return {
        "model": display_name,
        "metrics": payload,
        "preds": {"eval": np.asarray(eval_preds)},
        "df_eval": frame.copy().reset_index(drop=True),
        "y_eval": y_eval,
        "bundle_name": frame.attrs.get("dataset_name", display_name),
        "cache_status": "computed",
        "cache_source": None,
    }


def build_sequence_eval_bundle_from_df(
    df: pd.DataFrame,
    dataset_name: str,
    vocab: dict,
    cfg: dict = None,
    drop_duplicates: bool = False,
):
    cfg = cfg or CFG
    frame = prepare_encoded_text_frame(
        df,
        cfg=cfg,
        label_mapping=class2id,
        drop_duplicates=drop_duplicates,
        clean_text="clean",
    )
    target_col = resolve_target_col(frame, cfg)
    seq_eval = texts_to_sequences(frame["text_clean"], vocab, cfg["max_seq_len"])
    y_eval = np.asarray(frame[target_col], dtype=np.int64)
    loader_eval = DataLoader(
        TextSequenceDataset(seq_eval, y_eval),
        batch_size=cfg["dl_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return {
        "name": dataset_name,
        "df_eval": frame.copy().reset_index(drop=True),
        "loader_eval": loader_eval,
        "seq_eval": seq_eval,
        "y_eval": y_eval,
    }


def evaluate_torch_model_on_bundle(
    display_name: str,
    model,
    eval_bundle: dict,
    label_encoder=None,
    output_dir: Path = None,
    data_paths=None,
    checkpoint_path: Path = None,
    cfg: dict = None,
    force_recompute: bool = False,
    plot_confusion: bool = False,
    quiet_cache: bool = False,
):
    cfg = cfg or CFG
    y_eval = np.asarray(eval_bundle["y_eval"], dtype=np.int64)

    if output_dir is not None and not force_recompute:
        cached_payload, cached_preds, cached_path = load_cached_eval_artifacts(output_dir)
        if cached_payload is not None:
            eval_metrics = register_results_metrics(display_name, "eval", cached_payload.get("eval", {}))
            if plot_confusion and len(cached_preds) == len(y_eval):
                plot_confusion_matrix_clf(y_eval, cached_preds, display_name, label_encoder)
            if not quiet_cache:
                print(f"{display_name}: loaded cached metrics -> {cached_path}")
            return {
                "model": display_name,
                "metrics": {"eval": eval_metrics},
                "preds": {"eval": np.asarray(cached_preds)},
                "df_eval": eval_bundle["df_eval"].copy().reset_index(drop=True),
                "y_eval": y_eval,
                "bundle_name": eval_bundle["name"],
                "cache_status": "cached",
                "cache_source": str(cached_path) if cached_path is not None else None,
            }

    if model is None:
        raise ValueError(f"Model is required to evaluate {display_name} when no cache is available.")

    model.to(cfg["device"])
    model.eval()
    eval_preds = predict(model, eval_bundle["loader_eval"], cfg["device"])
    eval_metrics = evaluate_clf(
        display_name,
        y_eval,
        eval_preds,
        label_encoder=label_encoder,
        split="eval",
    )

    if plot_confusion:
        plot_confusion_matrix_clf(y_eval, eval_preds, display_name, label_encoder)

    meta = {
        "dataset": eval_bundle["name"],
    }
    if checkpoint_path is not None:
        meta["best_checkpoint"] = str(checkpoint_path)
    if isinstance(data_paths, (list, tuple)):
        meta["data_paths"] = [str(p) for p in data_paths]
    elif data_paths is not None:
        meta["data_path"] = str(data_paths)

    payload = {"eval": _coerce_metric_dict(eval_metrics)}
    if output_dir is not None:
        payload = persist_eval_artifacts(output_dir, eval_metrics, eval_preds, meta=meta)

    return {
        "model": display_name,
        "metrics": payload,
        "preds": {"eval": np.asarray(eval_preds)},
        "df_eval": eval_bundle["df_eval"].copy().reset_index(drop=True),
        "y_eval": y_eval,
        "bundle_name": eval_bundle["name"],
        "cache_status": "computed",
        "cache_source": None,
    }
