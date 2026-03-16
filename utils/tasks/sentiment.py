"""Sentiment-task-specific helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ── Sentiment label mapping ───────────────────────────────────────────────────

LABEL_NAMES: dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
    6: "neutral",
}


# ── Inference helper ──────────────────────────────────────────────────────────


@torch.no_grad()
def predict_one(
    text: str,
    model,
    tokenizer,
    label_names: dict,
    cfg: dict,
) -> tuple[str, float]:
    """
    Run a single forward pass through a HuggingFace sequence-classification model.

    Parameters
    ----------
    text        : raw input text
    model       : loaded AutoModelForSequenceClassification (already on device)
    tokenizer   : matching AutoTokenizer
    label_names : {int: str} mapping of class ids to human-readable labels
    cfg         : config dict; must contain "device" and optionally "max_len"

    Returns
    -------
    (label_str, confidence) tuple
    """
    device = torch.device(cfg["device"])
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.get("max_len", 256),
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    conf, pred_id = torch.max(probs, dim=-1)
    return label_names[int(pred_id)], float(conf)


# ── Artifact restoration ──────────────────────────────────────────────────────


def restore_final_transformer_test_artifacts(cfg: dict = None) -> dict:
    """
    Locate the best transformer checkpoint for the sentiment task and load
    its test predictions and metadata into the notebook's global namespace.

    Relies on notebook-level helpers (``ensure_split_artifacts_loaded``,
    ``resolve_final_transformer_selection``, ``load_saved_label_encoder``,
    ``load_saved_transformer_summary``) that must be defined before calling
    this function.

    Parameters
    ----------
    cfg : config dict (falls back to the notebook-level ``CFG`` global)

    Returns
    -------
    dict with keys "experiment", "best_model", "preds_path"
    """
    import builtins
    import inspect

    # Resolve caller globals so this function can read/write notebook state.
    frame = inspect.currentframe()
    caller_globals = frame.f_back.f_globals if frame and frame.f_back else {}
    cfg = cfg or caller_globals.get("CFG", {})

    def _g(name, default=None):
        return caller_globals.get(name, default)

    def _s(name, value):
        caller_globals[name] = value

    # Ensure splits are loaded (notebook helper).
    ensure_fn = _g("ensure_split_artifacts_loaded")
    if ensure_fn is not None:
        ensure_fn(cfg)

    # Load label encoder if not yet present.
    if "le" not in caller_globals:
        load_le_fn = _g("load_saved_label_encoder")
        if load_le_fn is not None:
            shared_model_dir = cfg["model_dir"] / "shared"
            _s("le", load_le_fn(shared_model_dir, _g("class2id")))

    # Pick the best transformer run.
    resolve_fn = _g("resolve_final_transformer_selection")
    if resolve_fn is None:
        raise RuntimeError("resolve_final_transformer_selection is not defined.")
    best_name, best_path = resolve_fn(cfg)
    _s("FINAL_TRANSFORMER_NAME", best_name)
    _s("TRANSFORMER_FINAL_BEST_PATH", best_path)

    # Populate test split data.
    split_key = _g("train_pool_key", cfg.get("train_pool_key", "train_pool"))
    splits = _g("splits", {})
    split = splits.get(split_key, {})
    _s("trf_df_test", split.get("df_test", None))
    _s("trf_y_test", np.asarray(split.get("y_test", []), dtype=np.int64))

    # Try to load cached test predictions.
    load_summary_fn = _g("load_saved_transformer_summary")
    experiment_dir = None
    if load_summary_fn is not None:
        summary_df = load_summary_fn(cfg)
        if not summary_df.empty and {"experiment", "experiment_dir"}.issubset(summary_df.columns):
            matched = summary_df.loc[summary_df["experiment"] == best_name]
            if not matched.empty:
                experiment_dir = matched.iloc[0].get("experiment_dir")

    preds_candidates: list[Path] = []
    if experiment_dir:
        runs_dir = Path(experiment_dir) / "runs"
        preds_candidates.append(runs_dir / "test_preds_latest.npy")
        preds_candidates.extend(sorted(runs_dir.glob("test_preds_*.npy")))

    exp_root = Path(best_path).parent
    runs_dir = exp_root / "runs"
    preds_candidates.append(runs_dir / "test_preds_latest.npy")
    preds_candidates.extend(sorted(runs_dir.glob("test_preds_*.npy")))

    seen: set = set()
    for candidate in preds_candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            _s("trf_test_preds", np.load(candidate))
            return {"experiment": best_name, "best_model": best_path, "preds_path": candidate}

    _s("trf_test_preds", np.asarray([], dtype=np.int64))
    return {"experiment": best_name, "best_model": best_path, "preds_path": None}
