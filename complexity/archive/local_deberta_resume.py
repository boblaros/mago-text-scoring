from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


def default_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def ensure_history_schema(history: dict[str, Any] | None) -> dict[str, list[float]]:
    base = dict(history or {})
    base.setdefault("train_loss", [])
    base.setdefault("val_loss", [])
    base.setdefault("val_acc", [])
    base.setdefault("val_f1", [])
    base.setdefault("epoch_seconds", [])
    return base


def history_for_plot(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "train_loss": list(history.get("train_loss", [])),
        "val_loss": list(history.get("val_loss", [])),
        "val_f1_macro": list(history.get("val_f1", [])),
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if src.is_dir():
        try:
            os.symlink(src.resolve(), dst, target_is_directory=True)
        except Exception:
            shutil.copytree(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)


def _mirror_dir_contents(src_dir: Path, dst_dir: Path) -> bool:
    if not src_dir.exists() or not src_dir.is_dir():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied_any = False
    for item in src_dir.iterdir():
        if item.name == ".DS_Store":
            continue
        _link_or_copy(item, dst_dir / item.name)
        copied_any = True
    return copied_any


def bootstrap_best_model(exp_root: Path, best_model_dir: Path) -> Path:
    best_model_dir.mkdir(parents=True, exist_ok=True)
    if (best_model_dir / "config.json").exists():
        return best_model_dir

    candidates = [
        exp_root / "archive" / "best_model",
        exp_root / "best_model",
    ]
    for candidate in candidates:
        if candidate.resolve() == best_model_dir.resolve():
            continue
        if _mirror_dir_contents(candidate, best_model_dir):
            return best_model_dir
    return best_model_dir


def latest_checkpoint_dir(checkpoint_dir: Path) -> Path | None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint_epoch_")
        ],
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def save_history(history: dict[str, list[float]], history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


class DebertaTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = np.asarray(texts, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros((1, self.max_length), dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def build_model(
    model_name: str,
    label_names: list[str],
) -> AutoModelForSequenceClassification:
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in id2label.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def build_optimizer(model, learning_rate: float, weight_decay: float) -> AdamW:
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    params = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(no_decay_key in name for no_decay_key in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(no_decay_key in name for no_decay_key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=learning_rate, eps=1e-8)


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if bool(token_type_ids.any().item()):
                    model_kwargs["token_type_ids"] = token_type_ids
                outputs = model(**model_kwargs)

            total_loss += float(outputs.loss.item())
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    labels_arr = np.asarray(all_labels, dtype=np.int64)
    preds_arr = np.asarray(all_preds, dtype=np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "f1_macro": float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)),
    }
    avg_loss = total_loss / max(len(loader), 1)
    metrics["loss"] = float(avg_loss)
    return avg_loss, metrics, preds_arr, labels_arr


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    metrics_snapshot: dict[str, float],
    checkpoint_dir: Path,
    keep_last_n_ckpts: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    state = {
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "metrics": metrics_snapshot,
    }
    torch.save(state, ckpt_path / "training_state.pt")

    checkpoints = sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint_epoch_")
        ],
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    while len(checkpoints) > keep_last_n_ckpts:
        shutil.rmtree(checkpoints.pop(0), ignore_errors=True)
    return ckpt_path


def run_deberta_resume_experiment(
    *,
    display_name: str,
    exp_paths: dict[str, Any],
    data_bundle: dict[str, Any],
    label_names: list[str],
    epochs: int,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    max_length: int = 256,
    fp16: bool = True,
    save_every_n_epochs: int = 1,
    keep_last_n_ckpts: int = 2,
    seed: int = 42,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device | None = None,
    metrics_stem: str | None = None,
    transformer_metrics_dir: str | Path | None = None,
) -> dict[str, Any]:
    exp_root = Path(exp_paths["root"])
    best_model_dir = Path(exp_paths["best_model"])
    checkpoint_dir = Path(exp_paths["checkpoints"])
    runs_dir = Path(exp_paths["runs"])
    exp_root.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    history_path = exp_root / "training_history.json"
    bootstrap_best_model(exp_root, best_model_dir)

    if device is None:
        torch_device = default_device()
    elif isinstance(device, torch.device):
        torch_device = device
    else:
        torch_device = torch.device(device)
    use_fp16 = bool(fp16 and torch_device.type == "cuda")

    set_reproducibility(seed)

    tokenizer_source = str(best_model_dir) if (best_model_dir / "tokenizer.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    train_dataset = DebertaTextDataset(
        data_bundle["X_train_raw"],
        data_bundle["y_train"],
        tokenizer,
        max_length=max_length,
    )
    val_dataset = DebertaTextDataset(
        data_bundle["X_val_raw"],
        data_bundle["y_val"],
        tokenizer,
        max_length=max_length,
    )
    test_dataset = DebertaTextDataset(
        data_bundle["X_test_raw"],
        data_bundle["y_test"],
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    eval_batch_size = max(batch_size, batch_size * 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
    )

    latest_ckpt = latest_checkpoint_dir(checkpoint_dir)
    history = ensure_history_schema(
        json.loads(history_path.read_text()) if history_path.exists() else {}
    )
    best_val_f1 = max(history["val_f1"]) if history["val_f1"] else -1.0
    best_epoch = history["val_f1"].index(best_val_f1) + 1 if history["val_f1"] else 0

    if latest_ckpt is not None and (latest_ckpt / "training_state.pt").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(latest_ckpt), use_fast=True)
        train_dataset.tokenizer = tokenizer
        val_dataset.tokenizer = tokenizer
        test_dataset.tokenizer = tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(str(latest_ckpt))
    else:
        model = build_model(model_name, label_names)

    model = model.to(torch_device)
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    total_steps = math.ceil(len(train_loader) / max(grad_accum_steps, 1)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    start_epoch = 1
    restored_state = {
        "model": False,
        "optimizer": False,
        "scheduler": False,
        "scaler": False,
    }
    if latest_ckpt is not None and (latest_ckpt / "training_state.pt").exists():
        state = torch.load(latest_ckpt / "training_state.pt", map_location=torch_device)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state.get("scaler_state", {}))
        start_epoch = int(state.get("epoch", 0)) + 1
        restored_state = {
            "model": True,
            "optimizer": True,
            "scheduler": True,
            "scaler": True,
        }

    ran_remaining_epochs = False
    for epoch in range(start_epoch, epochs + 1):
        ran_remaining_epochs = True
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)
        epoch_start = datetime.now()
        optimizer.zero_grad()
        progress = tqdm(
            train_loader,
            total=n_batches,
            desc=f"{display_name} | epoch {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            labels = batch["labels"].to(torch_device)
            token_type_ids = batch["token_type_ids"].to(torch_device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if bool(token_type_ids.any().item()):
                    model_kwargs["token_type_ids"] = token_type_ids
                outputs = model(**model_kwargs)
                loss = outputs.loss / max(grad_accum_steps, 1)

            scaler.scale(loss).backward()
            if step % max(grad_accum_steps, 1) == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += float(outputs.loss.item())
            if step % max(1, n_batches // 20) == 0 or step == n_batches:
                progress.set_postfix(
                    {
                        "loss": f"{epoch_loss / step:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "best_f1": f"{best_val_f1:.4f}" if best_val_f1 >= 0 else "--",
                    },
                    refresh=False,
                )
        progress.close()

        avg_train_loss = epoch_loss / max(n_batches, 1)
        _, val_metrics, _, _ = evaluate_model(model, val_loader, torch_device, use_fp16)
        epoch_seconds = float((datetime.now() - epoch_start).total_seconds())

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["f1_macro"]))
        history["epoch_seconds"].append(epoch_seconds)
        save_history(history, history_path)

        if val_metrics["f1_macro"] > best_val_f1 or not (best_model_dir / "config.json").exists():
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))

        if epoch % save_every_n_epochs == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics_snapshot={
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["accuracy"]),
                    "val_f1": float(val_metrics["f1_macro"]),
                    "epoch_seconds": epoch_seconds,
                },
                checkpoint_dir=checkpoint_dir,
                keep_last_n_ckpts=keep_last_n_ckpts,
            )

    best_model_source = best_model_dir if (best_model_dir / "config.json").exists() else latest_ckpt
    if best_model_source is None:
        raise FileNotFoundError(
            f"No DeBERTa best model found under {best_model_dir} or {checkpoint_dir}."
        )

    eval_tokenizer = AutoTokenizer.from_pretrained(str(best_model_source), use_fast=True)
    val_dataset.tokenizer = eval_tokenizer
    test_dataset.tokenizer = eval_tokenizer
    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_source)).to(torch_device)

    _, val_metrics, val_preds, val_labels = evaluate_model(best_model, val_loader, torch_device, use_fp16)
    _, test_metrics, test_preds, test_labels = evaluate_model(best_model, test_loader, torch_device, use_fp16)

    metrics_stem = metrics_stem or slugify(display_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_info = {
        "checkpoint_path": str(latest_ckpt) if latest_ckpt is not None else None,
        "completed_epoch_before_resume": max(len(history["train_loss"]) - (1 if ran_remaining_epochs else 0), 0)
        if latest_ckpt is not None
        else 0,
        "start_epoch": start_epoch,
        "target_epochs": epochs,
        "epoch_5_completed": len(history["train_loss"]) >= epochs,
        "ran_remaining_epochs": ran_remaining_epochs,
        "restored_state": restored_state,
        # The original notebook did not persist RNG/sampler state, so exact replay is not available.
        "exact_resume_possible": False if latest_ckpt is not None else True,
        "exact_resume_limitations": (
            [
                "python random state was not saved",
                "numpy random state was not saved",
                "torch random state was not saved",
                "dataloader shuffle state was not saved",
            ]
            if latest_ckpt is not None
            else []
        ),
    }

    run_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_model_dir": str(best_model_dir),
        "checkpoints_dir": str(checkpoint_dir),
        "experiment_dir": str(exp_root),
        "history_file": str(history_path),
        "resume_info": resume_info,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
    with (runs_dir / f"metrics_{ts}.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metrics, handle, indent=2)
    with (runs_dir / "metrics_latest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metrics, handle, indent=2)

    np.save(runs_dir / f"val_preds_{ts}.npy", val_preds)
    np.save(runs_dir / f"test_preds_{ts}.npy", test_preds)
    np.save(runs_dir / "val_preds_latest.npy", val_preds)
    np.save(runs_dir / "test_preds_latest.npy", test_preds)

    if transformer_metrics_dir is not None:
        metrics_dir_path = Path(transformer_metrics_dir)
        metrics_dir_path.mkdir(parents=True, exist_ok=True)
        with (metrics_dir_path / f"{metrics_stem}_latest.json").open("w", encoding="utf-8") as handle:
            json.dump(run_metrics, handle, indent=2)

    return {
        "name": display_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "y_val": val_labels,
        "y_test": test_labels,
        "df_test": data_bundle["df_test"].copy().reset_index(drop=True),
        "best_dir": best_model_dir,
        "paths": {
            "root": exp_root,
            "best_model": best_model_dir,
            "checkpoints": checkpoint_dir,
            "runs": runs_dir,
        },
        "history": history,
        "history_plot": history_for_plot(history),
        "resume_info": resume_info,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
