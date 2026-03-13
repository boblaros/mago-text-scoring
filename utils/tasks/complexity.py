
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from ..data import TextSequenceDataset, bootstrap_dir_from_legacy, bootstrap_file_from_legacy, build_transformer_eval_bundle_from_df, load_pickle_with_fallback, now_ts, resolve_experiment_paths, size_tag_from_split, slugify, texts_to_sequences
from ..metrics import RESULTS, _coerce_metric_dict
from ..models.architectures import BiLSTMAttention, EmbeddingMLP, TextCNN
from ..models.loading import _load_latest_run_array, load_saved_cross_dataset_summaries, load_saved_label_encoder, load_saved_results_df, load_saved_transformer_run, load_saved_transformer_summary, merge_transformer_summary_frames, resolve_final_transformer_selection, transformer_summary_from_runs
from ..plots import plot_history
from ..training import load_torch_checkpoint, run_transformer_experiment

CFG: dict = {}
debarta_metrics_dir = Path(".")


UNIFIED_LABELS = ['A1-A2', 'B1-B2', 'C1-C2']
DATA_DIR = Path('data')
OUTPUT_DIR = DATA_DIR / 'prepared'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CYRILLIC_LOOKALIKES = str.maketrans({'с': 'c', 'С': 'C'})
SCORE_CLASS_BANDS = {
    'A1-A2': (0.0, 1 / 3),
    'B1-B2': (1 / 3, 2 / 3),
    'C1-C2': (2 / 3, 1.0),
}
LEXILE_CLASS_BANDS = {
    'A1-A2': {'max': 700},
    'B1-B2': {'min': 700, 'max': 1100},
    'C1-C2': {'min': 1100},
}
DEFAULT_DATASET_SLUGS = [
    'uk-key-stage-readability-for-english-texts',
    'asap-2-0',
    'asap-aes',
    'cerf-levelled-english-texts',
    'one-stop-corpus-english',
    'cambridge-english-readability',
    'commonlit-readability-prize',
]
DEBERTA_DISPLAY_NAME = 'DeBERTaV3_train_pool'
DEBERTA_MODEL_NAME = 'microsoft/deberta-v3-base'
DEBERTA_METRICS_STEM = 'deberta_v3_base_train_pool'


def get_parquet_engine() -> str:
    for engine in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(engine) is not None:
            return engine
    raise ImportError(
        "Writing parquet files requires `pyarrow` or `fastparquet` in the notebook kernel."
    )


def normalize_slug(value: str) -> str:
    normalized = value.translate(CYRILLIC_LOOKALIKES)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.replace("_", "-").lower()


def resolve_dataset_dir(slug: str) -> Path:
    exact_path = DATA_DIR / slug
    if exact_path.exists():
        return exact_path

    normalized_target = normalize_slug(slug)
    candidates = [
        path for path in DATA_DIR.iterdir()
        if path.is_dir() and normalize_slug(path.name) == normalized_target
    ]
    if candidates:
        return candidates[0]

    fuzzy_candidates = [
        path for path in DATA_DIR.iterdir()
        if path.is_dir() and normalized_target in normalize_slug(path.name)
    ]
    if fuzzy_candidates:
        return fuzzy_candidates[0]

    raise FileNotFoundError(f"Could not resolve dataset directory for: {slug}")


PARQUET_ENGINE = get_parquet_engine()
try:
    DATASET_DIRS = {name: resolve_dataset_dir(name) for name in DEFAULT_DATASET_SLUGS}
except FileNotFoundError:
    # Keep imports usable even before the notebook is started from its task directory.
    DATASET_DIRS = {name: DATA_DIR / name for name in DEFAULT_DATASET_SLUGS}


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def infer_observation_format(series: pd.Series, word_threshold: int = 3, sample_size: int = 200) -> str:
    sample = series.fillna("").astype(str).str.strip()
    sample = sample[sample.ne("")].head(sample_size)
    if sample.empty:
        return "unknown"
    token_counts = sample.str.split().str.len()
    if (token_counts <= word_threshold).mean() >= 0.85:
        return "isolated_words"
    return "full_texts_or_excerpts"


def summarize_text_lengths(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    token_counts = cleaned.str.split().str.len()
    return token_counts.describe(percentiles=[0.25, 0.50, 0.75]).round(2)


def show_basic_overview(df: pd.DataFrame, text_col: str, label_cols: list[str]) -> None:
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Text column:", text_col)
    print("Label columns:", label_cols)
    print("Observation format:", infer_observation_format(df[text_col]))
    print("Text length statistics (word counts):")
    display(summarize_text_lengths(df[text_col]))

    for label_col in label_cols:
        if label_col in df.columns:
            print(f"Value counts for {label_col}:")
            display(df[label_col].value_counts(dropna=False).rename("count"))

    preview_cols = [text_col] + [col for col in label_cols if col in df.columns]
    preview = df[preview_cols].head(3).copy()
    preview[text_col] = (
        preview[text_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 200)
    )
    print("Preview:")
    display(preview)


def build_standardized_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    source_dataset: str,
) -> pd.DataFrame:
    standardized = df[[text_col, label_col]].copy()
    standardized = standardized.rename(columns={text_col: "text", label_col: "label"})
    standardized["source_dataset"] = source_dataset
    return standardized


def save_standardized_dataset(df: pd.DataFrame, output_name: str) -> tuple[pd.DataFrame, Path]:
    standardized = df.loc[:, ["text", "label", "source_dataset"]].copy()
    standardized["text"] = (
        standardized["text"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    standardized["label"] = standardized["label"].astype("string").str.strip()
    standardized = standardized[
        standardized["text"].ne("") & standardized["label"].isin(UNIFIED_LABELS)
    ].reset_index(drop=True)

    output_path = OUTPUT_DIR / output_name
    standardized.to_parquet(output_path, index=False, engine=PARQUET_ENGINE)

    print(f"Saved {output_path} with shape {standardized.shape}")
    print("Unified label distribution:")
    display(standardized["label"].value_counts().reindex(UNIFIED_LABELS, fill_value=0))
    display(standardized.head(3))
    return standardized, output_path


def min_max_normalize(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    if min_value == max_value:
        return pd.Series([0.5] * len(series), index=series.index, dtype="Float64")
    return (series - min_value) / (max_value - min_value)


def normalized_score_to_label(value: float | pd._libs.missing.NAType) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    for label, (lower_bound, upper_bound) in SCORE_CLASS_BANDS.items():
        if value <= upper_bound:
            return label
    return pd.NA


def generate_rule_based_sentence(word: str, unified_label: str) -> str:
    word = str(word).strip()
    templates = {
        "A1-A2": f"This short example uses the word '{word}' in a simple sentence.",
        "B1-B2": f"In an everyday context, the word '{word}' appears naturally in this sentence.",
        "C1-C2": f"In a more formal and abstract context, the word '{word}' is used precisely in this sentence.",
    }
    return templates.get(unified_label, f"The word '{word}' appears in this example sentence.")


def build_synthetic_sentence_dataset(
    words_df: pd.DataFrame,
    word_col: str,
    label_col: str,
    source_dataset: str,
    output_name: str,
    generator=generate_rule_based_sentence,
) -> tuple[pd.DataFrame, Path]:
    synthetic_df = words_df[[word_col, label_col]].copy()
    synthetic_df["text"] = synthetic_df.apply(
        lambda row: generator(row[word_col], row[label_col]),
        axis=1,
    )
    synthetic_df["source_dataset"] = source_dataset
    return save_standardized_dataset(
        synthetic_df[["text", label_col, "source_dataset"]].rename(columns={label_col: "label"}),
        output_name,
    )


def extract_lexile_value(value: object) -> float | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    numbers = [int(match) for match in re.findall(r"\d+", str(value))]
    if not numbers:
        return pd.NA
    return float(sum(numbers) / len(numbers))


def lexile_to_label(value: float | pd._libs.missing.NAType) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    if value <= LEXILE_CLASS_BANDS["A1-A2"]["max"]:
        return "A1-A2"
    if value <= LEXILE_CLASS_BANDS["B1-B2"]["max"]:
        return "B1-B2"
    return "C1-C2"


def load_folder_corpus(root_dir: Path, folder_names: list[str], label_column_name: str) -> pd.DataFrame:
    rows = []
    for folder_name in folder_names:
        folder_path = root_dir / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")
        for text_path in sorted(folder_path.glob("*.txt"), key=natural_sort_key):
            rows.append(
                {
                    "document_id": f"{folder_name}/{text_path.stem}",
                    label_column_name: folder_name,
                    "text": read_text_file(text_path),
                }
            )
    return pd.DataFrame(rows)


def load_simple_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "doc": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def column_index(column_letters: str) -> int:
        index = 0
        for character in column_letters:
            if character.isalpha():
                index = index * 26 + (ord(character.upper()) - 64)
        return index - 1

    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root:
                text_parts = [
                    text_node.text or ""
                    for text_node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                ]
                shared_strings.append("".join(text_parts))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.find("main:sheets", namespace)
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships.findall("rel:Relationship", namespace)
        }

        target_sheet = None
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target_sheet = "xl/" + relationship_map[relationship_id]
                break
        if target_sheet is None:
            raise KeyError(f"Sheet '{sheet_name}' not found in {path}")

        sheet_root = ET.fromstring(archive.read(target_sheet))
        rows = []
        for row in sheet_root.findall(".//main:sheetData/main:row", namespace):
            values_by_position = {}
            for cell in row.findall("main:c", namespace):
                reference = cell.attrib.get("r", "")
                column_letters = "".join(character for character in reference if character.isalpha())
                position = column_index(column_letters)
                cell_type = cell.attrib.get("t")
                value = None

                raw_value = cell.find("main:v", namespace)
                if raw_value is not None:
                    value = shared_strings[int(raw_value.text)] if cell_type == "s" else raw_value.text

                inline_string = cell.find("main:is", namespace)
                if inline_string is not None:
                    text_parts = [
                        text_node.text or ""
                        for text_node in inline_string.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    ]
                    value = "".join(text_parts)

                values_by_position[position] = value

            if values_by_position:
                width = max(values_by_position) + 1
                rows.append([values_by_position.get(index) for index in range(width)])

    if not rows:
        return pd.DataFrame()

    width = max(len(row) for row in rows)
    normalized_rows = [row + [None] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    data = normalized_rows[1:]
    return pd.DataFrame(data, columns=header)


def load_clear_corpus(path: Path, sheet_name: str = "Data") -> pd.DataFrame:
    if importlib.util.find_spec("openpyxl") is not None:
        return pd.read_excel(path, sheet_name=sheet_name)
    print("openpyxl is not available; using the XML fallback loader for CLEAR_corpus_final.xlsx.")
    return load_simple_xlsx_sheet(path, sheet_name=sheet_name)


def _deberta_history_for_plot(history: dict | None) -> dict:
    history = history or {}
    return {
        "train_loss": list(history.get("train_loss", [])),
        "val_loss": list(history.get("val_loss", [])),
        "val_f1_macro": list(history.get("val_f1", history.get("val_f1_macro", []))),
    }


def _deberta_load_history(history_path) -> dict:
    history_path = Path(history_path) if history_path is not None else None
    if history_path is None or not history_path.exists():
        return {}
    try:
        return json.loads(history_path.read_text())
    except Exception as e:
        print(f"Could not load DeBERTa history from {history_path}: {e}")
        return {}


def _deberta_latest_checkpoint_dir(checkpoint_dir: Path):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.search(r"checkpoint(?:_epoch_|-)(\d+)$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    return sorted(candidates, key=lambda item: item[0])[-1][1] if candidates else None


def _deberta_materialize_best_model(exp_paths: dict):
    best_model_dir = Path(exp_paths["best_model"])
    best_model_dir.mkdir(parents=True, exist_ok=True)
    if (best_model_dir / "config.json").exists():
        return best_model_dir

    archive_best_dir = Path(exp_paths["root"]) / "archive" / "best_model"
    bootstrap_dir_from_legacy(best_model_dir, [archive_best_dir])
    if (best_model_dir / "config.json").exists():
        return best_model_dir

    latest_ckpt = _deberta_latest_checkpoint_dir(exp_paths["checkpoints"])
    if latest_ckpt is not None:
        bootstrap_dir_from_legacy(best_model_dir, [latest_ckpt])

    return best_model_dir if (best_model_dir / "config.json").exists() else None


def _deberta_resolve_model_source(exp_paths: dict):
    best_model_dir = _deberta_materialize_best_model(exp_paths)
    if best_model_dir is not None:
        return best_model_dir

    latest_ckpt = _deberta_latest_checkpoint_dir(exp_paths["checkpoints"])
    if latest_ckpt is not None and (latest_ckpt / "config.json").exists():
        return latest_ckpt

    archive_best_dir = Path(exp_paths["root"]) / "archive" / "best_model"
    if (archive_best_dir / "config.json").exists():
        return archive_best_dir

    return None


def _deberta_build_cached_payload(metrics_payload: dict, data_bundle: dict, exp_paths: dict) -> dict:
    history_path = metrics_payload.get("history_file") or (Path(exp_paths["root"]) / "training_history.json")
    history = _deberta_load_history(history_path)
    runs_dir = Path(exp_paths["runs"])
    val_preds, _ = _load_latest_run_array(runs_dir, "val_preds")
    test_preds, _ = _load_latest_run_array(runs_dir, "test_preds")
    best_dir = _deberta_materialize_best_model(exp_paths)

    if best_dir is None and metrics_payload.get("best_model_dir"):
        candidate_best_dir = Path(metrics_payload["best_model_dir"])
        if candidate_best_dir.exists():
            best_dir = candidate_best_dir

    return {
        "name": DEBERTA_DISPLAY_NAME,
        "val_metrics": metrics_payload.get("val", {}),
        "test_metrics": metrics_payload.get("test", {}),
        "val_preds": np.asarray(val_preds),
        "test_preds": np.asarray(test_preds),
        "y_val": np.asarray(data_bundle["y_val"], dtype=np.int64),
        "y_test": np.asarray(data_bundle["y_test"], dtype=np.int64),
        "df_test": data_bundle["df_test"].copy().reset_index(drop=True),
        "best_dir": best_dir,
        "paths": exp_paths,
        "history": history,
        "history_plot": _deberta_history_for_plot(history),
        "resume_info": metrics_payload.get("resume_info", {}),
        "best_epoch": metrics_payload.get("best_epoch"),
        "best_val_f1": metrics_payload.get("best_val_f1"),
    }


def _deberta_load_saved_payload(data_bundle: dict, exp_paths: dict):
    cached_payload = load_saved_transformer_run(DEBERTA_DISPLAY_NAME, data_bundle, CFG)
    if isinstance(cached_payload, dict) and cached_payload.get("val_metrics") and cached_payload.get("test_metrics"):
        cached_payload["best_dir"] = _deberta_materialize_best_model(exp_paths) or cached_payload.get("best_dir")
        cached_payload["paths"] = exp_paths
        if not cached_payload.get("history_plot"):
            cached_payload["history_plot"] = _deberta_history_for_plot(cached_payload.get("history", {}))
        return cached_payload

    candidate_metric_paths = [
        Path(exp_paths["runs"]) / "metrics_latest.json",
        debarta_metrics_dir / f"{DEBERTA_METRICS_STEM}_latest.json",
    ]
    for candidate in candidate_metric_paths:
        if not candidate.exists():
            continue
        try:
            metrics_payload = json.loads(candidate.read_text())
        except Exception as e:
            print(f"Could not load DeBERTa metrics from {candidate}: {e}")
            continue
        if isinstance(metrics_payload, dict) and "val" in metrics_payload and "test" in metrics_payload:
            return _deberta_build_cached_payload(metrics_payload, data_bundle, exp_paths)

    return None


def _deberta_has_full_cached_outputs(payload: dict, data_bundle: dict) -> bool:
    val_preds = np.asarray(payload.get("val_preds", []), dtype=np.int64)
    test_preds = np.asarray(payload.get("test_preds", []), dtype=np.int64)
    return (
        len(val_preds) == len(np.asarray(data_bundle["y_val"], dtype=np.int64))
        and len(test_preds) == len(np.asarray(data_bundle["y_test"], dtype=np.int64))
        and len(test_preds) > 0
    )


class _DebertaTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = np.asarray(texts, dtype=object)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
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


def _deberta_seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _deberta_build_model(model_source, label_names):
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in id2label.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        str(model_source),
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _deberta_build_optimizer(model, learning_rate: float, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    params = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(key in name for key in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(key in name for key in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=learning_rate, eps=1e-8)


def _deberta_evaluate_model(model, loader, device, use_fp16: bool):
    model.eval()
    all_preds = []
    all_labels = []
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
        "loss": float(total_loss / max(len(loader), 1)),
    }
    return metrics, preds_arr, labels_arr


def _deberta_save_history(history: dict, history_path: Path):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _deberta_save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    metrics_snapshot: dict,
    checkpoint_dir: Path,
    keep_last_n_ckpts: int,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metrics": metrics_snapshot,
        },
        ckpt_path / "training_state.pt",
    )

    checkpoints = sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and re.search(r"checkpoint(?:_epoch_|-)(\d+)$", path.name)
        ],
        key=lambda path: int(re.search(r"(\d+)$", path.name).group(1)),
    )
    while len(checkpoints) > keep_last_n_ckpts:
        shutil.rmtree(checkpoints.pop(0), ignore_errors=True)


def _run_deberta_notebook_experiment(
    *,
    display_name: str,
    exp_paths: dict,
    data_bundle: dict,
    label_names: list,
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
    model_name: str | None = None,
    device=None,
    metrics_stem: str | None = None,
    transformer_metrics_dir: Path | None = None,
):
    model_name = model_name or globals().get("DEBERTA_MODEL_NAME", "microsoft/deberta-v3-base")
    metrics_stem = metrics_stem or globals().get("DEBERTA_METRICS_STEM", "deberta_v3_base_train_pool")
    if transformer_metrics_dir is None:
        transformer_metrics_dir = globals().get("debarta_metrics_dir", CFG["output_paths"]["metrics"] / "transformers")

    exp_root = Path(exp_paths["root"])
    best_model_dir = Path(exp_paths["best_model"])
    checkpoint_dir = Path(exp_paths["checkpoints"])
    runs_dir = Path(exp_paths["runs"])
    exp_root.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    history_path = exp_root / "training_history.json"
    history = _deberta_load_history(history_path)
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", [])
    history.setdefault("val_acc", [])
    history.setdefault("val_f1", [])
    history.setdefault("epoch_seconds", [])

    if device is None:
        torch_device = torch.device(CFG["device"])
    elif isinstance(device, torch.device):
        torch_device = device
    else:
        torch_device = torch.device(device)
    use_fp16 = bool(fp16 and torch_device.type == "cuda")

    _deberta_seed_everything(seed)

    latest_ckpt = _deberta_latest_checkpoint_dir(checkpoint_dir)
    materialized_best_model = _deberta_materialize_best_model(exp_paths)
    latest_state_path = latest_ckpt / "training_state.pt" if latest_ckpt is not None else None

    tokenizer_source = None
    for candidate in [latest_ckpt, materialized_best_model, model_name]:
        if candidate is None:
            continue
        candidate_path = Path(candidate) if isinstance(candidate, Path) else None
        if candidate_path is None or (candidate_path / "tokenizer.json").exists():
            tokenizer_source = candidate
            break
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)

    train_dataset = _DebertaTextDataset(
        data_bundle["X_train_raw"],
        data_bundle["y_train"],
        tokenizer,
        max_length=max_length,
    )
    val_dataset = _DebertaTextDataset(
        data_bundle["X_val_raw"],
        data_bundle["y_val"],
        tokenizer,
        max_length=max_length,
    )
    test_dataset = _DebertaTextDataset(
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

    model_source = model_name
    if latest_ckpt is not None and latest_state_path is not None and latest_state_path.exists():
        model_source = latest_ckpt
    elif materialized_best_model is not None:
        model_source = materialized_best_model
    elif latest_ckpt is not None:
        model_source = latest_ckpt

    model = _deberta_build_model(model_source, label_names).to(torch_device)
    optimizer = _deberta_build_optimizer(model, learning_rate, weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / max(grad_accum_steps, 1)) * max(epochs, 1))
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
    exact_resume_limitations = []

    if latest_ckpt is not None and latest_state_path is not None and latest_state_path.exists():
        state = torch.load(latest_state_path, map_location=torch_device)
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
        exact_resume_limitations = [
            "python random state was not saved",
            "numpy random state was not saved",
            "torch random state was not saved",
            "dataloader shuffle state was not saved",
        ]
    elif latest_ckpt is not None:
        match = re.search(r"checkpoint(?:_epoch_|-)(\d+)$", latest_ckpt.name)
        if match:
            start_epoch = int(match.group(1)) + 1
            exact_resume_limitations = [
                "optimizer state was not saved",
                "scheduler state was not saved",
                "grad scaler state was not saved",
                "python random state was not saved",
                "numpy random state was not saved",
                "torch random state was not saved",
                "dataloader shuffle state was not saved",
            ]
    elif materialized_best_model is not None:
        start_epoch = epochs + 1

    best_val_f1 = max(history["val_f1"]) if history["val_f1"] else -1.0
    best_epoch = history["val_f1"].index(best_val_f1) + 1 if history["val_f1"] else 0

    ran_remaining_epochs = False
    for epoch in range(start_epoch, epochs + 1):
        ran_remaining_epochs = True
        model.train()
        epoch_loss = 0.0
        epoch_start = datetime.now()
        optimizer.zero_grad()
        progress = tqdm(
            train_loader,
            total=len(train_loader),
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
            if step % max(grad_accum_steps, 1) == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += float(outputs.loss.item())
            if step % max(1, len(train_loader) // 20) == 0 or step == len(train_loader):
                progress.set_postfix(
                    {
                        "loss": f"{epoch_loss / step:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                        "best_f1": f"{best_val_f1:.4f}" if best_val_f1 >= 0 else "--",
                    },
                    refresh=False,
                )
        progress.close()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics, _, _ = _deberta_evaluate_model(model, val_loader, torch_device, use_fp16)
        epoch_seconds = float((datetime.now() - epoch_start).total_seconds())

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["f1_macro"]))
        history["epoch_seconds"].append(epoch_seconds)
        _deberta_save_history(history, history_path)

        if val_metrics["f1_macro"] > best_val_f1 or not (best_model_dir / "config.json").exists():
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))

        if epoch % save_every_n_epochs == 0:
            _deberta_save_checkpoint(
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

    best_model_source = _deberta_resolve_model_source(exp_paths)
    if best_model_source is None:
        raise FileNotFoundError(
            f"No reusable DeBERTa model was found under {best_model_dir} or {checkpoint_dir}."
        )

    eval_tokenizer = AutoTokenizer.from_pretrained(str(best_model_source), use_fast=True)
    val_dataset.tokenizer = eval_tokenizer
    test_dataset.tokenizer = eval_tokenizer
    best_model = _deberta_build_model(best_model_source, label_names).to(torch_device)

    val_metrics, val_preds, val_labels = _deberta_evaluate_model(best_model, val_loader, torch_device, use_fp16)
    test_metrics, test_preds, test_labels = _deberta_evaluate_model(best_model, test_loader, torch_device, use_fp16)
    if best_val_f1 < 0:
        best_val_f1 = float(val_metrics["f1_macro"])
        if best_epoch == 0 and start_epoch > epochs:
            best_epoch = epochs

    ts = now_ts()
    resume_info = {
        "checkpoint_path": str(latest_ckpt) if latest_ckpt is not None else None,
        "completed_epoch_before_resume": max(start_epoch - 1, 0),
        "start_epoch": start_epoch,
        "target_epochs": epochs,
        "epoch_5_completed": bool(start_epoch > epochs or len(history["train_loss"]) >= epochs),
        "ran_remaining_epochs": ran_remaining_epochs,
        "restored_state": restored_state,
        "exact_resume_possible": bool(latest_ckpt is None or (latest_state_path is not None and latest_state_path.exists())),
        "exact_resume_limitations": exact_resume_limitations,
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
    with open(runs_dir / f"metrics_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)
    with open(runs_dir / "metrics_latest.json", "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)

    np.save(runs_dir / f"val_preds_{ts}.npy", val_preds)
    np.save(runs_dir / f"test_preds_{ts}.npy", test_preds)
    np.save(runs_dir / "val_preds_latest.npy", val_preds)
    np.save(runs_dir / "test_preds_latest.npy", test_preds)

    if transformer_metrics_dir is not None:
        transformer_metrics_dir = Path(transformer_metrics_dir)
        transformer_metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(transformer_metrics_dir / f"{metrics_stem}_latest.json", "w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2)

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
        "history_plot": _deberta_history_for_plot(history),
        "resume_info": resume_info,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }


def load_saved_deep_model_for_inference(model_label: str, split_key: str | None = None, cfg: dict | None = None):
    cfg = cfg or CFG
    ensure_split_artifacts_loaded(cfg)
    split_key = split_key or globals().get("train_pool_key", cfg.get("train_pool_key", "train_pool"))
    shared_model_dir = cfg["model_dir"] / "shared"

    if "vocab" not in globals():
        vocab_path = shared_model_dir / "vocab.pkl"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Shared vocabulary not found: {vocab_path}")
        globals()["vocab"] = joblib.load(vocab_path)

    deep_model_specs = build_deep_model_specs(vocab=globals()["vocab"], embedding_matrix=None, cfg=cfg)
    if model_label not in deep_model_specs:
        raise KeyError(f"Unknown deep model for inference: {model_label}")

    model_spec = deep_model_specs[model_label]
    exp_paths = resolve_experiment_paths(
        model_family=model_spec["family_slug"],
        dataset=cfg.get("experiment_dataset_slug", slugify(cfg.get("task", "dataset"))),
        size_tag=size_tag_from_split(split_key),
        model_root=cfg["model_dir"],
    )
    legacy_dir = cfg["model_dir"] / f"{model_label.replace(' ', '_')}_[{split_key}]"
    best_ckpt_path = bootstrap_file_from_legacy(
        exp_paths["best_model"] / "best.pt",
        [legacy_dir / "best.pt"],
    )
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found for {model_label}: {best_ckpt_path}")

    model_obj = model_spec["builder"]()
    load_torch_checkpoint(model_obj, best_ckpt_path, cfg["device"])
    model_obj.to(cfg["device"])
    model_obj.eval()
    return {
        "name": model_label,
        "model": model_obj,
        "vocab": globals()["vocab"],
        "checkpoint": best_ckpt_path,
    }


def ensure_split_artifacts_loaded(cfg: dict = None):
    cfg = cfg or CFG
    split_artifacts_local = globals().get("split_artifacts")

    if not split_artifacts_local:
        split_artifacts_local = load_pickle_with_fallback(cfg["split_artifact_path"])
        globals()["split_artifacts"] = split_artifacts_local

    globals()["splits"] = split_artifacts_local["splits"]
    globals()["train_pool_key"] = cfg.get("train_pool_key", "train_pool")
    globals()["train_pool_split"] = globals()["splits"][globals()["train_pool_key"]]
    globals()["train_pool_sources"] = split_artifacts_local.get(
        "train_pool_sources",
        cfg.get("train_pool_datasets", []),
    )
    globals()["external_eval_datasets"] = split_artifacts_local["external_eval_datasets"]

    external_eval_summary_df_local = split_artifacts_local.get("external_eval_summary_df")
    if external_eval_summary_df_local is None:
        external_eval_summary_df_local = pd.DataFrame(
            [
                {
                    "dataset": name,
                    "rows": len(df),
                    "classes": ", ".join(sorted(df[cfg["label_col"]].dropna().astype(str).unique().tolist())),
                }
                for name, df in globals()["external_eval_datasets"].items()
            ]
        ).sort_values(["rows", "dataset"], ascending=[False, True]).reset_index(drop=True)
    globals()["external_eval_summary_df"] = external_eval_summary_df_local

    if "external_eval_bundles" not in globals() or not globals()["external_eval_bundles"]:
        globals()["external_eval_bundles"] = {
            name: build_transformer_eval_bundle_from_df(df, dataset_name=name)
            for name, df in globals()["external_eval_datasets"].items()
        }

    return split_artifacts_local


def build_deep_model_specs(vocab: dict, embedding_matrix=None, cfg: dict = None) -> dict:
    cfg = cfg or CFG
    return {
        "GloVe MLP": {
            "family_slug": "glove_mlp",
            "builder": lambda: EmbeddingMLP(
                embedding_matrix=embedding_matrix,
                vocab_size=len(vocab),
                embed_dim=cfg["embedding_dim"],
                num_classes=cfg["num_labels"],
                hidden_dim=256,
                dropout=cfg["dropout"],
            ),
        },
        "GloVe CNN": {
            "family_slug": "glove_cnn",
            "builder": lambda: TextCNN(
                embedding_matrix=embedding_matrix,
                vocab_size=len(vocab),
                embed_dim=cfg["embedding_dim"],
                num_classes=cfg["num_labels"],
                num_filters=cfg["cnn_num_filters"],
                kernel_sizes=cfg["cnn_kernel_sizes"],
                dropout=cfg["dropout"],
            ),
        },
        "GloVe BiLSTM": {
            "family_slug": "glove_bilstm",
            "builder": lambda: BiLSTMAttention(
                embedding_matrix=embedding_matrix,
                vocab_size=len(vocab),
                embed_dim=cfg["embedding_dim"],
                num_classes=cfg["num_labels"],
                hidden_dim=cfg["lstm_hidden_dim"],
                num_layers=cfg["lstm_num_layers"],
                dropout=cfg["dropout"],
            ),
        },
    }


def restore_results_from_saved_artifacts(
    cfg: dict = None,
    reset: bool = True,
    include_eval: bool = True,
    include_learning_curves: bool = True,
):
    cfg = cfg or CFG
    metrics_dir = cfg["output_paths"]["metrics"]

    if reset:
        RESULTS.clear()

    restored = 0

    def _register(display_name: str, split: str, metrics: dict):
        nonlocal restored
        clean_metrics = _coerce_metric_dict(metrics)
        if not clean_metrics:
            return
        RESULTS[f"{display_name} | {split}"] = clean_metrics
        restored += 1

    def _load_df(csv_name: str, pkl_name: str = None):
        csv_path = metrics_dir / csv_name
        pkl_path = metrics_dir / pkl_name if pkl_name else None
        if pkl_path is not None and pkl_path.exists():
            try:
                obj = joblib.load(pkl_path)
                if isinstance(obj, pd.DataFrame):
                    return obj
            except Exception as e:
                print(f"Could not load {pkl_path}: {e}")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    baseline_eval_df_local = _load_df("baseline_eval_metrics.csv", "baseline_eval_metrics.pkl")
    if not baseline_eval_df_local.empty:
        globals()["baseline_eval_df"] = baseline_eval_df_local
        for row in baseline_eval_df_local.itertuples(index=False):
            model_tag = getattr(row, "model_tag", None)
            if model_tag is None or pd.isna(model_tag):
                split_key = getattr(row, "split_key", globals().get("train_pool_key", cfg.get("train_pool_key", "train_pool")))
                model_tag = f"{row.model} [{split_key}]"
            metrics = {
                "accuracy": float(getattr(row, "accuracy", np.nan)),
                "f1_macro": float(getattr(row, "f1_macro", np.nan)),
                "f1_weighted": float(getattr(row, "f1_weighted", np.nan)),
            }
            roc_auc = getattr(row, "roc_auc", np.nan)
            if pd.notna(roc_auc):
                metrics["roc_auc"] = float(roc_auc)
            _register(str(model_tag), str(row.split), metrics)

    if include_learning_curves:
        baseline_curve_df_local = _load_df("baseline_curve_metrics.csv", "baseline_curve_metrics.pkl")
        if baseline_curve_df_local.empty:
            baseline_curve_df_local = _load_df("classical_lc_metrics.csv", "classical_lc_metrics.pkl")
        if not baseline_curve_df_local.empty:
            globals()["baseline_curve_df"] = baseline_curve_df_local
            for row in baseline_curve_df_local.itertuples(index=False):
                RESULTS[f"LC | {row.model} | {row.split_key}"] = {"f1_macro": float(row.f1_macro)}

    deep_val_eval_df_local = _load_df("deep_eval_metrics_val.csv", "deep_eval_metrics_val.pkl")
    if not deep_val_eval_df_local.empty:
        globals()["deep_val_eval_df"] = deep_val_eval_df_local
        for row in deep_val_eval_df_local.itertuples(index=False):
            _register(
                f"{row.model} [{row.split_key}]",
                "val",
                {
                    "accuracy": float(getattr(row, "accuracy", np.nan)),
                    "f1_macro": float(getattr(row, "f1_macro", np.nan)),
                    "f1_weighted": float(getattr(row, "f1_weighted", np.nan)),
                },
            )

    deep_test_eval_df_local = _load_df("deep_eval_metrics_test.csv", "deep_eval_metrics_test.pkl")
    if not deep_test_eval_df_local.empty:
        globals()["deep_test_eval_df"] = deep_test_eval_df_local
        for row in deep_test_eval_df_local.itertuples(index=False):
            _register(
                f"{row.model} [{row.split_key}]",
                "test",
                {
                    "accuracy": float(getattr(row, "accuracy", np.nan)),
                    "f1_macro": float(getattr(row, "f1_macro", np.nan)),
                    "f1_weighted": float(getattr(row, "f1_weighted", np.nan)),
                },
            )

    deep_internal_eval_df_local = _load_df("deep_eval_metrics_internal.csv", "deep_eval_metrics_internal.pkl")
    if not deep_internal_eval_df_local.empty:
        globals()["deep_internal_eval_df"] = deep_internal_eval_df_local

    if include_learning_curves:
        deep_curve_df_local = _load_df("deep_curve_metrics.csv", "deep_curve_metrics.pkl")
        if not deep_curve_df_local.empty:
            globals()["deep_curve_df"] = deep_curve_df_local
            for row in deep_curve_df_local.itertuples(index=False):
                RESULTS[f"LC | {row.model} | {row.split_key}"] = {"f1_macro": float(row.f1_macro)}

        combined_curve_df_local = _load_df("combined_curve_metrics.csv", "combined_curve_metrics.pkl")
        if not combined_curve_df_local.empty:
            globals()["combined_curve_df"] = combined_curve_df_local

    transformer_summary_df_local = load_saved_transformer_summary(cfg)
    if not transformer_summary_df_local.empty:
        globals()["transformer_summary_df"] = transformer_summary_df_local
        if {"experiment", "best_model"}.issubset(transformer_summary_df_local.columns):
            best_row = (
                transformer_summary_df_local
                .sort_values(["val_f1_macro", "test_f1_macro", "experiment"], ascending=[False, False, True])
                .iloc[0]
            )
            best_model_path = Path(best_row["best_model"])
            if best_model_path.exists():
                globals()["FINAL_TRANSFORMER_NAME"] = str(best_row["experiment"])
                globals()["TRANSFORMER_FINAL_BEST_PATH"] = best_model_path

    transformer_metrics_dir = metrics_dir / "transformers"
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
                experiment = name_map.get(p.stem.replace("_latest", ""), p.stem.replace("_latest", ""))
                _register(experiment, "val", payload.get("val", {}))
                _register(experiment, "test", payload.get("test", {}))
            except Exception as e:
                print(f"Could not restore transformer metrics from {p}: {e}")

    if include_eval:
        cross_dataset_summary_df_local, cross_dataset_family_summary_df_local = load_saved_cross_dataset_summaries(cfg)
        if not cross_dataset_summary_df_local.empty:
            globals()["cross_dataset_summary_df"] = cross_dataset_summary_df_local
            globals()["baseline_external_eval_df"] = cross_dataset_summary_df_local.query("family == 'classical'").reset_index(drop=True)
            globals()["deep_external_eval_df"] = cross_dataset_summary_df_local.query("family == 'deep'").reset_index(drop=True)
            globals()["transformer_external_summary_df"] = cross_dataset_summary_df_local.query("family == 'transformer'").reset_index(drop=True)
            for row in cross_dataset_summary_df_local.itertuples(index=False):
                _register(
                    f"{row.model} [{row.dataset}]",
                    "eval",
                    {
                        "accuracy": float(getattr(row, "accuracy", np.nan)),
                        "f1_macro": float(getattr(row, "f1_macro", np.nan)),
                        "f1_weighted": float(getattr(row, "f1_weighted", np.nan)),
                    },
                )
        if not cross_dataset_family_summary_df_local.empty:
            globals()["cross_dataset_family_summary_df"] = cross_dataset_family_summary_df_local

    return restored


def restore_final_transformer_test_artifacts(cfg: dict = None):
    cfg = cfg or CFG
    ensure_split_artifacts_loaded(cfg)

    shared_model_dir = cfg["model_dir"] / "shared"
    if "le" not in globals():
        globals()["le"] = load_saved_label_encoder(shared_model_dir, globals().get("class2id"))

    best_name, best_path = resolve_final_transformer_selection(cfg)
    globals()["FINAL_TRANSFORMER_NAME"] = best_name
    globals()["TRANSFORMER_FINAL_BEST_PATH"] = best_path

    split_key = globals().get("train_pool_key", cfg.get("train_pool_key", "train_pool"))
    split = globals()["splits"][split_key]
    globals()["trf_df_test"] = split["df_test"].copy().reset_index(drop=True)
    globals()["trf_y_test"] = np.asarray(split["y_test"], dtype=np.int64)

    transformer_summary_df_local = load_saved_transformer_summary(cfg)
    experiment_dir = None
    if not transformer_summary_df_local.empty and {"experiment", "experiment_dir"}.issubset(transformer_summary_df_local.columns):
        matched = transformer_summary_df_local.loc[
            transformer_summary_df_local["experiment"] == best_name
        ]
        if not matched.empty:
            experiment_dir = matched.iloc[0].get("experiment_dir")

    preds_candidates = []
    if experiment_dir:
        runs_dir = Path(experiment_dir) / "runs"
        preds_candidates.append(runs_dir / "test_preds_latest.npy")
        preds_candidates.extend(sorted(runs_dir.glob("test_preds_*.npy")))

    exp_root = Path(best_path).parent
    runs_dir = exp_root / "runs"
    preds_candidates.append(runs_dir / "test_preds_latest.npy")
    preds_candidates.extend(sorted(runs_dir.glob("test_preds_*.npy")))

    seen = set()
    for candidate in preds_candidates:
        candidate = Path(candidate)
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.exists():
            globals()["trf_test_preds"] = np.load(candidate)
            return {
                "experiment": best_name,
                "best_model": best_path,
                "preds_path": candidate,
            }

    globals()["trf_test_preds"] = np.asarray([], dtype=np.int64)
    return {
        "experiment": best_name,
        "best_model": best_path,
        "preds_path": None,
    }




def load_transformer_registry(cfg: dict = None, transformer_runs: dict | None = None, summary_df: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = cfg or CFG
    frames = []
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        frames.append(summary_df.copy())
    if transformer_runs:
        frames.append(transformer_summary_from_runs(transformer_runs))
    frames.append(load_saved_transformer_summary(cfg))

    df = merge_transformer_summary_frames(*frames)
    required = {'experiment', 'best_model'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Transformer registry is missing columns: {sorted(missing)}')

    return (
        df.dropna(subset=['best_model'])
        .loc[df['best_model'].astype(str).str.len() > 0]
        .drop_duplicates(subset=['experiment'], keep='first')
        .sort_values(['val_f1_macro', 'test_f1_macro', 'experiment'], ascending=[False, False, True])
        .reset_index(drop=True)
    )
