#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return rel(dst)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_slashes(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def infer_family(model_name: str) -> str:
    name = model_name.lower()
    if "deberta" in name or "distilbert" in name or "roberta" in name or "bert" in name:
        return "transformer"
    if "glove" in name:
        return "deep"
    if "tf-idf" in name or "majority" in name:
        return "classical"
    return "unknown"


def normalize_display_name(name: str) -> str:
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_train_size(token: str) -> int | None:
    token = token.lower()
    if token == "train_pool":
        return 31305
    match = re.search(r"(\d+)(k)?", token)
    if not match:
        return None
    value = int(match.group(1))
    if match.group(2):
        value *= 1000
    return value


def parse_results_json(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payload = load_json(results_path)
    for key, metrics in payload.items():
        if " | " not in key:
            continue
        model, split = key.rsplit(" | ", 1)
        rows.append(
            {
                "model": model,
                "display_name": normalize_display_name(model),
                "family": infer_family(model),
                "split": split,
                "accuracy": clean_float(metrics.get("accuracy")),
                "f1_macro": clean_float(metrics.get("f1_macro")),
                "f1_weighted": clean_float(metrics.get("f1_weighted")),
                "loss": clean_float(metrics.get("loss")),
                "source_file": rel(results_path),
            }
        )
    return rows


def parse_pipe_results_csv(results_path: Path) -> list[dict[str, Any]]:
    df = load_csv(results_path)
    first_col = df.columns[0]
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        model_split = str(item[first_col]).strip()
        if not model_split or model_split.startswith("LC | "):
            continue
        if " | " not in model_split:
            continue
        model, split = model_split.rsplit(" | ", 1)
        rows.append(
            {
                "model": model,
                "display_name": normalize_display_name(model),
                "family": infer_family(model),
                "split": split,
                "accuracy": clean_float(item.get("accuracy")),
                "f1_macro": clean_float(item.get("f1_macro")),
                "f1_weighted": clean_float(item.get("f1_weighted")),
                "loss": clean_float(item.get("loss")),
                "source_file": rel(results_path),
            }
        )
    return rows


def parse_learning_curve_csv(curve_path: Path) -> list[dict[str, Any]]:
    df = load_csv(curve_path)
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        split_key = str(item.get("split_key", "")).strip()
        rows.append(
            {
                "model": str(item["model"]).strip(),
                "display_name": normalize_display_name(str(item["model"]).strip()),
                "family": infer_family(str(item["model"]).strip()),
                "split_key": split_key,
                "train_size": int(item["train_size"]) if not pd.isna(item["train_size"]) else None,
                "f1_macro": clean_float(item.get("f1_macro")),
                "source_file": rel(curve_path),
            }
        )
    return rows


def parse_hf_log_history(entries: list[dict[str, Any]], source_file: Path) -> dict[str, Any]:
    train_events: list[dict[str, Any]] = []
    eval_events: list[dict[str, Any]] = []
    final_summary: dict[str, Any] | None = None
    for entry in entries:
        base = {
            "epoch": clean_float(entry.get("epoch")),
            "step": clean_float(entry.get("step")),
        }
        if "eval_loss" in entry or "eval_f1_macro" in entry or "eval_accuracy" in entry:
            eval_events.append(
                {
                    **base,
                    "eval_loss": clean_float(entry.get("eval_loss")),
                    "eval_f1_macro": clean_float(entry.get("eval_f1_macro")),
                    "eval_accuracy": clean_float(entry.get("eval_accuracy")),
                    "learning_rate": clean_float(entry.get("learning_rate")),
                    "grad_norm": clean_float(entry.get("grad_norm")),
                }
            )
        elif "loss" in entry:
            train_events.append(
                {
                    **base,
                    "loss": clean_float(entry.get("loss")),
                    "learning_rate": clean_float(entry.get("learning_rate")),
                    "grad_norm": clean_float(entry.get("grad_norm")),
                }
            )
        elif "train_loss" in entry:
            final_summary = {
                "train_loss": clean_float(entry.get("train_loss")),
                "train_runtime": clean_float(entry.get("train_runtime")),
                "train_steps_per_second": clean_float(entry.get("train_steps_per_second")),
                "train_samples_per_second": clean_float(entry.get("train_samples_per_second")),
                "epoch": clean_float(entry.get("epoch")),
                "step": clean_float(entry.get("step")),
            }

    best_eval = None
    eval_with_metric = [item for item in eval_events if item.get("eval_f1_macro") is not None]
    if eval_with_metric:
        best_eval = max(eval_with_metric, key=lambda item: item["eval_f1_macro"])

    return {
        "source_file": rel(source_file),
        "x_axis": "step",
        "train_events": train_events,
        "eval_events": eval_events,
        "best_eval_event": best_eval,
        "final_summary": final_summary,
    }


def parse_epoch_log(log_path: Path) -> dict[str, Any]:
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+\|\s+tr_loss=(?P<tr_loss>[0-9.]+)\s+\|\s+"
        r"va_loss=(?P<va_loss>[0-9.]+)\s+\|\s+val_f1=(?P<val_f1>[0-9.]+)\s+\|\s+(?P<seconds>[0-9.]+)s"
    )
    points: list[dict[str, Any]] = []
    notes: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            points.append(
                {
                    "epoch": int(match.group("epoch")),
                    "total_epochs": int(match.group("epochs")),
                    "train_loss": float(match.group("tr_loss")),
                    "val_loss": float(match.group("va_loss")),
                    "val_f1_macro": float(match.group("val_f1")),
                    "epoch_seconds": float(match.group("seconds")),
                }
            )
        elif "Early stopping triggered" in line or "Restored best checkpoint" in line:
            notes.append(line.split("|", 2)[-1].strip())
    best_point = max(points, key=lambda item: item["val_f1_macro"]) if points else None
    return {
        "source_file": rel(log_path),
        "x_axis": "epoch",
        "points": points,
        "best_eval_event": best_point,
        "notes": notes,
    }


def copy_confusion_images(
    dashboard_dir: Path,
    image_sources: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for image_id, src in image_sources:
        if not src.exists():
            continue
        dst = dashboard_dir / "confusion" / f"{image_id}.png"
        outputs.append(
            {
                "id": image_id,
                "file": copy_file(src, dst),
                "source_file": rel(src),
            }
        )
    return outputs


def copy_supporting_images(
    dashboard_dir: Path,
    image_sources: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for image_id, src in image_sources:
        if not src.exists():
            continue
        dst = dashboard_dir / "figures" / "images" / f"{image_id}.png"
        outputs.append(
            {
                "id": image_id,
                "file": copy_file(src, dst),
                "source_file": rel(src),
            }
        )
    return outputs


def make_bar_figure(
    title: str,
    x_values: list[Any],
    y_values: list[Any],
    *,
    x_title: str,
    y_title: str,
    text_values: list[str] | None = None,
    color: str = "#0f766e",
    orientation: str = "v",
) -> dict[str, Any]:
    data = [
        {
            "type": "bar",
            "orientation": orientation,
            "x": x_values if orientation == "v" else y_values,
            "y": y_values if orientation == "v" else x_values,
            "marker": {"color": color},
            "text": text_values,
            "textposition": "auto",
        }
    ]
    layout = {
        "title": {"text": title},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "xaxis": {"title": {"text": x_title}, "automargin": True},
        "yaxis": {"title": {"text": y_title}, "automargin": True},
    }
    return {"data": data, "layout": layout}


def make_multi_line_figure(
    title: str,
    series: list[dict[str, Any]],
    *,
    x_title: str,
    y_title: str,
) -> dict[str, Any]:
    colors = [
        "#0f766e",
        "#b45309",
        "#1d4ed8",
        "#be123c",
        "#7c3aed",
        "#475569",
    ]
    data = []
    for index, item in enumerate(series):
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": item["name"],
                "x": item["x"],
                "y": item["y"],
                "line": {"color": colors[index % len(colors)], "width": 2},
                "marker": {"size": 6},
            }
        )
    layout = {
        "title": {"text": title},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "xaxis": {"title": {"text": x_title}, "automargin": True},
        "yaxis": {"title": {"text": y_title}, "automargin": True},
        "legend": {"orientation": "h", "y": 1.12},
    }
    return {"data": data, "layout": layout}


def write_figure(dashboard_dir: Path, figure_id: str, figure: dict[str, Any]) -> str:
    path = dashboard_dir / "figures" / f"{figure_id}.plotly.json"
    write_json(path, figure)
    return rel(path)


def build_source_audit(
    domain_root: Path,
    prod_dir: Path,
    selected_sources: list[dict[str, Any]],
    notes: list[str],
) -> dict[str, Any]:
    candidate_categories: dict[str, list[str]] = defaultdict(list)
    interesting_exts = {".json", ".csv", ".pkl", ".parquet", ".png", ".jpg", ".jpeg", ".svg", ".log", ".npy", ".ipynb"}
    for scan_root in [domain_root, prod_dir]:
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == ".DS_Store":
                continue
            suffix = path.suffix.lower()
            path_str = rel(path)
            name = path.name.lower()
            if suffix not in interesting_exts:
                continue
            if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                category = "plot_images"
            elif suffix == ".log":
                category = "logs"
            elif suffix == ".ipynb":
                category = "notebooks"
            elif suffix == ".npy":
                category = "prediction_arrays"
            elif suffix in {".pkl", ".parquet"}:
                category = "structured_data"
            elif "metric" in name or "result" in name or "summary" in name or "cfg_" in name:
                category = "metrics_and_configs"
            else:
                category = "other_structured_files"
            candidate_categories[category].append(path_str)

    artifact_counts = {key: len(value) for key, value in candidate_categories.items()}
    return {
        "generated_at": GENERATED_AT,
        "scanned_roots": [rel(domain_root), rel(prod_dir)],
        "artifact_counts": artifact_counts,
        "candidate_artifacts_by_category": dict(candidate_categories),
        "selected_dashboard_sources": selected_sources,
        "notes": notes,
    }


def build_overview(
    *,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    production_benchmark_name: str,
    cross_dataset_rows: list[dict[str, Any]] | None,
    section_status: dict[str, str],
    notes: list[str],
) -> dict[str, Any]:
    test_rows = [row for row in benchmark_rows if row.get("split") == "test" and row.get("f1_macro") is not None]
    ranked_rows = sorted(test_rows, key=lambda row: row["f1_macro"], reverse=True)
    rank_info = None
    for index, row in enumerate(ranked_rows, start=1):
        if row["model"] == production_benchmark_name:
            rank_info = {
                "rank": index,
                "out_of": len(ranked_rows),
                "metric": "f1_macro",
                "score": row["f1_macro"],
            }
            break

    cross_dataset_summary = None
    if cross_dataset_rows:
        valid_rows = [row for row in cross_dataset_rows if row.get("f1_macro") is not None]
        if valid_rows:
            cross_dataset_summary = {
                "datasets": len(valid_rows),
                "mean_accuracy": round(sum(row["accuracy"] for row in valid_rows) / len(valid_rows), 6),
                "mean_f1_macro": round(sum(row["f1_macro"] for row in valid_rows) / len(valid_rows), 6),
                "mean_f1_weighted": round(sum(row["f1_weighted"] for row in valid_rows) / len(valid_rows), 6),
            }

    return {
        "generated_at": GENERATED_AT,
        "model": {
            "model_id": metadata.get("model_id"),
            "display_name": metadata.get("display_name"),
            "domain": metadata.get("domain"),
            "framework_type": metadata.get("framework", {}).get("type"),
            "framework_architecture": metadata.get("framework", {}).get("architecture")
            or metadata.get("framework", {}).get("backbone")
            or metadata.get("framework", {}).get("base_model"),
        },
        "evaluation_highlights": {
            "validation": evaluation.get("splits", {}).get("val"),
            "test": evaluation.get("splits", {}).get("test"),
            "benchmark_rank": rank_info,
        },
        "cross_dataset_highlights": cross_dataset_summary,
        "artifact_status": section_status,
        "notes": notes,
    }


def section_record(
    section_id: str,
    title: str,
    status: str,
    files: list[str],
    *,
    description: str,
    reason: str | None = None,
    charts: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": section_id,
        "title": title,
        "status": status,
        "description": description,
        "files": files,
    }
    if reason:
        payload["reason"] = reason
    if charts:
        payload["charts"] = charts
    return payload


def build_manifest(
    *,
    dashboard_dir: Path,
    metadata: dict[str, Any],
    overview_file: str,
    source_audit_file: str,
    sections: list[dict[str, Any]],
    selected_sources: list[dict[str, Any]],
    notes: list[str],
) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": GENERATED_AT,
        "dashboard_root": rel(dashboard_dir),
        "model": {
            "model_id": metadata.get("model_id"),
            "display_name": metadata.get("display_name"),
            "domain": metadata.get("domain"),
            "description": metadata.get("description"),
        },
        "entrypoints": {
            "overview": overview_file,
            "source_audit": source_audit_file,
        },
        "sections": sections,
        "selected_sources": selected_sources,
        "notes": notes,
    }
    write_json(dashboard_dir / "dashboard-manifest.json", manifest)


def sort_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        return (row.get("f1_macro") or -math.inf, row.get("accuracy") or -math.inf)

    return sorted(rows, key=key, reverse=True)


def top_test_rows(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row.get("split") == "test"]
    return sort_benchmark_rows(test_rows)[:limit]


def model_metadata(prod_dir: Path, extra_runtime_config: Path | None = None) -> dict[str, Any]:
    metadata = load_yaml(prod_dir / "model-config.yaml")
    runtime_config = None
    if extra_runtime_config and extra_runtime_config.exists():
        runtime_config = load_json(extra_runtime_config)
    artifacts_present = sorted(
        rel(path)
        for path in prod_dir.iterdir()
        if path.name != "dashboard" and not path.name.startswith(".ipynb_checkpoints")
    )
    metadata["prod_model_dir"] = rel(prod_dir)
    metadata["runtime_config"] = runtime_config
    metadata["artifacts_present"] = artifacts_present
    return metadata


def prepare_metadata_files(
    dashboard_dir: Path,
    metadata: dict[str, Any],
    experiment_config: dict[str, Any] | None,
) -> list[str]:
    files = []
    model_path = dashboard_dir / "metadata" / "model.json"
    write_json(model_path, metadata)
    files.append(rel(model_path))
    if experiment_config is not None:
        config_path = dashboard_dir / "metadata" / "experiment-config.json"
        write_json(config_path, experiment_config)
        files.append(rel(config_path))
    return files


def build_abuse_dashboard() -> None:
    prod_dir = REPO_ROOT / "app" / "app-models" / "prod-model-abuse"
    dashboard_dir = prod_dir / "dashboard"
    domain_root = REPO_ROOT / "abuse"

    notes = [
        "Dashboard metadata uses prod-model YAML labels as the source of truth.",
        "The original abuse CFG reports num_labels=3, while the production model-config and exported transformer config expose 5 labels.",
    ]
    selected_sources: list[dict[str, Any]] = []

    metadata = model_metadata(prod_dir, prod_dir / "config.json")
    experiment_config = load_json(domain_root / "outputs" / "cfg_20260222_1241.json")
    metadata_files = prepare_metadata_files(dashboard_dir, metadata, experiment_config)
    selected_sources.extend(
        [
            {"category": "metadata", "path": rel(prod_dir / "model-config.yaml"), "reason": "Production model identity and label mapping."},
            {"category": "metadata", "path": rel(prod_dir / "config.json"), "reason": "Exported transformer architecture config."},
            {"category": "config", "path": rel(domain_root / "outputs" / "cfg_20260222_1241.json"), "reason": "Training/evaluation runtime configuration for the archived BERT run."},
        ]
    )

    benchmark_rows = parse_results_json(domain_root / "outputs" / "results_20260222_1241.json")
    selected_sources.append(
        {"category": "metrics", "path": rel(domain_root / "outputs" / "results_20260222_1241.json"), "reason": "Unified validation/test benchmark table across abuse models."}
    )

    production_name = "BERT Fine-tune"
    eval_row_map = {row["split"]: row for row in benchmark_rows if row["model"] == production_name}
    evaluation = {
        "model": production_name,
        "source_files": [rel(domain_root / "outputs" / "results_20260222_1241.json")],
        "splits": {
            split: {
                key: value
                for key, value in row.items()
                if key in {"accuracy", "f1_macro", "f1_weighted", "loss"}
            }
            for split, row in eval_row_map.items()
        },
    }
    evaluation_path = dashboard_dir / "metrics" / "primary-evaluation.json"
    write_json(evaluation_path, evaluation)

    benchmark_test = top_test_rows(benchmark_rows, limit=10)
    for row in benchmark_test:
        row["is_production"] = row["model"] == production_name
    benchmark_path = dashboard_dir / "metrics" / "benchmark-test.json"
    write_json(benchmark_path, benchmark_test)

    training_history = parse_hf_log_history(
        load_json(domain_root / "logs" / "trainer_log_history_20260222_124054.json"),
        domain_root / "logs" / "trainer_log_history_20260222_124054.json",
    )
    selected_sources.append(
        {"category": "training_history", "path": rel(domain_root / "logs" / "trainer_log_history_20260222_124054.json"), "reason": "Per-step Hugging Face training/evaluation history for the BERT run."}
    )
    curves_path = dashboard_dir / "curves" / "training-history.json"
    write_json(curves_path, training_history)

    confusion_files = copy_confusion_images(
        dashboard_dir,
        [("test-confusion-matrix", domain_root / "outputs" / "cm_BERT_Fine-tune.png")],
    )
    if confusion_files:
        selected_sources.append(
            {"category": "confusion_matrix", "path": rel(domain_root / "outputs" / "cm_BERT_Fine-tune.png"), "reason": "Saved test confusion matrix image for the production abuse transformer."}
        )

    benchmark_figure = make_bar_figure(
        "Abuse Test F1 Macro Benchmark",
        [row["display_name"] for row in benchmark_test],
        [row["f1_macro"] for row in benchmark_test],
        x_title="Model",
        y_title="F1 Macro",
        text_values=[f"{row['f1_macro']:.3f}" for row in benchmark_test],
    )
    benchmark_fig_file = write_figure(dashboard_dir, "benchmark-test-f1", benchmark_figure)

    loss_figure = make_multi_line_figure(
        "Abuse BERT Training Loss",
        [
            {"name": "Train Loss", "x": [item["step"] for item in training_history["train_events"]], "y": [item["loss"] for item in training_history["train_events"]]},
            {"name": "Eval Loss", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_loss"] for item in training_history["eval_events"]]},
        ],
        x_title="Training Step",
        y_title="Loss",
    )
    loss_fig_file = write_figure(dashboard_dir, "training-loss", loss_figure)

    eval_metric_figure = make_multi_line_figure(
        "Abuse BERT Evaluation Metrics",
        [
            {"name": "Eval F1 Macro", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_f1_macro"] for item in training_history["eval_events"]]},
            {"name": "Eval Accuracy", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_accuracy"] for item in training_history["eval_events"]]},
        ],
        x_title="Training Step",
        y_title="Score",
    )
    eval_metric_fig_file = write_figure(dashboard_dir, "eval-metrics", eval_metric_figure)

    section_status = {
        "metadata": "available",
        "summary": "available",
        "evaluation": "available",
        "benchmark": "available",
        "training_curves": "available",
        "confusion_matrix": "image_only" if confusion_files else "missing",
        "class_distribution": "missing",
        "samples": "missing",
        "cross_dataset": "missing",
    }

    source_audit = build_source_audit(domain_root, prod_dir, selected_sources, notes)
    source_audit_path = dashboard_dir / "summary" / "source-audit.json"
    write_json(source_audit_path, source_audit)

    overview = build_overview(
        metadata=metadata,
        evaluation=evaluation,
        benchmark_rows=benchmark_rows,
        production_benchmark_name=production_name,
        cross_dataset_rows=None,
        section_status=section_status,
        notes=notes,
    )
    overview_path = dashboard_dir / "summary" / "overview.json"
    write_json(overview_path, overview)

    sections = [
        section_record("metadata", "Model Metadata", "available", metadata_files, description="Production model identity, runtime export config, and label mapping."),
        section_record("summary", "Summary", "available", [rel(overview_path), rel(source_audit_path)], description="Dashboard readiness overview and full source audit."),
        section_record("evaluation", "Primary Evaluation", "available", [rel(evaluation_path)], description="Validation and test metrics for the production BERT abuse model."),
        section_record("benchmark", "Benchmark", "available", [rel(benchmark_path), benchmark_fig_file], description="Top internal test benchmark rows across archived abuse models.", charts=["benchmark-test-f1"]),
        section_record("training_curves", "Training Curves", "available", [rel(curves_path), loss_fig_file, eval_metric_fig_file], description="Per-step BERT training and evaluation history from the saved trainer log.", charts=["training-loss", "eval-metrics"]),
        section_record(
            "confusion_matrix",
            "Confusion Matrix",
            "image_only" if confusion_files else "missing",
            [item["file"] for item in confusion_files],
            description="Saved confusion matrix image for the production abuse transformer.",
            reason=None if confusion_files else "No raw prediction arrays or numeric confusion matrix payload were found for the production model.",
        ),
        section_record(
            "class_distribution",
            "Class Distribution",
            "missing",
            [],
            description="Class distribution for the production abuse dataset.",
            reason="No source dataset or saved label-count artifact is present in the repository for the production abuse run.",
        ),
        section_record(
            "samples",
            "Prediction Samples",
            "missing",
            [],
            description="Example abuse predictions for dashboard display.",
            reason="No saved text-level prediction sample artifact was found for the production abuse model.",
        ),
        section_record(
            "cross_dataset",
            "Cross Dataset Evaluation",
            "missing",
            [],
            description="External evaluation results outside the main abuse split.",
            reason="No cross-dataset evaluation artifacts were found for the abuse domain.",
        ),
    ]

    build_manifest(
        dashboard_dir=dashboard_dir,
        metadata=metadata,
        overview_file=rel(overview_path),
        source_audit_file=rel(source_audit_path),
        sections=sections,
        selected_sources=selected_sources,
        notes=notes,
    )


def build_age_dashboard() -> None:
    prod_dir = REPO_ROOT / "app" / "app-models" / "prod-model-age"
    dashboard_dir = prod_dir / "dashboard"
    domain_root = REPO_ROOT / "age"

    notes = [
        "Primary evaluation uses the production DistilBERT_1000k metrics JSON and prod model-config YAML.",
        "The repository does not contain the raw 1,000,000-sample reddit parquet or saved 1000k split artifact, so class distributions and numeric confusion matrices for the prod model cannot be derived confidently.",
        "Checkpoint trainer_state.json and post-hoc metrics_latest.json disagree on the absolute best validation score; the dashboard uses metrics_latest.json for primary evaluation and trainer_state.json only for the training curve.",
    ]
    selected_sources: list[dict[str, Any]] = []

    metadata = model_metadata(prod_dir, prod_dir / "config.json")
    experiment_config = load_json(domain_root / "outputs" / "metrics" / "cfg_20260311_1817.json")
    metadata_files = prepare_metadata_files(dashboard_dir, metadata, experiment_config)
    selected_sources.extend(
        [
            {"category": "metadata", "path": rel(prod_dir / "model-config.yaml"), "reason": "Production model identity and label mapping."},
            {"category": "metadata", "path": rel(prod_dir / "config.json"), "reason": "Exported DistilBERT architecture config with explicit labels."},
            {"category": "config", "path": rel(domain_root / "outputs" / "metrics" / "cfg_20260311_1817.json"), "reason": "Latest age evaluation configuration for the prod-facing metrics layer."},
        ]
    )

    primary_metrics_path = domain_root / "outputs" / "metrics" / "transformers" / "distilbert_1000k_latest.json"
    primary_metrics = load_json(primary_metrics_path)
    evaluation = {
        "model": "DistilBERT_1000k",
        "source_files": [rel(primary_metrics_path)],
        "splits": {split: primary_metrics[split] for split in ["val", "test"] if split in primary_metrics},
        "artifact_paths": {
            key: normalize_slashes(value)
            for key, value in primary_metrics.items()
            if key not in {"val", "test"}
        },
    }
    evaluation_path = dashboard_dir / "metrics" / "primary-evaluation.json"
    write_json(evaluation_path, evaluation)
    selected_sources.append(
        {"category": "metrics", "path": rel(primary_metrics_path), "reason": "Primary val/test metrics for the production age model."}
    )

    benchmark_rows = parse_pipe_results_csv(domain_root / "outputs" / "metrics" / "results_age-classification.csv")
    benchmark_test = top_test_rows(benchmark_rows, limit=10)
    for row in benchmark_test:
        row["is_production"] = row["model"] == "DistilBERT_1000k"
    benchmark_path = dashboard_dir / "metrics" / "benchmark-test.json"
    write_json(benchmark_path, benchmark_test)
    selected_sources.append(
        {"category": "benchmark", "path": rel(domain_root / "outputs" / "metrics" / "results_age-classification.csv"), "reason": "Combined internal age leaderboard across transformer, deep, and classical models."}
    )

    trainer_state_path = domain_root / "models" / "distilbert_reddit_1000k_20260223_233842" / "checkpoints" / "checkpoint-84375" / "trainer_state.json"
    training_history = parse_hf_log_history(load_json(trainer_state_path)["log_history"], trainer_state_path)
    curves_path = dashboard_dir / "curves" / "training-history.json"
    write_json(curves_path, training_history)
    selected_sources.append(
        {"category": "training_history", "path": rel(trainer_state_path), "reason": "Step-level DistilBERT training curve from checkpoint trainer state."}
    )

    learning_curve_rows = parse_learning_curve_csv(domain_root / "outputs" / "metrics" / "baseline_curve_metrics.csv")
    learning_curve_path = dashboard_dir / "curves" / "learning-curve.json"
    write_json(learning_curve_path, learning_curve_rows)
    selected_sources.append(
        {"category": "learning_curve", "path": rel(domain_root / "outputs" / "metrics" / "baseline_curve_metrics.csv"), "reason": "Age classical sample-size curve data for dashboard comparison plots."}
    )

    cross_summary = load_csv(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv")
    cross_rows = cross_summary.to_dict(orient="records")
    for row in cross_rows:
        for metric in ["accuracy", "f1_macro", "f1_weighted"]:
            row[metric] = clean_float(row[metric])
        row["source_file"] = rel(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv")
    cross_path = dashboard_dir / "metrics" / "cross-dataset.json"
    write_json(cross_path, cross_rows)
    selected_sources.append(
        {"category": "cross_dataset", "path": rel(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv"), "reason": "Age external evaluation summary for the production DistilBERT model."}
    )

    confusion_files = copy_confusion_images(
        dashboard_dir,
        [
            ("internal-test-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_1000k.png"),
            ("external-blog-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_1000k_blog.png"),
            ("external-hippocorpus-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_1000k_hippocorpus.png"),
            ("external-pan-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_1000k_pan13_14_15.png"),
            ("external-tweets-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_1000k_tweets.png"),
        ],
    )
    if confusion_files:
        selected_sources.append(
            {"category": "confusion_matrix", "path": rel(domain_root / "outputs" / "plots" / "confusion_matrices"), "reason": "Saved internal and external confusion matrix images for DistilBERT_1000k."}
        )

    benchmark_fig_file = write_figure(
        dashboard_dir,
        "benchmark-test-f1",
        make_bar_figure(
            "Age Test F1 Macro Benchmark",
            [row["display_name"] for row in benchmark_test],
            [row["f1_macro"] for row in benchmark_test],
            x_title="Model",
            y_title="F1 Macro",
            text_values=[f"{row['f1_macro']:.3f}" for row in benchmark_test],
        ),
    )

    loss_fig_file = write_figure(
        dashboard_dir,
        "training-loss",
        make_multi_line_figure(
            "Age DistilBERT Training Loss",
            [
                {"name": "Train Loss", "x": [item["step"] for item in training_history["train_events"]], "y": [item["loss"] for item in training_history["train_events"]]},
                {"name": "Eval Loss", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_loss"] for item in training_history["eval_events"]]},
            ],
            x_title="Training Step",
            y_title="Loss",
        ),
    )

    eval_metric_fig_file = write_figure(
        dashboard_dir,
        "eval-metrics",
        make_multi_line_figure(
            "Age DistilBERT Evaluation Metrics",
            [
                {"name": "Eval F1 Macro", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_f1_macro"] for item in training_history["eval_events"]]},
                {"name": "Eval Accuracy", "x": [item["step"] for item in training_history["eval_events"]], "y": [item["eval_accuracy"] for item in training_history["eval_events"]]},
            ],
            x_title="Training Step",
            y_title="Score",
        ),
    )

    learning_series = []
    for model_name, group in pd.DataFrame(learning_curve_rows).groupby("display_name"):
        ordered = group.sort_values("train_size")
        learning_series.append(
            {"name": model_name, "x": ordered["train_size"].tolist(), "y": ordered["f1_macro"].tolist()}
        )
    learning_curve_fig_file = write_figure(
        dashboard_dir,
        "learning-curve-f1",
        make_multi_line_figure(
            "Age Classical Learning Curves",
            learning_series,
            x_title="Train Size",
            y_title="Validation F1 Macro",
        ),
    )

    cross_fig_file = write_figure(
        dashboard_dir,
        "cross-dataset-f1",
        make_bar_figure(
            "Age External Evaluation F1 Macro",
            [row["dataset"] for row in cross_rows],
            [row["f1_macro"] for row in cross_rows],
            x_title="Dataset",
            y_title="F1 Macro",
            text_values=[f"{row['f1_macro']:.3f}" for row in cross_rows],
            color="#b45309",
        ),
    )

    section_status = {
        "metadata": "available",
        "summary": "available",
        "evaluation": "available",
        "benchmark": "available",
        "training_curves": "available",
        "learning_curves": "available",
        "confusion_matrix": "image_only" if confusion_files else "missing",
        "class_distribution": "missing",
        "samples": "missing",
        "cross_dataset": "available",
    }

    source_audit = build_source_audit(domain_root, prod_dir, selected_sources, notes)
    source_audit_path = dashboard_dir / "summary" / "source-audit.json"
    write_json(source_audit_path, source_audit)

    overview = build_overview(
        metadata=metadata,
        evaluation=evaluation,
        benchmark_rows=benchmark_rows,
        production_benchmark_name="DistilBERT_1000k",
        cross_dataset_rows=cross_rows,
        section_status=section_status,
        notes=notes,
    )
    overview_path = dashboard_dir / "summary" / "overview.json"
    write_json(overview_path, overview)

    sections = [
        section_record("metadata", "Model Metadata", "available", metadata_files, description="Production model identity, export config, and matching evaluation config."),
        section_record("summary", "Summary", "available", [rel(overview_path), rel(source_audit_path)], description="Dashboard readiness overview and full source audit."),
        section_record("evaluation", "Primary Evaluation", "available", [rel(evaluation_path)], description="Validation and test metrics for the production age DistilBERT model."),
        section_record("benchmark", "Benchmark", "available", [rel(benchmark_path), benchmark_fig_file], description="Top internal test benchmark rows across age models.", charts=["benchmark-test-f1"]),
        section_record("training_curves", "Training Curves", "available", [rel(curves_path), loss_fig_file, eval_metric_fig_file], description="Step-level DistilBERT training and evaluation history from checkpoint trainer state.", charts=["training-loss", "eval-metrics"]),
        section_record("learning_curves", "Learning Curves", "available", [rel(learning_curve_path), learning_curve_fig_file], description="Sample-size learning curve data across classical age baselines.", charts=["learning-curve-f1"]),
        section_record("cross_dataset", "Cross Dataset Evaluation", "available", [rel(cross_path), cross_fig_file], description="External evaluation summary across blog, hippocorpus, PAN, and tweets.", charts=["cross-dataset-f1"]),
        section_record(
            "confusion_matrix",
            "Confusion Matrix",
            "image_only" if confusion_files else "missing",
            [item["file"] for item in confusion_files],
            description="Saved confusion matrix images for internal and external DistilBERT_1000k evaluation runs.",
            reason=None if confusion_files else "No confusion matrix images were found for the production age model.",
        ),
        section_record(
            "class_distribution",
            "Class Distribution",
            "missing",
            [],
            description="Class distribution for the production age training/eval splits.",
            reason="The raw 1,000,000-sample age dataset/split artifact is not present in the repository, so numeric prod-model class counts cannot be derived confidently.",
        ),
        section_record(
            "samples",
            "Prediction Samples",
            "missing",
            [],
            description="Example age predictions for dashboard display.",
            reason="No saved text-level prediction sample artifact was found for the production age model.",
        ),
    ]

    build_manifest(
        dashboard_dir=dashboard_dir,
        metadata=metadata,
        overview_file=rel(overview_path),
        source_audit_file=rel(source_audit_path),
        sections=sections,
        selected_sources=selected_sources,
        notes=notes,
    )


def build_complexity_dashboard() -> None:
    prod_dir = REPO_ROOT / "app" / "app-models" / "prod-model-complexity"
    dashboard_dir = prod_dir / "dashboard"
    domain_root = REPO_ROOT / "complexity"

    notes = [
        "Primary evaluation uses the production GloVe BiLSTM metrics_latest.json referenced by prod-model-complexity/model-config.yaml.",
        "Complexity is the richest dashboard domain in the repo: internal benchmark data, external evaluation summaries, split parquets, manual sample predictions, and prod-model training logs are all present.",
        "Confusion matrices exist only as saved images for the production model; no numeric matrix artifact was found.",
    ]
    selected_sources: list[dict[str, Any]] = []

    metadata = model_metadata(prod_dir, None)
    experiment_config = load_json(domain_root / "outputs" / "metrics" / "cfg_20260310_2125.json")
    metadata_files = prepare_metadata_files(dashboard_dir, metadata, experiment_config)
    selected_sources.extend(
        [
            {"category": "metadata", "path": rel(prod_dir / "model-config.yaml"), "reason": "Production model identity and architecture metadata."},
            {"category": "metadata", "path": rel(domain_root / "models" / "shared" / "label_classes.json"), "reason": "Production label classes for the BiLSTM complexity model."},
            {"category": "config", "path": rel(domain_root / "outputs" / "metrics" / "cfg_20260310_2125.json"), "reason": "Complexity evaluation config paired with the production BiLSTM summary."},
        ]
    )

    primary_metrics_path = domain_root / "models" / "glove_bilstm_complexity_train_pool_20260307_163358" / "runs" / "metrics_latest.json"
    primary_metrics = load_json(primary_metrics_path)
    evaluation = {
        "model": "GloVe BiLSTM [train_pool]",
        "source_files": [rel(primary_metrics_path)],
        "splits": {split: primary_metrics[split] for split in ["val", "test"] if split in primary_metrics},
        "artifact_paths": {
            key: normalize_slashes(value)
            for key, value in primary_metrics.items()
            if key not in {"val", "test"}
        },
    }
    evaluation_path = dashboard_dir / "metrics" / "primary-evaluation.json"
    write_json(evaluation_path, evaluation)
    selected_sources.append(
        {"category": "metrics", "path": rel(primary_metrics_path), "reason": "Primary val/test metrics for the production complexity BiLSTM model."}
    )

    benchmark_rows = parse_pipe_results_csv(domain_root / "outputs" / "metrics" / "results_complexity-classification.csv")
    benchmark_test = top_test_rows(benchmark_rows, limit=10)
    for row in benchmark_test:
        row["is_production"] = row["model"] == "GloVe BiLSTM [train_pool]"
    benchmark_path = dashboard_dir / "metrics" / "benchmark-test.json"
    write_json(benchmark_path, benchmark_test)
    selected_sources.append(
        {"category": "benchmark", "path": rel(domain_root / "outputs" / "metrics" / "results_complexity-classification.csv"), "reason": "Combined internal complexity leaderboard."}
    )

    training_history = parse_epoch_log(domain_root / "logs" / "complexity-classification_GloVe_BiLSTM_[train_pool].log")
    curves_path = dashboard_dir / "curves" / "training-history.json"
    write_json(curves_path, training_history)
    selected_sources.append(
        {"category": "training_history", "path": rel(domain_root / "logs" / "complexity-classification_GloVe_BiLSTM_[train_pool].log"), "reason": "Epoch-level prod-model BiLSTM training log."}
    )

    cross_summary = load_csv(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv")
    cross_rows = [
        {
            **record,
            "accuracy": clean_float(record["accuracy"]),
            "f1_macro": clean_float(record["f1_macro"]),
            "f1_weighted": clean_float(record["f1_weighted"]),
            "source_file": rel(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv"),
        }
        for record in cross_summary.to_dict(orient="records")
        if str(record["model"]).strip() == "GloVe BiLSTM"
    ]
    cross_path = dashboard_dir / "metrics" / "cross-dataset.json"
    write_json(cross_path, cross_rows)
    selected_sources.append(
        {"category": "cross_dataset", "path": rel(domain_root / "outputs" / "cross_dataset_eval" / "summary_latest.csv"), "reason": "External evaluation summary rows filtered to the production BiLSTM model."}
    )

    train_df = pd.read_parquet(domain_root / "data" / "train_pool_train.parquet")
    val_df = pd.read_parquet(domain_root / "data" / "train_pool_val.parquet")
    test_df = pd.read_parquet(domain_root / "data" / "train_pool_test.parquet")

    def distribution_rows(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
        total = len(frame)
        counts = frame["label"].value_counts().sort_index()
        return [
            {
                "split": split,
                "label": label,
                "count": int(count),
                "share": round(count / total, 6),
            }
            for label, count in counts.items()
        ]

    class_distribution = {
        "overall": distribution_rows(pd.concat([train_df, val_df, test_df], ignore_index=True), "overall"),
        "splits": distribution_rows(train_df, "train") + distribution_rows(val_df, "val") + distribution_rows(test_df, "test"),
    }
    class_distribution_path = dashboard_dir / "distributions" / "class-distribution.json"
    write_json(class_distribution_path, class_distribution)
    selected_sources.append(
        {"category": "distribution", "path": rel(domain_root / "data"), "reason": "Explicit train/val/test parquet splits enable reliable class and dataset distribution extraction."}
    )

    def source_distribution_rows(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
        total = len(frame)
        counts = frame["source_dataset"].value_counts().sort_index()
        return [
            {
                "split": split,
                "source_dataset": label,
                "count": int(count),
                "share": round(count / total, 6),
            }
            for label, count in counts.items()
        ]

    source_distribution = {
        "overall": source_distribution_rows(pd.concat([train_df, val_df, test_df], ignore_index=True), "overall"),
        "splits": source_distribution_rows(train_df, "train")
        + source_distribution_rows(val_df, "val")
        + source_distribution_rows(test_df, "test"),
    }
    source_distribution_path = dashboard_dir / "distributions" / "source-dataset-distribution.json"
    write_json(source_distribution_path, source_distribution)

    samples_df = load_csv(domain_root / "outputs" / "metrics" / "manual_example_predictions.csv")
    sample_rows = []
    for record in samples_df.to_dict(orient="records"):
        sample_rows.append(
            {
                "example_id": int(record["example_id"]),
                "text": record["text"],
                "production_prediction": {
                    "model": "GloVe BiLSTM",
                    "label": record["GloVe_BiLSTM_label"],
                    "confidence": clean_float(record["GloVe_BiLSTM_conf"]),
                },
                "reference_prediction": {
                    "model": "DeBERTaV3",
                    "label": record["DeBERTaV3_label"],
                    "confidence": clean_float(record["DeBERTaV3_conf"]),
                },
            }
        )
    samples_path = dashboard_dir / "samples" / "prediction-samples.json"
    write_json(samples_path, sample_rows)
    selected_sources.append(
        {"category": "samples", "path": rel(domain_root / "outputs" / "metrics" / "manual_example_predictions.csv"), "reason": "Manual prediction examples include the production BiLSTM alongside a transformer reference model."}
    )

    confusion_files = copy_confusion_images(
        dashboard_dir,
        [("internal-test-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_glove_bilstm_train_pool.png")],
    )
    if confusion_files:
        selected_sources.append(
            {"category": "confusion_matrix", "path": rel(domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_glove_bilstm_train_pool.png"), "reason": "Saved internal confusion matrix image for the production BiLSTM model."}
        )

    benchmark_fig_file = write_figure(
        dashboard_dir,
        "benchmark-test-f1",
        make_bar_figure(
            "Complexity Test F1 Macro Benchmark",
            [row["display_name"] for row in benchmark_test],
            [row["f1_macro"] for row in benchmark_test],
            x_title="Model",
            y_title="F1 Macro",
            text_values=[f"{row['f1_macro']:.3f}" for row in benchmark_test],
        ),
    )

    loss_fig_file = write_figure(
        dashboard_dir,
        "training-loss",
        make_multi_line_figure(
            "Complexity BiLSTM Training Loss",
            [
                {"name": "Train Loss", "x": [item["epoch"] for item in training_history["points"]], "y": [item["train_loss"] for item in training_history["points"]]},
                {"name": "Validation Loss", "x": [item["epoch"] for item in training_history["points"]], "y": [item["val_loss"] for item in training_history["points"]]},
            ],
            x_title="Epoch",
            y_title="Loss",
        ),
    )

    eval_metric_fig_file = write_figure(
        dashboard_dir,
        "eval-metrics",
        make_multi_line_figure(
            "Complexity BiLSTM Validation F1",
            [{"name": "Validation F1 Macro", "x": [item["epoch"] for item in training_history["points"]], "y": [item["val_f1_macro"] for item in training_history["points"]]}],
            x_title="Epoch",
            y_title="F1 Macro",
        ),
    )

    cross_fig_file = write_figure(
        dashboard_dir,
        "cross-dataset-f1",
        make_bar_figure(
            "Complexity BiLSTM External F1 Macro",
            [row["dataset"] for row in cross_rows],
            [row["f1_macro"] for row in cross_rows],
            x_title="Dataset",
            y_title="F1 Macro",
            text_values=[f"{row['f1_macro']:.3f}" for row in cross_rows],
            color="#b45309",
        ),
    )

    overall_class_rows = class_distribution["overall"]
    class_distribution_fig_file = write_figure(
        dashboard_dir,
        "class-distribution",
        make_bar_figure(
            "Complexity Overall Class Distribution",
            [row["label"] for row in overall_class_rows],
            [row["count"] for row in overall_class_rows],
            x_title="Class",
            y_title="Count",
            text_values=[str(row["count"]) for row in overall_class_rows],
            color="#1d4ed8",
        ),
    )

    overall_source_rows = source_distribution["overall"]
    source_distribution_fig_file = write_figure(
        dashboard_dir,
        "source-dataset-distribution",
        make_bar_figure(
            "Complexity Overall Source Dataset Distribution",
            [row["source_dataset"] for row in overall_source_rows],
            [row["count"] for row in overall_source_rows],
            x_title="Source Dataset",
            y_title="Count",
            text_values=[str(row["count"]) for row in overall_source_rows],
            color="#475569",
        ),
    )

    section_status = {
        "metadata": "available",
        "summary": "available",
        "evaluation": "available",
        "benchmark": "available",
        "training_curves": "available",
        "learning_curves": "not_applicable",
        "confusion_matrix": "image_only" if confusion_files else "missing",
        "class_distribution": "available",
        "samples": "available",
        "cross_dataset": "available",
    }

    source_audit = build_source_audit(domain_root, prod_dir, selected_sources, notes)
    source_audit_path = dashboard_dir / "summary" / "source-audit.json"
    write_json(source_audit_path, source_audit)

    overview = build_overview(
        metadata=metadata,
        evaluation=evaluation,
        benchmark_rows=benchmark_rows,
        production_benchmark_name="GloVe BiLSTM [train_pool]",
        cross_dataset_rows=cross_rows,
        section_status=section_status,
        notes=notes,
    )
    overview_path = dashboard_dir / "summary" / "overview.json"
    write_json(overview_path, overview)

    sections = [
        section_record("metadata", "Model Metadata", "available", metadata_files, description="Production model identity, label classes, and paired evaluation config."),
        section_record("summary", "Summary", "available", [rel(overview_path), rel(source_audit_path)], description="Dashboard readiness overview and full source audit."),
        section_record("evaluation", "Primary Evaluation", "available", [rel(evaluation_path)], description="Validation and test metrics for the production complexity BiLSTM model."),
        section_record("benchmark", "Benchmark", "available", [rel(benchmark_path), benchmark_fig_file], description="Top internal test benchmark rows across complexity models.", charts=["benchmark-test-f1"]),
        section_record("training_curves", "Training Curves", "available", [rel(curves_path), loss_fig_file, eval_metric_fig_file], description="Epoch-level BiLSTM training log parsed into dashboard-ready curve data.", charts=["training-loss", "eval-metrics"]),
        section_record(
            "learning_curves",
            "Learning Curves",
            "not_applicable",
            [],
            description="Sample-size learning curves for the complexity domain.",
            reason="The complexity benchmark is built around a fixed train_pool dataset rather than multiple prod-relevant train sizes.",
        ),
        section_record("cross_dataset", "Cross Dataset Evaluation", "available", [rel(cross_path), cross_fig_file], description="External evaluation summary for the production BiLSTM model across five datasets.", charts=["cross-dataset-f1"]),
        section_record(
            "confusion_matrix",
            "Confusion Matrix",
            "image_only" if confusion_files else "missing",
            [item["file"] for item in confusion_files],
            description="Saved internal confusion matrix image for the production BiLSTM model.",
            reason=None if confusion_files else "No confusion matrix image was found for the production complexity model.",
        ),
        section_record("class_distribution", "Class Distribution", "available", [rel(class_distribution_path), rel(source_distribution_path), class_distribution_fig_file, source_distribution_fig_file], description="Reliable class and source-dataset distributions derived from explicit train/val/test parquet splits.", charts=["class-distribution", "source-dataset-distribution"]),
        section_record("samples", "Prediction Samples", "available", [rel(samples_path)], description="Manual text-level prediction samples for the production BiLSTM, with a transformer reference prediction."),
    ]

    build_manifest(
        dashboard_dir=dashboard_dir,
        metadata=metadata,
        overview_file=rel(overview_path),
        source_audit_file=rel(source_audit_path),
        sections=sections,
        selected_sources=selected_sources,
        notes=notes,
    )


def build_sentiment_dashboard() -> None:
    prod_dir = REPO_ROOT / "app" / "app-models" / "prod-model-sentiment"
    dashboard_dir = prod_dir / "dashboard"
    domain_root = REPO_ROOT / "sentiment"

    notes = [
        "Primary evaluation uses the production DistilBERT_60k metrics JSON and prod-model YAML labels.",
        "The prod-model export config stores numeric label ids only; the human-readable label names come from prod-model-sentiment/model-config.yaml.",
        "No matching DistilBERT_60k training-history, prediction-array, or raw dataset split artifacts were found in the repository. The available trainer_log_history file belongs to a different bert-base-uncased run and was intentionally excluded from prod-model curves.",
    ]
    selected_sources: list[dict[str, Any]] = []

    metadata = model_metadata(prod_dir, prod_dir / "config.json")
    metadata_files = prepare_metadata_files(dashboard_dir, metadata, None)
    selected_sources.extend(
        [
            {"category": "metadata", "path": rel(prod_dir / "model-config.yaml"), "reason": "Production model identity and human-readable label mapping."},
            {"category": "metadata", "path": rel(prod_dir / "config.json"), "reason": "Exported DistilBERT architecture config for the production model."},
        ]
    )

    primary_metrics_path = domain_root / "outputs" / "metrics" / "transformers" / "distilbert_60k_latest.json"
    primary_metrics = load_json(primary_metrics_path)
    evaluation = {
        "model": "DistilBERT_60k",
        "source_files": [rel(primary_metrics_path)],
        "splits": {split: primary_metrics[split] for split in ["val", "test"] if split in primary_metrics},
        "artifact_paths": {
            key: normalize_slashes(value)
            for key, value in primary_metrics.items()
            if key not in {"val", "test"}
        },
    }
    evaluation_path = dashboard_dir / "metrics" / "primary-evaluation.json"
    write_json(evaluation_path, evaluation)
    selected_sources.append(
        {"category": "metrics", "path": rel(primary_metrics_path), "reason": "Primary val/test metrics for the production sentiment DistilBERT model."}
    )

    benchmark_rows = parse_pipe_results_csv(domain_root / "outputs" / "metrics" / "results_sentiment_analysis.csv")
    benchmark_rows.extend(
        [
            {
                "model": "DistilBERT_60k",
                "display_name": "DistilBERT 60k",
                "family": "transformer",
                "split": split,
                "accuracy": clean_float(metrics.get("accuracy")),
                "f1_macro": clean_float(metrics.get("f1_macro")),
                "f1_weighted": clean_float(metrics.get("f1_weighted")),
                "loss": clean_float(metrics.get("loss")),
                "source_file": rel(primary_metrics_path),
            }
            for split, metrics in primary_metrics.items()
            if split in {"val", "test"}
        ]
    )
    roberta_60k_path = domain_root / "outputs" / "metrics" / "transformers" / "roberta_60k_latest.json"
    roberta_60k = load_json(roberta_60k_path)
    benchmark_rows.extend(
        [
            {
                "model": "RoBERTa_60k",
                "display_name": "RoBERTa 60k",
                "family": "transformer",
                "split": split,
                "accuracy": clean_float(metrics.get("accuracy")),
                "f1_macro": clean_float(metrics.get("f1_macro")),
                "f1_weighted": clean_float(metrics.get("f1_weighted")),
                "loss": clean_float(metrics.get("loss")),
                "source_file": rel(roberta_60k_path),
            }
            for split, metrics in roberta_60k.items()
            if split in {"val", "test"}
        ]
    )
    selected_sources.extend(
        [
            {"category": "benchmark", "path": rel(domain_root / "outputs" / "metrics" / "results_sentiment_analysis.csv"), "reason": "Combined sentiment leaderboard for deep and classical baselines."},
            {"category": "benchmark", "path": rel(roberta_60k_path), "reason": "Transformer benchmark reference for the 60k sentiment setup."},
        ]
    )

    benchmark_test = top_test_rows(benchmark_rows, limit=10)
    for row in benchmark_test:
        row["is_production"] = row["model"] == "DistilBERT_60k"
    benchmark_path = dashboard_dir / "metrics" / "benchmark-test.json"
    write_json(benchmark_path, benchmark_test)

    learning_curve_rows = parse_learning_curve_csv(domain_root / "outputs" / "metrics" / "baseline_curve_metrics.csv")
    learning_curve_path = dashboard_dir / "curves" / "learning-curve.json"
    write_json(learning_curve_path, learning_curve_rows)
    selected_sources.append(
        {"category": "learning_curve", "path": rel(domain_root / "outputs" / "metrics" / "baseline_curve_metrics.csv"), "reason": "Sentiment sample-size curve data across classical baselines."}
    )

    confusion_files = copy_confusion_images(
        dashboard_dir,
        [("internal-test-confusion-matrix", domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_60k.png")],
    )
    if confusion_files:
        selected_sources.append(
            {"category": "confusion_matrix", "path": rel(domain_root / "outputs" / "plots" / "confusion_matrices" / "cm_distilbert_60k.png"), "reason": "Saved internal confusion matrix image for the production sentiment model."}
        )

    benchmark_fig_file = write_figure(
        dashboard_dir,
        "benchmark-test-f1",
        make_bar_figure(
            "Sentiment Test F1 Macro Benchmark",
            [row["display_name"] for row in benchmark_test],
            [row["f1_macro"] for row in benchmark_test],
            x_title="Model",
            y_title="F1 Macro",
            text_values=[f"{row['f1_macro']:.3f}" for row in benchmark_test],
        ),
    )

    learning_series = []
    for model_name, group in pd.DataFrame(learning_curve_rows).groupby("display_name"):
        ordered = group.sort_values("train_size")
        learning_series.append(
            {"name": model_name, "x": ordered["train_size"].tolist(), "y": ordered["f1_macro"].tolist()}
        )
    learning_curve_fig_file = write_figure(
        dashboard_dir,
        "learning-curve-f1",
        make_multi_line_figure(
            "Sentiment Classical Learning Curves",
            learning_series,
            x_title="Train Size",
            y_title="Validation F1 Macro",
        ),
    )

    section_status = {
        "metadata": "available",
        "summary": "available",
        "evaluation": "available",
        "benchmark": "available",
        "training_curves": "missing",
        "learning_curves": "available",
        "confusion_matrix": "image_only" if confusion_files else "missing",
        "class_distribution": "missing",
        "samples": "missing",
        "cross_dataset": "missing",
    }

    source_audit = build_source_audit(domain_root, prod_dir, selected_sources, notes)
    source_audit_path = dashboard_dir / "summary" / "source-audit.json"
    write_json(source_audit_path, source_audit)

    overview = build_overview(
        metadata=metadata,
        evaluation=evaluation,
        benchmark_rows=benchmark_rows,
        production_benchmark_name="DistilBERT_60k",
        cross_dataset_rows=None,
        section_status=section_status,
        notes=notes,
    )
    overview_path = dashboard_dir / "summary" / "overview.json"
    write_json(overview_path, overview)

    sections = [
        section_record("metadata", "Model Metadata", "available", metadata_files, description="Production model identity, export config, and human-readable label mapping."),
        section_record("summary", "Summary", "available", [rel(overview_path), rel(source_audit_path)], description="Dashboard readiness overview and full source audit."),
        section_record("evaluation", "Primary Evaluation", "available", [rel(evaluation_path)], description="Validation and test metrics for the production sentiment DistilBERT model."),
        section_record("benchmark", "Benchmark", "available", [rel(benchmark_path), benchmark_fig_file], description="Top internal test benchmark rows across sentiment models.", charts=["benchmark-test-f1"]),
        section_record(
            "training_curves",
            "Training Curves",
            "missing",
            [],
            description="Training curve data for the production DistilBERT_60k model.",
            reason="No production DistilBERT_60k trainer history artifact was found in the repository. The available trainer_log_history belongs to a different bert-base-uncased run and was excluded.",
        ),
        section_record("learning_curves", "Learning Curves", "available", [rel(learning_curve_path), learning_curve_fig_file], description="Sample-size learning curve data across classical sentiment baselines.", charts=["learning-curve-f1"]),
        section_record(
            "confusion_matrix",
            "Confusion Matrix",
            "image_only" if confusion_files else "missing",
            [item["file"] for item in confusion_files],
            description="Saved internal confusion matrix image for the production sentiment model.",
            reason=None if confusion_files else "No confusion matrix image was found for the production sentiment model.",
        ),
        section_record(
            "class_distribution",
            "Class Distribution",
            "missing",
            [],
            description="Class distribution for the production sentiment training/eval splits.",
            reason="The repository does not include the production sentiment dataset or saved label-count artifacts.",
        ),
        section_record(
            "samples",
            "Prediction Samples",
            "missing",
            [],
            description="Example sentiment predictions for dashboard display.",
            reason="No saved text-level prediction sample artifact was found for the production sentiment model.",
        ),
        section_record(
            "cross_dataset",
            "Cross Dataset Evaluation",
            "missing",
            [],
            description="External evaluation results outside the main sentiment split.",
            reason="No cross-dataset evaluation artifacts were found for the sentiment domain.",
        ),
    ]

    build_manifest(
        dashboard_dir=dashboard_dir,
        metadata=metadata,
        overview_file=rel(overview_path),
        source_audit_file=rel(source_audit_path),
        sections=sections,
        selected_sources=selected_sources,
        notes=notes,
    )


def main() -> None:
    build_abuse_dashboard()
    build_age_dashboard()
    build_complexity_dashboard()
    build_sentiment_dashboard()
    print("Dashboard data build completed.")


if __name__ == "__main__":
    main()
