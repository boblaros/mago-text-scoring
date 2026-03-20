from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


SERIALIZED_ARTIFACT_EXTENSIONS = {".bin", ".joblib", ".pkl"}


def load_serialized_artifact(path: Path) -> Any:
    errors: list[str] = []

    try:
        import joblib  # type: ignore
    except ModuleNotFoundError:
        joblib = None

    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception as exc:  # pragma: no cover - exercised via fallback paths too.
            errors.append(f"joblib: {exc}")

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    joined = "; ".join(errors) or "unknown deserialization error"
    raise RuntimeError(f"Could not load serialized artifact '{path.name}': {joined}")


def load_optional_label_encoder(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() not in SERIALIZED_ARTIFACT_EXTENSIONS:
        return None

    try:
        return load_serialized_artifact(path)
    except Exception:
        return None


def load_label_records(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}

    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = load_serialized_artifact(path)
    except Exception:
        return {}

    parsed = parse_label_records_payload(payload)
    return parsed or {}


def parse_label_records_payload(payload: Any) -> dict[int, str] | None:
    if isinstance(payload, list):
        records: dict[int, str] = {}
        for index, item in enumerate(payload):
            if isinstance(item, str) and item.strip():
                records[index] = item.strip()
                continue
            if isinstance(item, dict):
                raw_id = item.get("id", index)
                raw_name = item.get("display_name") or item.get("name")
                try:
                    label_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if isinstance(raw_name, str) and raw_name.strip():
                    records[label_id] = raw_name.strip()
        return records or None

    if not isinstance(payload, dict):
        return None

    if "id2label" in payload:
        return parse_label_records_payload(payload["id2label"])
    if "label2id" in payload:
        return parse_label_records_payload(payload["label2id"])
    if "classes" in payload and isinstance(payload["classes"], list):
        return parse_label_records_payload(payload["classes"])

    int_key_mapping: dict[int, str] = {}
    for key, value in payload.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            int_key_mapping = {}
            break
        if isinstance(value, str) and value.strip():
            int_key_mapping[label_id] = value.strip()
        elif isinstance(value, dict):
            raw_name = value.get("display_name") or value.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                int_key_mapping[label_id] = raw_name.strip()
    if int_key_mapping:
        return int_key_mapping

    int_value_mapping: dict[int, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            int_value_mapping = {}
            break
        try:
            label_id = int(value)
        except (TypeError, ValueError):
            int_value_mapping = {}
            break
        int_value_mapping[label_id] = key.strip()
    if int_value_mapping:
        return int_value_mapping

    return None


def derive_manifest_labels(
    *,
    label_classes_path: Path | None = None,
    label_map_path: Path | None = None,
    label_encoder_path: Path | None = None,
) -> list[dict[str, object]] | None:
    label_records = load_label_records(label_classes_path) or load_label_records(label_map_path)
    if not label_records:
        label_records = _derive_labels_from_encoder(label_encoder_path)
    if not label_records:
        return None

    return [
        {
            "id": label_id,
            "name": label_name,
            "display_name": label_name,
        }
        for label_id, label_name in sorted(label_records.items())
    ]


def _derive_labels_from_encoder(path: Path | None) -> dict[int, str]:
    encoder = load_optional_label_encoder(path)
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        return {}

    records: dict[int, str] = {}
    try:
        for index, class_name in enumerate(classes):
            normalized_name = str(class_name).strip()
            if normalized_name:
                records[index] = normalized_name
    except Exception:
        return {}
    return records


def decode_label_value(
    value: Any,
    *,
    manifest_labels: dict[int, str] | None = None,
    artifact_labels: dict[int, str] | None = None,
    label_encoder: Any | None = None,
) -> str:
    normalized_value = _normalize_scalar(value)

    if label_encoder is not None:
        decoded = _decode_with_label_encoder(normalized_value, label_encoder)
        if decoded is not None:
            return decoded

    if isinstance(normalized_value, int):
        if manifest_labels and normalized_value in manifest_labels:
            return manifest_labels[normalized_value]
        if artifact_labels and normalized_value in artifact_labels:
            return artifact_labels[normalized_value]

    return str(normalized_value)


def _decode_with_label_encoder(value: Any, label_encoder: Any) -> str | None:
    if hasattr(label_encoder, "inverse_transform"):
        try:
            decoded = label_encoder.inverse_transform([value])
            if len(decoded):
                return str(decoded[0])
        except Exception:
            pass

    classes = getattr(label_encoder, "classes_", None)
    if classes is not None and isinstance(value, int):
        try:
            if 0 <= value < len(classes):
                return str(classes[value])
        except Exception:
            return None

    return None


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value
