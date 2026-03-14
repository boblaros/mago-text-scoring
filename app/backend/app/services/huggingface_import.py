from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, parse, request


USER_AGENT = "mago-text-scoring/0.1"
CORE_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
)
OPTIONAL_TOKENIZER_FILENAMES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "added_tokens.json",
)
OPTIONAL_CONFIG_FILENAMES = ("generation_config.json",)


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoFile:
    path: str
    category: str
    required: bool
    size_bytes: int | None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class HuggingFaceInspection:
    repo_id: str
    repo_url: str
    detected_framework_type: str | None
    detected_task: str | None
    framework_library: str | None
    architecture: str | None
    backbone: str | None
    base_model: str | None
    labels: list[dict[str, Any]]
    model_payload: dict[str, Any]
    required_files: list[HuggingFaceRepoFile]
    download_plan: dict[str, list[HuggingFaceRepoFile]]
    estimated_download_size_bytes: int | None
    disk_free_bytes: int
    memory_total_bytes: int | None
    memory_estimate_bytes: int | None
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def runtime_supported(self) -> bool:
        return (
            self.detected_framework_type == "transformers"
            and self.detected_task == "sequence-classification"
        )

    @property
    def compatible(self) -> bool:
        return self.runtime_supported and not self.blocking_reasons

    @property
    def ready_to_import(self) -> bool:
        return self.compatible


class HuggingFaceImportService:
    def __init__(self, settings, *, timeout_seconds: float = 20.0) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds

    def inspect(self, repo_input: str) -> HuggingFaceInspection:
        repo_id = self.parse_repo_id(repo_input)
        repo_url = f"https://huggingface.co/{repo_id}"

        try:
            repo_payload = self._fetch_json(
                f"https://huggingface.co/api/models/{parse.quote(repo_id, safe='/')}"
            )
        except error.HTTPError as exc:
            if exc.code == 404:
                raise ValueError("Hugging Face repo not found.")
            if exc.code in {401, 403}:
                raise ValueError(
                    "This Hugging Face repo is private, gated, or requires authentication."
                )
            raise ValueError("Unable to inspect the Hugging Face repo right now.") from exc
        except error.URLError as exc:
            raise ValueError(
                "Unable to reach Hugging Face right now. Check network access and try again."
            ) from exc

        siblings = {
            str(item.get("rfilename")): item
            for item in repo_payload.get("siblings", [])
            if item.get("rfilename")
        }
        config_payload: dict[str, Any] = {}
        if "config.json" in siblings:
            try:
                config_payload = self._fetch_json(
                    f"https://huggingface.co/{repo_id}/resolve/main/config.json?download=1"
                )
            except Exception:
                config_payload = {}

        detected_task = self._detect_task(repo_payload, config_payload)
        detected_framework_type = self._detect_framework_type(repo_payload, siblings, config_payload)
        framework_library = str(repo_payload.get("library_name") or "huggingface")
        architecture = _first_string(config_payload.get("architectures"))
        base_model = _first_non_empty_string(
            config_payload.get("_name_or_path"),
            repo_payload.get("cardData", {}).get("base_model"),
        )
        backbone = base_model
        label_suggestions = _extract_labels(config_payload)
        model_payload = _extract_model_payload(config_payload, repo_id)

        required_files: list[HuggingFaceRepoFile] = []
        download_plan: dict[str, list[HuggingFaceRepoFile]] = {
            "weights": [],
            "tokenizer": [],
            "config": [],
        }
        warnings: list[str] = []
        blocking_reasons: list[str] = []

        sharded_weights = _first_available(
            siblings,
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if sharded_weights is not None:
            blocking_reasons.append(
                "Sharded Hugging Face checkpoints are not supported by this import flow yet."
            )

        weight_path = _first_available(siblings, "model.safetensors", "pytorch_model.bin")
        if weight_path is None:
            required_files.append(
                HuggingFaceRepoFile(
                    path="model.safetensors | pytorch_model.bin",
                    category="weights",
                    required=True,
                    size_bytes=None,
                    message="Missing transformer weight file.",
                )
            )
            blocking_reasons.append("Required transformer weights were not found in the repo.")
        else:
            weight_file = _repo_file_from_sibling(siblings[weight_path], "weights", required=True)
            required_files.append(weight_file)
            download_plan["weights"].append(weight_file)

        if "config.json" not in siblings:
            required_files.append(
                HuggingFaceRepoFile(
                    path="config.json",
                    category="config",
                    required=True,
                    size_bytes=None,
                    message="Missing transformer config.json.",
                )
            )
            blocking_reasons.append("Required config.json was not found in the repo.")
        else:
            config_file = _repo_file_from_sibling(siblings["config.json"], "config", required=True)
            required_files.append(config_file)
            download_plan["config"].append(config_file)

        core_tokenizer_paths = [
            path for path in CORE_TOKENIZER_FILENAMES if path in siblings
        ]
        if not core_tokenizer_paths:
            required_files.append(
                HuggingFaceRepoFile(
                    path="tokenizer.json | vocab.txt | tokenizer.model",
                    category="tokenizer",
                    required=True,
                    size_bytes=None,
                    message="Missing tokenizer assets.",
                )
            )
            blocking_reasons.append("Required tokenizer assets were not found in the repo.")
        else:
            for tokenizer_path in core_tokenizer_paths:
                tokenizer_file = _repo_file_from_sibling(
                    siblings[tokenizer_path], "tokenizer", required=True
                )
                required_files.append(tokenizer_file)
                download_plan["tokenizer"].append(tokenizer_file)

        for optional_name in OPTIONAL_TOKENIZER_FILENAMES:
            if optional_name in siblings:
                download_plan["tokenizer"].append(
                    _repo_file_from_sibling(siblings[optional_name], "tokenizer", required=False)
                )

        for optional_name in OPTIONAL_CONFIG_FILENAMES:
            if optional_name in siblings:
                download_plan["config"].append(
                    _repo_file_from_sibling(siblings[optional_name], "config", required=False)
                )

        if detected_framework_type != "transformers":
            blocking_reasons.append(
                "Only transformer repositories are supported for Hugging Face import right now."
            )
        if detected_task != "sequence-classification":
            blocking_reasons.append(
                "Only sequence-classification models are supported for Hugging Face import."
            )

        estimated_download_size_bytes = _sum_sizes(download_plan)
        if estimated_download_size_bytes is None:
            warnings.append(
                "Download size could not be estimated precisely because one or more repo files do not report a size."
            )

        model_root = self._settings.model_discovery_root
        model_root.mkdir(parents=True, exist_ok=True)
        disk_free_bytes = shutil.disk_usage(model_root).free
        if estimated_download_size_bytes is not None:
            required_disk = int(estimated_download_size_bytes * 1.15)
            if disk_free_bytes < required_disk:
                blocking_reasons.append(
                    "Not enough free disk space is available for this import."
                )

        memory_total_bytes = _system_memory_bytes()
        weight_size_bytes = next(
            (file.size_bytes for file in download_plan["weights"] if file.size_bytes is not None),
            None,
        )
        memory_estimate_bytes = (
            int(weight_size_bytes * 2.5) if weight_size_bytes is not None else None
        )
        if (
            memory_estimate_bytes is not None
            and memory_total_bytes is not None
            and memory_estimate_bytes > memory_total_bytes
        ):
            blocking_reasons.append(
                "Estimated runtime memory demand exceeds the memory available on this system."
            )
        elif (
            memory_estimate_bytes is not None
            and memory_total_bytes is not None
            and memory_estimate_bytes > int(memory_total_bytes * 0.75)
        ):
            warnings.append(
                "Estimated runtime memory demand is high relative to available system memory."
            )

        if not label_suggestions:
            warnings.append(
                "No label mapping was detected in the remote config, so labels should be confirmed before import."
            )

        return HuggingFaceInspection(
            repo_id=repo_id,
            repo_url=repo_url,
            detected_framework_type=detected_framework_type,
            detected_task=detected_task,
            framework_library=framework_library,
            architecture=architecture,
            backbone=backbone,
            base_model=base_model,
            labels=label_suggestions,
            model_payload=model_payload,
            required_files=required_files,
            download_plan=download_plan,
            estimated_download_size_bytes=estimated_download_size_bytes,
            disk_free_bytes=disk_free_bytes,
            memory_total_bytes=memory_total_bytes,
            memory_estimate_bytes=memory_estimate_bytes,
            warnings=warnings,
            blocking_reasons=list(dict.fromkeys(blocking_reasons)),
        )

    def download_to_directory(
        self,
        inspection: HuggingFaceInspection,
        destination_dir: Path,
    ) -> dict[str, object]:
        destination_dir.mkdir(parents=True, exist_ok=True)
        artifact_manifest: dict[str, object] = {
            "weights": [],
            "tokenizer": [],
            "config": [],
            "vocabulary": [],
            "label_map_file": None,
            "label_classes_file": None,
            "label_encoder_file": None,
        }

        for slot, files in inspection.download_plan.items():
            stored_paths: list[str] = []
            for file in files:
                destination = _unique_destination(destination_dir, Path(file.path).name)
                self._download_file(
                    f"https://huggingface.co/{inspection.repo_id}/resolve/main/{parse.quote(file.path, safe='/')}?download=1",
                    destination,
                )
                stored_paths.append(destination.name)
            artifact_manifest[slot] = stored_paths

        return artifact_manifest

    def parse_repo_id(self, repo_input: str) -> str:
        value = repo_input.strip()
        if value.startswith("http://") or value.startswith("https://"):
            parsed = parse.urlparse(value)
            if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
                raise ValueError("Only huggingface.co model URLs are supported.")
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) < 2:
                raise ValueError("Paste a full model URL or a repo id like org/model-name.")
            return f"{parts[0]}/{parts[1]}"

        parts = [part for part in value.split("/") if part]
        if len(parts) != 2:
            raise ValueError("Paste a full model URL or a repo id like org/model-name.")
        return f"{parts[0]}/{parts[1]}"

    def _detect_framework_type(
        self,
        repo_payload: dict[str, Any],
        siblings: dict[str, dict[str, Any]],
        config_payload: dict[str, Any],
    ) -> str | None:
        library_name = str(repo_payload.get("library_name") or "").lower()
        if library_name == "transformers":
            return "transformers"
        if "config.json" in siblings and (
            config_payload.get("model_type") or config_payload.get("architectures")
        ):
            return "transformers"
        return None

    def _detect_task(
        self,
        repo_payload: dict[str, Any],
        config_payload: dict[str, Any],
    ) -> str | None:
        pipeline_tag = str(repo_payload.get("pipeline_tag") or "").lower()
        if pipeline_tag == "text-classification":
            return "sequence-classification"
        if pipeline_tag:
            return pipeline_tag
        problem_type = str(config_payload.get("problem_type") or "").lower()
        if "classification" in problem_type:
            return "sequence-classification"
        return None

    def _download_file(self, url: str, destination: Path) -> None:
        req = request.Request(url, headers={"User-Agent": USER_AGENT})
        with request.urlopen(req, timeout=self._timeout_seconds) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        req = request.Request(url, headers={"User-Agent": USER_AGENT})
        with request.urlopen(req, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _extract_labels(config_payload: dict[str, Any]) -> list[dict[str, Any]]:
    id2label = config_payload.get("id2label")
    if not isinstance(id2label, dict):
        return []

    labels: list[dict[str, Any]] = []
    for raw_id, raw_label in sorted(
        id2label.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            label_id = len(labels)
        display_name = str(raw_label).strip()
        labels.append(
            {
                "id": label_id,
                "name": _slugify_label(display_name or f"class_{label_id}"),
                "display_name": display_name or f"Class {label_id}",
            }
        )
    return labels


def _extract_model_payload(
    config_payload: dict[str, Any],
    repo_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"source_repo": repo_id}
    for key in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_labels",
        "vocab_size",
        "max_position_embeddings",
    ):
        if key in config_payload:
            payload[key] = config_payload[key]
    return payload


def _slugify_label(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "label"


def _first_available(
    siblings: dict[str, dict[str, Any]],
    *names: str,
) -> str | None:
    for name in names:
        if name in siblings:
            return name
    return None


def _repo_file_from_sibling(
    sibling: dict[str, Any],
    category: str,
    *,
    required: bool,
) -> HuggingFaceRepoFile:
    return HuggingFaceRepoFile(
        path=str(sibling.get("rfilename")),
        category=category,
        required=required,
        size_bytes=_coerce_size(sibling.get("size")),
    )


def _coerce_size(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _sum_sizes(download_plan: dict[str, list[HuggingFaceRepoFile]]) -> int | None:
    total = 0
    saw_known_size = False
    for files in download_plan.values():
        for file in files:
            if file.size_bytes is None:
                return None
            total += file.size_bytes
            saw_known_size = True
    return total if saw_known_size else 0


def _system_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(pages, int):
        return None
    total = page_size * pages
    return total if total > 0 else None


def _first_string(value: Any) -> str | None:
    if isinstance(value, list) and value:
        first = value[0]
        return str(first) if isinstance(first, str) and first.strip() else None
    return str(value) if isinstance(value, str) and value.strip() else None


def _first_non_empty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _unique_destination(root: Path, filename: str) -> Path:
    candidate = root / filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while candidate.exists():
        candidate = root / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate
