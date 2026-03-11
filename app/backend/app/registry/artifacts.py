from __future__ import annotations

from pathlib import Path

from app.registry.contracts import ModelManifest, ResolvedArtifacts


def resolve_artifacts(model_dir: Path, manifest: ModelManifest) -> ResolvedArtifacts:
    resolved = ResolvedArtifacts()
    base_dir = manifest.artifacts.base_dir

    resolved.weights = _resolve_many(model_dir, manifest.artifacts.weights, base_dir, resolved)
    resolved.tokenizer = _resolve_many(
        model_dir, manifest.artifacts.tokenizer, base_dir, resolved
    )
    resolved.config = _resolve_many(model_dir, manifest.artifacts.config, base_dir, resolved)
    resolved.vocabulary = _resolve_many(
        model_dir, manifest.artifacts.vocabulary, base_dir, resolved
    )
    resolved.label_map_file = _resolve_optional(
        model_dir, manifest.artifacts.label_map_file, base_dir, resolved
    )
    resolved.label_classes_file = _resolve_optional(
        model_dir, manifest.artifacts.label_classes_file, base_dir, resolved
    )
    resolved.label_encoder_file = _resolve_optional(
        model_dir, manifest.artifacts.label_encoder_file, base_dir, resolved
    )
    if not resolved.weights:
        resolved.missing.append("weights")
    return resolved


def _resolve_many(
    model_dir: Path,
    items: list[str],
    base_dir: str | None,
    resolved: ResolvedArtifacts,
) -> list[Path]:
    matches: list[Path] = []
    for item in items:
        path = _resolve_single(model_dir, item, base_dir, resolved)
        if path is not None:
            matches.append(path)
    return matches


def _resolve_optional(
    model_dir: Path,
    item: str | None,
    base_dir: str | None,
    resolved: ResolvedArtifacts,
) -> Path | None:
    if not item:
        return None
    return _resolve_single(model_dir, item, base_dir, resolved)


def _resolve_single(
    model_dir: Path,
    configured_path: str,
    base_dir: str | None,
    resolved: ResolvedArtifacts,
) -> Path | None:
    candidate_path = Path(configured_path)
    candidates: list[Path] = []

    if candidate_path.is_absolute():
        candidates.append(candidate_path)

    if base_dir:
        candidates.append(model_dir / base_dir / candidate_path)
        candidates.append(model_dir / base_dir / candidate_path.name)

    candidates.append(model_dir / candidate_path)
    candidates.append(model_dir / candidate_path.name)

    if candidate_path.parts and candidate_path.parts[0] == "models":
        candidates.append(model_dir / Path(*candidate_path.parts[1:]))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            if candidate.name != candidate_path.name or candidate_path.parent != Path("."):
                resolved.notes.append(
                    f"Resolved '{configured_path}' to '{candidate.relative_to(model_dir)}'."
                )
            return candidate

    recursive_matches = sorted(model_dir.rglob(candidate_path.name))
    if recursive_matches:
        candidate = recursive_matches[0]
        resolved.notes.append(
            f"Resolved '{configured_path}' by basename search to '{candidate.relative_to(model_dir)}'."
        )
        return candidate

    resolved.missing.append(configured_path)
    return None

