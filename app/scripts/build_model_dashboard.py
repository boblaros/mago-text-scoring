#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
APP_ROOT = SCRIPT_PATH.parents[1]
BACKEND_ROOT = APP_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.registry.dashboard_builder import build_configured_dashboards, build_model_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build config-driven model dashboards from model-config.yaml files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Build a single model dashboard from this model directory.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=APP_ROOT / "app-models",
        help="Scan this root for model-config.yaml files when --model-dir is not set.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print models that were skipped because they do not use generic-v1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.model_dir is not None:
        result = build_model_dashboard(args.model_dir)
        return _print_single_result(result, show_skipped=True)

    generated = 0
    skipped = 0
    for result in build_configured_dashboards(args.model_root):
        if result.generated:
            generated += 1
            print(f"generated {result.model_id}: {result.manifest_path}")
            continue
        skipped += 1
        if args.show_skipped:
            reason = "; ".join(result.notes) if result.notes else "skipped"
            print(f"skipped {result.model_id}: {reason}")

    print(f"summary: generated={generated} skipped={skipped}")
    return 0


def _print_single_result(result, *, show_skipped: bool) -> int:
    if result.generated:
        print(f"generated {result.model_id}: {result.manifest_path}")
        return 0
    if show_skipped:
        reason = "; ".join(result.notes) if result.notes else "skipped"
        print(f"skipped {result.model_id}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
