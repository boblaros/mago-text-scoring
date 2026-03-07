from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


PAN14_ROOT = Path("data/pan14")
OUTPUT_PATH = PAN14_ROOT / "pan14.parquet"
TMP_OUTPUT_PATH = PAN14_ROOT / "pan14.parquet.tmp"
TOKENIZER_CANDIDATES = (
    Path("models/distilbert_reddit_500k_20260303_022503/best_model"),
    Path("models/distilbert_reddit_1000k_20260223_233842/best_model"),
)
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 128

# Only exact mappings are kept. The PAN14 25-34 bucket straddles the target bins.
AGE_MAP = {
    "18-24": "18-29",
    "35-49": "30-49",
    "50-64": "50-64",
    "65-xx": "65+",
}


def resolve_tokenizer_path() -> Path:
    for candidate in TOKENIZER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No local tokenizer found. Checked: "
        + ", ".join(str(path) for path in TOKENIZER_CANDIDATES)
    )


def load_truth(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        author_id, _gender, age_group = line.split(":::")
        mapping[author_id] = age_group
    return mapping


def iter_author_documents(top_level_dir: Path):
    truth_files = sorted(top_level_dir.rglob("truth.txt"))
    if len(truth_files) != 1:
        raise RuntimeError(
            f"Expected exactly one truth.txt under {top_level_dir}, found {len(truth_files)}"
        )

    truth = load_truth(truth_files[0])
    xml_files = sorted(p for p in top_level_dir.rglob("*.xml") if p.name != "truth.txt")

    for xml_path in xml_files:
        author_id = xml_path.stem
        if author_id not in truth:
            raise KeyError(f"Missing truth label for {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        corpus_type = root.attrib.get("type", "unknown")
        age_group = truth[author_id]
        mapped_age = AGE_MAP.get(age_group)

        for doc in root.findall("./documents/document"):
            text = doc.text or ""
            yield corpus_type, age_group, mapped_age, text


def truncate_batch(tokenizer, texts: list[str]) -> tuple[list[str], int, int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    processed_texts = list(texts)
    truncated_indices: list[int] = []
    end_token_indices: dict[int, int] = {}
    truncated_rows = 0
    corrected_rows = 0

    for idx, (text, offsets) in enumerate(zip(texts, encoded["offset_mapping"])):
        if len(offsets) > MAX_SEQ_LENGTH:
            end_idx = MAX_SEQ_LENGTH - 1
            end_char = offsets[end_idx][1]
            processed_texts[idx] = text[:end_char]
            truncated_indices.append(idx)
            end_token_indices[idx] = end_idx
            truncated_rows += 1

    if truncated_indices:
        lengths = tokenizer(
            [processed_texts[idx] for idx in truncated_indices],
            add_special_tokens=False,
            return_length=True,
            truncation=False,
            padding=False,
        )["length"]

        for idx, length in zip(truncated_indices, lengths):
            while length > MAX_SEQ_LENGTH:
                end_token_indices[idx] -= 1
                end_char = encoded["offset_mapping"][idx][end_token_indices[idx]][1]
                processed_texts[idx] = texts[idx][:end_char]
                length = tokenizer(
                    [processed_texts[idx]],
                    add_special_tokens=False,
                    return_length=True,
                    truncation=False,
                    padding=False,
                )["length"][0]
                corrected_rows += 1

    return processed_texts, truncated_rows, corrected_rows


def flush_batch(writer, tokenizer, texts: list[str], ages: list[str]) -> tuple[int, int]:
    processed_texts, truncated_rows, corrected_rows = truncate_batch(tokenizer, texts)
    table = pa.table({"text": processed_texts, "age": ages})
    writer.write_table(table)
    return truncated_rows, corrected_rows


def main() -> None:
    warnings.filterwarnings("ignore")

    tokenizer_path = resolve_tokenizer_path()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    tokenizer.model_max_length = 10**9

    schema = pa.schema(
        [
            ("text", pa.string()),
            ("age", pa.string()),
        ]
    )

    if TMP_OUTPUT_PATH.exists():
        TMP_OUTPUT_PATH.unlink()

    writer = pq.ParquetWriter(str(TMP_OUTPUT_PATH), schema, compression="snappy")

    kept_rows = 0
    truncated_rows = 0
    corrected_rows = 0
    empty_docs = 0
    unmappable_docs = 0
    total_docs = 0
    kept_by_age: Counter[str] = Counter()
    kept_by_type: Counter[str] = Counter()
    dropped_age_groups: Counter[str] = Counter()
    batch_texts: list[str] = []
    batch_ages: list[str] = []

    try:
        for top_level_dir in sorted(p for p in PAN14_ROOT.iterdir() if p.is_dir()):
            print(f"Processing {top_level_dir.name} ...", flush=True)

            per_dir_docs = 0
            per_dir_kept = 0

            for corpus_type, age_group, mapped_age, text in iter_author_documents(top_level_dir):
                total_docs += 1
                per_dir_docs += 1

                if not text.strip():
                    empty_docs += 1
                    continue

                if mapped_age is None:
                    unmappable_docs += 1
                    dropped_age_groups[age_group] += 1
                    continue

                batch_texts.append(text)
                batch_ages.append(mapped_age)
                kept_rows += 1
                per_dir_kept += 1
                kept_by_age[mapped_age] += 1
                kept_by_type[corpus_type] += 1

                if len(batch_texts) >= BATCH_SIZE:
                    batch_truncated, batch_corrected = flush_batch(
                        writer, tokenizer, batch_texts, batch_ages
                    )
                    truncated_rows += batch_truncated
                    corrected_rows += batch_corrected
                    batch_texts.clear()
                    batch_ages.clear()

            print(
                f"  docs_seen={per_dir_docs:,} kept={per_dir_kept:,} "
                f"empty_skipped={empty_docs:,} unmappable_skipped={unmappable_docs:,}",
                flush=True,
            )

        if batch_texts:
            batch_truncated, batch_corrected = flush_batch(
                writer, tokenizer, batch_texts, batch_ages
            )
            truncated_rows += batch_truncated
            corrected_rows += batch_corrected
    finally:
        writer.close()

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    TMP_OUTPUT_PATH.replace(OUTPUT_PATH)

    print("\nSaved:", OUTPUT_PATH, flush=True)
    print(f"Tokenizer: {tokenizer_path}", flush=True)
    print(f"Total documents seen: {total_docs:,}", flush=True)
    print(f"Kept rows: {kept_rows:,}", flush=True)
    print(f"Skipped empty docs: {empty_docs:,}", flush=True)
    print(f"Skipped unmappable docs: {unmappable_docs:,}", flush=True)
    print(f"Truncated rows: {truncated_rows:,}", flush=True)
    print(f"Correction steps: {corrected_rows:,}", flush=True)
    print("Kept rows by age:", dict(sorted(kept_by_age.items())), flush=True)
    print("Kept rows by type:", dict(sorted(kept_by_type.items())), flush=True)
    print("Dropped age groups:", dict(sorted(dropped_age_groups.items())), flush=True)


if __name__ == "__main__":
    main()
