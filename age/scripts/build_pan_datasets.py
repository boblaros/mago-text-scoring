from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


MAX_SEQ_LENGTH = 128
BATCH_SIZE = 128
PAN19_REFERENCE_YEAR = 2019
BASE_DATA_DIR = Path("data")
TOKENIZER_CANDIDATES = (
    Path("models/distilbert_reddit_500k_20260303_022503/best_model"),
    Path("models/distilbert_reddit_1000k_20260223_233842/best_model"),
)
SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("age", pa.string()),
    ]
)

PAN13_AGE_MAP = {
    "10s": "13-17",
    "20s": "18-29",
    "30s": "30-49",
}

# Conservative mapping: keep only bins that do not cross target boundaries.
PAN15_AGE_MAP = {
    "18-24": "18-29",
    "35-49": "30-49",
}


@dataclass
class BuildStats:
    rows_kept: int = 0
    rows_dropped_empty: int = 0
    rows_dropped_unmappable: int = 0
    rows_truncated: int = 0
    correction_steps: int = 0
    age_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.age_counts is None:
            self.age_counts = Counter()


def resolve_tokenizer_path() -> Path:
    for candidate in TOKENIZER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No local tokenizer found. Checked: "
        + ", ".join(str(path) for path in TOKENIZER_CANDIDATES)
    )


def make_tokenizer():
    tokenizer_path = resolve_tokenizer_path()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    tokenizer.model_max_length = 10**9
    return tokenizer, tokenizer_path


def iter_document_elements(root: ET.Element):
    docs = root.findall("./documents/document")
    if docs:
        return docs
    return root.findall("./document")


def iter_json_texts(value):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_json_texts(item)


def text_from_element(elem: ET.Element) -> str:
    return "".join(elem.itertext()).strip()


def truncate_batch(tokenizer, texts: list[str]) -> tuple[list[str], int, int]:
    processed_texts = list(texts)
    long_indices = [idx for idx, text in enumerate(texts) if len(text) > MAX_SEQ_LENGTH]
    if not long_indices:
        return processed_texts, 0, 0

    long_texts = [texts[idx] for idx in long_indices]
    encoded = tokenizer(
        long_texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    truncated_indices: list[int] = []
    end_token_indices: dict[int, int] = {}
    offsets_by_index: dict[int, list[tuple[int, int]]] = {}
    truncated_rows = 0
    corrected_rows = 0

    for idx, text, offsets in zip(long_indices, long_texts, encoded["offset_mapping"]):
        offsets_by_index[idx] = offsets
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
                end_char = offsets_by_index[idx][end_token_indices[idx]][1]
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


def write_rows(output_path: Path, rows, tokenizer) -> BuildStats:
    stats = BuildStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    writer = pq.ParquetWriter(str(tmp_output_path), SCHEMA, compression="snappy")
    batch_texts: list[str] = []
    batch_ages: list[str] = []

    try:
        for text, age in rows:
            if not text.strip():
                stats.rows_dropped_empty += 1
                continue
            if age is None:
                stats.rows_dropped_unmappable += 1
                continue

            batch_texts.append(text)
            batch_ages.append(age)
            stats.rows_kept += 1
            stats.age_counts[age] += 1

            if len(batch_texts) >= BATCH_SIZE:
                processed_texts, truncated_rows, corrected_rows = truncate_batch(
                    tokenizer, batch_texts
                )
                writer.write_table(pa.table({"text": processed_texts, "age": batch_ages}))
                stats.rows_truncated += truncated_rows
                stats.correction_steps += corrected_rows
                batch_texts.clear()
                batch_ages.clear()

        if batch_texts:
            processed_texts, truncated_rows, corrected_rows = truncate_batch(
                tokenizer, batch_texts
            )
            writer.write_table(pa.table({"text": processed_texts, "age": batch_ages}))
            stats.rows_truncated += truncated_rows
            stats.correction_steps += corrected_rows
    finally:
        writer.close()

    if output_path.exists():
        output_path.unlink()
    tmp_output_path.replace(output_path)
    return stats


def build_pan13_rows():
    source_path = BASE_DATA_DIR / "pan13" / "source.zip"
    with ZipFile(source_path) as outer:
        training_name = next(
            name
            for name in outer.namelist()
            if name.endswith("pan13-author-profiling-training-corpus-2013-01-09.zip")
        )
        training_bytes = outer.read(training_name)

    with ZipFile(BytesIO(training_bytes)) as inner:
        for name in inner.namelist():
            if not name.endswith(".xml") or "/en/" not in name:
                continue
            root = ET.fromstring(inner.read(name))
            mapped_age = PAN13_AGE_MAP.get(root.attrib.get("age_group"))
            for conv in root.findall("./conversations/conversation"):
                yield text_from_element(conv), mapped_age


def load_pan15_truth(inner_zip: ZipFile) -> dict[str, str]:
    truth_name = next(name for name in inner_zip.namelist() if name.endswith("truth.txt"))
    truth: dict[str, str] = {}
    for raw_line in inner_zip.read(truth_name).decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        author_id, _gender, age_group, *_rest = line.split(":::")
        truth[author_id] = age_group
    return truth


def build_pan15_rows():
    source_path = BASE_DATA_DIR / "pan15" / "source.zip"
    with ZipFile(source_path) as outer:
        english_name = next(
            name
            for name in outer.namelist()
            if "english" in name and name.endswith(".zip")
        )
        english_bytes = outer.read(english_name)

    with ZipFile(BytesIO(english_bytes)) as inner:
        truth = load_pan15_truth(inner)
        for name in inner.namelist():
            if not name.endswith(".xml"):
                continue
            root = ET.fromstring(inner.read(name))
            author_id = root.attrib.get("id") or Path(name).stem
            mapped_age = PAN15_AGE_MAP.get(truth.get(author_id))
            for doc in iter_document_elements(root):
                yield text_from_element(doc), mapped_age


def map_pan19_birthyear(birthyear: int | None) -> str | None:
    if birthyear is None:
        return None

    # Only keep birth years whose possible 2019 ages stay within one target bin.
    if 2002 <= birthyear <= 2005:
        return "13-17"
    if 1990 <= birthyear <= 2000:
        return "18-29"
    if 1970 <= birthyear <= 1988:
        return "30-49"
    if 1955 <= birthyear <= 1968:
        return "50-64"
    if birthyear <= 1953:
        return "65+"
    return None


def find_single_member(zip_file: ZipFile, expected_name: str) -> str:
    matches = [
        name for name in zip_file.namelist() if Path(name).name == expected_name and not name.endswith("/")
    ]
    if not matches:
        raise FileNotFoundError(f"Could not find {expected_name} inside {zip_file.filename}")
    if len(matches) > 1:
        raise ValueError(f"Found multiple {expected_name} members inside {zip_file.filename}: {matches}")
    return matches[0]


def build_pan19_rows():
    source_path = BASE_DATA_DIR / "pan19" / "source.zip"
    with ZipFile(source_path) as archive:
        feeds_name = find_single_member(archive, "feeds.ndjson")
        labels_name = find_single_member(archive, "labels.ndjson")

        age_by_id: dict[str, str | None] = {}
        with archive.open(labels_name) as labels_file:
            for raw_line in labels_file:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                record = json.loads(line)
                age_by_id[str(record["id"])] = map_pan19_birthyear(record.get("birthyear"))

        with archive.open(feeds_name) as feeds_file:
            for raw_line in feeds_file:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                record = json.loads(line)
                mapped_age = age_by_id.get(str(record["id"]))
                for text in iter_json_texts(record.get("text")):
                    yield text, mapped_age


def build_empty_pan16(output_path: Path, note_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "text": pa.array([], type=pa.string()),
                "age": pa.array([], type=pa.string()),
            }
        ),
        output_path,
        compression="snappy",
    )
    note_path.write_text(
        (
            "PAN16 was not convertible into a real text dataset from the official training archive.\n"
            "The English and Spanish XML files contain only tweet IDs/URLs, not tweet bodies.\n"
            "This placeholder parquet is intentionally empty.\n"
        ),
        encoding="utf-8",
    )


def print_stats(name: str, output_path: Path, stats: BuildStats) -> None:
    print(f"{name} -> {output_path}", flush=True)
    print(f"  rows_kept={stats.rows_kept:,}", flush=True)
    print(f"  rows_dropped_empty={stats.rows_dropped_empty:,}", flush=True)
    print(f"  rows_dropped_unmappable={stats.rows_dropped_unmappable:,}", flush=True)
    print(f"  rows_truncated={stats.rows_truncated:,}", flush=True)
    print(f"  correction_steps={stats.correction_steps:,}", flush=True)
    print(f"  age_counts={dict(sorted(stats.age_counts.items()))}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("pan13", "pan15", "pan16", "pan19"),
        default=("pan13", "pan15", "pan16", "pan19"),
        help="Datasets to build.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()

    tokenizer, tokenizer_path = make_tokenizer()
    print(f"Tokenizer: {tokenizer_path}", flush=True)

    pan13_output = BASE_DATA_DIR / "pan13" / "pan13.parquet"
    pan15_output = BASE_DATA_DIR / "pan15" / "pan15.parquet"
    pan16_output = BASE_DATA_DIR / "pan16" / "pan16.parquet"
    pan16_note = BASE_DATA_DIR / "pan16" / "pan16_build_note.txt"
    pan19_output = BASE_DATA_DIR / "pan19" / "pan19.parquet"

    if "pan13" in args.datasets:
        pan13_stats = write_rows(pan13_output, build_pan13_rows(), tokenizer)
        print_stats("PAN13", pan13_output, pan13_stats)

    if "pan15" in args.datasets:
        pan15_stats = write_rows(pan15_output, build_pan15_rows(), tokenizer)
        print_stats("PAN15", pan15_output, pan15_stats)

    if "pan16" in args.datasets:
        build_empty_pan16(pan16_output, pan16_note)
        print(f"PAN16 -> {pan16_output}", flush=True)
        print(f"  note={pan16_note}", flush=True)

    if "pan19" in args.datasets:
        pan19_stats = write_rows(pan19_output, build_pan19_rows(), tokenizer)
        print_stats("PAN19", pan19_output, pan19_stats)


if __name__ == "__main__":
    main()
