
from __future__ import annotations

import json
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ..data import first_existing_path
from ..metrics import RESULTS

CFG: dict = {}
class2id: dict = {}

AGE_BINS = [6, 13, 18, 30, 50, 65, float('inf')]
AGE_LABELS = ['6-12', '13-17', '18-29', '30-49', '50-64', '65+']
MODEL_CLASSES = ['13-17', '18-29', '30-49', '50-64', '65+']
PAN_PARQUETS = {
    "pan13": Path("data/pan13/pan13.parquet"),
    "pan15": Path("data/pan15/pan15.parquet"),
    "pan16": Path("data/pan16/pan16.parquet"),
}
PAN_NOTES = {
    "pan13": "English only; one row per <conversation>; mapped 10s/20s/30s to target bins.",
    "pan15": "English only; one row per <document>; kept only safely mappable age bins.",
    "pan16": "Empty placeholder: this local PAN16 archive contains tweet URLs/IDs instead of tweet text.",
    "pan19": "Celebrity profiling; one tweet per row; age derived conservatively from birthyear relative to 2019.",
}
SEED = 0.42
STANDARDIZED_BASE_DIR = Path('./outputs/standardized')
STANDARDIZED_BASE_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_SCHEMA_COLUMNS = ['text', 'label', 'label_name', 'source', 'doc_id', 'meta']
LABEL_MAPPING_CONFIG = {'blog': {}, 'hippocorpus': {}, 'pan14': {}}

def ensure_parent_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def age_distribution_parquet(parquet_path):
    dataset = ds.dataset(str(parquet_path), format="parquet")
    table = dataset.to_table(columns=["age"])
    vc = pc.value_counts(table["age"])

    counts = pd.DataFrame({
        "age": vc.field("values").to_pandas(),
        "count": vc.field("counts").to_pandas(),
    }).sort_values("count", ascending=False)
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts.reset_index(drop=True)


def process_tsv_to_parquet(
    input_path,
    output_path,
    text_builder,
    age_col="DMGAgeAtPost",
    usecols=None,
    chunk_size=100_000,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8")) - 1
    total_chunks = math.ceil(total_lines / chunk_size)

    writer = None
    reader = pd.read_csv(input_path, sep="	", chunksize=chunk_size, usecols=usecols)

    for chunk in tqdm(reader, total=total_chunks, desc=f"Processing {input_path.name}"):
        chunk["text"] = text_builder(chunk)
        age_num = pd.to_numeric(chunk[age_col], errors="coerce")
        chunk["age"] = pd.cut(age_num, bins=AGE_BINS, labels=AGE_LABELS, right=False)

        chunk = chunk[["text", "age"]].dropna(subset=["age"])
        chunk["text"] = chunk["text"].fillna("").astype(str)
        chunk["age"] = chunk["age"].astype(str)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    return output_path


def sql_quote(value):
    return str(value).replace("'", "''")


def sql_age_order(column="age"):
    parts = [f"WHEN {column} = '{age}' THEN {idx}" for idx, age in enumerate(MODEL_CLASSES)]
    return "CASE " + " ".join(parts) + " ELSE 999 END"


def pan_quick_eda(dataset_name, parquet_path, sample_per_class=2):
    parquet_sql = sql_quote(parquet_path.as_posix())
    con = duckdb.connect(database=":memory:")

    class_df = con.execute(
        f"""
        SELECT
            age,
            COUNT(*) AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percent,
            ROUND(AVG(LENGTH(text)), 2) AS avg_chars,
            MAX(LENGTH(text)) AS max_chars,
            SUM(CASE WHEN LENGTH(TRIM(text)) = 0 THEN 1 ELSE 0 END) AS empty_texts
        FROM read_parquet('{parquet_sql}')
        GROUP BY age
        ORDER BY {sql_age_order('age')}
        """
    ).fetchdf()

    meta_df = pd.DataFrame([
        {
            "dataset": dataset_name,
            "rows": pq.ParquetFile(parquet_path).metadata.num_rows,
            "present_classes": len(class_df),
            "largest_class_percent": round(class_df["percent"].max(), 2),
            "file_size_mb": round(parquet_path.stat().st_size / (1024 * 1024), 2),
            "note": PAN_NOTES.get(dataset_name, ""),
        }
    ])

    example_frames = []
    for age in class_df["age"].tolist():
        example_df = con.execute(
            f"""
            SELECT
                age,
                LENGTH(text) AS char_len,
                LEFT(REGEXP_REPLACE(text, '\\s+', ' ', 'g'), 180) AS preview
            FROM read_parquet('{parquet_sql}')
            WHERE age = '{sql_quote(age)}'
              AND LENGTH(TRIM(text)) > 0
            LIMIT {sample_per_class}
            """
        ).fetchdf()
        if not example_df.empty:
            example_frames.append(example_df)

    examples_df = (
        pd.concat(example_frames, ignore_index=True)
        if example_frames
        else pd.DataFrame(columns=["age", "char_len", "preview"])
    )
    return meta_df, class_df, examples_df


def map_to_5_classes(raw_label, *, dataset_name):
    """Map source-specific raw labels to the unified 5-class target.

    Returns:
        tuple[int, str] | None
            (label, label_name) if mapping is configured, otherwise None.
    """
    # TODO: inspect label distribution / age ranges for each dataset.
    # TODO: define bin edges / class definitions for exactly 5 target classes.
    # TODO: verify no leakage / consistent interpretation across all datasets.
    if raw_label is None or pd.isna(raw_label):
        return None

    rules = LABEL_MAPPING_CONFIG.get(dataset_name, {})
    key = str(raw_label).strip()
    return rules.get(key)


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_json_value(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def _meta_json_from_row(row, columns):
    payload = {}
    for col in columns:
        val = _safe_json_value(row[col])
        if val is not None:
            payload[col] = val
    return json.dumps(payload, ensure_ascii=False)


def read_tabular_file(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)

    raise ValueError(f"Unsupported file format: {path}")


def load_tabular_files(dataset_dir, *, preferred_globs):
    dataset_dir = Path(dataset_dir)
    files = []
    for pattern in preferred_globs:
        files.extend(dataset_dir.rglob(pattern))

    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise FileNotFoundError(f"No matching files found under {dataset_dir}")

    frames = []
    for path in files:
        try:
            frame = read_tabular_file(path)
            frame["__source_file"] = str(path)
            frames.append(frame)
        except Exception as exc:
            warnings.warn(f"Skipping unreadable file {path}: {exc}")

    if not frames:
        raise RuntimeError(f"No readable files found under {dataset_dir}")

    return pd.concat(frames, ignore_index=True)


def load_blog_raw(dataset_dir):
    return load_tabular_files(
        dataset_dir,
        preferred_globs=[
            "blogtext.csv",
            "*.csv",
            "*.parquet",
            "*.tsv",
            "*.jsonl",
            "*.json",
        ],
    )


def load_hippocorpus_raw(dataset_dir):
    return load_tabular_files(
        dataset_dir,
        preferred_globs=[
            "hippoCorpusV2.csv",
            "*.csv",
            "*.parquet",
            "*.tsv",
            "*.jsonl",
            "*.json",
        ],
    )


def extract_pan14_text(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        warnings.warn(f"XML parse failed for {xml_path}: {exc}")
        return ""

    parts = []
    for doc in root.findall(".//document"):
        text = "".join(doc.itertext()).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_pan14_raw(dataset_dir):
    dataset_dir = Path(dataset_dir)
    truth_files = sorted(dataset_dir.rglob("truth.txt"))
    if not truth_files:
        raise FileNotFoundError(f"No truth.txt files found under {dataset_dir}")

    records = []
    for truth_path in truth_files:
        corpus_root = truth_path.parent
        xml_by_author = {p.stem: p for p in corpus_root.glob("*.xml")}
        if not xml_by_author:
            xml_by_author = {p.stem: p for p in corpus_root.rglob("*.xml")}

        with open(truth_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(":::")
                if len(parts) < 3:
                    warnings.warn(f"Malformed truth row: {truth_path}:{line_no} -> {line}")
                    continue

                author_id, gender, raw_label = parts[0].strip(), parts[1].strip(), parts[2].strip()
                xml_path = xml_by_author.get(author_id)
                if xml_path is None:
                    warnings.warn(f"Missing XML for author {author_id} in {truth_path}")
                    continue

                text = extract_pan14_text(xml_path)
                records.append(
                    {
                        "doc_id": author_id,
                        "text": text,
                        "raw_label": raw_label,
                        "gender": gender,
                        "domain": corpus_root.name,
                        "truth_file": str(truth_path),
                        "xml_file": str(xml_path),
                    }
                )

    if not records:
        raise RuntimeError("PAN14 parsing produced 0 rows")

    return pd.DataFrame(records)


def standardize_to_unified_schema(
    raw_df,
    *,
    dataset_name,
    text_candidates,
    label_candidates,
    doc_id_candidates,
):
    raw_df = raw_df.copy()

    text_col = _first_existing_column(raw_df, text_candidates)
    if text_col is None:
        raise KeyError(f"{dataset_name}: no text column found in {text_candidates}")

    label_col = _first_existing_column(raw_df, label_candidates)
    doc_id_col = _first_existing_column(raw_df, doc_id_candidates)

    work = pd.DataFrame(index=raw_df.index)
    work["text"] = raw_df[text_col].fillna("").astype(str).str.strip()

    if label_col is None:
        work["raw_label"] = pd.NA
    else:
        work["raw_label"] = raw_df[label_col]

    if doc_id_col is None:
        if "__source_file" in raw_df.columns:
            source_stem = raw_df["__source_file"].map(lambda p: Path(p).stem)
            work["doc_id"] = source_stem + ":" + raw_df.index.astype(str)
        else:
            work["doc_id"] = f"{dataset_name}:" + raw_df.index.astype(str)
    else:
        work["doc_id"] = raw_df[doc_id_col].astype(str)

    reserved = {
        text_col,
        label_col,
        doc_id_col,
        "text",
        "label",
        "label_name",
        "source",
        "doc_id",
        "meta",
    }
    meta_cols = [c for c in raw_df.columns if c not in reserved and c is not None]

    if meta_cols:
        work["meta"] = raw_df.apply(lambda row: _meta_json_from_row(row, meta_cols), axis=1)
    else:
        work["meta"] = "{}"

    mapped = work["raw_label"].apply(lambda v: map_to_5_classes(v, dataset_name=dataset_name))
    work["label"] = mapped.apply(lambda x: x[0] if isinstance(x, tuple) else None)
    work["label_name"] = mapped.apply(lambda x: x[1] if isinstance(x, tuple) else None)

    mapped_mask = work["label"].notna()
    dropped = int((~mapped_mask).sum())
    if dropped:
        warnings.warn(
            f"{dataset_name}: dropped {dropped:,} rows because map_to_5_classes has no rule for their raw labels"
        )

    standardized = work.loc[mapped_mask, ["text", "label", "label_name", "doc_id", "meta"]].copy()
    standardized["label"] = standardized["label"].astype(int)
    standardized["source"] = dataset_name
    standardized = standardized[UNIFIED_SCHEMA_COLUMNS]
    standardized = standardized[standardized["text"].str.len() > 0].reset_index(drop=True)

    return standardized


def save_standardized_dataset(dataset_name, standardized_df, *, rows_loaded):
    output_dir = STANDARDIZED_BASE_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "data.parquet"
    csv_path = output_dir / "data.csv"
    stats_path = output_dir / "stats.json"

    standardized_df.to_parquet(parquet_path, index=False)
    standardized_df.to_csv(csv_path, index=False)

    class_counts = {
        str(int(k)): int(v)
        for k, v in standardized_df["label"].value_counts().sort_index().to_dict().items()
    } if not standardized_df.empty else {}

    stats = {
        "dataset": dataset_name,
        "total_rows_loaded": int(rows_loaded),
        "rows_mapped_to_5_classes": int(len(standardized_df)),
        "rows_dropped_unmapped": int(rows_loaded - len(standardized_df)),
        "class_counts": class_counts,
    }

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "stats": str(stats_path),
    }, stats


def load_blog_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    blog_df = pd.read_parquet(path).copy()

    required_cols = [CFG["text_col"], CFG["label_col"]]
    missing_cols = [col for col in required_cols if col not in blog_df.columns]
    if missing_cols:
        raise KeyError(f"Blog dataset is missing required columns: {missing_cols}")

    blog_df = blog_df[required_cols].dropna(subset=required_cols).copy()
    blog_df[CFG["text_col"]] = blog_df[CFG["text_col"]].astype(str)
    blog_df[CFG["label_col"]] = blog_df[CFG["label_col"]].astype(str)
    blog_df = blog_df[blog_df[CFG["label_col"]].isin(class2id)].reset_index(drop=True)
    return blog_df


def load_hippocorpus_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    hippocorpus_df = pd.read_parquet(path).copy()

    required_cols = [CFG["text_col"], CFG["label_col"]]
    missing_cols = [col for col in required_cols if col not in hippocorpus_df.columns]
    if missing_cols:
        raise KeyError(f"Hippocorpus dataset is missing required columns: {missing_cols}")

    hippocorpus_df = hippocorpus_df[required_cols].dropna(subset=required_cols).copy()
    hippocorpus_df[CFG["text_col"]] = hippocorpus_df[CFG["text_col"]].astype(str)
    hippocorpus_df[CFG["label_col"]] = hippocorpus_df[CFG["label_col"]].astype(str)
    hippocorpus_df = hippocorpus_df[hippocorpus_df[CFG["label_col"]].isin(class2id)].reset_index(drop=True)
    return hippocorpus_df


def load_pan_dataset(paths) -> pd.DataFrame:
    text_candidates = [CFG["text_col"], "text_raw", "content", "body", "document"]
    label_candidates = [CFG["label_col"], "label_name", "age_group", "raw_label", "class", "target"]
    frames = []

    for path in [Path(p) for p in paths]:
        pan_df = pd.read_parquet(path).copy()
        text_col = next((col for col in text_candidates if col in pan_df.columns), None)
        label_col = next((col for col in label_candidates if col in pan_df.columns), None)

        if text_col is None or label_col is None:
            raise KeyError(
                f"{path.name}: expected text/label columns from {text_candidates} and {label_candidates}, "
                f"got {list(pan_df.columns)}"
            )

        pan_df = pan_df[[text_col, label_col]].rename(
            columns={text_col: CFG["text_col"], label_col: CFG["label_col"]}
        )
        pan_df = pan_df.dropna(subset=[CFG["text_col"], CFG["label_col"]]).copy()
        pan_df[CFG["text_col"]] = pan_df[CFG["text_col"]].astype(str)
        pan_df[CFG["label_col"]] = pan_df[CFG["label_col"]].astype(str)
        pan_df = pan_df[pan_df[CFG["label_col"]].isin(class2id)].reset_index(drop=True)
        pan_df["pan_source"] = path.stem
        frames.append(pan_df)

    if not frames:
        raise ValueError("No PAN datasets were loaded.")

    return pd.concat(frames, ignore_index=True)


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

    def _coerce_metric_dict(metrics: dict) -> dict:
        clean = {}
        for key, value in (metrics or {}).items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if pd.isna(numeric):
                continue
            clean[str(key)] = numeric
        return clean

    def _should_keep(result_name: str) -> bool:
        if (not include_learning_curves) and result_name.startswith("LC |"):
            return False
        if (not include_eval) and result_name.endswith(" | eval"):
            return False
        return True

    latest_json = sorted(metrics_dir.glob("results_*.json"))
    if latest_json:
        try:
            with open(latest_json[-1], "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for result_name, metrics in payload.items():
                    result_name = str(result_name)
                    if not _should_keep(result_name):
                        continue
                    clean_metrics = _coerce_metric_dict(metrics if isinstance(metrics, dict) else {})
                    if not clean_metrics:
                        continue
                    RESULTS[result_name] = clean_metrics
                    restored += 1
                return restored
        except Exception as e:
            print(f"Could not restore RESULTS from {latest_json[-1]}: {e}")

    csv_path = metrics_dir / f"results_{cfg['task']}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            name_col = "Model | Split" if "Model | Split" in df.columns else df.columns[0]
            for row in df.to_dict(orient="records"):
                result_name = str(row.pop(name_col, ""))
                if not result_name or not _should_keep(result_name):
                    continue
                clean_metrics = _coerce_metric_dict(row)
                if not clean_metrics:
                    continue
                RESULTS[result_name] = clean_metrics
                restored += 1
        except Exception as e:
            print(f"Could not restore RESULTS from {csv_path}: {e}")

    return restored
