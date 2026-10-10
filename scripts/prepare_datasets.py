"""
scripts/prepare_datasets.py
==============================

Loads the raw LIAR and MultiFC fact-checking datasets from `data/raw/`,
normalizes their heterogeneous multi-class veracity labels into a unified
binary (`REAL`=0 / `FAKE`=1) or ternary (`REAL`=0 / `UNCERTAIN`=1 / `FAKE`=2)
target space, cleans the claim/statement text, and writes train/val/test
JSONL splits to `data/processed/`.

Expected raw layout
--------------------
    data/raw/liar/
        train.tsv, valid.tsv (or val.tsv), test.tsv     # original LIAR release, no header
    data/raw/multifc/
        <one or more>.tsv                                # MultiFC release, no header

If a dataset directory is absent, that dataset is simply skipped (with a
warning) so this script can run against whichever raw data is available.

Label mapping
-------------
LIAR (6-way): pants-fire, false, barely-true, half-true, mostly-true, true

    binary : {true, mostly-true} -> REAL(0)
             {false, barely-true, pants-fire} -> FAKE(1)
             {half-true} -> dropped (ambiguous)
    ternary: {true, mostly-true} -> REAL(0)
             {half-true} -> UNCERTAIN(1)
             {false, barely-true, pants-fire} -> FAKE(2)

MultiFC labels are free-text strings sourced from 30+ different
fact-checking outlets (e.g. "true", "mostly true", "false", "pants on
fire!", "mixture", "unproven", ...). These are normalized via keyword
matching in `normalize_multifc_label` into the same REAL / UNCERTAIN / FAKE
buckets; labels that cannot be confidently mapped are dropped.

Output
------
    data/processed/{train,val,test}.jsonl

Each line: {"id": str, "text": str, "label": int, "label_name": str,
            "source_dataset": "liar" | "multifc"}
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ============================================================================
# Text cleaning
# ============================================================================
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Unescapes HTML entities, strips HTML tags/URLs/control characters, collapses whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _URL_PATTERN.sub(" ", text)
    text = _CONTROL_CHAR_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


# ============================================================================
# Label schema
# ============================================================================
LABEL_NAMES_BINARY = {0: "REAL", 1: "FAKE"}
LABEL_NAMES_TERNARY = {0: "REAL", 1: "UNCERTAIN", 2: "FAKE"}


def label_names_for_scheme(scheme: str) -> dict[int, str]:
    return LABEL_NAMES_TERNARY if scheme == "ternary" else LABEL_NAMES_BINARY


# ============================================================================
# LIAR dataset
# ============================================================================
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "speaker_job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context",
]

_LIAR_REAL_LABELS = {"true", "mostly-true"}
_LIAR_FAKE_LABELS = {"false", "barely-true", "pants-fire"}
_LIAR_UNCERTAIN_LABELS = {"half-true"}


def map_liar_label(raw_label: str, scheme: str) -> Optional[int]:
    label = str(raw_label).strip().lower()
    if label in _LIAR_REAL_LABELS:
        return 0
    if label in _LIAR_UNCERTAIN_LABELS:
        return 1 if scheme == "ternary" else None
    if label in _LIAR_FAKE_LABELS:
        return 2 if scheme == "ternary" else 1
    logger.debug("Unrecognized LIAR label '%s'; dropping example.", raw_label)
    return None


def load_liar_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, names=LIAR_COLUMNS,
        quoting=3, dtype=str, keep_default_na=False,
    )
    logger.info("Loaded LIAR split '%s': %d rows.", path.name, len(df))
    return df


def process_liar(raw_dir: Path, scheme: str) -> pd.DataFrame:
    liar_dir = raw_dir / "liar"
    if not liar_dir.exists():
        logger.warning("LIAR directory not found at %s; skipping.", liar_dir)
        return pd.DataFrame(columns=["id", "text", "label", "label_name", "source_dataset"])

    split_files = {}
    for candidate in ["train.tsv", "valid.tsv", "val.tsv", "test.tsv"]:
        p = liar_dir / candidate
        if p.exists():
            key = "val" if candidate in ("valid.tsv", "val.tsv") else candidate.replace(".tsv", "")
            split_files[key] = p

    if not split_files:
        logger.warning("No LIAR .tsv files found under %s; skipping.", liar_dir)
        return pd.DataFrame(columns=["id", "text", "label", "label_name", "source_dataset"])

    frames = []
    for split_name, path in split_files.items():
        df = load_liar_split(path)
        df["_split_hint"] = split_name
        frames.append(df)
    liar_df = pd.concat(frames, ignore_index=True)

    liar_df["label"] = liar_df["label"].apply(lambda x: map_liar_label(x, scheme))
    before = len(liar_df)
    liar_df = liar_df.dropna(subset=["label"])
    liar_df["label"] = liar_df["label"].astype(int)
    logger.info("LIAR: dropped %d/%d examples with unmapped/ambiguous labels.", before - len(liar_df), before)

    label_names = label_names_for_scheme(scheme)
    out = pd.DataFrame({
        "id": "liar_" + liar_df["id"].astype(str),
        "text": liar_df["statement"].apply(clean_text),
        "label": liar_df["label"],
        "label_name": liar_df["label"].map(label_names),
        "source_dataset": "liar",
        "_split_hint": liar_df["_split_hint"],
    })
    out = out[out["text"].str.len() > 0]
    return out


# ============================================================================
# MultiFC dataset
# ============================================================================
MULTIFC_COLUMNS = [
    "claim_id", "claim", "label", "claim_url", "reason", "category",
    "speaker", "checker", "tags", "claim_entities", "article_title",
    "publish_date", "claim_date", "entities",
]

# Keyword-based normalization: MultiFC aggregates ~40 distinct fact-checker-
# specific label vocabularies (e.g. "full flop", "truth-o-meter", "pants on
# fire!", "mostly true", "unproven", "mixture", ...). We match the most
# common unambiguous patterns first (longest/most-specific substrings first)
# and drop anything that doesn't clearly resolve.
_MULTIFC_FAKE_PATTERNS = (
    "pants on fire", "false", "fake", "incorrect", "bogus", "hoax",
    "distorts", "distorted", "misleading", "not true", "untrue",
    "no evidence", "fiction", "scam",
)
_MULTIFC_REAL_PATTERNS = (
    "true", "correct", "accurate", "confirmed", "verified", "fact",
)
_MULTIFC_UNCERTAIN_PATTERNS = (
    "mixture", "mixed", "half true", "half-true", "partly true", "partly false",
    "unproven", "unverified", "in-between", "in between", "inconclusive",
    "understated", "overstated", "exaggerat",
)


def normalize_multifc_label(raw_label: str, scheme: str) -> Optional[int]:
    label = str(raw_label).strip().lower()
    if not label:
        return None

    # Order matters: check the most specific "uncertain" and "fake" patterns
    # before the broad "real" patterns, since e.g. "not true" and "mostly
    # true" both contain the substring "true".
    for pattern in _MULTIFC_UNCERTAIN_PATTERNS:
        if pattern in label:
            return 1 if scheme == "ternary" else None

    for pattern in _MULTIFC_FAKE_PATTERNS:
        if pattern in label:
            return 2 if scheme == "ternary" else 1

    for pattern in _MULTIFC_REAL_PATTERNS:
        if pattern in label:
            return 0

    logger.debug("Unrecognized MultiFC label '%s'; dropping example.", raw_label)
    return None


def load_multifc_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path, sep="\t", header=None, names=MULTIFC_COLUMNS,
            quoting=3, dtype=str, keep_default_na=False, on_bad_lines="skip",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse MultiFC file '%s': %s", path, exc)
        return pd.DataFrame(columns=MULTIFC_COLUMNS)
    logger.info("Loaded MultiFC file '%s': %d rows.", path.name, len(df))
    return df


def process_multifc(raw_dir: Path, scheme: str) -> pd.DataFrame:
    multifc_dir = raw_dir / "multifc"
    if not multifc_dir.exists():
        logger.warning("MultiFC directory not found at %s; skipping.", multifc_dir)
        return pd.DataFrame(columns=["id", "text", "label", "label_name", "source_dataset"])

    tsv_files = sorted(multifc_dir.glob("*.tsv"))
    if not tsv_files:
        logger.warning("No MultiFC .tsv files found under %s; skipping.", multifc_dir)
        return pd.DataFrame(columns=["id", "text", "label", "label_name", "source_dataset"])

    frames = [load_multifc_file(p) for p in tsv_files]
    multifc_df = pd.concat(frames, ignore_index=True)

    multifc_df["mapped_label"] = multifc_df["label"].apply(lambda x: normalize_multifc_label(x, scheme))
    before = len(multifc_df)
    multifc_df = multifc_df.dropna(subset=["mapped_label"])
    multifc_df["mapped_label"] = multifc_df["mapped_label"].astype(int)
    logger.info(
        "MultiFC: dropped %d/%d examples with unmapped/ambiguous labels.",
        before - len(multifc_df), before,
    )

    label_names = label_names_for_scheme(scheme)
    ids = multifc_df["claim_id"].astype(str)
    ids = ids.mask(ids.str.strip() == "", pd.Series(range(len(multifc_df))).astype(str))

    out = pd.DataFrame({
        "id": "multifc_" + ids,
        "text": multifc_df["claim"].apply(clean_text),
        "label": multifc_df["mapped_label"],
        "label_name": multifc_df["mapped_label"].map(label_names),
        "source_dataset": "multifc",
        "_split_hint": None,
    })
    out = out[out["text"].str.len() > 0]
    return out


# ============================================================================
# Split assembly
# ============================================================================
@dataclass
class SplitSizes:
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42


def assemble_splits(combined: pd.DataFrame, sizes: SplitSizes) -> dict[str, pd.DataFrame]:
    """
    Rows carrying a `_split_hint` from their source dataset (LIAR ships with
    predefined train/val/test files) are respected as-is. Rows without a
    hint (e.g. MultiFC) are stratified-split by label into train/val/test
    according to `sizes`.
    """
    has_hint = combined["_split_hint"].notna() & (combined["_split_hint"] != "")
    hinted = combined[has_hint]
    unhinted = combined[~has_hint]

    splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        subset = hinted[hinted["_split_hint"] == split_name]
        if len(subset) > 0:
            splits[split_name].append(subset)

    if len(unhinted) > 0:
        train_val, test = train_test_split(
            unhinted, test_size=sizes.test_size, random_state=sizes.seed,
            stratify=unhinted["label"] if unhinted["label"].nunique() > 1 else None,
        )
        relative_val_size = sizes.val_size / max(1.0 - sizes.test_size, 1e-6)
        train, val = train_test_split(
            train_val, test_size=relative_val_size, random_state=sizes.seed,
            stratify=train_val["label"] if train_val["label"].nunique() > 1 else None,
        )
        splits["train"].append(train)
        splits["val"].append(val)
        splits["test"].append(test)

    result = {}
    for split_name, frames in splits.items():
        if frames:
            df = pd.concat(frames, ignore_index=True).drop(columns=["_split_hint"], errors="ignore")
            df = df.sample(frac=1.0, random_state=sizes.seed).reset_index(drop=True)  # shuffle
            result[split_name] = df
        else:
            result[split_name] = pd.DataFrame(columns=["id", "text", "label", "label_name", "source_dataset"])
    return result


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(df), path)


def report_label_distribution(df: pd.DataFrame, split_name: str) -> None:
    counts = Counter(df["label_name"])
    total = sum(counts.values())
    parts = ", ".join(f"{name}={count} ({count / max(total, 1):.1%})" for name, count in sorted(counts.items()))
    logger.info("[%s] n=%d | %s", split_name, total, parts if parts else "(empty)")


# ============================================================================
# CLI entry point
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare LIAR + MultiFC into unified REAL/FAKE (or REAL/UNCERTAIN/FAKE) "
                    "train/val/test JSONL splits."
    )
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing liar/ and multifc/")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for JSONL splits")
    parser.add_argument("--label_scheme", type=str, default="binary", choices=["binary", "ternary"])
    parser.add_argument("--val_size", type=float, default=0.1, help="Val fraction for datasets without predefined splits")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction for datasets without predefined splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", type=str, default="liar,multifc",
        help="Comma-separated subset of {liar,multifc} to process",
    )
    parser.add_argument("--min_words", type=int, default=3, help="Drop examples with fewer than this many words")
    parser.add_argument("--max_words", type=int, default=200, help="Drop examples with more than this many words")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    requested_datasets = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}

    frames = []
    if "liar" in requested_datasets:
        frames.append(process_liar(raw_dir, args.label_scheme))
    if "multifc" in requested_datasets:
        frames.append(process_multifc(raw_dir, args.label_scheme))

    frames = [f for f in frames if len(f) > 0]
    if not frames:
        raise RuntimeError(
            f"No usable data found under '{raw_dir}' for requested datasets "
            f"{sorted(requested_datasets)}. Ensure raw files are placed at "
            f"data/raw/liar/*.tsv and/or data/raw/multifc/*.tsv."
        )

    combined = pd.concat(frames, ignore_index=True)

    word_counts = combined["text"].str.split().apply(len)
    before = len(combined)
    combined = combined[(word_counts >= args.min_words) & (word_counts <= args.max_words)]
    logger.info(
        "Filtered by length [%d, %d] words: kept %d/%d examples.",
        args.min_words, args.max_words, len(combined), before,
    )

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"])
    logger.info("Deduplicated exact-text duplicates: kept %d/%d examples.", len(combined), before)

    sizes = SplitSizes(val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    splits = assemble_splits(combined, sizes)

    logger.info("Final label scheme: %s -> %s", args.label_scheme, label_names_for_scheme(args.label_scheme))
    for split_name, df in splits.items():
        report_label_distribution(df, split_name)
        write_jsonl(df.drop(columns=["_split_hint"], errors="ignore"), output_dir / f"{split_name}.jsonl")

    manifest = {
        "label_scheme": args.label_scheme,
        "label_names": label_names_for_scheme(args.label_scheme),
        "datasets_included": sorted(requested_datasets),
        "split_sizes": {name: len(df) for name, df in splits.items()},
        "min_words": args.min_words,
        "max_words": args.max_words,
        "seed": args.seed,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
