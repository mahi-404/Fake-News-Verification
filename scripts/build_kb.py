"""
scripts/build_kb.py
======================

Builds the retrieval knowledge base (Module 3's corpus) from raw Wikipedia
or FEVER reference-passage dumps: chunks each source document into
~100-word passages and indexes them with `HybridRetriever`
(dense FAISS + sparse BM25), persisting the result to `data/kb_index/`.

Supported input formats
------------------------
--source_format kilt
    KILT-style Wikipedia JSONL: one JSON object per line with at least
    {"wikipedia_id" | "id", "wikipedia_title" | "title", "text"}, where
    "text" is either a single string of the full article or a list of
    paragraph strings.

--source_format fever
    FEVER wiki-pages JSONL: one JSON object per line with
    {"id": <page title>, "lines": <raw sentence-indexed string>}, where
    "lines" is FEVER's native format:
        "0\\tFirst sentence.\\t<hyperlink spans>\\n1\\tSecond sentence.\\t..."

--source_format plain_jsonl
    Generic fallback: one JSON object per line with a configurable
    `--text_field` (default "text") and optional `--title_field`
    (default "title").

Usage
-----
    python scripts/build_kb.py \\
        --input_path data/raw/kb_source/kilt_wikipedia_sample.jsonl \\
        --source_format kilt \\
        --output_dir data/kb_index \\
        --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================================
# Chunking
# ============================================================================
def chunk_text_into_passages(
    text: str,
    chunk_size_words: int = 100,
    overlap_words: int = 0,
) -> list[str]:
    """
    Splits `text` into a list of passages of approximately
    `chunk_size_words` whitespace-delimited words each, with an optional
    sliding-window overlap between consecutive chunks.
    """
    words = text.split()
    if not words:
        return []

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    stride = chunk_size_words - overlap_words
    chunks = []
    for start in range(0, len(words), stride):
        chunk_words = words[start:start + chunk_size_words]
        if not chunk_words:
            continue
        # Avoid emitting a tiny trailing fragment (< 20% of target size)
        # that duplicates content already covered by overlap.
        if len(chunk_words) < max(chunk_size_words * 0.2, 1) and start > 0:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
    return chunks


# ============================================================================
# Source format parsers
# ============================================================================
def _clean(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def parse_fever_lines(raw_lines: str) -> list[str]:
    """
    Parses FEVER's native `lines` field into an ordered list of sentence
    strings, discarding the leading sentence-index column and any trailing
    hyperlink-span annotation columns.
    """
    sentences = []
    for raw_line in raw_lines.split("\n"):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) < 2:
            continue
        sentence_text = fields[1].strip()
        if sentence_text:
            sentences.append(sentence_text)
    return sentences


def iter_kilt_documents(input_path: Path) -> Iterator[tuple[str, str, str]]:
    """Yields (doc_id, title, full_text) tuples from a KILT-style Wikipedia JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d of %s.", line_no, input_path)
                continue

            doc_id = str(record.get("wikipedia_id") or record.get("id") or line_no)
            title = record.get("wikipedia_title") or record.get("title") or doc_id

            raw_text = record.get("text")
            if isinstance(raw_text, list):
                full_text = " ".join(str(p) for p in raw_text)
            else:
                full_text = str(raw_text or "")

            full_text = _clean(full_text)
            if full_text:
                yield doc_id, str(title), full_text


def iter_fever_documents(input_path: Path) -> Iterator[tuple[str, str, str]]:
    """Yields (doc_id, title, full_text) tuples from a FEVER wiki-pages JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d of %s.", line_no, input_path)
                continue

            doc_id = str(record.get("id", line_no))
            title = doc_id.replace("_", " ")

            raw_lines = record.get("lines", "")
            sentences = parse_fever_lines(raw_lines)
            full_text = _clean(" ".join(sentences))
            if full_text:
                yield doc_id, title, full_text


def iter_plain_jsonl_documents(
    input_path: Path, text_field: str, title_field: Optional[str],
) -> Iterator[tuple[str, str, str]]:
    """Yields (doc_id, title, full_text) tuples from a generic JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d of %s.", line_no, input_path)
                continue

            doc_id = str(record.get("id", line_no))
            title = str(record.get(title_field, doc_id)) if title_field else doc_id
            full_text = _clean(str(record.get(text_field, "")))
            if full_text:
                yield doc_id, title, full_text


_SOURCE_ITERATORS = {
    "kilt": iter_kilt_documents,
    "fever": iter_fever_documents,
}


# ============================================================================
# Passage assembly
# ============================================================================
def build_passages(
    input_path: Path,
    source_format: str,
    chunk_size_words: int,
    overlap_words: int,
    max_documents: Optional[int] = None,
    max_passages: Optional[int] = None,
    text_field: str = "text",
    title_field: Optional[str] = "title",
) -> list[dict]:
    """
    Streams documents from `input_path` (per `source_format`), chunks each
    into ~`chunk_size_words`-word passages, and returns a flat list of
    passage dicts ready for `HybridRetriever.build_index`:
        {"id": str, "text": str, "source": str, "title": str, "doc_id": str}
    """
    if source_format == "plain_jsonl":
        doc_iterator = iter_plain_jsonl_documents(input_path, text_field, title_field)
    elif source_format in _SOURCE_ITERATORS:
        doc_iterator = _SOURCE_ITERATORS[source_format](input_path)
    else:
        raise ValueError(f"Unsupported source_format '{source_format}'.")

    passages: list[dict] = []
    num_documents = 0

    for doc_id, title, full_text in doc_iterator:
        if max_documents is not None and num_documents >= max_documents:
            break
        num_documents += 1

        chunks = chunk_text_into_passages(full_text, chunk_size_words, overlap_words)
        for chunk_idx, chunk_text in enumerate(chunks):
            passages.append(
                {
                    "id": f"{doc_id}_{chunk_idx}",
                    "text": chunk_text,
                    "source": title,
                    "title": title,
                    "doc_id": doc_id,
                }
            )
            if max_passages is not None and len(passages) >= max_passages:
                logger.info("Reached max_passages=%d; stopping ingestion.", max_passages)
                return passages

        if num_documents % 1000 == 0:
            logger.info("Processed %d documents -> %d passages so far...", num_documents, len(passages))

    logger.info(
        "Finished chunking: %d documents -> %d passages (chunk_size=%d words, overlap=%d words).",
        num_documents, len(passages), chunk_size_words, overlap_words,
    )
    return passages


# ============================================================================
# CLI entry point
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk raw Wikipedia/FEVER reference passages into ~100-word "
                    "passages and build the HybridRetriever knowledge-base index."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw source JSONL file")
    parser.add_argument(
        "--source_format", type=str, default="kilt",
        choices=["kilt", "fever", "plain_jsonl"],
    )
    parser.add_argument("--text_field", type=str, default="text", help="Used only for --source_format plain_jsonl")
    parser.add_argument("--title_field", type=str, default="title", help="Used only for --source_format plain_jsonl")
    parser.add_argument("--chunk_size_words", type=int, default=100)
    parser.add_argument("--chunk_overlap_words", type=int, default=0)
    parser.add_argument("--max_documents", type=int, default=None, help="Cap on number of source documents to ingest")
    parser.add_argument("--max_passages", type=int, default=None, help="Cap on total number of passages produced")
    parser.add_argument("--output_dir", type=str, default="data/kb_index", help="Where to persist the built index")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml (retrieval settings)")
    parser.add_argument(
        "--passages_only", action="store_true",
        help="Only chunk and write passages.jsonl to --output_dir; skip embedding/indexing "
             "(useful for inspecting chunking output before committing to a full build).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input source file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    passages = build_passages(
        input_path=input_path,
        source_format=args.source_format,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.chunk_overlap_words,
        max_documents=args.max_documents,
        max_passages=args.max_passages,
        text_field=args.text_field,
        title_field=args.title_field,
    )

    if not passages:
        raise RuntimeError(
            f"No passages were produced from '{input_path}' with source_format="
            f"'{args.source_format}'. Check the file format and field names."
        )

    if args.passages_only:
        passages_path = output_dir / "passages.jsonl"
        with open(passages_path, "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("--passages_only set: wrote %d chunked passages to %s (no index built).",
                    len(passages), passages_path)
        return

    # Imported lazily: HybridRetriever's own heavy deps (sentence-transformers,
    # faiss, rank_bm25) are only touched once we actually build the index.
    from efnd.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever.from_config_yaml(args.config)
    retriever.build_index(passages)
    retriever.save(output_dir)

    manifest = {
        "input_path": str(input_path),
        "source_format": args.source_format,
        "chunk_size_words": args.chunk_size_words,
        "chunk_overlap_words": args.chunk_overlap_words,
        "num_passages": len(passages),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Knowledge base index built and saved to %s (%d passages).", output_dir, len(passages))


if __name__ == "__main__":
    main()
