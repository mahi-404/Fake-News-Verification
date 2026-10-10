"""
efnd.retrieval.hybrid_retriever
=================================

Module 3: Hybrid Evidence Retrieval.

Combines dense semantic retrieval (BGE-large-en-v1.5 embeddings indexed
via FAISS IndexFlatIP over L2-normalized vectors, i.e. cosine similarity)
with sparse lexical retrieval (BM25Okapi over tokenized passages), fused
via Reciprocal Rank Fusion (RRF):

    RRF(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))

with k = 60 per `config.yaml -> retrieval.fusion.rrf_k`.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _simple_tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievalConfig:
    dense_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    normalize_embeddings: bool = True
    dense_batch_size: int = 64
    dense_top_k: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    sparse_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    device: str = "cpu"
    cache_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "RetrievalConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        r = full_cfg.get("retrieval", {})
        dense = r.get("dense", {})
        sparse = r.get("sparse", {})
        fusion = r.get("fusion", {})
        return cls(
            dense_model_name=dense.get("model_name", "BAAI/bge-large-en-v1.5"),
            embedding_dim=dense.get("embedding_dim", 1024),
            normalize_embeddings=dense.get("normalize_embeddings", True),
            dense_batch_size=dense.get("batch_size", 64),
            dense_top_k=dense.get("top_k", 20),
            bm25_k1=sparse.get("k1", 1.5),
            bm25_b=sparse.get("b", 0.75),
            sparse_top_k=sparse.get("top_k", 20),
            rrf_k=fusion.get("rrf_k", 60),
            final_top_k=fusion.get("final_top_k", 5),
            device=full_cfg.get("device", "cpu"),
            cache_dir=full_cfg.get("paths", {}).get("cache_dir"),
        )


class HybridRetriever:
    """
    Hybrid dense + sparse evidence retriever with RRF fusion.

    Passages are dicts of the form:
        {"id": str, "text": str, "source": str, ...extra metadata...}

    Usage
    -----
    >>> retriever = HybridRetriever.from_config_yaml("config.yaml")
    >>> retriever.build_index(passages)
    >>> retriever.save("data/kb_index")
    >>> retriever2 = HybridRetriever.from_config_yaml("config.yaml")
    >>> retriever2.load("data/kb_index")
    >>> results = retriever2.retrieve("The earth is flat.", top_k=5)
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._encoder = None  # lazy-loaded SentenceTransformer
        self._faiss_index = None
        self._bm25 = None
        self._passages: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    @classmethod
    def from_config_yaml(cls, config_path: str | Path) -> "HybridRetriever":
        return cls(config=RetrievalConfig.from_yaml(config_path))

    # ------------------------------------------------------------------
    # Encoder loading
    # ------------------------------------------------------------------
    def _resolve_encoder_device(self) -> str:
        requested = (self.config.device or "cpu").lower()
        if requested in {"cuda", "gpu"}:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested for dense encoder but unavailable; using CPU."
                    )
                    return "cpu"
            except Exception:  # noqa: BLE001
                return "cpu"
        if requested == "mps":
            try:
                import torch

                if not (
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                ):
                    logger.warning("MPS requested for dense encoder but unavailable; using CPU.")
                    return "cpu"
            except Exception:  # noqa: BLE001
                return "cpu"
        return requested if requested != "auto" else "cpu"

    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer

        device = self._resolve_encoder_device()
        self._encoder = SentenceTransformer(
            self.config.dense_model_name,
            device=device,
            cache_folder=self.config.cache_dir,
        )
        logger.info("Loaded dense encoder: %s (device=%s)", self.config.dense_model_name, device)
        return self._encoder

    def _encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        encoder = self._load_encoder()
        # BGE models recommend a query instruction prefix for asymmetric search.
        if is_query:
            texts = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
        embeddings = encoder.encode(
            texts,
            batch_size=self.config.dense_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def build_index(self, passages: list[dict[str, str]]) -> None:
        """
        Build both the dense FAISS index and the sparse BM25 index over
        `passages`. Each passage dict must contain at least a "text" key;
        an "id" key is auto-generated (as the list index) if absent.
        """
        if not passages:
            raise ValueError("`passages` must be a non-empty list of passage dicts.")

        import faiss

        self._passages = []
        for idx, p in enumerate(passages):
            if "text" not in p or not p["text"].strip():
                raise ValueError(f"Passage at index {idx} is missing non-empty 'text'.")
            passage = dict(p)
            passage.setdefault("id", str(idx))
            self._passages.append(passage)

        texts = [p["text"] for p in self._passages]

        # --- Dense index ---
        logger.info("Encoding %d passages with dense encoder...", len(texts))
        embeddings = self._encode(texts, is_query=False)
        dim = embeddings.shape[1]
        if dim != self.config.embedding_dim:
            logger.warning(
                "Configured embedding_dim=%d but encoder produced dim=%d; "
                "using actual dim.", self.config.embedding_dim, dim,
            )
            self.config.embedding_dim = dim

        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)
        logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d).", len(texts), dim)

        # --- Sparse index ---
        from rank_bm25 import BM25Okapi

        self._tokenized_corpus = [_simple_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        logger.info("Built BM25Okapi index over %d tokenized passages.", len(texts))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, index_dir: str | Path) -> None:
        if self._faiss_index is None or self._bm25 is None:
            raise RuntimeError("No index to save; call build_index() first.")

        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(index_dir / "dense.faiss"))

        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "tokenized_corpus": self._tokenized_corpus}, f
            )

        with open(index_dir / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self._passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        with open(index_dir / "retrieval_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2)

        logger.info("Saved hybrid index to %s", index_dir)

    def load(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)

        dense_path = index_dir / "dense.faiss"
        bm25_path = index_dir / "bm25.pkl"
        passages_path = index_dir / "passages.jsonl"

        if not (dense_path.exists() and bm25_path.exists() and passages_path.exists()):
            raise FileNotFoundError(f"Incomplete or missing index files in {index_dir}")

        self._faiss_index = faiss.read_index(str(dense_path))

        with open(bm25_path, "rb") as f:
            state = pickle.load(f)
            self._bm25 = state["bm25"]
            self._tokenized_corpus = state["tokenized_corpus"]

        self._passages = []
        with open(passages_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._passages.append(json.loads(line))

        logger.info(
            "Loaded hybrid index from %s (%d passages).", index_dir, len(self._passages)
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, claim: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the top-k evidence passages for `claim`, fused via RRF
        over the dense and sparse rankings.

        Returns a list of dicts:
            {
                "passage": <passage dict>,
                "rrf_score": float,
                "dense_rank": Optional[int],
                "sparse_rank": Optional[int],
                "dense_score": Optional[float],
                "sparse_score": Optional[float],
            }
        sorted by descending `rrf_score`.
        """
        if self._faiss_index is None or self._bm25 is None:
            raise RuntimeError(
                "Index not built or loaded; call build_index() or load() first."
            )
        if not claim or not claim.strip():
            return []

        top_k = top_k or self.config.final_top_k

        dense_ranked = self._dense_search(claim)
        sparse_ranked = self._sparse_search(claim)

        fused = self._reciprocal_rank_fusion(dense_ranked, sparse_ranked)
        fused_sorted = sorted(fused.items(), key=lambda kv: kv[1]["rrf_score"], reverse=True)

        results = []
        for passage_idx, info in fused_sorted[:top_k]:
            results.append(
                {
                    "passage": self._passages[passage_idx],
                    "rrf_score": round(info["rrf_score"], 6),
                    "dense_rank": info.get("dense_rank"),
                    "sparse_rank": info.get("sparse_rank"),
                    "dense_score": info.get("dense_score"),
                    "sparse_score": info.get("sparse_score"),
                }
            )
        return results

    def _dense_search(self, claim: str) -> list[tuple[int, float]]:
        """Returns list of (passage_index, similarity_score), rank-ordered desc."""
        n_results = min(self.config.dense_top_k, len(self._passages))
        if n_results == 0:
            return []
        query_embedding = self._encode([claim], is_query=True)
        scores, indices = self._faiss_index.search(query_embedding, n_results)
        ranked = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
        return ranked

    def _sparse_search(self, claim: str) -> list[tuple[int, float]]:
        """Returns list of (passage_index, bm25_score), rank-ordered desc."""
        n_results = min(self.config.sparse_top_k, len(self._passages))
        if n_results == 0:
            return []
        tokenized_query = _simple_tokenize(claim)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n_results]
        ranked = [(int(idx), float(scores[idx])) for idx in top_indices]
        return ranked

    def _reciprocal_rank_fusion(
        self,
        dense_ranked: list[tuple[int, float]],
        sparse_ranked: list[tuple[int, float]],
    ) -> dict[int, dict]:
        """
        Computes RRF(d) = sum_m 1 / (rrf_k + rank_m(d)) for each passage
        appearing in either ranked list. Ranks are 1-indexed.
        """
        k = self.config.rrf_k
        fused: dict[int, dict] = {}

        for rank, (passage_idx, score) in enumerate(dense_ranked, start=1):
            entry = fused.setdefault(
                passage_idx,
                {"rrf_score": 0.0, "dense_rank": None, "sparse_rank": None,
                 "dense_score": None, "sparse_score": None},
            )
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["dense_rank"] = rank
            entry["dense_score"] = round(score, 6)

        for rank, (passage_idx, score) in enumerate(sparse_ranked, start=1):
            entry = fused.setdefault(
                passage_idx,
                {"rrf_score": 0.0, "dense_rank": None, "sparse_rank": None,
                 "dense_score": None, "sparse_score": None},
            )
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["sparse_rank"] = rank
            entry["sparse_score"] = round(score, 6)

        return fused


# ----------------------------------------------------------------------
# Standalone verification block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    _SAMPLE_PASSAGES = [
        {
            "id": "kilt_849201",
            "text": "There is no clinical evidence that lemon or citrus destroys "
                    "respiratory pathogens inside human cells.",
            "source": "WHO Factsheet",
        },
        {
            "id": "kilt_112233",
            "text": "The World Health Organization stated in March 2020 that no "
                    "citrus-based remedy has been shown to neutralize airborne "
                    "viral pathogens.",
            "source": "WHO Statement",
        },
        {
            "id": "kilt_998877",
            "text": "Regular handwashing with soap and water for at least 20 seconds "
                    "significantly reduces transmission of respiratory viruses.",
            "source": "CDC Guidelines",
        },
        {
            "id": "kilt_554433",
            "text": "The Eiffel Tower, completed in 1889, is located in Paris, France, "
                    "and stands approximately 330 meters tall.",
            "source": "Encyclopedia Britannica",
        },
        {
            "id": "kilt_667788",
            "text": "Vitamin C supplementation has not been shown in randomized "
                    "controlled trials to prevent or cure the common cold in the "
                    "general population.",
            "source": "Cochrane Review",
        },
    ]

    repo_root = Path(__file__).resolve().parents[2]
    config_yaml_path = repo_root / "config" / "config.yaml"

    if config_yaml_path.exists():
        retriever = HybridRetriever.from_config_yaml(config_yaml_path)
    else:
        logger.warning(
            "config.yaml not found at %s; using default RetrievalConfig.",
            config_yaml_path,
        )
        retriever = HybridRetriever(config=RetrievalConfig())

    retriever.build_index(_SAMPLE_PASSAGES)

    index_output_dir = repo_root / "data" / "kb_index" / "demo"
    retriever.save(index_output_dir)

    reloaded = HybridRetriever(config=retriever.config)
    reloaded.load(index_output_dir)

    query_claim = "Drinking hot lemon water completely destroys respiratory viruses."
    top_results = reloaded.retrieve(query_claim, top_k=3)

    print(f"\nTop {len(top_results)} fused evidence passages for claim:\n  \"{query_claim}\"\n")
    for rank, result in enumerate(top_results, start=1):
        print(json.dumps(result, indent=2))
        print("-" * 80)
