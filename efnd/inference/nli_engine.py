"""
efnd.inference.nli_engine
===========================

Module 4: Claim-Evidence NLI Cross-Encoder.

Wraps `cross-encoder/nli-deberta-v3-base` for batched Natural Language
Inference over (claim, evidence_passage) pairs, producing calibrated
Softmax probabilities in the canonical order
[p_entailment, p_contradiction, p_neutral], regardless of the underlying
checkpoint's native label ordering.

Also provides `compute_claim_features`, which aggregates per-pair NLI
probabilities and hybrid retrieval scores into a fixed-size per-claim
feature tensor consumed by Module 5 (Gated Cross-Attention Fusion).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

CANONICAL_LABEL_ORDER = ["entailment", "contradiction", "neutral"]


@dataclass
class NLIConfig:
    model_name: str = "cross-encoder/nli-deberta-v3-base"
    max_seq_length: int = 256
    batch_size: int = 32
    label_order: list[str] = field(default_factory=lambda: list(CANONICAL_LABEL_ORDER))
    device: str = "auto"           # "auto" | "cuda" | "cpu" | "mps"
    mixed_precision: str = "no"    # "no" | "fp16" | "bf16"
    cache_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "NLIConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        nli_cfg = full_cfg.get("nli", {})
        return cls(
            model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base"),
            max_seq_length=nli_cfg.get("max_seq_length", 256),
            batch_size=nli_cfg.get("batch_size", 32),
            label_order=nli_cfg.get("label_order", list(CANONICAL_LABEL_ORDER)),
            device=full_cfg.get("device", "auto"),
            mixed_precision=full_cfg.get("mixed_precision", "no"),
            cache_dir=full_cfg.get("paths", {}).get("cache_dir"),
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device='cuda' but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
            logger.warning("Requested device='mps' but MPS is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(requested)


class NLIEngine:
    """
    Batched NLI inference engine over (claim, evidence) pairs.

    Usage
    -----
    >>> engine = NLIEngine.from_config_yaml("config.yaml")
    >>> pairs = [("The earth is flat.", "The earth is an oblate spheroid.")]
    >>> probs = engine.predict(pairs)  # np.ndarray, shape (1, 3) -> [ent, con, neu]
    """

    def __init__(self, config: NLIConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self._tokenizer = None
        self._model = None
        self._label_permutation: Optional[list[int]] = None  # model_idx -> canonical_idx
        self._amp_dtype = self._resolve_amp_dtype()

    @classmethod
    def from_config_yaml(cls, config_path: str | Path) -> "NLIEngine":
        return cls(config=NLIConfig.from_yaml(config_path))

    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda":
            return None
        if self.config.mixed_precision == "fp16":
            return torch.float16
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, cache_dir=self.config.cache_dir
        )
        self._model.to(self.device)
        self._model.eval()

        self._label_permutation = self._build_label_permutation()

        logger.info(
            "Loaded NLI cross-encoder '%s' on device=%s (id2label=%s)",
            self.config.model_name, self.device,
            self._model.config.id2label,
        )

    def _build_label_permutation(self) -> list[int]:
        """
        Builds an index array `perm` such that, given raw model logits/probs
        ordered by the model's native `id2label`, `probs[:, perm]` reorders
        them into `self.config.label_order` (canonically
        [entailment, contradiction, neutral]).
        """
        id2label = self._model.config.id2label
        native_order = [id2label[i].lower() for i in range(len(id2label))]

        # Normalize common label spelling variants across NLI checkpoints.
        _ALIASES = {
            "contradiction": "contradiction",
            "contradicts": "contradiction",
            "entailment": "entailment",
            "entails": "entailment",
            "neutral": "neutral",
            "not_entailment": "neutral",
        }
        normalized_native = [_ALIASES.get(lbl, lbl) for lbl in native_order]

        target_order = [lbl.lower() for lbl in self.config.label_order]

        missing = set(target_order) - set(normalized_native)
        if missing:
            raise ValueError(
                f"Model '{self.config.model_name}' native labels {normalized_native} "
                f"do not cover required labels {target_order}. Missing: {missing}"
            )

        # perm[i] = index in native ordering of the i-th target label
        perm = [normalized_native.index(lbl) for lbl in target_order]
        return perm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """
        Runs batched NLI inference over `pairs` of (claim, evidence_passage).

        Returns
        -------
        np.ndarray of shape (len(pairs), 3), columns ordered as
        [p_entailment, p_contradiction, p_neutral] (or as configured via
        `config.label_order`), each row summing to 1.0.
        """
        if not pairs:
            return np.zeros((0, len(self.config.label_order)), dtype=np.float32)

        self._load_model()

        all_probs: list[np.ndarray] = []
        batch_size = self.config.batch_size

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            claims = [p[0] for p in batch]
            evidences = [p[1] for p in batch]

            encoded = self._tokenizer(
                claims,
                evidences,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            if self._amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    logits = self._model(**encoded).logits
            else:
                logits = self._model(**encoded).logits

            probs = F.softmax(logits.float(), dim=-1)
            probs = probs[:, self._label_permutation]
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0).astype(np.float32)

    def predict_dict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Convenience wrapper returning a list of {label: probability} dicts."""
        probs = self.predict(pairs)
        return [
            {label: float(p) for label, p in zip(self.config.label_order, row)}
            for row in probs
        ]

    def to(self, device: str) -> "NLIEngine":
        """Move the underlying model to a new device (e.g. for multi-GPU pipelines)."""
        self.device = _resolve_device(device)
        self._amp_dtype = self._resolve_amp_dtype()
        if self._model is not None:
            self._model.to(self.device)
        return self


# ----------------------------------------------------------------------
# Module 5 input preparation: per-claim feature aggregation
# ----------------------------------------------------------------------
@dataclass
class ClaimFeatureBundle:
    """Aggregated NLI + retrieval features for a single claim, ready for
    consumption by the Gated Cross-Attention Fusion network (Module 5)."""
    claim_id: Optional[str]
    mean_entailment: float
    mean_contradiction: float
    mean_neutral: float
    max_entailment: float
    max_contradiction: float
    retrieval_margin: float
    retrieval_weighted_entailment: float
    retrieval_weighted_contradiction: float
    alignment_vector: list[float] = field(default_factory=list)
    num_evidence: int = 0

    def to_array(self) -> np.ndarray:
        """Fixed-order numeric feature vector for tensorization."""
        return np.array(
            [
                self.mean_entailment,
                self.mean_contradiction,
                self.mean_neutral,
                self.max_entailment,
                self.max_contradiction,
                self.retrieval_margin,
                self.retrieval_weighted_entailment,
                self.retrieval_weighted_contradiction,
                float(self.num_evidence),
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "mean_entailment": self.mean_entailment,
            "mean_contradiction": self.mean_contradiction,
            "mean_neutral": self.mean_neutral,
            "max_entailment": self.max_entailment,
            "max_contradiction": self.max_contradiction,
            "retrieval_margin": self.retrieval_margin,
            "retrieval_weighted_entailment": self.retrieval_weighted_entailment,
            "retrieval_weighted_contradiction": self.retrieval_weighted_contradiction,
            "alignment_vector": self.alignment_vector,
            "num_evidence": self.num_evidence,
        }


def compute_claim_features(
    nli_probs: np.ndarray,
    retrieval_scores: Sequence[float],
    claim_id: Optional[str] = None,
    label_order: Optional[list[str]] = None,
) -> ClaimFeatureBundle:
    """
    Aggregates per-evidence NLI probability rows and retrieval scores into
    a single per-claim feature bundle.

    Parameters
    ----------
    nli_probs : np.ndarray, shape (k, 3)
        NLI probabilities for the k retrieved evidence passages of one claim,
        columns ordered per `label_order` (default canonical:
        [entailment, contradiction, neutral]).
    retrieval_scores : Sequence[float], length k
        Fusion (e.g. RRF) or dense/sparse scores associated with each of the
        k evidence passages, same order as `nli_probs` rows.
    claim_id : Optional[str]
        Identifier of the claim these features belong to.
    label_order : Optional[list[str]]
        Column ordering of `nli_probs`. Defaults to canonical order.

    Returns
    -------
    ClaimFeatureBundle
        - mean/max entailment & contradiction, mean neutral
        - retrieval_margin: score gap between top-1 and top-2 evidence
          (0.0 if fewer than 2 evidence items; measures retrieval confidence)
        - retrieval-weighted entailment/contradiction: softmax-weighted
          average of stance probabilities by retrieval score
        - alignment_vector: elementwise product of top-evidence NLI
          distribution and its normalized retrieval weight (interpretable
          per-class alignment signal)
        - num_evidence: number of evidence passages aggregated
    """
    label_order = label_order or list(CANONICAL_LABEL_ORDER)
    ent_idx = label_order.index("entailment")
    con_idx = label_order.index("contradiction")
    neu_idx = label_order.index("neutral")

    nli_probs = np.asarray(nli_probs, dtype=np.float32)
    scores = np.asarray(list(retrieval_scores), dtype=np.float32)

    if nli_probs.ndim != 2 or nli_probs.shape[1] != len(label_order):
        raise ValueError(
            f"nli_probs must have shape (k, {len(label_order)}); got {nli_probs.shape}"
        )
    if nli_probs.shape[0] != scores.shape[0]:
        raise ValueError(
            f"nli_probs has {nli_probs.shape[0]} rows but retrieval_scores has "
            f"{scores.shape[0]} entries; they must align 1:1."
        )

    k = nli_probs.shape[0]
    if k == 0:
        return ClaimFeatureBundle(
            claim_id=claim_id,
            mean_entailment=0.0,
            mean_contradiction=0.0,
            mean_neutral=1.0,
            max_entailment=0.0,
            max_contradiction=0.0,
            retrieval_margin=0.0,
            retrieval_weighted_entailment=0.0,
            retrieval_weighted_contradiction=0.0,
            alignment_vector=[0.0, 0.0, 1.0],
            num_evidence=0,
        )

    mean_entailment = float(nli_probs[:, ent_idx].mean())
    mean_contradiction = float(nli_probs[:, con_idx].mean())
    mean_neutral = float(nli_probs[:, neu_idx].mean())
    max_entailment = float(nli_probs[:, ent_idx].max())
    max_contradiction = float(nli_probs[:, con_idx].max())

    sorted_scores = np.sort(scores)[::-1]
    retrieval_margin = float(sorted_scores[0] - sorted_scores[1]) if k >= 2 else 0.0

    # Softmax-normalize retrieval scores into weights for a weighted average.
    if np.ptp(scores) > 1e-9:
        weights = _softmax(scores)
    else:
        weights = np.full(k, 1.0 / k, dtype=np.float32)

    retrieval_weighted_entailment = float(np.dot(weights, nli_probs[:, ent_idx]))
    retrieval_weighted_contradiction = float(np.dot(weights, nli_probs[:, con_idx]))

    top_idx = int(np.argmax(scores))
    top_weight = float(weights[top_idx])
    alignment_vector = (nli_probs[top_idx] * top_weight).tolist()

    return ClaimFeatureBundle(
        claim_id=claim_id,
        mean_entailment=mean_entailment,
        mean_contradiction=mean_contradiction,
        mean_neutral=mean_neutral,
        max_entailment=max_entailment,
        max_contradiction=max_contradiction,
        retrieval_margin=retrieval_margin,
        retrieval_weighted_entailment=retrieval_weighted_entailment,
        retrieval_weighted_contradiction=retrieval_weighted_contradiction,
        alignment_vector=alignment_vector,
        num_evidence=k,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()


# ----------------------------------------------------------------------
# Standalone verification block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    config_yaml_path = repo_root / "config" / "config.yaml"

    if config_yaml_path.exists():
        engine = NLIEngine.from_config_yaml(config_yaml_path)
    else:
        logger.warning(
            "config.yaml not found at %s; using default NLIConfig.", config_yaml_path
        )
        engine = NLIEngine(config=NLIConfig())

    claim_text = "Drinking hot lemon water completely destroys respiratory viruses."
    evidence_passages = [
        "There is no clinical evidence that lemon or citrus destroys respiratory "
        "pathogens inside human cells.",
        "The World Health Organization stated in March 2020 that no citrus-based "
        "remedy has been shown to neutralize airborne viral pathogens.",
        "Lemons are a good source of vitamin C and are commonly used in cooking.",
    ]
    retrieval_scores_demo = [0.891, 0.734, 0.201]

    pairs_input = [(claim_text, ev) for ev in evidence_passages]
    nli_probabilities = engine.predict(pairs_input)

    print(f"\nNLI probabilities [entailment, contradiction, neutral] on device={engine.device}:\n")
    for evidence, probs in zip(evidence_passages, nli_probabilities):
        print(f"  {probs.round(4).tolist()}  <-  {evidence[:70]}...")

    claim_features = compute_claim_features(
        nli_probabilities, retrieval_scores_demo, claim_id="claim_1"
    )

    print("\nAggregated claim feature bundle:\n")
    print(json.dumps(claim_features.to_dict(), indent=2))
    print("\nFeature array:", claim_features.to_array())
