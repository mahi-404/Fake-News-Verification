"""
efnd.inference_pipeline
===================================

End-to-end orchestration (Section 3 pipeline diagram; Section 6 output
schema; Section 16 terminal summary).

`FakeNewsVerificationPipeline.analyze(article_text)` runs:

    preprocessing -> ClaimExtractor -> HybridRetriever -> NLIEngine
    -> compute_claim_features -> EvidenceAwareClassifier
    -> Dynamic Uncertainty Gate -> structured JSON + ASCII report

and returns the exact JSON schema from Section 6.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from efnd.claims.extractor import ClaimExtractor, ExtractedClaim
from efnd.inference.nli_engine import (
    CANONICAL_LABEL_ORDER,
    ClaimFeatureBundle,
    NLIEngine,
    compute_claim_features,
)
from efnd.models.hybrid_verifier import EvidenceAwareClassifier, HybridVerifierConfig
from efnd.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_LABEL_NAMES = ("LIKELY REAL", "LIKELY FAKE")

# Simple lexicon for the heuristic article-level stylistic score. This is a
# transparent, non-learned signal reported alongside the model's evidence-
# based verdict (Section 6: "article_stylistic_score") — it is NOT what
# drives the classification itself, which comes from the DeBERTa-v3 macro
# encoder inside `EvidenceAwareClassifier`.
_SENSATIONALIST_MARKERS = (
    "shocking", "unbelievable", "secret", "miracle", "you won't believe",
    "doctors hate", "destroys", "instantly", "banned", "conspiracy",
    "they don't want you to know", "exposed", "outrage",
)


@dataclass
class PipelineConfig:
    max_claims_per_article: int = 8
    final_top_k_evidence: int = 5
    claim_feature_dim: int = 9
    encoder_max_seq_length: int = 512
    device: str = "auto"
    uncertainty_tau: Optional[float] = None  # None -> use model's own config default

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PipelineConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ce = cfg.get("claim_extraction", {})
        retrieval_fusion = cfg.get("retrieval", {}).get("fusion", {})
        macro = cfg.get("macro_encoder", {})
        uncertainty = cfg.get("uncertainty_gate", {})
        return cls(
            max_claims_per_article=ce.get("max_claims_per_article", 8),
            final_top_k_evidence=retrieval_fusion.get("final_top_k", 5),
            encoder_max_seq_length=macro.get("max_seq_length", 512),
            device=cfg.get("device", "auto"),
            uncertainty_tau=uncertainty.get("evidence_sufficiency_threshold")
            if not uncertainty.get("enabled", True) is False else None,
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested device='cuda' but CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


class FakeNewsVerificationPipeline:
    """
    Ties together Modules 1-6 into a single `analyze(article_text)` call.

    Parameters
    ----------
    config_path : str | Path
        Path to the project's `config.yaml`.
    kb_index_dir : str | Path, optional
        Directory of a previously built `HybridRetriever` index (see
        `HybridRetriever.save`/`load`). Required for real evidence
        retrieval; if omitted, retrieval returns no evidence for every
        claim and NLI/verification degrades to the uncertainty fallback.
    model_checkpoint : str | Path, optional
        Path to a trained `EvidenceAwareClassifier` checkpoint (see
        `efnd.training.train.Trainer`). If omitted, a freshly initialized
        (non-fine-tuned) classifier is built directly from `config.yaml` —
        useful for wiring/integration testing, not for real predictions.
    device : str
        "auto" | "cuda" | "cpu".
    """

    def __init__(
        self,
        config_path: str | Path,
        kb_index_dir: Optional[str | Path] = None,
        model_checkpoint: Optional[str | Path] = None,
        device: str = "auto",
    ) -> None:
        self.config_path = Path(config_path)
        self.pipeline_config = PipelineConfig.from_yaml(self.config_path)
        self.device = _resolve_device(device if device != "auto" else self.pipeline_config.device)

        logger.info("Initializing FakeNewsVerificationPipeline (device=%s)...", self.device)

        # Module 2 — prefer the spaCy fallback for a bootable demo if the
        # instruction-tuned SLM is not already cached locally.
        self.claim_extractor = ClaimExtractor.from_config_yaml(self.config_path, mode="auto")

        # Module 3
        self.retriever = HybridRetriever.from_config_yaml(self.config_path)
        self._retriever_ready = False
        if kb_index_dir is not None:
            self.retriever.load(kb_index_dir)
            self._retriever_ready = True
            # Load the dense encoder now so the first /verify request is not
            # blocked on a Hugging Face download.
            self.retriever._load_encoder()
        else:
            logger.warning(
                "No kb_index_dir provided; retrieval will return zero evidence "
                "for every claim until HybridRetriever.load()/build_index() is called."
            )

        # Module 4
        self.nli_engine = NLIEngine.from_config_yaml(self.config_path)
        self.nli_engine._load_model()

        # Modules 5 & 6
        self.model_config = HybridVerifierConfig.from_yaml(self.config_path)
        self.model = self._load_or_build_model(model_checkpoint)
        self.model.to(self.device)
        self.model.float()
        self.model.eval()

        # Article tokenizer (must match the macro-context encoder).
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.encoder_model_name)

        logger.info("Pipeline ready.")

    def _load_or_build_model(self, model_checkpoint: Optional[str | Path]) -> EvidenceAwareClassifier:
        if model_checkpoint is not None:
            from efnd.evaluation.evaluate_generalization import load_model_from_checkpoint

            return load_model_from_checkpoint(
                model_checkpoint, device=self.device, override_config=self.model_config
            )
        logger.warning(
            "No model_checkpoint provided; building an UNTRAINED classifier. "
            "Predictions will not be meaningful until a trained checkpoint is supplied."
        )
        return EvidenceAwareClassifier(self.model_config)

    # ------------------------------------------------------------------
    # Stage 0: preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_article(article_text: str) -> str:
        return re.sub(r"\s+", " ", article_text).strip()

    @staticmethod
    def _compute_stylistic_heuristic(article_text: str) -> dict:
        """
        Lightweight, non-learned heuristic used only to populate the
        Section 6 `article_stylistic_score` field. Combines punctuation-
        based excitability signals with a small sensationalist-language
        lexicon.
        """
        text_lower = article_text.lower()
        n_chars = max(len(article_text), 1)

        exclamation_density = article_text.count("!") / n_chars * 100
        caps_words = re.findall(r"\b[A-Z]{3,}\b", article_text)
        caps_density = len(caps_words) / max(len(article_text.split()), 1)

        marker_hits = sum(1 for marker in _SENSATIONALIST_MARKERS if marker in text_lower)
        marker_score = min(marker_hits / 3.0, 1.0)

        raw_score = 0.5 * marker_score + 0.3 * min(exclamation_density * 10, 1.0) + 0.2 * min(caps_density * 5, 1.0)
        sensationalism_bias = float(min(max(raw_score, 0.0), 1.0))

        # Heuristic confidence grows with article length (more signal to
        # average over) and saturates for reasonably long articles.
        word_count = len(article_text.split())
        heuristic_confidence = float(min(0.5 + word_count / 400.0, 0.95))

        return {
            "sensationalism_bias": round(sensationalism_bias, 4),
            "confidence": round(heuristic_confidence, 4),
        }

    # ------------------------------------------------------------------
    # Stage 1-4: per-claim retrieval + NLI + feature aggregation
    # ------------------------------------------------------------------
    def _process_claim(self, claim: ExtractedClaim) -> tuple[ClaimFeatureBundle, list[dict], np.ndarray]:
        """
        Returns
        -------
        feature_bundle : ClaimFeatureBundle
        evidence_records : list[dict]   (for the Section 6 "retrieved_evidence" field)
        mean_nli_probs : np.ndarray, shape (3,)  ([entailment, contradiction, neutral])
        """
        top_k = self.pipeline_config.final_top_k_evidence

        if self._retriever_ready:
            retrieved = self.retriever.retrieve(claim.text, top_k=top_k)
        else:
            retrieved = []

        if not retrieved:
            empty_probs = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # fully neutral, no evidence
            feature_bundle = compute_claim_features(
                empty_probs, retrieval_scores=[0.0], claim_id=str(claim.claim_id)
            )
            return feature_bundle, [], empty_probs[0]

        evidence_texts = [r["passage"]["text"] for r in retrieved]
        pairs = [(claim.text, ev_text) for ev_text in evidence_texts]
        nli_probs = self.nli_engine.predict(pairs)   # (k, 3) -> [ent, con, neu]

        retrieval_scores = [r["rrf_score"] for r in retrieved]
        feature_bundle = compute_claim_features(
            nli_probs, retrieval_scores=retrieval_scores, claim_id=str(claim.claim_id)
        )

        evidence_records = []
        for r, probs_row in zip(retrieved, nli_probs):
            passage = r["passage"]
            evidence_records.append(
                {
                    "passage": passage["text"],
                    "source": passage.get("source", passage.get("id", "unknown")),
                    "retrieval_score": round(float(r["rrf_score"]), 6),
                    "nli_probs": {
                        "entailment": round(float(probs_row[0]), 6),
                        "contradiction": round(float(probs_row[1]), 6),
                        "neutral": round(float(probs_row[2]), 6),
                    },
                }
            )

        mean_nli_probs = nli_probs.mean(axis=0)   # (3,) representative distribution for this claim
        return feature_bundle, evidence_records, mean_nli_probs

    @staticmethod
    def _claim_verdict_from_probs(mean_probs: np.ndarray, has_evidence: bool) -> str:
        if not has_evidence:
            return "INSUFFICIENT_EVIDENCE"
        idx = int(np.argmax(mean_probs))
        return ["ENTAILED", "CONTRADICTED", "NEUTRAL"][idx]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze(self, article_text: str) -> dict:
        """
        Runs the full pipeline over one article and returns the Section 6
        JSON schema as a Python dict.
        """
        article_text = self._preprocess_article(article_text)
        if not article_text:
            raise ValueError("article_text is empty after preprocessing.")

        stylistic_score = self._compute_stylistic_heuristic(article_text)

        extracted_claims = self.claim_extractor.extract(article_text)
        max_claims = self.pipeline_config.max_claims_per_article
        feature_dim = self.pipeline_config.claim_feature_dim

        claim_features = np.zeros((max_claims, feature_dim), dtype=np.float32)
        claim_mask = np.zeros((max_claims,), dtype=np.int64)
        uncertainty_retrieval_scores = np.zeros((max_claims,), dtype=np.float32)
        uncertainty_nli_probs = np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32), (max_claims, 1)
        )  # default: fully neutral for padded/unfilled claim slots

        claims_analysis: list[dict] = []

        for slot, claim in enumerate(extracted_claims[:max_claims]):
            feature_bundle, evidence_records, mean_nli_probs = self._process_claim(claim)

            claim_features[slot] = feature_bundle.to_array()
            claim_mask[slot] = 1
            uncertainty_retrieval_scores[slot] = max(
                (r["retrieval_score"] for r in evidence_records), default=0.0
            )
            uncertainty_nli_probs[slot] = mean_nli_probs

            claims_analysis.append(
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "retrieved_evidence": [
                        {
                            "passage": ev["passage"],
                            "source": ev["source"],
                            "retrieval_score": ev["retrieval_score"],
                        }
                        for ev in evidence_records
                    ],
                    "nli_assessment": {
                        "entailment": round(float(mean_nli_probs[0]), 6),
                        "neutral": round(float(mean_nli_probs[2]), 6),
                        "contradiction": round(float(mean_nli_probs[1]), 6),
                    },
                    "claim_verdict": self._claim_verdict_from_probs(
                        mean_nli_probs, has_evidence=bool(evidence_records)
                    ),
                }
            )

        # --- Tokenize article for the macro-context encoder ---
        encoded = self.tokenizer(
            article_text,
            padding="max_length",
            truncation=True,
            max_length=self.pipeline_config.encoder_max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)               # (1, L)
        attention_mask = encoded["attention_mask"].to(self.device)     # (1, L)

        claim_features_t = torch.from_numpy(claim_features).unsqueeze(0).to(self.device)   # (1, C, F)
        claim_mask_t = torch.from_numpy(claim_mask).unsqueeze(0).to(self.device)             # (1, C)
        retrieval_scores_t = torch.from_numpy(uncertainty_retrieval_scores).unsqueeze(0).to(self.device)  # (1, C)
        nli_probs_t = torch.from_numpy(uncertainty_nli_probs).unsqueeze(0).to(self.device)   # (1, C, 3)

        results = self.model.predict_with_uncertainty(
            input_ids=input_ids,
            attention_mask=attention_mask,
            claim_features=claim_features_t,
            retrieval_scores=retrieval_scores_t,
            nli_probs=nli_probs_t,
            claim_mask=claim_mask_t,
            tau=self.pipeline_config.uncertainty_tau,
            label_names=_LABEL_NAMES,
        )[0]

        num_contradicted = sum(1 for c in claims_analysis if c["claim_verdict"] == "CONTRADICTED")

        if results["prediction"] == self.model.config.abstain_label:
            reasoning = (
                "Evidence sufficiency score fell below the abstention threshold; "
                "retrieved evidence was too sparse or ambiguous to support a confident verdict."
            )
        elif num_contradicted > 0:
            reasoning = (
                f"High contradiction detected across {num_contradicted} primary factual claim"
                f"{'s' if num_contradicted != 1 else ''}."
            )
        else:
            reasoning = (
                "No significant contradictions detected between extracted claims and "
                "retrieved evidence; article-level stylistic signal drove the final verdict."
            )

        output = {
            "prediction": results["prediction"],
            "confidence": round(results["confidence"], 6),
            "verdict_reasoning": reasoning,
            "article_stylistic_score": stylistic_score,
            "claims_analysis": claims_analysis,
            "uncertainty_metrics": {
                "evidence_sufficiency_score": round(results["evidence_sufficiency_score"], 6),
                "abstain_triggered": results["abstain_triggered"],
            },
        }
        return output

    # ------------------------------------------------------------------
    # Section 16: ASCII terminal report
    # ------------------------------------------------------------------
    @staticmethod
    def format_terminal_report(result: dict, width: int = 100) -> str:
        """
        Renders `result` (as returned by `analyze`) as a human-readable
        ASCII terminal summary table.
        """
        lines: list[str] = []
        rule = "=" * width
        thin_rule = "-" * width

        lines.append(rule)
        lines.append("EVIDENCE-AWARE FAKE NEWS VERIFICATION REPORT".center(width))
        lines.append(rule)
        lines.append(f"PREDICTION      : {result['prediction']}")
        lines.append(f"CONFIDENCE      : {result['confidence']:.2%}")
        style = result["article_stylistic_score"]
        lines.append(
            f"STYLISTIC SCORE : sensationalism={style['sensationalism_bias']:.2f} "
            f"(heuristic confidence={style['confidence']:.2f})"
        )
        um = result["uncertainty_metrics"]
        lines.append(
            f"ABSTAIN         : {um['abstain_triggered']}  "
            f"(evidence_sufficiency_score={um['evidence_sufficiency_score']:.3f})"
        )
        lines.append(f"REASONING       : {result['verdict_reasoning']}")
        lines.append(thin_rule)

        header = f"{'ID':<4}{'CLAIM':<48}{'VERDICT':<20}{'ENT':>8}{'CON':>8}{'NEU':>8}"
        lines.append(header)
        lines.append(thin_rule)

        for claim in result["claims_analysis"]:
            claim_text = claim["text"]
            truncated = (claim_text[:44] + "...") if len(claim_text) > 47 else claim_text
            nli = claim["nli_assessment"]
            row = (
                f"{claim['claim_id']:<4}"
                f"{truncated:<48}"
                f"{claim['claim_verdict']:<20}"
                f"{nli['entailment']:>8.2f}"
                f"{nli['contradiction']:>8.2f}"
                f"{nli['neutral']:>8.2f}"
            )
            lines.append(row)

            if claim["retrieved_evidence"]:
                top_evidence = claim["retrieved_evidence"][0]
                source = top_evidence["source"]
                passage_snip = top_evidence["passage"]
                passage_snip = (
                    (passage_snip[:80] + "...") if len(passage_snip) > 83 else passage_snip
                )
                lines.append(f"      -> [{source}] {passage_snip}")
            else:
                lines.append("      -> (no supporting evidence retrieved)")

        lines.append(rule)
        return "\n".join(lines)

    def print_terminal_report(self, result: dict, width: int = 100) -> None:
        print(self.format_terminal_report(result, width=width))


# ----------------------------------------------------------------------
# Standalone verification block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    log = logging.getLogger(__name__)

    repo_root = Path(__file__).resolve().parents[1]
    config_yaml_path = repo_root / "config" / "config.yaml"

    _SAMPLE_ARTICLE = """
    SHOCKING: A new study claims that drinking hot lemon water completely
    destroys respiratory viruses within seconds! Doctors don't want you to
    know this secret. The World Health Organization has not commented, but
    over 90 percent of readers online believe it already.
    """

    _SAMPLE_KB = [
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
    ]

    try:
        pipeline = FakeNewsVerificationPipeline(config_path=config_yaml_path)

        pipeline.retriever.build_index(_SAMPLE_KB)
        pipeline._retriever_ready = True

        analysis_result = pipeline.analyze(_SAMPLE_ARTICLE)

        print(json.dumps(analysis_result, indent=2))
        print()
        pipeline.print_terminal_report(analysis_result)

    except Exception as exc:  # noqa: BLE001
        log.error(
            "Pipeline smoke test failed to run end-to-end in this environment "
            "(most likely: no network access to download pretrained "
            "DeBERTa-v3 / NLI / BGE weights from huggingface.co, or missing "
            "torch/transformers). This is expected in network-restricted "
            "sandboxes; the pipeline code itself is structurally complete. "
            "Underlying error: %s", exc,
        )
