"""
efnd.claims.extractor
======================

Module 2: Claim Extraction & Filtering.

Segments a news article into an ordered list of discrete, check-worthy
factual claims, per `config.yaml -> claim_extraction`.

Two extraction backends are supported:

1. ``slm`` (primary): zero-shot prompt-based extraction using a local
   instruction-tuned small language model (e.g. Qwen2.5-1.5B-Instruct or
   flan-t5-base) via a Hugging Face `transformers.pipeline`.
2. ``rule_based`` (fallback): spaCy dependency-parse filtering that keeps
   declarative sentences containing at least one named entity and one
   numeric/temporal/quantitative expression, which correlates strongly
   with check-worthy factual assertions.

The fallback is used automatically if SLM loading or generation fails,
and can also be selected explicitly via ``mode="rule_based"``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

logger = logging.getLogger(__name__)

_SPACY_ENT_LABELS_NUMERIC = {
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL",
}
_SPACY_ENT_LABELS_SUBJECT = {
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "PRODUCT", "LAW",
}

_CLAIM_EXTRACTION_PROMPT_TEMPLATE = """You are a fact-checking assistant. Read the news article below and extract only the discrete, objectively verifiable factual claims it makes. 

Rules:
- Ignore opinions, rhetorical questions, subjective commentary, and non-verifiable filler.
- Each claim must be a single, self-contained, declarative sentence.
- Do not include claims longer than {max_claim_tokens} tokens or shorter than {min_claim_tokens} tokens.
- Extract at most {max_claims} claims, ordered by how central they are to the article's main assertions.
- Output strictly as a JSON array of strings, with no additional commentary, no markdown fences, and no explanation.

Article:
\"\"\"
{article_text}
\"\"\"

JSON array of claims:"""


@dataclass
class ExtractedClaim:
    claim_id: int
    text: str
    source_sentence_span: Optional[tuple[int, int]] = None
    extraction_method: str = "slm"
    checkworthiness_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_sentence_span": self.source_sentence_span,
            "extraction_method": self.extraction_method,
            "checkworthiness_score": self.checkworthiness_score,
        }


@dataclass
class ClaimExtractionConfig:
    method: Literal["instruction_tuned_slm", "claimbuster"] = "instruction_tuned_slm"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_claims_per_article: int = 8
    min_claim_tokens: int = 4
    max_claim_tokens: int = 64
    remove_questions: bool = True
    remove_opinions: bool = True
    checkworthiness_threshold: float = 0.55
    temperature: float = 0.0
    max_new_tokens: int = 256
    do_sample: bool = False
    device: str = "cpu"
    cache_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ClaimExtractionConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        ce = full_cfg.get("claim_extraction", {})
        gen = ce.get("generation", {})
        filt = ce.get("filtering", {})
        return cls(
            method=ce.get("method", "instruction_tuned_slm"),
            model_name=ce.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            max_claims_per_article=ce.get("max_claims_per_article", 8),
            min_claim_tokens=ce.get("min_claim_tokens", 4),
            max_claim_tokens=ce.get("max_claim_tokens", 64),
            remove_questions=filt.get("remove_questions", True),
            remove_opinions=filt.get("remove_opinions", True),
            checkworthiness_threshold=filt.get("checkworthiness_threshold", 0.55),
            temperature=gen.get("temperature", 0.0),
            max_new_tokens=gen.get("max_new_tokens", 256),
            do_sample=gen.get("do_sample", False),
            device=full_cfg.get("device", "cpu"),
            cache_dir=full_cfg.get("paths", {}).get("cache_dir"),
        )


_OPINION_MARKERS = (
    "i think", "i believe", "in my opinion", "it seems", "arguably",
    "should", "must", "could be", "might be", "we feel", "clearly",
    "obviously", "shockingly", "disgustingly", "amazingly",
)


class ClaimExtractor:
    """
    Extracts check-worthy factual claims from a news article.

    Parameters
    ----------
    config : ClaimExtractionConfig
        Parsed claim-extraction configuration (see `from_config_yaml`).
    mode : {"auto", "slm", "rule_based"}
        "auto" (default) attempts SLM extraction and transparently falls
        back to rule-based extraction on any failure (model load error,
        generation error, malformed output). "slm" and "rule_based" force
        a specific backend and raise on failure instead of falling back.
    lazy_load : bool
        If True (default), the SLM pipeline and spaCy model are loaded
        lazily on first use rather than at construction time.
    """

    def __init__(
        self,
        config: ClaimExtractionConfig,
        mode: Literal["auto", "slm", "rule_based"] = "auto",
        lazy_load: bool = True,
    ) -> None:
        self.config = config
        self.mode = mode
        self._slm_pipeline = None
        self._nlp = None
        if not lazy_load:
            if mode in ("auto", "slm"):
                self._load_slm()
            if mode in ("auto", "rule_based"):
                self._load_spacy()

    @classmethod
    def from_config_yaml(
        cls,
        config_path: str | Path,
        mode: Literal["auto", "slm", "rule_based"] = "auto",
        lazy_load: bool = True,
    ) -> "ClaimExtractor":
        config = ClaimExtractionConfig.from_yaml(config_path)
        return cls(config=config, mode=mode, lazy_load=lazy_load)

    # ------------------------------------------------------------------
    # Lazy backend loaders
    # ------------------------------------------------------------------
    def _load_slm(self) -> None:
        if self._slm_pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            is_seq2seq = "flan-t5" in self.config.model_name.lower() or \
                "t5" in self.config.model_name.lower()
            task = "text2text-generation" if is_seq2seq else "text-generation"
            model_kwargs = {}
            if self.config.cache_dir:
                model_kwargs["cache_dir"] = self.config.cache_dir
            # In auto mode, do not download a multi-GB instruct model just to
            # boot the pipeline; fall back to spaCy if it is not cached.
            if self.mode == "auto":
                model_kwargs["local_files_only"] = True
            self._slm_pipeline = hf_pipeline(
                task=task,
                model=self.config.model_name,
                device_map="auto" if self.config.device != "cpu" else None,
                model_kwargs=model_kwargs,
            )
            self._slm_task = task
            logger.info(
                "Loaded SLM claim extraction pipeline: %s (task=%s)",
                self.config.model_name, task,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load SLM pipeline '%s': %s", self.config.model_name, exc
            )
            self._slm_pipeline = False  # sentinel: load attempted and failed

    def _load_spacy(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy

            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found; attempting download."
                )
                from spacy.cli import download as spacy_download

                spacy_download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model 'en_core_web_sm' for rule-based fallback.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load spaCy backend: %s", exc)
            self._nlp = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, article_text: str) -> list[ExtractedClaim]:
        """
        Extract check-worthy claims from `article_text`.

        Returns a list of `ExtractedClaim`, capped at
        `config.max_claims_per_article`, ordered by extraction rank.
        """
        article_text = self._normalize_whitespace(article_text)
        if not article_text.strip():
            return []

        if self.mode == "slm":
            return self._extract_via_slm(article_text)
        if self.mode == "rule_based":
            return self._extract_via_rules(article_text)

        # auto mode: try SLM first, fall back to rules on any failure
        try:
            claims = self._extract_via_slm(article_text)
            if claims:
                return claims
            logger.info("SLM extraction returned no claims; falling back to rules.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SLM extraction failed (%s); falling back to rule-based.", exc)
        return self._extract_via_rules(article_text)

    # ------------------------------------------------------------------
    # Backend 1: SLM-based extraction
    # ------------------------------------------------------------------
    def _extract_via_slm(self, article_text: str) -> list[ExtractedClaim]:
        self._load_slm()
        if not self._slm_pipeline:
            raise RuntimeError("SLM pipeline is unavailable.")

        prompt = _CLAIM_EXTRACTION_PROMPT_TEMPLATE.format(
            max_claim_tokens=self.config.max_claim_tokens,
            min_claim_tokens=self.config.min_claim_tokens,
            max_claims=self.config.max_claims_per_article,
            article_text=article_text[:8000],
        )

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = max(self.config.temperature, 1e-5)

        if self._slm_task == "text2text-generation":
            output = self._slm_pipeline(prompt, **gen_kwargs)
            raw_text = output[0]["generated_text"]
        else:
            output = self._slm_pipeline(
                prompt,
                return_full_text=False,
                **gen_kwargs,
            )
            raw_text = output[0]["generated_text"]

        claims_list = self._parse_json_claims(raw_text)
        claims_list = self._filter_claims_text(claims_list)
        claims_list = claims_list[: self.config.max_claims_per_article]

        return [
            ExtractedClaim(
                claim_id=idx + 1,
                text=text,
                extraction_method="slm",
                checkworthiness_score=None,
            )
            for idx, text in enumerate(claims_list)
        ]

    @staticmethod
    def _parse_json_claims(raw_text: str) -> list[str]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        json_candidate = match.group(0) if match else cleaned

        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            logger.warning("Failed to parse SLM output as JSON array; got: %.200s", raw_text)

        # Best-effort line-based fallback if JSON parsing fails.
        lines = [
            re.sub(r'^\s*[-*\d.)\]"\']+\s*', "", ln).strip().strip('",')
            for ln in cleaned.splitlines()
        ]
        return [ln for ln in lines if ln]

    # ------------------------------------------------------------------
    # Backend 2: Rule-based spaCy fallback
    # ------------------------------------------------------------------
    def _extract_via_rules(self, article_text: str) -> list[ExtractedClaim]:
        self._load_spacy()
        if not self._nlp:
            raise RuntimeError("spaCy backend is unavailable for rule-based extraction.")

        doc = self._nlp(article_text)
        candidates: list[tuple[str, tuple[int, int], float]] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            token_count = sum(1 for t in sent if not t.is_space)
            if token_count < self.config.min_claim_tokens or \
                    token_count > self.config.max_claim_tokens:
                continue

            if self.config.remove_questions and sent_text.rstrip().endswith("?"):
                continue

            if self.config.remove_opinions and any(
                marker in sent_text.lower() for marker in _OPINION_MARKERS
            ):
                continue

            root = sent.root
            is_declarative = root.pos_ in ("VERB", "AUX")
            if not is_declarative:
                continue

            ent_labels = {ent.label_ for ent in sent.ents}
            has_subject_entity = bool(ent_labels & _SPACY_ENT_LABELS_SUBJECT)
            has_numeric_entity = bool(ent_labels & _SPACY_ENT_LABELS_NUMERIC)

            if not (has_subject_entity and has_numeric_entity):
                # Relax to: at least one named entity plus a digit token,
                # to avoid discarding valid claims lacking a NUMERIC label.
                has_any_entity = len(sent.ents) > 0
                has_digit = any(ch.isdigit() for ch in sent_text)
                if not (has_any_entity and has_digit):
                    continue

            score = self._checkworthiness_score(sent, ent_labels)
            if score < self.config.checkworthiness_threshold:
                continue

            candidates.append((sent_text, (sent.start_char, sent.end_char), score))

        candidates.sort(key=lambda c: c[2], reverse=True)
        top_candidates = candidates[: self.config.max_claims_per_article]

        return [
            ExtractedClaim(
                claim_id=idx + 1,
                text=text,
                source_sentence_span=span,
                extraction_method="rule_based",
                checkworthiness_score=round(score, 4),
            )
            for idx, (text, span, score) in enumerate(top_candidates)
        ]

    @staticmethod
    def _checkworthiness_score(sent, ent_labels: set[str]) -> float:
        """
        Heuristic checkworthiness score in [0, 1] combining entity density,
        numeric/temporal presence, and sentence length normalization.
        """
        n_tokens = max(sum(1 for t in sent if not t.is_space), 1)
        n_ents = len(sent.ents)
        entity_density = min(n_ents / max(n_tokens / 8, 1), 1.0)

        numeric_bonus = 0.25 if (ent_labels & _SPACY_ENT_LABELS_NUMERIC) else 0.0
        subject_bonus = 0.25 if (ent_labels & _SPACY_ENT_LABELS_SUBJECT) else 0.0

        length_penalty = 1.0
        if n_tokens < 6:
            length_penalty = 0.7
        elif n_tokens > 45:
            length_penalty = 0.85

        score = (0.5 * entity_density + numeric_bonus + subject_bonus) * length_penalty
        return max(0.0, min(score, 1.0))

    # ------------------------------------------------------------------
    # Shared filtering / utilities
    # ------------------------------------------------------------------
    def _filter_claims_text(self, claims: list[str]) -> list[str]:
        filtered = []
        seen = set()
        for text in claims:
            text = text.strip().strip('"').strip()
            if not text:
                continue
            token_count = len(text.split())
            if token_count < self.config.min_claim_tokens:
                continue
            if token_count > self.config.max_claim_tokens:
                continue
            if self.config.remove_questions and text.rstrip().endswith("?"):
                continue
            if self.config.remove_opinions and any(
                marker in text.lower() for marker in _OPINION_MARKERS
            ):
                continue
            dedup_key = text.lower()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            filtered.append(text)
        return filtered

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------
# Standalone verification block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    _SAMPLE_ARTICLE = """
    A new study published this week claims that drinking hot lemon water
    completely destroys respiratory viruses within 30 seconds of contact.
    The report, allegedly funded by 12 independent research groups in 2021,
    says over 90 percent of patients recovered within two days.
    Health officials in California have not confirmed these findings.
    Some readers might think this sounds too good to be true.
    Is this claim actually backed by peer-reviewed evidence?
    Experts obviously find this suspicious given the lack of clinical trials.
    The World Health Organization stated in March 2020 that no citrus-based
    remedy has been shown to neutralize airborne viral pathogens.
    """

    repo_root = Path(__file__).resolve().parents[2]
    config_yaml_path = repo_root / "config" / "config.yaml"

    if config_yaml_path.exists():
        extractor = ClaimExtractor.from_config_yaml(config_yaml_path, mode="auto")
    else:
        logger.warning(
            "config.yaml not found at %s; using default ClaimExtractionConfig.",
            config_yaml_path,
        )
        extractor = ClaimExtractor(config=ClaimExtractionConfig(), mode="auto")

    extracted_claims = extractor.extract(_SAMPLE_ARTICLE)

    print(f"\nExtracted {len(extracted_claims)} claim(s):\n")
    for claim in extracted_claims:
        print(json.dumps(claim.to_dict(), indent=2))
