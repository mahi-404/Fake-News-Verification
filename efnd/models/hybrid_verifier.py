"""
efnd.models.hybrid_verifier
==============================

Modules 5 & 6: Gated Cross-Attention Fusion + Classification Head with
Uncertainty Gate.

Shape legend used throughout (in comments):
    B  = batch size
    L  = article token sequence length
    C  = number of claims per article (padded/truncated to a fixed max)
    F  = raw per-claim feature dimension (output of
         `efnd.inference.nli_engine.ClaimFeatureBundle.to_array()`, default 9)
    D  = shared model / fusion hidden dimension (default 768, matches
         DeBERTa-v3-base hidden size)
    H  = number of cross-attention heads
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

CANONICAL_LABEL_ORDER = ["entailment", "contradiction", "neutral"]


@dataclass
class HybridVerifierConfig:
    # Macro-context encoder (Module 1)
    encoder_model_name: str = "microsoft/deberta-v3-base"
    encoder_max_seq_length: int = 512
    encoder_hidden_dim: int = 768
    encoder_dropout: float = 0.1
    encoder_pooling: str = "cls"          # "cls" | "mean"
    load_pretrained_encoder: bool = True

    # Claim feature input (Module 4 output)
    claim_feature_dim: int = 9

    # Gated cross-attention fusion (Module 5)
    article_embedding_dim: int = 768
    claim_embedding_dim: int = 768
    fused_dim: int = 768
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Classification head (Module 6)
    num_labels: int = 2
    hidden_layers: list[int] = field(default_factory=lambda: [768, 256])
    activation: str = "gelu"
    classifier_dropout: float = 0.2
    label_smoothing: float = 0.05

    # Uncertainty gate (Module 6)
    uncertainty_enabled: bool = True
    evidence_sufficiency_threshold: float = 0.45   # tau
    uncertainty_aggregation: str = "min"           # "min" | "mean"
    abstain_label: str = "UNCERTAIN"

    # Ablation flags (Section 5.2 ablation studies; toggled from train.py CLI)
    ablate_retrieval: bool = False   # zero out claim-evidence feature tensors
    ablate_gating: bool = False      # replace gated cross-attention with naive concat

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "HybridVerifierConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        macro = cfg.get("macro_encoder", {})
        fusion = cfg.get("fusion_network", {})
        classifier = cfg.get("classifier", {})
        uncertainty = cfg.get("uncertainty_gate", {})

        return cls(
            encoder_model_name=macro.get("model_name", "microsoft/deberta-v3-base"),
            encoder_max_seq_length=macro.get("max_seq_length", 512),
            encoder_hidden_dim=macro.get("hidden_dim", 768),
            encoder_dropout=macro.get("dropout", 0.1),
            encoder_pooling=macro.get("pooling_strategy", "cls"),
            article_embedding_dim=fusion.get("article_embedding_dim", 768),
            claim_embedding_dim=fusion.get("claim_embedding_dim", 768),
            fused_dim=fusion.get("fused_dim", 768),
            num_attention_heads=fusion.get("num_attention_heads", 8),
            attention_dropout=fusion.get("attention_dropout", 0.1),
            num_labels=classifier.get("num_labels", 2),
            hidden_layers=classifier.get("hidden_layers", [768, 256]),
            activation=classifier.get("activation", "gelu"),
            classifier_dropout=classifier.get("dropout", 0.2),
            label_smoothing=classifier.get("label_smoothing", 0.05),
            uncertainty_enabled=uncertainty.get("enabled", True),
            evidence_sufficiency_threshold=uncertainty.get(
                "evidence_sufficiency_threshold", 0.45
            ),
            abstain_label=uncertainty.get("abstain_label", "UNCERTAIN"),
        )


def _get_activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation '{name}'.")
    return activations[name.lower()]


# ============================================================================
# Module 5: Gated Cross-Attention Fusion
# ============================================================================
class GatedCrossAttentionFusion(nn.Module):
    """
    Cross-attends claim-evidence representations against the macro article
    representation, then interpolates between the raw article embedding and
    the cross-attended (evidence-aware) representation via a learned scalar
    gate:

        Q = W_q h_art                              # (B, 1, D)
        K = W_k H_claims,  V = W_v H_claims         # (B, C, D)
        h_cross = MultiHeadAttention(Q, K, V)       # (B, 1, D) -> (B, D)
        lambda  = sigmoid(W_g [h_art ; h_cross] + b_g)   # (B, 1)
        h_fused = lambda * h_art + (1 - lambda) * h_cross    # (B, D)

    `lambda` close to 1 means the model relies primarily on the article's
    own stylistic/rhetorical signal; close to 0 means it defers to the
    claim-evidence verification signal.
    """

    def __init__(self, config: HybridVerifierConfig) -> None:
        super().__init__()
        self.config = config
        D = config.fused_dim
        H = config.num_attention_heads
        if D % H != 0:
            raise ValueError(f"fused_dim ({D}) must be divisible by num_attention_heads ({H}).")

        self.head_dim = D // H
        self.num_heads = H

        # Input projections: article -> queries, claims -> keys/values.
        self.article_in_proj = nn.Linear(config.article_embedding_dim, D)
        self.claim_in_proj = nn.Linear(config.claim_embedding_dim, D)

        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.layer_norm = nn.LayerNorm(D)

        # Gating network: takes concat[h_art, h_cross] -> scalar in [0, 1].
        self.gate_proj = nn.Linear(D * 2, 1)

    def forward(
        self,
        h_art: torch.Tensor,          # (B, article_embedding_dim)
        h_claims: torch.Tensor,       # (B, C, claim_embedding_dim)
        claim_mask: Optional[torch.Tensor] = None,  # (B, C), 1=valid, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_fused : torch.Tensor, shape (B, D)
            Gated fusion of article context and evidence cross-attention.
        gate_lambda : torch.Tensor, shape (B, 1)
            The learned interpolation scalar (for logging/interpretability).
        """
        B = h_art.size(0)
        C = h_claims.size(1)
        D = self.config.fused_dim

        h_art_proj = self.article_in_proj(h_art)          # (B, D)
        h_claims_proj = self.claim_in_proj(h_claims)       # (B, C, D)

        q = self.W_q(h_art_proj).unsqueeze(1)              # (B, 1, D)
        k = self.W_k(h_claims_proj)                        # (B, C, D)
        v = self.W_v(h_claims_proj)                        # (B, C, D)

        # Reshape into heads: (B, num_heads, seq, head_dim)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, hd)
        k = k.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, C, hd)
        v = v.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, C, hd)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn_scores: (B, H, 1, C)

        if claim_mask is not None:
            # claim_mask: (B, C) -> (B, 1, 1, C), masked positions get -inf.
            mask = claim_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)      # (B, H, 1, C)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # guard all-masked rows
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)        # (B, H, 1, hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, D)  # (B, 1, D)
        attn_output = attn_output.squeeze(1)                # (B, D)

        h_cross = self.out_proj(attn_output)                # (B, D)
        h_cross = self.layer_norm(h_cross + h_art_proj)      # residual + LN, (B, D)

        gate_input = torch.cat([h_art_proj, h_cross], dim=-1)   # (B, 2D)
        gate_lambda = torch.sigmoid(self.gate_proj(gate_input))  # (B, 1)

        h_fused = gate_lambda * h_art_proj + (1.0 - gate_lambda) * h_cross  # (B, D)

        return h_fused, gate_lambda


class NaiveConcatFusion(nn.Module):
    """
    Ablation variant of `GatedCrossAttentionFusion` used when
    `config.ablate_gating=True`: replaces learned cross-attention + gating
    with masked mean-pooling over claims followed by simple concatenation
    and a linear projection back to `fused_dim`. No attention, no gate.
    """

    def __init__(self, config: HybridVerifierConfig) -> None:
        super().__init__()
        D = config.fused_dim
        self.article_in_proj = nn.Linear(config.article_embedding_dim, D)
        self.claim_in_proj = nn.Linear(config.claim_embedding_dim, D)
        self.concat_proj = nn.Linear(D * 2, D)
        self.layer_norm = nn.LayerNorm(D)

    def forward(
        self,
        h_art: torch.Tensor,                          # (B, article_embedding_dim)
        h_claims: torch.Tensor,                        # (B, C, claim_embedding_dim)
        claim_mask: Optional[torch.Tensor] = None,     # (B, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = h_art.size(0)
        h_art_proj = self.article_in_proj(h_art)                 # (B, D)
        h_claims_proj = self.claim_in_proj(h_claims)              # (B, C, D)

        if claim_mask is not None:
            mask = claim_mask.unsqueeze(-1).float()               # (B, C, 1)
            summed = (h_claims_proj * mask).sum(dim=1)             # (B, D)
            counts = mask.sum(dim=1).clamp(min=1e-9)               # (B, 1)
            h_claims_pooled = summed / counts                      # (B, D)
        else:
            h_claims_pooled = h_claims_proj.mean(dim=1)             # (B, D)

        concat = torch.cat([h_art_proj, h_claims_pooled], dim=-1)  # (B, 2D)
        h_fused = self.layer_norm(self.concat_proj(concat))         # (B, D)

        # No learned gate in this ablation; report a fixed neutral value
        # (0.5) so downstream logging/interpretability code keeps working.
        gate_lambda = torch.full((B, 1), 0.5, device=h_art.device, dtype=h_fused.dtype)

        return h_fused, gate_lambda


# ============================================================================
# Module 6: Classification Head with Uncertainty Gate
# ============================================================================
@dataclass
class UncertaintyResult:
    evidence_sufficiency_score: torch.Tensor   # (B,)
    abstain_mask: torch.Tensor                 # (B,) bool, True => override to UNCERTAIN
    per_claim_sufficiency: torch.Tensor        # (B, C)


class EvidenceAwareClassifier(nn.Module):
    """
    End-to-end classifier: encodes the raw article with DeBERTa-v3, fuses it
    with pre-extracted per-claim NLI/retrieval feature tensors via
    `GatedCrossAttentionFusion`, and projects to Real/Fake logits. Also
    exposes `calculate_uncertainty` to override confident predictions with
    an `UNCERTAIN` abstention label when evidence is weak or ambiguous.
    """

    def __init__(self, config: HybridVerifierConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder, self.encoder_config = self._build_encoder(config)
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)

        # Projects raw per-claim feature vectors (F-dim) up to the shared
        # claim_embedding_dim consumed by the fusion module's K/V projections.
        self.claim_feature_proj = nn.Sequential(
            nn.Linear(config.claim_feature_dim, config.claim_embedding_dim),
            nn.LayerNorm(config.claim_embedding_dim),
            _get_activation(config.activation),
        )

        self.fusion = NaiveConcatFusion(config) if config.ablate_gating else GatedCrossAttentionFusion(config)

        # MLP classification head over h_fused -> num_labels logits.
        layers: list[nn.Module] = []
        in_dim = config.fused_dim
        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(config.activation))
            layers.append(nn.Dropout(config.classifier_dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.num_labels))
        self.classifier_head = nn.Sequential(*layers)

    @staticmethod
    def _build_encoder(config: HybridVerifierConfig):
        """
        Loads the DeBERTa-v3 macro-context encoder. Falls back to a
        randomly-initialized encoder from the checkpoint's architecture
        config (no network download) if `load_pretrained_encoder=False`
        or if the pretrained weights cannot be fetched (e.g. offline
        environment), so the module remains importable/testable without
        network access.
        """
        from transformers import AutoConfig, AutoModel

        hf_config = AutoConfig.from_pretrained(config.encoder_model_name)
        if config.load_pretrained_encoder:
            try:
                encoder = AutoModel.from_pretrained(config.encoder_model_name)
                return encoder, hf_config
            except Exception:
                pass  # fall through to random init below
        encoder = AutoModel.from_config(hf_config)
        return encoder, hf_config

    def encode_article(
        self,
        input_ids: torch.Tensor,          # (B, L)
        attention_mask: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Returns pooled macro-context article embedding h_art: (B, encoder_hidden_dim)."""
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state   # (B, L, encoder_hidden_dim)

        if self.config.encoder_pooling == "cls":
            pooled = last_hidden_state[:, 0, :]                  # (B, encoder_hidden_dim)
        elif self.config.encoder_pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
            summed = (last_hidden_state * mask).sum(dim=1)       # (B, encoder_hidden_dim)
            counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
            pooled = summed / counts
        else:
            raise ValueError(f"Unsupported pooling strategy '{self.config.encoder_pooling}'.")

        return self.encoder_dropout(pooled)                      # (B, encoder_hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,                 # (B, L)
        attention_mask: torch.Tensor,             # (B, L)
        claim_features: torch.Tensor,             # (B, C, F)
        claim_mask: Optional[torch.Tensor] = None,  # (B, C)
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
            logits      : (B, num_labels)
            probs       : (B, num_labels)
            h_art       : (B, article_embedding_dim)
            h_fused     : (B, fused_dim)
            gate_lambda : (B, 1)
        """
        h_art = self.encode_article(input_ids, attention_mask)        # (B, D_art)

        if self.config.ablate_retrieval:
            claim_features = torch.zeros_like(claim_features)         # (B, C, F), zeroed evidence

        h_claims = self.claim_feature_proj(claim_features)            # (B, C, D_claim)

        h_fused, gate_lambda = self.fusion(h_art, h_claims, claim_mask)  # (B, D), (B, 1)

        logits = self.classifier_head(h_fused)                        # (B, num_labels)
        probs = F.softmax(logits, dim=-1)                             # (B, num_labels)

        return {
            "logits": logits,
            "probs": probs,
            "h_art": h_art,
            "h_fused": h_fused,
            "gate_lambda": gate_lambda,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,   # (B, num_labels)
        labels: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits, labels, label_smoothing=self.config.label_smoothing
        )

    # ------------------------------------------------------------------
    # Uncertainty Gate
    # ------------------------------------------------------------------
    def calculate_uncertainty(
        self,
        retrieval_scores: torch.Tensor,   # (B, C) - max/top retrieval score per claim
        nli_probs: torch.Tensor,          # (B, C, 3) - [ent, con, neu] per claim
        tau: Optional[float] = None,
        claim_mask: Optional[torch.Tensor] = None,  # (B, C)
    ) -> UncertaintyResult:
        """
        Computes the evidence sufficiency score per claim:

            S_ev = max_j(RetrievalScore_j) * (1 - H(p_NLI) / log(3))

        and aggregates across a claim's evidence (already reduced to one
        row of `nli_probs` / one scalar of `retrieval_scores` per claim, as
        produced by `compute_claim_features`), then aggregates across all
        claims in the article (`config.uncertainty_aggregation`: "min" is
        the conservative default -- a single poorly-evidenced claim is
        enough to trigger abstention). Predictions are overridden to
        `UNCERTAIN` wherever the aggregated score falls below `tau`.

        Returns
        -------
        UncertaintyResult
            evidence_sufficiency_score : (B,)  aggregated article-level S_ev
            abstain_mask               : (B,)  bool, True -> override to UNCERTAIN
            per_claim_sufficiency      : (B, C) per-claim S_ev values
        """
        tau = tau if tau is not None else self.config.evidence_sufficiency_threshold

        eps = 1e-12
        probs_clamped = nli_probs.clamp(min=eps)                        # (B, C, 3)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)    # (B, C)
        normalized_entropy = entropy / math.log(3)                      # (B, C)

        per_claim_sufficiency = retrieval_scores * (1.0 - normalized_entropy)  # (B, C)

        if claim_mask is not None:
            mask_bool = claim_mask.to(dtype=torch.bool)                 # (B, C)
            neutral_fill = torch.ones_like(per_claim_sufficiency)       # treat pad as "sufficient"
            per_claim_sufficiency = torch.where(mask_bool, per_claim_sufficiency, neutral_fill)

        if self.config.uncertainty_aggregation == "min":
            article_sufficiency, _ = per_claim_sufficiency.min(dim=-1)  # (B,)
        elif self.config.uncertainty_aggregation == "mean":
            article_sufficiency = per_claim_sufficiency.mean(dim=-1)    # (B,)
        else:
            raise ValueError(
                f"Unsupported uncertainty_aggregation "
                f"'{self.config.uncertainty_aggregation}'."
            )

        abstain_mask = article_sufficiency < tau                        # (B,) bool

        return UncertaintyResult(
            evidence_sufficiency_score=article_sufficiency,
            abstain_mask=abstain_mask,
            per_claim_sufficiency=per_claim_sufficiency,
        )

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        claim_features: torch.Tensor,             # (B, C, F)
        retrieval_scores: torch.Tensor,           # (B, C)
        nli_probs: torch.Tensor,                  # (B, C, 3)
        claim_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        label_names: tuple[str, str] = ("LIKELY_REAL", "LIKELY_FAKE"),
    ) -> list[dict]:
        """
        Full forward pass + uncertainty override, returning one result dict
        per article with a human-readable final verdict.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, claim_features, claim_mask)
            uncertainty = self.calculate_uncertainty(
                retrieval_scores, nli_probs, tau=tau, claim_mask=claim_mask
            )

        probs = outputs["probs"]                     # (B, num_labels)
        pred_indices = probs.argmax(dim=-1)           # (B,)
        confidences = probs.gather(-1, pred_indices.unsqueeze(-1)).squeeze(-1)  # (B,)

        results = []
        B = probs.size(0)
        for i in range(B):
            if self.config.uncertainty_enabled and bool(uncertainty.abstain_mask[i]):
                verdict = self.config.abstain_label
            else:
                verdict = label_names[int(pred_indices[i])]

            results.append(
                {
                    "prediction": verdict,
                    "confidence": float(confidences[i]),
                    "gate_lambda": float(outputs["gate_lambda"][i]),
                    "evidence_sufficiency_score": float(
                        uncertainty.evidence_sufficiency_score[i]
                    ),
                    "abstain_triggered": bool(uncertainty.abstain_mask[i]),
                    "raw_probs": probs[i].tolist(),
                }
            )
        return results


# ----------------------------------------------------------------------
# Standalone verification block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    log = logging.getLogger(__name__)

    repo_root = Path(__file__).resolve().parents[2]
    config_yaml_path = repo_root / "config" / "config.yaml"

    if config_yaml_path.exists():
        model_config = HybridVerifierConfig.from_yaml(config_yaml_path)
    else:
        log.warning("config.yaml not found at %s; using default HybridVerifierConfig.",
                    config_yaml_path)
        model_config = HybridVerifierConfig()

    # Force a network-free random-init encoder for this offline smoke test.
    model_config.load_pretrained_encoder = False

    torch.manual_seed(42)
    model = EvidenceAwareClassifier(model_config)
    model.eval()

    B, L, C, F_DIM = 2, 64, 4, model_config.claim_feature_dim

    dummy_input_ids = torch.randint(low=0, high=1000, size=(B, L))
    dummy_attention_mask = torch.ones(B, L, dtype=torch.long)
    dummy_claim_features = torch.randn(B, C, F_DIM)
    dummy_claim_mask = torch.ones(B, C, dtype=torch.long)
    dummy_claim_mask[1, 2:] = 0  # simulate article 2 having only 2 real claims

    dummy_retrieval_scores = torch.rand(B, C)
    dummy_nli_probs = F.softmax(torch.randn(B, C, 3), dim=-1)

    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            claim_features=dummy_claim_features,
            claim_mask=dummy_claim_mask,
        )

    print("logits shape:      ", tuple(outputs["logits"].shape))
    print("probs shape:       ", tuple(outputs["probs"].shape))
    print("h_art shape:       ", tuple(outputs["h_art"].shape))
    print("h_fused shape:     ", tuple(outputs["h_fused"].shape))
    print("gate_lambda values:", outputs["gate_lambda"].squeeze(-1).tolist())

    results = model.predict_with_uncertainty(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        claim_features=dummy_claim_features,
        retrieval_scores=dummy_retrieval_scores,
        nli_probs=dummy_nli_probs,
        claim_mask=dummy_claim_mask,
    )

    print("\nFinal verdicts with uncertainty gating:\n")
    for i, res in enumerate(results):
        print(f"Article {i}: {res}")
