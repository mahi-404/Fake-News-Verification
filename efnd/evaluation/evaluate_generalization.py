"""
efnd.evaluation.evaluate_generalization
==========================================

Section 5.3: Cross-Domain Generalization Evaluation.

Loads a saved model checkpoint (as produced by `efnd.training.train.Trainer`)
and runs zero-shot evaluation on an external, out-of-domain (OOD) validation
set (e.g. train on MultiFC/PolitiFact/GossipCop, zero-shot test on
LIAR-New / COVID-19 misinformation corpus).

Reports: Macro-F1, Balanced Accuracy, ROC-AUC, Brier Score, and ECE, plus
uncertainty-gate diagnostics (abstain rate, mean evidence sufficiency).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from efnd.evaluation.metrics import compute_classification_metrics
from efnd.models.hybrid_verifier import EvidenceAwareClassifier, HybridVerifierConfig
from efnd.training.train import EFNDDataset, collate_fn

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    override_config: Optional[HybridVerifierConfig] = None,
) -> EvidenceAwareClassifier:
    """
    Rebuilds `EvidenceAwareClassifier` from a checkpoint's saved
    `model_config` dict (unless `override_config` is given) and loads its
    trained weights. The encoder is *not* re-downloaded from pretrained
    weights during this rebuild (`load_pretrained_encoder=False`), since
    `load_state_dict` overwrites all encoder parameters with the trained
    (or checkpointed) values regardless.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if override_config is not None:
        model_config = override_config
    else:
        saved_cfg_dict = checkpoint.get("model_config")
        if saved_cfg_dict is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' has no 'model_config'; "
                f"pass --config to reconstruct the model architecture."
            )
        model_config = HybridVerifierConfig(**saved_cfg_dict)

    model_config.load_pretrained_encoder = False
    model = EvidenceAwareClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint '%s' (epoch=%s, val_macro_f1=%.4f at save time).",
        checkpoint_path, checkpoint.get("epoch"), checkpoint.get("val_macro_f1", float("nan")),
    )
    return model


@torch.no_grad()
def run_inference(
    model: EvidenceAwareClassifier,
    data_loader: DataLoader,
    device: torch.device,
    uncertainty_tau: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """
    Runs a full forward pass over `data_loader`, returning stacked numpy
    arrays: labels, preds, pos_probs (P(LIKELY_FAKE)), gate_lambda,
    evidence_sufficiency_score, abstain_mask.
    """
    all_labels, all_preds, all_pos_probs = [], [], []
    all_gate_lambda, all_sufficiency, all_abstain = [], [], []

    for batch in data_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            claim_features=batch["claim_features"],
            claim_mask=batch["claim_mask"],
        )
        uncertainty = model.calculate_uncertainty(
            retrieval_scores=batch["retrieval_scores"],
            nli_probs=batch["nli_probs"],
            tau=uncertainty_tau,
            claim_mask=batch["claim_mask"],
        )

        probs = outputs["probs"]                      # (B, num_labels)
        preds = probs.argmax(dim=-1)                   # (B,)

        all_labels.append(batch["labels"].cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        if probs.size(-1) == 2:
            all_pos_probs.append(probs[:, 1].cpu().numpy())
        all_gate_lambda.append(outputs["gate_lambda"].squeeze(-1).cpu().numpy())
        all_sufficiency.append(uncertainty.evidence_sufficiency_score.cpu().numpy())
        all_abstain.append(uncertainty.abstain_mask.cpu().numpy())

    return {
        "labels": np.concatenate(all_labels),
        "preds": np.concatenate(all_preds),
        "pos_probs": np.concatenate(all_pos_probs) if all_pos_probs else None,
        "gate_lambda": np.concatenate(all_gate_lambda),
        "evidence_sufficiency_score": np.concatenate(all_sufficiency),
        "abstain_mask": np.concatenate(all_abstain),
    }


def evaluate_generalization(
    checkpoint_path: str | Path,
    ood_data_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 2,
    device_str: str = "cuda",
    ece_num_bins: int = 15,
    uncertainty_tau: Optional[float] = None,
) -> dict:
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str == "cuda" and device.type == "cpu":
        logger.warning("Requested device='cuda' but CUDA unavailable; using CPU.")

    model = load_model_from_checkpoint(checkpoint_path, device)

    ood_dataset = EFNDDataset(ood_data_path)
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    results = run_inference(model, ood_loader, device, uncertainty_tau=uncertainty_tau)

    metrics = compute_classification_metrics(
        labels=results["labels"],
        preds=results["preds"],
        pos_probs=results["pos_probs"],
        ece_num_bins=ece_num_bins,
    )

    abstain_rate = float(results["abstain_mask"].mean())
    mean_sufficiency = float(results["evidence_sufficiency_score"].mean())
    mean_gate_lambda = float(results["gate_lambda"].mean())

    report = {
        "checkpoint": str(checkpoint_path),
        "ood_dataset": str(ood_data_path),
        "n_examples": metrics.n_examples,
        "metrics": metrics.to_dict(),
        "uncertainty_gate": {
            "abstain_rate": abstain_rate,
            "mean_evidence_sufficiency_score": mean_sufficiency,
        },
        "gated_fusion": {
            "mean_gate_lambda": mean_gate_lambda,
        },
    }

    logger.info(
        "OOD eval | n=%d | macro_f1=%.4f | balanced_acc=%.4f | roc_auc=%s | "
        "brier=%s | ece=%s | abstain_rate=%.4f",
        report["n_examples"], metrics.macro_f1, metrics.balanced_accuracy,
        f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A",
        f"{metrics.brier_score:.4f}" if metrics.brier_score is not None else "N/A",
        f"{metrics.ece:.4f}" if metrics.ece is not None else "N/A",
        abstain_rate,
    )

    return report


# ============================================================================
# CLI entry point
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot out-of-domain generalization evaluation "
                    "(Section 5.3) for a trained EFND checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--ood_data", type=str, required=True, help="Path to preprocessed OOD .pt dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ece_num_bins", type=int, default=15)
    parser.add_argument(
        "--uncertainty_tau", type=float, default=None,
        help="Override the checkpoint's evidence_sufficiency_threshold for this eval run.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write the JSON report")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    report = evaluate_generalization(
        checkpoint_path=args.checkpoint,
        ood_data_path=args.ood_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        ece_num_bins=args.ece_num_bins,
        uncertainty_tau=args.uncertainty_tau,
    )

    print(json.dumps(report, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote generalization report to %s", output_path)


if __name__ == "__main__":
    main()
