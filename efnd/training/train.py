"""
efnd.training.train
======================

Training & evaluation entry point for the Evidence-Aware Hybrid Transformer
(Section 5 of the specification).

Features
--------
- Mixed-precision training via `torch.cuda.amp.autocast` + `GradScaler`.
- Gradient accumulation.
- AdamW with linear warmup + cosine decay (`transformers.get_cosine_schedule_with_warmup`).
- Early stopping monitored on validation Macro-F1.
- Ablation flags (CLI or config): `--ablate_retrieval`, `--ablate_gating`.
- Clean checkpointing (best + rotating recent) and JSONL training logs.

Expected data format
---------------------
This script consumes a *preprocessed* tensor cache produced upstream by the
claim extraction (Module 2), hybrid retrieval (Module 3), and NLI (Module 4)
components. Each example is a `.pt` file (or an entry in a single packed
`.pt` list) containing:

    {
        "input_ids":        LongTensor (L,)
        "attention_mask":   LongTensor (L,)
        "claim_features":   FloatTensor (C, F)   # per-claim aggregated features
        "claim_mask":       LongTensor (C,)      # 1 = real claim, 0 = pad
        "retrieval_scores": FloatTensor (C,)     # top/aggregated retrieval score per claim
        "nli_probs":        FloatTensor (C, 3)   # [entailment, contradiction, neutral]
        "label":            LongTensor ()        # 0 = LIKELY_REAL, 1 = LIKELY_FAKE
    }

`C` (max claims per article) and `L` (max sequence length) must already be
padded consistently by the upstream preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import yaml

from efnd.models.hybrid_verifier import EvidenceAwareClassifier, HybridVerifierConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Config
# ============================================================================
@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_epochs: int = 10
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 3
    save_total_limit: int = 3
    logging_steps: int = 50
    mixed_precision: str = "bf16"   # "no" | "fp16" | "bf16"
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        t = cfg.get("training", {})
        es = t.get("early_stopping", {})
        paths = cfg.get("paths", {})
        return cls(
            batch_size=t.get("batch_size", 16),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 2),
            num_epochs=t.get("num_epochs", 10),
            learning_rate=float(t.get("learning_rate", 2.0e-5)),
            weight_decay=t.get("weight_decay", 0.01),
            warmup_ratio=t.get("warmup_ratio", 0.06),
            max_grad_norm=t.get("max_grad_norm", 1.0),
            early_stopping_enabled=es.get("enabled", True),
            early_stopping_patience=es.get("patience", 3),
            save_total_limit=t.get("save_total_limit", 3),
            logging_steps=t.get("logging_steps", 50),
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            device=cfg.get("device", "cuda"),
            seed=cfg.get("seed", 42),
            checkpoint_dir=paths.get("checkpoint_dir", "checkpoints"),
            log_dir=paths.get("log_dir", "logs"),
        )


# ============================================================================
# Dataset
# ============================================================================
class EFNDDataset(Dataset):
    """
    Loads a list of preprocessed example dicts (see module docstring) from a
    single packed `.pt` file created by the upstream data pipeline.
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        self.examples: list[dict] = torch.load(self.data_path, map_location="cpu")
        if not isinstance(self.examples, list) or len(self.examples) == 0:
            raise ValueError(
                f"Expected a non-empty list of example dicts in {self.data_path}."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "claim_features": torch.stack([b["claim_features"] for b in batch], dim=0),
        "claim_mask": torch.stack([b["claim_mask"] for b in batch], dim=0),
        "retrieval_scores": torch.stack([b["retrieval_scores"] for b in batch], dim=0),
        "nli_probs": torch.stack([b["nli_probs"] for b in batch], dim=0),
        "labels": torch.stack([b["label"] for b in batch], dim=0),
    }


# ============================================================================
# Trainer
# ============================================================================
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    auc_roc: Optional[float]


class Trainer:
    def __init__(
        self,
        model: EvidenceAwareClassifier,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)

        self.checkpoint_dir = self.output_dir / self.config.checkpoint_dir
        self.log_dir = self.output_dir / self.config.log_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        total_update_steps = (
            len(self.train_loader) // self.config.gradient_accumulation_steps
        ) * self.config.num_epochs
        total_update_steps = max(total_update_steps, 1)
        warmup_steps = int(total_update_steps * self.config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

        self.amp_enabled = (
            self.device.type == "cuda" and self.config.mixed_precision in ("fp16", "bf16")
        )
        self.amp_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_enabled and self.amp_dtype == torch.float16))

        self.global_step = 0
        self.best_macro_f1 = -1.0
        self.epochs_without_improvement = 0
        self.saved_checkpoints: list[Path] = []

        self.log_file = self.log_dir / f"train_log_{int(time.time())}.jsonl"

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested device='cuda' but CUDA unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _build_optimizer(self) -> torch.optim.AdamW:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )

    def _log(self, record: dict) -> None:
        record["timestamp"] = time.time()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------
    def train(self) -> None:
        logger.info(
            "Starting training: %d epochs, device=%s, amp=%s(%s)",
            self.config.num_epochs, self.device, self.amp_enabled, self.config.mixed_precision,
        )
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(epoch, is_train=True)
            val_metrics = self._run_epoch(epoch, is_train=False)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f val_acc=%.4f "
                "val_macro_f1=%.4f val_auc=%s",
                epoch, train_metrics.loss, val_metrics.loss, val_metrics.accuracy,
                val_metrics.macro_f1,
                f"{val_metrics.auc_roc:.4f}" if val_metrics.auc_roc is not None else "N/A",
            )
            self._log(
                {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "train_loss": train_metrics.loss,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_macro_f1": val_metrics.macro_f1,
                    "val_auc_roc": val_metrics.auc_roc,
                }
            )

            improved = val_metrics.macro_f1 > self.best_macro_f1
            if improved:
                self.best_macro_f1 = val_metrics.macro_f1
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            if (
                self.config.early_stopping_enabled
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d (no val_macro_f1 improvement "
                    "for %d epochs). Best val_macro_f1=%.4f",
                    epoch, self.config.early_stopping_patience, self.best_macro_f1,
                )
                self._log({"event": "early_stop", "epoch": epoch, "best_macro_f1": self.best_macro_f1})
                break

        logger.info("Training complete. Best val_macro_f1=%.4f", self.best_macro_f1)

    def _run_epoch(self, epoch: int, is_train: bool) -> EpochMetrics:
        self.model.train(mode=is_train)
        loader = self.train_loader if is_train else self.val_loader

        total_loss = 0.0
        num_batches = 0
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_pos_probs: list[float] = []

        context = torch.enable_grad() if is_train else torch.no_grad()

        if is_train:
            self.optimizer.zero_grad(set_to_none=True)

        with context:
            for step, batch in enumerate(loader):
                batch = self._batch_to_device(batch)
                labels = batch["labels"]

                autocast_ctx = (
                    torch.cuda.amp.autocast(dtype=self.amp_dtype)
                    if self.amp_enabled
                    else _NullContext()
                )
                with autocast_ctx:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        claim_features=batch["claim_features"],
                        claim_mask=batch["claim_mask"],
                    )
                    loss = self.model.compute_loss(outputs["logits"], labels)

                if is_train:
                    loss_to_backprop = loss / self.config.gradient_accumulation_steps
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss_to_backprop).backward()
                    else:
                        loss_to_backprop.backward()

                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                        if self.global_step % self.config.logging_steps == 0:
                            lr = self.scheduler.get_last_lr()[0]
                            logger.info(
                                "  step=%d loss=%.4f lr=%.3e", self.global_step, loss.item(), lr
                            )
                            self._log(
                                {
                                    "event": "train_step",
                                    "global_step": self.global_step,
                                    "epoch": epoch,
                                    "loss": loss.item(),
                                    "lr": lr,
                                }
                            )

                total_loss += loss.item()
                num_batches += 1

                preds = outputs["probs"].argmax(dim=-1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())
                if outputs["probs"].size(-1) == 2:
                    all_pos_probs.extend(outputs["probs"][:, 1].detach().cpu().tolist())

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0

        auc_roc = None
        if all_pos_probs and len(set(all_labels)) > 1:
            auc_roc = roc_auc_score(all_labels, all_pos_probs)

        return EpochMetrics(loss=avg_loss, accuracy=accuracy, macro_f1=macro_f1, auc_roc=auc_roc)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_macro_f1": metrics.macro_f1,
            "val_accuracy": metrics.accuracy,
            "val_loss": metrics.loss,
            "model_config": self.model.config.__dict__,
        }

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info("Saved new best checkpoint: %s (val_macro_f1=%.4f)", best_path, metrics.macro_f1)

        epoch_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(state, epoch_path)
        self.saved_checkpoints.append(epoch_path)

        if len(self.saved_checkpoints) > self.config.save_total_limit:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug("Removed rotated checkpoint: %s", oldest)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


# ============================================================================
# CLI entry point
# ============================================================================
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Evidence-Aware Hybrid Transformer fake-news verifier."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--train_data", type=str, required=True, help="Path to preprocessed train .pt file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to preprocessed val .pt file")
    parser.add_argument("--output_dir", type=str, default="runs/default", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override config num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override config learning_rate")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--ablate_retrieval", action="store_true",
        help="Ablation: zero out claim-evidence feature tensors (disables retrieval signal).",
    )
    parser.add_argument(
        "--ablate_gating", action="store_true",
        help="Ablation: replace gated cross-attention fusion with naive concatenation.",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    train_config = TrainingConfig.from_yaml(args.config)
    model_config = HybridVerifierConfig.from_yaml(args.config)

    if args.num_epochs is not None:
        train_config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.seed is not None:
        train_config.seed = args.seed

    model_config.ablate_retrieval = args.ablate_retrieval
    model_config.ablate_gating = args.ablate_gating

    set_seed(train_config.seed)

    logger.info(
        "Ablation flags: ablate_retrieval=%s ablate_gating=%s",
        model_config.ablate_retrieval, model_config.ablate_gating,
    )

    train_dataset = EFNDDataset(args.train_data)
    val_dataset = EFNDDataset(args.val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EvidenceAwareClassifier(model_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"training": train_config.__dict__, "model": model_config.__dict__, "cli_args": vars(args)},
            f, indent=2,
        )

    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
