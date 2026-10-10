"""
app.api
=========

High-throughput FastAPI service wrapping `FakeNewsVerificationPipeline`.

The pipeline (DeBERTa encoder, retrieval index, NLI cross-encoder,
classifier) is loaded exactly once at process startup and reused across
requests. Because the underlying model calls are synchronous (blocking)
PyTorch/CPU-GPU work, each request is offloaded to a bounded
`ThreadPoolExecutor` so the FastAPI event loop stays responsive and
concurrent requests are naturally queued rather than serialized on the
event loop itself.

Run with:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1

(Use a single Uvicorn *process* worker per model replica -- the pipeline
holds GPU/CPU model state that should not be forked after CUDA init; scale
throughput via the `EFND_MAX_WORKERS` thread pool size and/or by running
multiple independent processes behind a load balancer, each pinned to a
different device.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from efnd.inference_pipeline import FakeNewsVerificationPipeline

logger = logging.getLogger("efnd.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ============================================================================
# Configuration (env-overridable; defaults read from config.yaml)
# ============================================================================
def _load_serving_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("serving", {})


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("EFND_CONFIG_PATH", REPO_ROOT / "config" / "config.yaml"))
KB_INDEX_DIR = os.environ.get("EFND_KB_INDEX_DIR")            # e.g. data/kb_index
MODEL_CHECKPOINT = os.environ.get("EFND_MODEL_CHECKPOINT")    # e.g. runs/best/checkpoints/best_model.pt
DEVICE = os.environ.get("EFND_DEVICE", "auto")

_serving_config = _load_serving_config(CONFIG_PATH)
_serving_defaults = _serving_config.get("api", {})
MAX_WORKERS = int(os.environ.get("EFND_MAX_WORKERS", 4))
REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get(
        "EFND_REQUEST_TIMEOUT_SECONDS",
        _serving_config.get("request_timeout_seconds", 30),
    )
)
MAX_ARTICLE_CHARS = int(os.environ.get("EFND_MAX_ARTICLE_CHARS", 20_000))


# ============================================================================
# Request / response schemas
# ============================================================================
class VerifyRequest(BaseModel):
    article: str = Field(..., description="Raw news article text to verify.", min_length=1)

    @field_validator("article")
    @classmethod
    def article_must_be_reasonable(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("article must not be empty or whitespace-only.")
        if len(stripped) > MAX_ARTICLE_CHARS:
            raise ValueError(
                f"article exceeds maximum allowed length of {MAX_ARTICLE_CHARS} characters."
            )
        return stripped


class RetrievedEvidence(BaseModel):
    passage: str
    source: str
    retrieval_score: float


class NLIAssessment(BaseModel):
    entailment: float
    neutral: float
    contradiction: float


class ClaimAnalysis(BaseModel):
    claim_id: int
    text: str
    retrieved_evidence: list[RetrievedEvidence]
    nli_assessment: NLIAssessment
    claim_verdict: str


class ArticleStylisticScore(BaseModel):
    sensationalism_bias: float
    confidence: float


class UncertaintyMetrics(BaseModel):
    evidence_sufficiency_score: float
    abstain_triggered: bool


class VerifyResponse(BaseModel):
    prediction: str
    confidence: float
    verdict_reasoning: str
    article_stylistic_score: ArticleStylisticScore
    claims_analysis: list[ClaimAnalysis]
    uncertainty_metrics: UncertaintyMetrics
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    retriever_ready: bool


# ============================================================================
# Application state & lifespan
# ============================================================================
class AppState:
    pipeline: Optional[FakeNewsVerificationPipeline] = None
    executor: Optional[ThreadPoolExecutor] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FakeNewsVerificationPipeline (this may take a while)...")
    state.pipeline = FakeNewsVerificationPipeline(
        config_path=CONFIG_PATH,
        kb_index_dir=KB_INDEX_DIR or None,
        model_checkpoint=MODEL_CHECKPOINT or None,
        device=DEVICE,
    )
    state.executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="efnd-inference"
    )
    logger.info("Pipeline loaded. Ready to serve. max_workers=%d", MAX_WORKERS)

    yield

    logger.info("Shutting down: releasing pipeline and executor.")
    if state.executor is not None:
        state.executor.shutdown(wait=True)
    state.pipeline = None
    state.executor = None


app = FastAPI(
    title="Evidence-Aware Fake News Verification API",
    description="Evidence-aware hybrid transformer for explainable fake news detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routes
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized.")
    return HealthResponse(
        status="ok",
        device=str(state.pipeline.device),
        retriever_ready=state.pipeline._retriever_ready,
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest, http_request: Request) -> VerifyResponse:
    if state.pipeline is None or state.executor is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized.")

    loop = asyncio.get_running_loop()
    start_time = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(state.executor, state.pipeline.analyze, request.article),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        logger.error("Request from %s timed out after %.1fs", http_request.client, REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(
            status_code=504,
            detail=f"Verification timed out after {REQUEST_TIMEOUT_SECONDS}s.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline inference failed.")
        raise HTTPException(status_code=500, detail=f"Internal verification error: {exc}") from exc

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    result["latency_ms"] = round(latency_ms, 2)

    return VerifyResponse(**result)


@app.get("/")
async def root() -> dict:
    return {
        "service": "Evidence-Aware Fake News Verification API",
        "endpoints": {"verify": "POST /verify", "health": "GET /health"},
    }


# ============================================================================
# Standalone launch
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    host = _serving_defaults.get("host", "0.0.0.0")
    port = int(os.environ.get("EFND_API_PORT", _serving_defaults.get("port", 8000)))

    uvicorn.run("app.api:app", host=host, port=port, workers=1, log_level="info")
