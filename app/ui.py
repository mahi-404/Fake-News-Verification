"""
app.ui
========

Streamlit dashboard for the Evidence-Aware Fake News Verification API.

Accepts either raw article text or a URL (best-effort `<p>`-tag extraction),
calls the FastAPI `/verify` endpoint, and renders:
    - A colored verdict badge (LIKELY REAL / LIKELY FAKE / UNCERTAIN)
    - Confidence + evidence-sufficiency metrics
    - An expander per extracted claim showing retrieved evidence passages,
      source metadata, and a stacked NLI stance bar
      (green = entailment/support, red = contradiction/refute, gray = neutral)

Run with:
    streamlit run app/ui.py
"""

from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup

API_BASE_URL_DEFAULT = os.environ.get("EFND_API_BASE_URL", "http://localhost:8000")

_BADGE_STYLES = {
    "LIKELY REAL": {"bg": "#1b7a3d", "fg": "#ffffff", "label": "LIKELY REAL"},
    "LIKELY FAKE": {"bg": "#b3261e", "fg": "#ffffff", "label": "LIKELY FAKE"},
    "UNCERTAIN": {"bg": "#6b6f76", "fg": "#ffffff", "label": "UNCERTAIN"},
}

_STANCE_COLORS = {
    "entailment": "#2e9e4f",     # green: support
    "contradiction": "#d1453b",  # red: refute
    "neutral": "#9aa0a6",        # gray: neutral
}

_VERDICT_ICONS = {
    "ENTAILED": "✅",
    "CONTRADICTED": "❌",
    "NEUTRAL": "➖",
    "INSUFFICIENT_EVIDENCE": "❔",
}


# ============================================================================
# Data fetching / API calls
# ============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def fetch_article_text_from_url(url: str) -> str:
    """
    Best-effort article text extraction: fetches the page and concatenates
    visible <p> tag text. This is a lightweight heuristic extractor, not a
    full readability/boilerplate-removal pipeline.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EFND-Verifier/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]  # drop boilerplate/nav fragments

    article_text = "\n\n".join(paragraphs)
    if not article_text.strip():
        raise ValueError(
            "Could not extract readable article text from this URL. "
            "Try pasting the article text directly instead."
        )
    return article_text


def call_verify_api(article_text: str, api_base_url: str, timeout: float = 180.0) -> dict:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/verify",
        json={"article": article_text},
        timeout=timeout,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")
    return response.json()


def check_api_health(api_base_url: str) -> Optional[dict]:
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


# ============================================================================
# Rendering helpers
# ============================================================================
def render_verdict_badge(prediction: str, confidence: float) -> None:
    style = _BADGE_STYLES.get(prediction, _BADGE_STYLES["UNCERTAIN"])
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:10px 22px;
            border-radius:999px;
            background-color:{style['bg']};
            color:{style['fg']};
            font-weight:700;
            font-size:1.15rem;
            letter-spacing:0.03em;
        ">
            {style['label']} &nbsp;·&nbsp; {confidence:.1%} confidence
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nli_stance_bar(nli: dict, height_px: int = 22) -> None:
    """
    Renders a horizontal stacked bar: green=entailment, red=contradiction,
    gray=neutral, each segment sized by its probability.
    """
    entailment = max(float(nli.get("entailment", 0.0)), 0.0)
    contradiction = max(float(nli.get("contradiction", 0.0)), 0.0)
    neutral = max(float(nli.get("neutral", 0.0)), 0.0)
    total = max(entailment + contradiction + neutral, 1e-9)

    ent_pct = entailment / total * 100
    con_pct = contradiction / total * 100
    neu_pct = neutral / total * 100

    st.markdown(
        f"""
        <div style="display:flex; width:100%; height:{height_px}px; border-radius:6px;
                    overflow:hidden; border:1px solid rgba(0,0,0,0.1);">
            <div style="width:{ent_pct}%; background-color:{_STANCE_COLORS['entailment']};"
                 title="Support: {entailment:.2f}"></div>
            <div style="width:{con_pct}%; background-color:{_STANCE_COLORS['contradiction']};"
                 title="Refute: {contradiction:.2f}"></div>
            <div style="width:{neu_pct}%; background-color:{_STANCE_COLORS['neutral']};"
                 title="Neutral: {neutral:.2f}"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem;
                    color:#6b6f76; margin-top:2px;">
            <span>🟢 Support {entailment:.2f}</span>
            <span>🔴 Refute {contradiction:.2f}</span>
            <span>⚪ Neutral {neutral:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_expander(claim: dict) -> None:
    verdict = claim.get("claim_verdict", "NEUTRAL")
    icon = _VERDICT_ICONS.get(verdict, "➖")
    claim_text = claim.get("text", "")
    header = f"{icon}  Claim {claim.get('claim_id')} — {verdict.replace('_', ' ').title()}"

    with st.expander(header, expanded=False):
        st.markdown(f"**Claim:** {claim_text}")
        st.markdown("**NLI stance distribution (vs. retrieved evidence):**")
        render_nli_stance_bar(claim.get("nli_assessment", {}))

        evidence_list = claim.get("retrieved_evidence", [])
        st.markdown(f"**Retrieved evidence ({len(evidence_list)}):**")
        if not evidence_list:
            st.caption("No supporting or refuting evidence was retrieved for this claim.")
        for evidence in evidence_list:
            st.markdown(
                f"""
                <div style="border-left:3px solid #d0d3d8; padding:8px 12px; margin-bottom:8px;
                            background-color:rgba(0,0,0,0.02); border-radius:4px;">
                    <div style="font-size:0.95rem;">{evidence.get('passage', '')}</div>
                    <div style="font-size:0.78rem; color:#6b6f76; margin-top:6px;">
                        📄 <b>{evidence.get('source', 'unknown source')}</b>
                        &nbsp;·&nbsp; retrieval score: {evidence.get('retrieval_score', 0.0):.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result(result: dict) -> None:
    st.subheader("Verdict")
    render_verdict_badge(result["prediction"], result["confidence"])
    st.write("")
    st.caption(result.get("verdict_reasoning", ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col2:
        stylistic = result.get("article_stylistic_score", {})
        st.metric("Sensationalism (heuristic)", f"{stylistic.get('sensationalism_bias', 0.0):.2f}")
    with col3:
        uncertainty = result.get("uncertainty_metrics", {})
        st.metric(
            "Evidence sufficiency",
            f"{uncertainty.get('evidence_sufficiency_score', 0.0):.2f}",
            delta="ABSTAINED" if uncertainty.get("abstain_triggered") else "confident",
            delta_color="off" if uncertainty.get("abstain_triggered") else "normal",
        )

    st.write("")
    claims = result.get("claims_analysis", [])
    st.subheader(f"Extracted claims ({len(claims)})")
    if not claims:
        st.info("No check-worthy factual claims were extracted from this article.")
    for claim in claims:
        render_claim_expander(claim)

    if "latency_ms" in result:
        st.caption(f"Inference latency: {result['latency_ms']:.0f} ms")


# ============================================================================
# Main app
# ============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Evidence-Aware Fake News Verifier",
        page_icon="🔎",
        layout="wide",
    )

    st.title("🔎 Evidence-Aware Fake News Verification")
    st.caption(
        "Deconstructs an article into check-worthy claims, retrieves evidence, "
        "and verifies each claim via Natural Language Inference before fusing "
        "everything into a final, explainable verdict."
    )

    with st.sidebar:
        st.header("Settings")
        api_base_url = st.text_input("API base URL", value=API_BASE_URL_DEFAULT)

        health = check_api_health(api_base_url)
        if health is not None:
            st.success(
                f"API online · device={health.get('device')} · "
                f"retriever_ready={health.get('retriever_ready')}"
            )
        else:
            st.error("API unreachable at this URL.")

        st.divider()
        input_mode = st.radio("Input mode", ["Paste text", "Fetch from URL"], index=0)

    article_text: Optional[str] = None

    if input_mode == "Paste text":
        article_text = st.text_area(
            "Article text",
            height=260,
            placeholder="Paste the full text of a news article here...",
        )
        submit = st.button("Verify article", type="primary", disabled=not article_text)
    else:
        url = st.text_input("Article URL", placeholder="https://example.com/some-news-story")
        submit = st.button("Fetch & verify", type="primary", disabled=not url)
        if submit and url:
            with st.spinner("Fetching article from URL..."):
                try:
                    article_text = fetch_article_text_from_url(url)
                    with st.expander("Extracted article text", expanded=False):
                        st.write(article_text)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to fetch article: {exc}")
                    article_text = None
                    submit = False

    if submit and article_text:
        with st.spinner("Extracting claims, retrieving evidence, and verifying..."):
            try:
                result = call_verify_api(article_text, api_base_url)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Verification failed: {exc}")
                return
        st.divider()
        render_result(result)


if __name__ == "__main__":
    main()
