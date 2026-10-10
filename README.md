# Evidence-Aware Fake News Verification (EFND)

An end-to-end, evidence-aware hybrid transformer framework for explainable fake news detection and claim verification. 

Unlike black-box models, this system deconstructs an article into check-worthy claims, retrieves factual evidence for each claim from a Knowledge Base (KB), verifies them using Natural Language Inference (NLI), and fuses the evidence with the article's stylistic signals to produce a highly explainable final verdict.

## 🚀 Features

- **Claim Extraction**: Extracts independent, check-worthy factual claims from raw news articles.
- **Hybrid Evidence Retrieval**: Uses Reciprocal Rank Fusion (RRF) to combine dense semantic retrieval (BGE-large) with sparse lexical retrieval (BM25) over a trusted Knowledge Base.
- **NLI Engine**: Evaluates entailment, contradiction, or neutrality of retrieved evidence against extracted claims using a DeBERTa-v3 cross-encoder.
- **Uncertainty Gating**: Automatically abstains (returns "INSUFFICIENT EVIDENCE") if the retrieved evidence is too weak to support a confident verdict.
- **Interactive UI**: A Streamlit dashboard for real-time article verification.
- **High-Throughput API**: A FastAPI backend designed to serve the pipeline concurrently.

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/mahi-404/Fake-News-Verification.git
cd Fake-News-Verification
```

### 3. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 4. Install Dependencies
Install the package and its dependencies in editable mode:
```bash
pip install -e .
```

---

## 🧠 Building the Knowledge Base (KB)

The system relies on a local Knowledge Base (KB) to fact-check claims. Out of the box, you must build the index before running the verifier.

1. **Add your corpus**: Place your raw facts in `data/raw/sample_corpus.jsonl`.
2. **Build the Index**:
   ```bash
   python scripts/build_kb.py --input_path data/raw/sample_corpus.jsonl --source_format plain_jsonl --output_dir data/kb_index
   ```
   *(This will chunk the text, compute embeddings, and build the FAISS and BM25 indices in `data/kb_index/`)*

---

## 💻 Running the Application

The project is split into a backend API (FastAPI) and a frontend UI (Streamlit).

### 1. Start the Backend API
You must specify the path to the Knowledge Base index directory:
```bash
# On Windows (PowerShell)
$env:EFND_KB_INDEX_DIR="data\kb_index"
uvicorn app.api:app --host 0.0.0.0 --port 8000

# On Linux/Mac
EFND_KB_INDEX_DIR="data/kb_index" uvicorn app.api:app --host 0.0.0.0 --port 8000
```
*Note: The first time you start the API, it will download the necessary HuggingFace models (DeBERTa, BGE, etc.). This may take a few minutes.*

### 2. Start the Frontend UI
Open a **new terminal window** (ensure the virtual environment is activated) and run:
```bash
streamlit run app/ui.py
```
This will launch the dashboard in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app/                  # FastAPI backend and Streamlit UI
├── config/               # YAML configuration files
├── data/                 # Raw datasets, processed files, and the KB index
├── efnd/                 # Core framework package
│   ├── claims/           # Claim extraction module
│   ├── retrieval/        # Hybrid semantic/sparse retrieval (Module 3)
│   ├── inference/        # NLI Cross-Encoder engine (Module 4)
│   ├── models/           # Gated Fusion & Classification (Modules 5 & 6)
│   └── training/         # Model training pipeline
├── scripts/              # Utilities for building KB and preparing datasets
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # This file
```

## 📄 License
This project is licensed under the MIT License.
