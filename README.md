# OmicsGPT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20842287.svg)](https://doi.org/10.5281/zenodo.20842287)

OmicsGPT is an open-source multi-agent framework for reproducible omics data analysis powered by large language models.

Powered by LangGraph, PyDESeq2, OncoKB, and PubMed.

This repository contains a Streamlit application built around a LangGraph `StateGraph` that ingests RNA-seq expression data, performs differential expression and pathway analysis, gathers external evidence, and synthesizes clinical-style reports.

## Demo

Try the public demo at [https://omicsai.org/OmicsGPT](https://omicsai.org/OmicsGPT).

## Overview

The app is designed for:

- translational researchers exploring transcriptomic biomarkers
- bioinformaticians performing differential expression and pathway analysis
- research teams needing rapid, evidence-grounded gene triage

## Architecture

The pipeline is implemented as a compiled LangGraph `StateGraph` with four sequential agent nodes:

```text
START -> Planner -> Executor -> Expert Review -> Writer -> END
```

## Features

- Differential expression analysis with PyDESeq2 or a T-test fallback
- GSEA pathway enrichment using gseapy
- OncoKB variant annotations
- Semantic PubMed retrieval with OpenAI embeddings and FAISS
- Clinical trials lookup
- STRING interaction networks
- Open Targets tractability and essentiality lookups
- AlphaFold structure visualization
- UniProt binding-site inspection
- PDF-based custom knowledge retrieval
- Human-in-the-loop evidence review
- Export to HTML and DOCX

## Docker Usage

The Docker image starts the Streamlit app on port 8501 and serves it under `/OmicsGPT`.

```bash
docker build -t omicsgpt:local .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  -e ONCOKB_API_KEY=your_oncokb_api_key_here \
  omicsgpt:local
```

If you prefer Streamlit secrets, mount a secrets file at `/app/.streamlit/secrets.toml`:

```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/secrets.toml <<'EOF'
OPENAI_API_KEY = "sk-..."
ONCOKB_API_KEY = "..."
EOF

docker run -p 8501:8501 \
  -v ~/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  omicsgpt:local
```

## Local Development

```bash
git clone https://github.com/your-org/OmicsGPT.git
cd OmicsGPT

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export ONCOKB_API_KEY=...

streamlit run ultimate_agent.py
```

## Prerequisites

- OpenAI API key in `st.secrets` as `OPENAI_API_KEY`
- OncoKB API key in `st.secrets` as `ONCOKB_API_KEY`
- Python 3.11+
- Docker for containerized deployment

## Sample Data

The repository includes sample breast cancer datasets in the root directory for quick testing:

- `bc_counts_transposed.csv`
- `bc_counts_transposed_condensed.csv`
- `bc_meta.csv`

## External Integrations

The executor agent orchestrates calls to:

- OncoKB
- PubMed / NCBI E-utilities
- ClinicalTrials.gov
- STRING-DB
- Open Targets Platform
- AlphaFold DB
- UniProt
- MSigDB / gseapy

## Tech Stack

- LangGraph for orchestration
- Streamlit for the UI
- OpenAI via `langchain_openai`
- PyDESeq2 for differential expression
- gseapy for pathway analysis
- NetworkX and PyVis for networks
- stmol and py3Dmol for structure visualization

## Citation

If you use OmicsGPT in your research, please cite the archived software release:

> OmicsGPT v1.0.0. Zenodo. https://doi.org/10.5281/zenodo.20842287