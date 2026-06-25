# 🧬 Agentic Precision Oncology Pipeline
Powered by LangGraph, PyDESeq2, OncoKB, and PubMed.

This repository contains an autonomous multi-agent graph architecture designed to ingest raw RNA-seq expression data, perform local pathway clustering, host an internal expert tumor board debate, and synthesize actionable clinical triage reports.

## 🚀 Quick Start & Testing
To test the live application, use the three sample breast cancer datasets provided right here in the root directory:
1. `bc_counts_transposed.csv` (Raw integer counts matrix)
2. `bc_counts_transposed_condensed.csv` (Condensed profile for rapid testing)
3. `bc_meta.csv` (Sample demographic and condition covariates)

Download these files to your local machine and upload them directly into the front-end file widgets to test the pipeline end-to-end.

Try out our [demo](https://langgraphorchestrator-mwtjfbe6ujxm9f6dbghom5.streamlit.app/)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Docker Usage](#-docker-usage)
- [Local Development](#-local-development)
- [Prerequisites](#-prerequisites)
- [Sample Data](#-sample-data)
- [Analysis Modes](#-analysis-modes)
- [External Integrations](#-external-integrations)
- [Tech Stack](#-tech-stack)

---

## 🔬 Overview

**OmicsGPT** is a production-grade, AI-native bioinformatics platform that transforms raw genomic count matrices into data-driven insights — with no programming required. At its core is a stateful LangGraph multi-agent orchestrator that autonomously plans, executes, peer-reviews, and writes a final structured research report, mirroring the reasoning process of a multi-expert molecular review board.

The platform is designed for:
- **Translational researchers** exploring transcriptomic biomarkers
- **Bioinformaticians** performing differential expression and pathway analysis
- **Research teams** needing rapid, evidence-grounded gene triage

---

## 🏗️ Architecture

The pipeline is implemented as a compiled **LangGraph `StateGraph`** with four sequential agent nodes:

```
START → [Planner] → [Executor] → [Expert Review] → [Writer] → END
```

| Node | Role |
|---|---|
| **Planner** | Parses the user's natural-language intent and generates a step-by-step tool execution plan |
| **Executor** | Runs each tool in the plan — querying OncoKB, PubMed, STRING, Open Targets, AlphaFold, and more |
| **Expert Review** | Hosts a simulated multi-expert debate, flagging lineage mismatches and evidence conflicts |
| **Writer** | Synthesizes all gathered evidence into a structured, publication-quality markdown report |

---

## ✨ Features

- **Differential Expression Analysis** — PyDESeq2-powered DESeq2 workflow on raw count matrices
- **GSEA Pathway Enrichment** — Gene Set Enrichment Analysis via `gseapy` with interactive plots
- **OncoKB Variant Annotations** — Oncogenicity and actionability classifications from MSKCC
- **Semantic PubMed Literature Search** — Intent-driven retrieval with OpenAI embedding-based semantic ranking and AI peer-review scoring
- **Clinical Trials Lookup** — Active and recruiting trials matched to gene + tumor type
- **Protein–Protein Interaction Networks** — STRING-DB interactomes rendered as interactive force-directed graphs
- **Drug Target Tractability** — Open Targets Platform tractability scores (small molecule, antibody, PROTAC)
- **AlphaFold Structure Viewer** — 3D protein structure rendering with binding site overlay via UniProt
- **GTEx / HPA Tissue Expression** — Normal tissue expression context via GPT-mediated proxy
- **RAG over Custom PDFs** — Upload institutional reports or preprints for in-context evidence retrieval
- **Multi-Agent Tumor Board Debate** — Expert consensus round before final report synthesis
- **Interactive Volcano Plots** — Plotly-powered interactive differential expression visualization
- **Exportable Reports** — Download synthesized reports as formatted documents

---

## 🐳 Docker Usage

The official pre-built image is published to Docker Hub and is the recommended way to run OmicsGPT anywhere — no Python environment setup required.

**Official Image URI:** `patrickroney44/omicsgpt:v1.0.0`

---

### Option 1 — Pull and Run from Docker Hub (Recommended)

```bash
# Pull the image
docker pull patrickroney44/omicsgpt:v1.0.0

# Run the application
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  patrickroney44/omicsgpt:v1.0.0
```

Then open your browser to **http://localhost:8501**.

---

### Option 2 — Run with a Secrets File (Recommended for Production)

Instead of passing secrets as environment variables, mount a Streamlit secrets file:

```bash
# Create your secrets file
mkdir -p ~/.streamlit
echo 'OPENAI_API_KEY = "sk-..."' > ~/.streamlit/secrets.toml

# Run with secrets mounted
docker run -p 8501:8501 \
  -v ~/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  patrickroney44/omicsgpt:v1.0.0
```

---

### Option 3 — Build Locally from Source

If you want to run a modified version from your local checkout:

```bash
# Clone the repository
git clone https://github.com/your-org/LangGraph_Orchestrator.git
cd LangGraph_Orchestrator

# Build the image
docker build -t omicsgpt:local .

# Run the locally built image
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  omicsgpt:local
```

---

### Option 4 — Docker Compose

For a reproducible, shareable deployment:

```yaml
# docker-compose.yml
version: "3.9"
services:
  omicsgpt:
    image: patrickroney44/omicsgpt:v1.0.0
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

```bash
OPENAI_API_KEY=sk-... docker compose up
```

---

## 💻 Local Development

To run directly from source without Docker:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/LangGraph_Orchestrator.git
cd LangGraph_Orchestrator

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=sk-...
# Or create .streamlit/secrets.toml:
# OPENAI_API_KEY = "sk-..."

# 5. Launch the app
streamlit run ultimate_agent.py
```

---

## 🔑 Prerequisites

| Requirement | Details |
|---|---|
| **OpenAI API Key** | Required. GPT-5 class models are used for planning, peer-review, and report synthesis. Get one at [platform.openai.com](https://platform.openai.com). |
| **Python 3.11+** | Required for local development |
| **Docker** | Required for containerized deployment (any version supporting `ENTRYPOINT`) |
| **Internet Access** | Required for live API calls to OncoKB, PubMed, STRING, Open Targets, AlphaFold, and ClinicalTrials.gov |

> **Note on costs:** Each full pipeline run makes multiple LLM calls for planning, embedding, peer-review scoring, tumor board consensus, and report writing. Monitor usage on your OpenAI dashboard.

---

## 📁 Sample Data

Three breast cancer datasets are included in the repository root to enable immediate end-to-end testing:

| File | Description |
|---|---|
| `bc_counts_transposed.csv` | Full raw integer RNA-seq count matrix (samples × genes) |
| `bc_counts_transposed_condensed.csv` | Condensed profile for rapid iteration and demos |
| `bc_meta.csv` | Sample metadata including condition labels and covariates |

Upload these files directly into the Streamlit file widgets to run the full pipeline without needing your own data.

---

## 🔬 Analysis Modes

The pipeline supports two primary modes, selectable in the UI:

| Mode | Description |
|---|---|
| **Triage** | Focuses on actionable evidence — OncoKB classifications, matching trials, and therapeutic implications |
| **Discovery** | Adds protein interaction network analysis (STRING), pathway co-expression context, and deeper mechanistic literature mining |

---

## 🌐 External Integrations

The executor agent orchestrates calls to the following public databases and APIs:

| Integration | Purpose |
|---|---|
| [OncoKB](https://www.oncokb.org/) | Variant oncogenicity and actionability classifications |
| [PubMed / NCBI E-utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/) | Literature retrieval and semantic ranking |
| [ClinicalTrials.gov](https://clinicaltrials.gov/) | Active trial matching by gene and tumor type |
| [STRING-DB](https://string-db.org/) | Protein–protein interaction networks |
| [Open Targets Platform](https://www.opentargets.org/) | Drug target tractability scores |
| [AlphaFold DB](https://alphafold.ebi.ac.uk/) | Predicted 3D protein structures |
| [UniProt](https://www.uniprot.org/) | Binding site annotations and protein metadata |
| [MSigDB / gseapy](https://www.gsea-msigdb.org/) | Gene set collections for pathway enrichment |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph) |
| **LLM** | [OpenAI GPT-5 class](https://platform.openai.com) via `langchain_openai` |
| **Embeddings & RAG** | OpenAI Embeddings + in-memory vector store |
| **Differential Expression** | [PyDESeq2](https://github.com/owkin/PyDESeq2) |
| **Pathway Analysis** | [gseapy](https://github.com/zqfang/GSEApy) |
| **Network Graphs** | [NetworkX](https://networkx.org/) + [PyVis](https://pyvis.readthedocs.io/) |
| **3D Structure** | [stmol](https://github.com/napoles-uach/stmol) + [py3Dmol](https://3dmol.csb.pitt.edu/) |
| **Front-End** | [Streamlit](https://streamlit.io/) |
| **Containerization** | Docker (Python 3.11-slim base) |

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20842287.svg)](https://doi.org/10.5281/zenodo.20842287)
