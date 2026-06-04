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