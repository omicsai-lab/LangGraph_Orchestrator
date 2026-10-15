# OmicsGPT vNext — V1 validated intake

## Scope

New, parallel `app.py` entry point. Existing `ultimate_agent.py`, Dockerfile, and deployed application are unchanged. V1 performs no DE, enrichment, LLM, or network calls. It uses the existing `omicsgpt/intake.py`, `csv_import.py`, and `design.py` modules.

## Researcher workflow

1. Declare bulk RNA-seq raw counts and upload counts/metadata CSVs.
2. Select metadata sample-ID column, grouping variable, numerator, reference, and optional categorical covariates.
3. Inspect source previews, alignment, validation, group sizes, source hashes, and proposed transformations.
4. Confirm the exact configuration. The fingerprint includes source hashes, sample order, orientation, contrast, covariates, and an explicit analysis settings object.
5. Preview the approved samples-by-genes matrix. No computation begins.

## Limitations and next milestones

- V1 does not run PyDESeq2 or replace the legacy application.
- This release does not support single-cell RNA-seq, multimodal analyses, extra unmatched samples, ambiguous orientation, or pairwise subsetting of three-or-more-level designs.
- The fingerprint prevents reuse of approval after configuration changes, but in-memory approval is not a permanent signed audit record.
- Design rank checks do not establish biological independence or validate study design.
- CSV parser currently assumes first column is the count matrix row identifier; explicit identifier-column mapping can be added later.
- Do not upload identifiable clinical data to an unapproved environment.
- Next: fast Streamlit smoke test, then provenance/run manifest and validated PyDESeq2 integration as separate milestones.

## Local checks

`python -m pytest -q tests/`

`python -m py_compile app.py omicsgpt/vnext_workflow.py`

For Docker local preview, from the project directory in PowerShell (Docker Desktop running):

`docker run --rm -p 127.0.0.1:8501:8501 --mount "type=bind,source=$($PWD.Path),target=/app" --entrypoint streamlit oncology-agent:v1 run /app/app.py --server.address=0.0.0.0 --server.port=8501`

Leave the terminal open and visit http://localhost:8501. The Docker image must already include dependencies and the `omicsgpt` modules must be present in the mounted project directory.

No changes to Dockerfile, requirements.txt, or production deployment are required for this preview.
