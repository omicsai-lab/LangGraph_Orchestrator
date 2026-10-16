# OmicsGPT — Scientific Architecture and Evidence Integration Principles

**Status:** Living design proposal (v1.0, 2026-10-16)  
**Scope:** Guides the next development stages; does not change current code or imply that planned capabilities are implemented.

## 1. Scientific objective
OmicsGPT aims to connect validated omics analyses to transparent, evidence-grounded hypothesis generation. It must distinguish **statistical association**, **biological interpretation**, **candidate prioritization**, and **biomarker/target validation**. A differentially expressed gene is not automatically a biomarker, causal driver, or therapeutic target.

**Core contribution to develop and evaluate:** a reproducible, evidence-aware framework integrating omics statistics, pathway/network context, and independently retrieved biological evidence to support uncertainty-aware candidate prioritization and source-traced scientific review.

## 2. Build simply; keep boundaries extensible
- **Current validated path:** bulk RNA-seq raw counts → strict intake/design → researcher approval → guarded PyDESeq2 → results, diagnostics, provenance. The existing 187-sample / 110-gene *condensed* benchmark is a development fixture, not a full-transcriptome performance demonstration.
- **Future adapters:** single-cell RNA-seq, proteomics, metabolomics, and cohort-matched multi-omics each require modality-appropriate validation, unit-of-replication definitions, and statistical methods. Never treat cells as independent biological replicates when donor-level dependence matters.
- **Common downstream objects:** analysis provenance, feature identifiers, statistical evidence, pathway associations, external evidence records, and candidate evidence profiles. Not all modalities have genes as primary features; mappings must be explicit and auditable.
- **Separation of concerns:** input contracts; scientific computation; evidence retrieval; evidence integration; agentic narrative; presentation. LLMs may summarize verified evidence but must not silently alter numerical analyses, identifiers, or provenance.

## 3. Simple staged roadmap
| Stage | Deliverable | Boundary / acceptance gate |
|---|---|---|
| **V2.5** | Results explorer: interactive volcano plot, searchable table, definitions, multiple deterministic candidate-pool selection strategies | Display controls never rerun DE; selection is exploratory, not final biomarker prioritization. |
| **V2.6** | Reproducible pathway enrichment (e.g., preranked GSEA) with documented identifier mapping and gene-set database/version | Preserve full ranked universe; report mapping failures, background assumptions, multiple testing and leading-edge membership. |
| **V2.7** | Bounded evidence retrieval and multidimensional gene/feature evidence profiles | Source URL/accession, retrieval time, species/tissue/disease context, study type, directness, uncertainty and contradictions. |
| **V2.8** | Evidence-integrated prioritization and controlled expert/agentic interpretation | Transparent decision criteria, sensitivity analysis, source-traced claims and human review; no unsupported single-score claims. |
| **Future research** | Empirically calibrated prioritization | External outcomes, independent validation, leakage control, correlated evidence handling and calibration evaluation required. |

These are milestones, not promises or fixed dates. Network methods may be introduced later as optional, independently validated evidence modules. Full-transcriptome and multi-dataset testing must precede broad robustness claims.

## 4. Candidate selection: interim approach
Separate **candidate-pool generation** from **final evidence-informed prioritization**. Initial deterministic, auditable options:
1. Balanced up/down: top N positive and top N negative log2FC among genes meeting user-specified eligibility criteria.
2. Statistical evidence: ascending adjusted p-value among eligible genes.
3. Effect magnitude: descending absolute log2FC among eligible genes.
4. Researcher-directed: explicitly selected genes, retaining the reason and selection timestamp.

Specify eligibility (including FDR threshold), ordering, stable tie-breaks (e.g., identifier), missing-value handling, and all selected IDs. Display selection strategy alongside the volcano plot. Preserve unselected genes and explain exclusions. Avoid treating selection as independent confirmation; large observed effects can be exaggerated by selection. Selection/display settings must not invalidate the DE model approval or trigger re-fitting.

## 5. Multidimensional evidence profile (future common contract)
Each evidence record should include **feature ID and namespace, species, modality, cohort or external source, biological context, evidence type, direction, quantitative estimate (if available), uncertainty, provenance, retrieval/database version, quality annotations, and missingness status**.

For each candidate, summarize distinct domains: (a) DE effect, uncertainty and diagnostics; (b) pathways/leading-edge; (c) network or interaction context when validated; (d) disease-specific literature; (e) normal-tissue context; (f) independent replication; (g) contradictory findings. Distinguish *not assessed*, *unavailable*, *negative evidence*, and *contradictory evidence*. Separate primary experimental results from curated annotations, predictions, and LLM hypotheses. Do not double-count overlapping pathways, publications or literature-derived network links as independent corroboration.

## 6. Evidence quality and prioritization
**Near term:** do not create a universal weighted score. Use structured profiles and explicit rule-based comparisons, with researcher oversight. A label such as “strong evidence” requires a documented rule and source trace. A gene with little literature must not automatically be scored as biologically unimportant.

**Longer term — empirical calibration (research hypothesis):** estimate the predictive utility of evidence domains against independent, task-specific validation outcomes (e.g., replication in held-out cohorts, orthogonal experimental validation). Candidate methods include regularized models, hierarchical Bayesian approaches, or learning-to-rank. Required safeguards: predefined target outcome, dataset splits by study/cohort and time where appropriate, prevention of annotation/literature leakage, treatment of missing-not-at-random evidence, evidence dependence, popularity bias, calibration, uncertainty intervals, and external generalization tests. Compare with DE-only and transparent rule-based baselines. Weighting that is learned from data is not automatically objective; its validity depends on outcomes and design.

## 7. Performance and reproducibility contracts
- Preserve uploaded bytes, validated analytical data, and presentation data separately. No silent coercion, imputation, gene dropping, or engine substitution.
- Cache by input hash + approved analysis configuration + software/database versions + evidence retrieval settings. Invalidate only affected downstream stages. A volcano threshold change should not rerun DE; new literature retrieval should not rerun DE.
- Provide per-stage progress, timing, cancellation/timeouts when supported, and actionable errors. No indefinite spinner. Limit retrieval scope and concurrency; allow users to defer costly modules.
- Provisional *development* targets (not guarantees): small-data intake <2 s; configuration rerender <1 s; condensed DE <15 s, excluding cold Docker startup. Benchmark representative full-transcriptome and future modalities before setting service limits.
- Record model fitting choices, internal fallbacks, outlier refits, package versions, data/config hashes, gene-set versions, and retrieval timestamps. Maintain repeatability tests, R DESeq2 comparisons, and independent validation.

## 8. Agentic safeguards
Agent outputs must cite evidence-record IDs; distinguish observations, external findings, mechanistic hypotheses, and recommendations for validation. Role-conditioned reviews should surface disagreements, contradictions, and unresolved evidence gaps. Keep the numerical pipeline deterministic; if multiple LLM outputs are used, evaluate agreement and evidence fidelity rather than averaging prose or treating consensus as proof. Never claim clinical readiness without clinical validation.

## 9. Immediate V2.5A design decisions to confirm before coding
- Volcano x-axis = estimated log2FC; y-axis = -log10(p-value) by default, with a clearly labeled adjusted-p-value alternative; handle p=0 and missing values explicitly.
- FDR 0.05 and absolute log2FC 1 are **exploratory display defaults**, not universally appropriate inferential cutoffs. Distinguish adjusted p-value selection from raw p-value plotting.
- Initial candidate strategies are optional and deterministic; selected points are visibly highlighted, with reasons. No pathway/literature weighting yet.
- Results explorer operates on a *copy* of completed DE outputs and never mutates approved results or model settings.
- Implement pure functions and tests before Streamlit integration; then test on condensed data and representative full-transcriptome data when available.

## 10. Open decisions / revision log
**Open:** desired default candidate selection (none versus balanced up/down), appropriate top-N limits, how to visualize FDR and effect thresholds, gene ID mapping standards, which pathway resource to implement first, external-evidence retrieval budgets, and validation outcomes for empirical calibration.

**2026-10-16 — v1.0:** Adopted simple V2.5–V2.8 roadmap; deferred weighted prioritization; retained empirically calibrated prioritization as a research direction; formalized modular evidence profiles, performance and provenance requirements.
