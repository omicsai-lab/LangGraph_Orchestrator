# OmicsGPT V2.6A — Pathway and Evidence Architecture Contract

Status: Proposed design for review; no computational implementation in this checkpoint.

## Scope
- Initial modality: bulk RNA-seq. Future single-cell, proteomics and multi-omics adapters must produce modality-labeled evidence records, not reuse bulk DE assumptions.
- Pathway inference: preranked GSEA using signed Wald statistics from an approved full-transcriptome differential-expression run.
- Primary pathway collection: locally stored, versioned human Reactome gene sets, with checksum and license/redistribution review.
- The 110-gene condensed benchmark is suitable for software tests but not genome-scale pathway interpretation.

## Data contracts
- Ranked universe: all eligible tested genes with finite signed Wald statistics, not only FDR-significant or highlighted genes.
- Validate unique identifiers, species, identifier namespace, duplicates, unmapped IDs, and pathway coverage; report exclusions and decisions. Do not silently map or discard ambiguous identifiers.
- Preserve original count-file and metadata hashes, filtering policy, ranking method, gene-set version and checksum, software versions, random seed, and computation parameters.
- Gene-set overlap and shared source evidence are not independent replications.

## Execution modes
### Hybrid / Fast (default)
- Local checksum-verified pathway resources; bounded targeted evidence requests for researcher-selected genes or pathways.
- Cache by scientific input fingerprint, evidence-source version, query parameters and retrieval mode.
- Display progress and source coverage; no live API dependency for primary GSEA.

### Deep Dive (opt-in)
- Broader documented search across *enabled and validated* evidence sources, with higher request/page budgets and more detailed review of contradictory and missing evidence.
- Deep Dive is bounded by explicit per-source caps, timeouts and an overall budget; it does not promise exhaustive literature coverage.
- Can resume or reuse cached Fast results, without rerunning DE or GSEA when scientific inputs are unchanged.
- Researcher may cancel; record partial completion and failed sources. No silent substitutions.

## Evidence records
Each record should include: source, source release/version, retrieval timestamp, stable source identifier/URL, organism, tissue/disease context, evidence type, claim/finding, supporting study or dataset, independence/dependency indicators, limitations, and retrieval status.

Distinguish: observed statistical association, curated annotation, experimental evidence, computational prediction, literature summary, and AI-generated hypothesis. Missing evidence is not negative evidence.

## Performance and safeguards
- Intake and plotting changes must not rerun statistical fitting or external searches.
- Display stage-level elapsed time, progress and failure reasons; benchmark full-transcriptome runtime before setting service-level targets.
- Cache external responses with provenance and a transparent refresh policy.
- Evidence retrieval failures cannot change approved statistics or be misrepresented as completed searches.
- No patient-identifiable data or uploaded cohort data are sent to third-party APIs without explicit authorization and appropriate governance.

## Scientific boundaries
- Candidate highlighting is exploratory, not validated biomarker ranking.
- Initial pathway analysis produces pathway-level evidence, not a composite gene score.
- Later empirically calibrated prioritization requires independent validation outcomes and assessment of evidence overlap, leakage and publication bias.

## Next checkpoint acceptance criteria
1. Inspect the full count matrix identifier namespace and gene universe; document any upstream filtering.
2. Validate ranked-universe and gene-set input schemas with unit tests and explicit failure modes.
3. Select a specific licensed Reactome resource release and verify checksum and mapping coverage.
4. Benchmark with full-transcriptome data, retaining the condensed dataset as a negative/smoke fixture.
5. Keep external API integration out of the first GSEA execution checkpoint.
