# Bulk RNA-seq intake contract v1 (proposed)

Status: specification for review, not yet implemented in `ultimate_agent.py`.

## Inputs and user choices

- User selects **Bulk RNA-seq raw gene counts**; other modalities are visibly unsupported.
- Required: count matrix, sample metadata, metadata sample-ID field (or explicitly selected index), grouping variable, numerator level, reference level.
- Optional: batch covariate; any extension to paired/repeated-measures designs requires separate specification.
- User confirms that rows/columns represent biological samples and genes as displayed. Cell-level matrices are not assumed to be independent bulk samples.
- Gene ID namespace is user-declared when needed for downstream annotation; do not silently map or collapse IDs.

## Inspect without mutation

- Preserve original uploaded bytes (or their hashes), source shape, labels, and types.
- Detect duplicate and missing sample/gene identifiers before indexing or alignment.
- Compare metadata IDs to both count row labels and count column labels.
- If metadata IDs match exactly one dimension, propose orientation; if neither or both, require manual resolution. Report unmatched/extra IDs; never silently drop them.
- Validate all count values: numeric, finite, nonnegative, and integer-valued. Missing, negative, fractional, nonnumeric, or infinite values block analysis. Report example coordinates.
- Zero-count genes are valid source data; report their number and apply any later filtering only in a documented analytical copy.
- Validate selected contrast levels are distinct, exist in metadata, and have adequate independent replicates under the selected design. Do not derive numerator/reference from row order.
- Report dimensions, matched samples, group counts, candidate batch fields, and potential resource limits before compute.

## Preparation after confirmation

- Create a separate samples-by-genes analytical matrix; transpose only when orientation is confirmed.
- Reorder rows to a confirmed metadata sample order without changing source files.
- Record every exclusion, filter, transformation, and the resulting dimensions.
- Stop on ambiguous design or data type. A large matrix is a warning requiring confirmation, not proof of single-cell data.

## User-visible outcomes

- **Ready for confirmation:** valid input, proposed orientation, matched IDs, explicit contrast.
- **Needs user resolution:** ambiguous orientation, extra/missing IDs, uncertain experimental unit, unsupported design.
- **Blocked:** invalid count values, duplicate IDs, invalid contrast, or unsupported modality.
- **Analysis started:** only after the user confirms the intake summary and design.

## Open methodological decisions before production integration

1. Whether extra count samples can be explicitly excluded after user confirmation, and how to record that selection.
2. Minimum replicate requirements and treatment of unreplicated/paired/batch-confounded designs.
3. CSV import conventions (ID columns, delimiters, Ensembl version suffixes) and supported formats in v1.
4. Resource warnings/limits for sample count, gene count, file size, memory, and execution time.
5. Gene prefiltering thresholds and ranking statistic for GSEA (separate statistical decision).

## Acceptance criteria

- No source data overwritten or silently coerced.
- Orientation and contrast cannot be inferred from row order alone.
- Every blocked case reports a meaningful reason.
- Validation runs without Streamlit, OpenAI, network access, or PyDESeq2.
- Validated analytical output is separable from raw input and accompanied by a transformation log.
