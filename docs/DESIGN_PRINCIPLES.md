# OmicsGPT design principles (draft v1)

Status: proposed; confirm before changing the production analysis.

1. **Human-directed scientific intent.** The researcher declares data modality, experimental units, study design, primary comparison, and biological purpose. The system may propose technical interpretations but must request confirmation for scientifically meaningful choices.
2. **Statistical integrity.** Validated computational tools perform estimation. LLMs do not invent, alter, or substitute inferential results. A failed engine does not silently switch methods.
3. **Reproducibility by design.** Preserve source data, immutable analysis settings, package/model identifiers, computation outputs, and provenance. A changed statistical specification creates a new analysis run.
4. **Typed evidence provenance.** Distinguish computed results, externally retrieved records, user-provided materials, and model-generated interpretations; never label a model summary as a database lookup.
5. **Structured, verifiable reasoning.** AI assessments should cite specific evidence, retain disagreements and uncertainty, and be testable against expert reference assessments. Agreement is not calibrated probability.
6. **Modular extensibility.** Shared intake manifests and typed interfaces permit future modalities, but unsupported modalities must not be routed into the bulk RNA-seq workflow.
7. **Transparent collaboration.** Show detected structure, assumptions, transformations, warnings, and exclusions; obtain explicit confirmation before expensive analysis.
8. **Empirical evaluation.** Compare orchestration, retrieval, role-conditioned review, and ensembling against matched baselines using frozen evidence and prespecified metrics.
9. **Explicit failure.** Invalid input, unsupported designs, absent evidence, and computation/API errors should be reported, not concealed by fallback or fabrication.
10. **Quality before complexity.** Prioritize reliability, testing, auditability, and clear scope before adding modalities, agents, or evidence channels.

## Current release boundary

Bulk RNA-seq count-based differential expression is the only workflow to implement and validate in this phase. Single-cell RNA-seq, DNA, proteomics, and multi-omics integration are planned interfaces, not currently supported analysis paths.

## Separation of responsibilities

- **Intake:** inspect and validate source files without modifying them.
- **Preparation:** explicitly orient and align validated data, retaining a transformation log.
- **Computation:** apply user-confirmed design and documented statistical filtering.
- **Retrieval:** preserve identifiers, queries, timestamps, source responses, and access status.
- **Interpretation:** synthesize typed evidence without silently recomputing or redefining the contrast.
