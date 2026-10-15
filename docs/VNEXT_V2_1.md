# OmicsGPT vNext V2.1 — independent differential-expression engine

This checkpoint introduces a **standalone** PyDESeq2 wrapper and filtering audit. It does not connect to Streamlit and does not authorize computation by itself.

- Requires explicit numerator/reference, aligned samples, and a prevalidated scientific design.
- Uses an explicit low-count filter: at least `min_count` counts in at least `ceil(min_fraction * number_of_samples)` samples.
- Keeps original input matrices unchanged and reports excluded gene IDs.
- Raises errors rather than substituting a t-test or modifying invalid values.
- Records the engine version, model specification, contrast, and elapsed runtime.

**Outstanding before production:** confirm filtering policy, implement authorization-to-engine integration, add an independent R DESeq2 comparison, validate categorical covariate handling and model diagnostics, and benchmark larger datasets. The current function must not be exposed as an unrestricted user action.
