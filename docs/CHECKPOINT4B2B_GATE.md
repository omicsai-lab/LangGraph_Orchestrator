# Checkpoint 4B.2b — Pure authorization gate (stage 1)

This package adds a fail-closed authorization primitive, **not yet a Streamlit integration**.

- `approve` requires explicit confirmation and fingerprints the exact configuration.
- `require_authorization` raises if no approval exists or the current configuration differs.
- The Streamlit button must be disabled when unapproved **and** the execution handler must call `require_authorization` immediately before starting computation.
- Approval must not be stored independently of the source uploads; always recompute the current configuration from the actual uploaded bytes and current design selections.
- A validated handoff must use the inspected and prepared data, not the legacy `pd.read_csv` + `fillna(0).round().astype(int)` route.
- Existing `ultimate_agent.py` is deliberately unchanged by this package. Do not claim the application is gated until the next integration checkpoint and UI tests pass.

The confirmation schema currently does not include every computation-affecting setting (e.g., DE engine and prefiltering settings). Before allowing production computation, expand the schema and its tests to cover those settings or bind an immutable versioned analysis specification separately.
