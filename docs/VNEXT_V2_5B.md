# OmicsGPT V2.5B — Interactive results explorer

This checkpoint adds a display-only Plotly volcano plot, explicit candidate highlighting, gene search, accessible definitions, and candidate CSV export to the already approved Streamlit results screen.

## Boundaries

- **No changes** to statistical fitting, intake, approval, run-state management, or analysis manifests.
- Results are rendered only when `current_result` matches the active approved configuration fingerprint.
- Presentation widgets do not invoke `execute_approved_analysis` or mutate scientific configuration.
- Selection rules come from the tested `omicsgpt.results_explorer` module; selection is exploratory, not biomarker validation.
- Balanced strategy selects up to N per direction (up to 2N total); other automatic strategies select up to N total.
- The volcano y-axis is raw or adjusted p-value, while point categories are based on adjusted p-value (FDR).
- Zero p-values are capped **only for display**; missing/unplottable values are counted and disclosed.
- Plotly is required by the Docker environment. All results remain session-local.

## Verification

1. Run `python -m py_compile app.py omicsgpt/results_panel.py`.
2. Run `python -m pytest -q tests/test_results_panel.py` and `python -m pytest -q tests/`.
3. Launch the existing Docker Streamlit workflow with the repository bind-mounted.
4. Execute the approved 187-sample benchmark, verify 75 retained genes, explore all selection methods, hover, search, and download.
5. Change display controls: verify **no model refit** and no approval invalidation.
6. Change the statistical filter: verify approval and prior results invalidate as before.

## Future work

Pathway, external evidence, and multidimensional profiles remain separate future milestones. Do not infer biological validity from DE ranking alone.

## V2.5B.1 presentation refinements

- Strategy explanations appear immediately below the strategy control. Balanced selects up to N **per direction**; other automatic strategies select up to N **total**. Manual selection does not apply the automatic FDR filter.
- Readable tables use scientific notation below 0.001; computational zeros and missing values are labeled. The numerical DE results and downloadable source CSV remain unchanged. Formatted table strings are presentation-only and should not be used for numerical computation.
- Gold star markers with contrasting borders highlight selected exploratory candidates across light and dark themes. Selection is also conveyed by shape.
- No changes to selection rules, statistical fitting, approval, or session-state logic.
