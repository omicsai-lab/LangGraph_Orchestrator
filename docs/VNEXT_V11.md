# OmicsGPT vNext V1.1 — Intake usability checkpoint

This release updates the **new** `app.py` only; it does not change `ultimate_agent.py`, statistical methods, or deployed Docker images.

- Sample-ID suggestions use exact matches against both count axes; grouping suggestions use repeated categorical levels. Both remain user-editable.
- The app reuses user selections across ordinary Streamlit reruns, resetting controls when uploaded file content changes.
- The group-size display is limited to eight groups and warns about identifier-like grouping variables.
- The intake review presents a compact summary while retaining detailed previews and hashes.
- An expandable panel reports server-side intake and script timing. These are **not** Docker startup or browser load times.
- Source data are never modified by the suggestion helper. Validation and explicit approval remain mandatory.

## Local verification

Run `python -m py_compile app.py omicsgpt/intake_suggestions.py` and `python -m pytest -q tests/`.

Start the **new** app in Docker and verify that `Sample` and `condition` are proposed with the breast cancer fixture. Change the sample-ID selection deliberately, then change an unrelated widget; confirm the override remains until the uploaded source files change. Correct the selection and confirm the 187-sample intake.

## Known limitations

The default contrast is still only a suggestion: researcher must explicitly review test/reference. Initial browser startup timing must be measured separately. No differential-expression or LLM calls are implemented in vNext V1.1.
