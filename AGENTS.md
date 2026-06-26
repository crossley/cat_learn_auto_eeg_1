# Agent Instructions

## Purpose and Traceability

This file tells coding agents how to work in this repository.

The priority is scientific traceability. Here, traceability means that a
competent scientist who is not a specialist programmer can read the code and
understand what was done.

Agents must preserve these principles:

- Do not modify raw data.
- Keep code simple, explicit, and auditable.
- Make outputs traceable to one script and one clear parameter set.
- Label exploratory variants clearly.
- Prefer clear failure over hidden fallback behavior.
- Follow `README.md` and existing project style before inventing patterns.

## Communication

- Be brief.
- Leave the user asking for more rather than saying too much.
- State assumptions only when needed.
- Avoid interpreting scientific results unless explicitly asked.
- Give commands clearly.
- Wrap normal prose around 70 characters when writing console-style text.
- Do not wrap commands, paths, code, tables, or tracebacks when wrapping would
  make them harder to use.

## Repository Map

Important source folders:

- `code/`: analysis, figure, and helper scripts.
- `tests/`: smoke tests and small validation tests.
- `scripts/`: maintenance scripts.
- `README.md`: project layout and current analysis inventory.

Important local data folders:

- `Behavioural/`
- `EEG/`
- `EEG_epo/`
- `EEG_epo_unmatched/`
- `task_eeg_preprocessed/`

Generated output folders:

- `output/`
- `figures/`
- `logs/`

Raw data folders must be treated as read-only unless the user explicitly says
otherwise.

## Code and Analysis Design

The project uses a flat script layout.

- Keep analysis and figure entry points directly in `code/`.
- Do not create analysis-family subdirectories unless explicitly approved.
- Do not reorganize the project into a package without explicit approval.
- Prefer one-purpose scripts with descriptive names.

- Keep analysis scripts readable to a scientist.
- Prefer direct readable code over small helper functions.
- Do not create helper functions for trivial operations.
- Do not hide simple calculations, such as standard errors, behind helpers.
- Use helper functions only when they genuinely improve human readability,
  reduce meaningful duplication, or reduce risk of error.
- Reuse existing loaders and established project conventions.
- Do not create alternative loaders or path-discovery logic without approval.
- Keep `_analysis.py` scripts for computation and saved outputs.
- Keep `_figure.py` scripts for plotting saved outputs.
- Do not use all-in-one runner scripts unless explicitly requested.
- Do not use CLI flags for ordinary analysis settings.
- Prefer constants and parameters near the top of scripts.

The project should be strict about data layout and assumptions.

- Fail fast when expected files, columns, metadata, or formats are missing.
- Do not silently search alternate locations.
- Do not infer missing metadata.
- Do not engineer complicated fallbacks around broken inputs.
- Do not patch around missing data unless the user explicitly approves it.

## Outputs, Figures, and Logs

Main analyses should use stable, descriptive output names.

Exploratory or diagnostic variants must include a clear label in filenames.
They should be easy to identify and remove later.

Do not overwrite main outputs with a variant unless the user explicitly asks.

Write outputs into the established folders:

- tables, arrays, caches, and summaries: `output/`
- figures: `figures/`
- run logs: `logs/`

The generated-output layout is also flat.

- Write output files directly in `output/`.
- Write figure files directly in `figures/`.
- Write log files directly in `logs/`.
- Do not create analysis-family subdirectories under these folders unless
  explicitly approved.
- Use filename prefixes and labels to show analysis family and variant.

Figure scripts should read saved outputs. Analysis scripts should write saved
outputs. Avoid scripts that do both unless there is a clear project precedent
or explicit user request.

When adding outputs, prefer a small set of useful files over many loosely
related files.

## Execution, Permissions, and Compute

Do not run heavy analyses unless the user asks.

When the user asks for code only, write code and give commands to run. Do not
run the analysis.

For long analyses, prefer giving the user a background command that writes a
log file. The user can ask for progress when desired.

Avoid repeated polling of long jobs in chat. It consumes context and is
expensive. Progress should be written to logs, not narrated repeatedly.

If an agent starts a long job, checks should be sparse and purposeful.

Use explicit worker counts and thread limits. Avoid accidental nested
parallelism. Prefer predictable compute behavior over maximum machine usage.

## Testing and Validation

Use validation proportional to the change.

For small code changes:

- run syntax checks where possible
- run a small smoke test if it is cheap
- check that expected output files are written when appropriate

For larger analysis changes:

- test on one or two subjects first
- check row counts and key columns
- inspect logs for skipped data or failures
- report any known limitations plainly

Do not treat a successful run as scientific validation. Report only the
engineering checks that were actually performed.

## Cleanup

Cleanup should be deliberate and reviewable.

- Never delete raw data.
- Use dry-run cleanup first.
- Prefer cleanup manifests over ad hoc deletion commands.
- Separate outputs into clear categories before deletion:
  - keep
  - remove
  - unknown / needs user decision
- Do not remove unknown files without user approval.
- Do not remove outputs from analyses that are still under discussion.

Cleanup scripts should default to dry-run mode.

## Git Discipline

- Check worktree status before commits or cleanup.
- Do not revert user changes unless explicitly requested.
- Do not commit raw data.
- Do not commit bulky caches unless explicitly requested.
- Keep commits focused.
- Summarize changed files and validation performed.

## Multi-Agent Coordination

Use family-level subagents for broad work. Do not use one subagent per script
unless the user explicitly asks.

The root agent coordinates:

- task interpretation
- shared conventions
- output naming
- final integration
- deciding what gets run
- cleanup manifests
- final summaries

Family agents may work within these areas:

- ERP
- MVPA
- Functional connectivity
- RSA
- Behavioral / decision-bound models
- HMM / state trajectory models
- RNN / sequence MVPA

Family agents should stay inside their analysis family.

Shared loaders, shared helpers, output naming conventions, and cleanup policy
require root-agent approval.

Subagents may classify cleanup candidates, but they should not delete files
directly.

Subagents should not run long analyses unless the user explicitly asks.
