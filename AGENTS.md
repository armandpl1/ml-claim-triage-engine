# AGENTS.md

Standards for anyone changing this repository, human or AI agent. Follow them
on every change.

## Workflow

1. **Branch off the latest `main`** and open a pull request. CI (lint, type
   check, tests) must pass before merging.
2. **Plan, build, verify, document, then commit**, in that order.
3. **Verify with real runs, not assumptions.** Run the pipeline and the tests.
   Never report a metric that wasn't measured.
4. **Before committing:** `make ci` (ruff, mypy, tests). If data logic changed,
   also run `make data`, then `make push-data`.
5. **Commit messages:** a short summary line, a blank line, then bullet points
   saying what changed and why.
6. **No AI attribution.** No `Co-Authored-By` or "Generated with" lines in
   commits or pull requests. Commits appear as the repository owner only.
7. **Never rewrite published history** (force push) without the owner's
   explicit go-ahead.

## Data standards

- **Fail fast.** Every dataset that feeds features or training passes its data
  contracts first (`src/validation/`). A contract failure logs one structured
  alert and exits non-zero. Never weaken a contract to make a run pass
  without documenting why (see the exposure range in the config).
- **Config, not code:** thresholds, paths and splits live in `configs/`.
- **Deterministic:** the same config and seed give identical data. Seed
  per-record randomness from a stable hash of the record's key, never from
  Python's built-in `hash()` or row order.
- **Idempotent:** every pipeline step can run twice with the same result.
- **Versioned by DVC:** data is never committed to git. Pipeline stages are
  declared in `dvc.yaml`.
- **Honest data:** synthetic parts (the adjuster notes) are labeled as
  synthetic in the README. No invented labels presented as real.

## ML standards (Steps 3–8)

- **Split before computing any statistic,** so no information leaks from
  validation or test data into training.
- **Training and serving share one preprocessing pipeline,** so features are
  computed the same way in both.
- **Log every experiment to MLflow:** parameters, metrics and artifacts.
- **Gate promotion:** a model replaces `@champion` only if the evaluation gate
  passes (metric, slices, latency).

## Python standards

- **Python 3.11 with uv.** Add packages with `uv add` (or `uv add --dev`),
  never pip.
- **ruff for lint and format, mypy for types on `src/`.** Both run as
  pre-commit hooks (`make hooks`).
- **Tests:** `tests/unit/` mirrors `src/`. Add `tests/integration/` and
  `tests/behavioral/` as the later steps land. New logic needs a test.
- **Logging:** use `src.common.logging` for structured JSON events, not
  `print`.

## Documentation standards

- **README.md** is the source of truth for design, commands and results.
  Update its step status table as steps land.
- **Plain language:** short sentences, no jargon without explanation.
- **Markdown tables** have their `|` characters aligned in the raw file.

## CI/CD standards

- **Pin actions to full commit SHAs,** with the version in a comment.
- **Least privilege:** read-only token by default.
- **Every job has a timeout** and uploads what's needed to debug a failure.
- **Don't rename CI job names** without also updating the branch ruleset's
  required checks.
- **Validate workflow changes** with actionlint:
  `uvx --from actionlint-py actionlint .github/workflows/*.yml`

## Harnesses

| Tool                                        | Use it to                                          |
| ------------------------------------------- | -------------------------------------------------- |
| `make help`                                 | List every command                                 |
| `make ci`                                   | Run everything CI runs, locally                    |
| `make data`                                 | Run the data pipeline (only changed stages re-run) |
| `dvc status`                                | See which pipeline stages are out of date          |
| `data/processed/validation_report.json`     | Latest contract results                            |
| `uv run python -m src.validation.contracts` | Re-check contracts on the warehouse                |
