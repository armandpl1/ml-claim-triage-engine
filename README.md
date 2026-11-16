# ml-claim-triage-engine

Real-time insurance claim triage: predict a claim's cost and whether it needs a senior adjuster, and classify the adjuster's note into a claim category. Everything runs locally on a laptop CPU with open-source tools.

| Status  | Step                                           |
| ------- | ---------------------------------------------- |
| Done    | 1. Engineering foundation                      |
| Done    | 2. Idempotent ingestion and data contracts     |
| Planned | 3. Leakage-free feature pipelines              |
| Planned | 4. Experiment tracking, tuning and fine-tuning |
| Planned | 5. Champion vs challenger model gate           |
| Planned | 6. Three-tier test suite                       |
| Planned | 7. FastAPI serving in Docker                   |
| Planned | 8. Observability, drift monitoring and CI/CD   |

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make install   # dependencies from uv.lock, plus the git hooks (ruff and mypy before every commit)
make data      # download, stage and validate the data (~2 min the first time)
make test      # all tests
make help      # every command
```

## Data

| Source                      | What it is                       | Rows    |
| --------------------------- | -------------------------------- | ------- |
| freMTPL2freq (OpenML 41214) | French motor insurance policies  | 678,013 |
| freMTPL2sev (OpenML 41215)  | Claim amounts, one row per claim | 26,639  |

The two are joined into one row per claim in `staging.claims` (26,444 claims). Every claim gets:

- **`claim_id`**, a deterministic primary key: `CLM-<policy_id>-<n>`, numbering a policy's claims by amount.
- **`is_high_cost`**, the triage target: amount above €5,000 (4.6% of claims). The threshold is in `configs/data_config.yaml`.
- **`adjuster_note`** and **`claim_category`**: a short note and its category (collision, theft, glass, weather, bodily injury), the NLP task.

**The notes are synthetic.** freMTPL2 has no text, so `src/ingestion/notes.py` generates one note per claim:
- **From real features:** the category's chances depend on the claim amount, vehicle age and population density. Bodily injury is far more likely on expensive claims, glass on cheap ones.
- **Severity wording:** the note's damage description follows the amount.
- **Realistic noise:** 8% of notes contain a typo or adjuster shorthand.
- **Deterministic:** each note is seeded by its `claim_id`, so the same claim always gets the same note on any machine.

## Step 1: Engineering foundation

| Piece                      | What it does                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `pyproject.toml`, uv       | Pinned dependencies (`uv.lock`), Python 3.11 (`.python-version`)                             |
| Pre-commit hooks           | ruff lint, ruff format and mypy run before every commit and in CI                            |
| DVC                        | Versions the data. `dvc.yaml` declares the pipeline; `dvc repro` re-runs only changed stages |
| DVC remote                 | `/tmp/dvc-storage`, a local folder standing in for cloud storage such as S3                  |
| `.github/workflows/ci.yml` | Lint and type check, then all tests, on every pull request                                   |

How Step 1 was verified:
- **Hooks block bad commits.** A badly formatted file was rejected by ruff, and a function returning text where it declares `int` was rejected by mypy. Neither reached the history.
- **Data round trip.** A fresh clone rebuilt the environment from `uv.lock`, pulled the data from the DVC remote byte-identical to the original, and passed every test.

`/tmp` is cleared on reboot, so the local DVC remote is temporary by design. Re-run `make data && make push-data` to repopulate it, or point the remote elsewhere with `dvc remote modify localstore url <path>`.

## Step 2: Idempotent ingestion and data contracts

**Ingestion** (`src/ingestion/fetch_data.py`):
1. Downloads both datasets from OpenML, cached as Parquet in `data/raw/`. Later runs skip the download (`--force` refreshes it).
2. Cleans columns: snake_case names, integer IDs, and quotes stripped from values such as `'Regular'`.
3. Joins claims to policies. **195 claims (0.7%) have no matching policy.** They are dropped and logged as a warning, not lost silently.
4. Adds `claim_id`, the target and the notes, then writes `raw.policies`, `raw.claim_amounts` and `staging.claims` to `data/warehouse.duckdb`. Tables are replaced on every run, so running twice gives identical tables.

**Contracts** (`src/validation/contracts.py`, built with Pandera). The pipeline stops before bad data reaches features or training. All thresholds are in `configs/data_config.yaml`.

| Kind    | Checks                                                                                                           |
| ------- | ---------------------------------------------------------------------------------------------------------------- |
| Column  | Types and nulls; ranges such as `driver_age` 16–100; allowed areas and fuel types; ID formats; unique `claim_id` |
| Dataset | At least 1,000 rows; high-cost share between 2% and 40%; every note category at least 3% of claims               |
| Lineage | At most 2% of source claims dropped as orphans                                                                   |

On failure, one structured JSON alert lists every broken check with example bad values, and the process exits with status 1:

```json
{"level": "ERROR", "logger": "validation", "event": "contract_failed", "table": "staging.claims",
 "failures": [{"column": "driver_age", "check": "in_range(16, 100)", "failing_rows": 1, "examples": ["15"]}]}
```

A report is written to `data/processed/validation_report.json` either way.

**Tests** (`tests/unit/`) prove each contract rejects its own kind of bad data:
- an underage driver
- a negative amount
- an unknown category
- a malformed, missing or duplicate ID
- an imbalanced target or a missing category
- too few rows, too many orphans

They also prove that ingestion is deterministic and independent of source row order.

**One finding about the real data:** 54 claims have more than one year of exposure (up to 2.0). That's a known quirk of freMTPL2, so the exposure contract allows up to 2.01 rather than failing on real data.

## Project layout

```
configs/data_config.yaml    sources, paths, target threshold, every contract threshold
src/common/                 config loading, structured JSON logging
src/ingestion/              fetch_data.py (download, clean, join, store), notes.py
src/validation/             contracts.py (Pandera contracts, fail-fast alerts)
tests/unit/                 ingestion, notes and contract tests
dvc.yaml, dvc.lock          data pipeline and the versions of its outputs
data/                       raw downloads, warehouse, reports (versioned by DVC, not git)
Makefile                    single-command execution (make help)
```
