# ml-claim-triage-engine

Real-time insurance claim triage: predict a claim's cost and whether it needs a senior adjuster, and classify the adjuster's note into a claim category. Everything runs locally on a laptop CPU with open-source tools.

| Status  | Step                                           |
| ------- | ---------------------------------------------- |
| Done    | 1. Engineering foundation                      |
| Done    | 2. Idempotent ingestion and data contracts     |
| Done    | 3. Leakage-free feature pipelines              |
| Done    | 4. Experiment tracking, tuning and fine-tuning |
| Done    | 5. Champion vs challenger model gate           |
| Done    | 6. Three-tier test suite                       |
| Done    | 7. FastAPI serving in Docker                   |
| Done    | 8. Observability, drift monitoring and CI/CD   |
| Planned | 9. LLMs vs the fine-tuned notes model          |
| Done    | Production storage: S3 (Terraform) + Postgres  |
| Done    | Batch scoring and delayed-label monitoring     |
| Done    | Orchestration with Dagster                     |
| Done    | Explainability, fairness and model card        |
| Done    | Point-in-time feature store (DuckDB + SQLite)  |

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make install   # dependencies from uv.lock, plus the git hooks (ruff and mypy before every commit)
make data      # download, stage, validate and split the data (~2 min the first time)
make train     # tune the triage model, fine-tune the notes model, benchmark (~1-2 h on a CPU)
make gate      # promote each challenger only if it beats the champion
make serve     # the API at http://localhost:8000/docs, models from the MLflow registry
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
- **Realistic noise**, so the task isn't trivially easy (with clean templates, any model scored 100%):
  - 10% of notes are vague ("Customer called to report damage at a supermarket car park. Assessment pending.") and 25% use wording that fits several categories.
  - 3% of labels are deliberately wrong, as when a claim is miscoded in the claim system.
  - About 3% of notes contain adjuster shorthand or a typo ("veh", "dmgd", "polcie"). The generator tries on 8% of notes, but only changes the note when the chosen word is in it.
- **Deterministic:** each note is seeded by its `claim_id`, so the same claim always gets the same note on any machine.

## Step 1: Engineering foundation

| Piece                      | What it does                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `pyproject.toml`, uv       | Pinned dependencies (`uv.lock`), Python 3.11 (`.python-version`)                             |
| Pre-commit hooks           | ruff lint, ruff format and mypy run before every commit and in CI                            |
| DVC                        | Versions the data. `dvc.yaml` declares the pipeline; `dvc repro` re-runs only changed stages |
| DVC remote                 | S3 bucket `claim-triage-dev-dvc` (KMS-encrypted, versioned), on the Moto emulator by default |
| `.github/workflows/ci.yml` | Lint and type check, then all tests, on every pull request                                   |

How Step 1 was verified:
- **Hooks block bad commits.** A badly formatted file was rejected by ruff, and a function returning text where it declares `int` was rejected by mypy. Neither reached the history.
- **Data round trip.** A fresh clone rebuilt the environment from `uv.lock`, pulled the data from the DVC remote byte-identical to the original, and passed every test.

The DVC remote lives on the AWS emulator (see Storage on S3 below), and the emulator's data is wiped when it stops. Run `make infra-up`, then `make push-data` to repopulate it. `.dvc/config` holds only the bucket and region, as for a real AWS account; `make infra-up` writes the emulator endpoint and its fake credentials to `.dvc/config.local`, which is not committed.

## Step 2: Idempotent ingestion and data contracts

**Ingestion** (`src/ingestion/fetch_data.py`):
1. Downloads both datasets from OpenML, cached as Parquet in `data/raw/`. Later runs skip the download (`--force` refreshes it).
2. Lands the downloads **untouched** in `raw.policies` and `raw.claim_amounts`, with the source's own column names and values, so any problem can be traced back to exactly what was received.
3. Cleans columns in memory: snake_case names, integer IDs, and quotes stripped from values such as `'Regular'`.
4. Joins claims to policies. **195 claims (0.7%) have no matching policy.** They are dropped and logged as a warning, not lost silently.
5. Adds `claim_id`, the target and the notes.
6. **Validates in memory.** If any contract fails, the run logs one alert and halts with exit status 1, and `staging.*` stays exactly as it was.
7. Only then writes `staging.policies` and `staging.claims`, in one transaction. Tables are replaced on every run, so running twice gives identical tables.

| Schema    | Tables                             | Contents                                                  |
| --------- | ---------------------------------- | --------------------------------------------------------- |
| `raw`     | `policies`, `claim_amounts`        | As downloaded, never modified                             |
| `staging` | `policies`, `claims` (primary key) | Cleaned and joined, written only after the contracts pass |

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

A report is written to `data/processed/validation_report.json` either way. The `validate` pipeline stage re-checks the stored tables and writes this report.

**Halt drill on the real data.** The pipeline was run through `dvc repro` with a deliberately strict config, requiring every driver to be at least 30 (`CLAIM_TRIAGE_DATA_CONFIG` points to an alternate config file). The results:
- DVC stopped with a non-zero exit status, and the `validate` stage never ran.
- One alert: `driver_age in_range(30, 100)`, with 4,256 failing claims and examples (24, 28, 22), followed by `pipeline_halted`.
- `staging.claims` was byte-for-byte unchanged.

The warehouse is a `persist: true` output in `dvc.yaml`. Without it, DVC would delete the last good warehouse before re-running the stage, so a failed run would leave no data at all.

**Tests** (`tests/unit/`) prove each contract rejects its own kind of bad data:
- an underage driver
- a negative amount
- an unknown category
- a malformed, missing or duplicate ID
- an imbalanced target or a missing category
- too few rows, too many orphans

They also prove that ingestion is deterministic and independent of source row order.

**One finding about the real data:** 54 claims have more than one year of exposure (up to 2.0). That's a known quirk of freMTPL2, so the exposure contract allows up to 2.01 rather than failing on real data.

## Step 3: Leakage-free feature pipelines

Leakage means a model sees information during training that it won't have when it predicts for real. Its scores then look better than it will perform. This step is built so leakage can't happen (`src/features/`, pipeline stage `features`):

1. **Split first.** Claims are split 70/15/15 into train, validation and test before any statistic is computed.
   - **Grouped by policy:** all claims of a policy land in the same split, so the model is never tested on a policy whose other claims it trained on.
   - **Stratified:** each split keeps the overall high-cost rate (4.62%, 4.61%, 4.64%).
2. **Fit on train only.** The preprocessor (median imputation for numbers, category codes for text columns) learns its values from the training split alone. Categories never seen in training map to "unknown" instead of failing.
3. **Exclude what isn't known at prediction time.** `claim_amount` (the answer), `claim_seq` (ranks a policy's claims by amount), and `claim_count` (counts claims that haven't happened yet). Each exclusion and its reason is in `configs/data_config.yaml`.
4. **One feature function for training and serving.** `log_density` is computed by the same code in both (`src/features/row.py`), and a test proves the API and the training pipeline produce identical model inputs.
5. **Save a reference snapshot** of the training inputs for drift monitoring (Step 8).

The stage writes `data/processed/feature_report.json` and fails if any policy appears in more than one split.

| Split      | Claims | High-cost share | Policies shared with another split |
| ---------- | ------ | --------------- | ---------------------------------- |
| Train      | 18,509 | 4.62%           | 0                                  |
| Validation | 3,966  | 4.61%           | 0                                  |
| Test       | 3,969  | 4.64%           | 0                                  |

The test split is touched only by the gate (Step 5), never during tuning.

## Step 4: Experiment tracking, tuning and fine-tuning

Every run is logged to **MLflow** (parameters, metrics, plots, the model), stored locally in `mlflow.db` and `mlartifacts/`. `make mlflow-ui` opens it. Trained models are registered in the MLflow **Model Registry**, and the newest version gets the alias `@challenger`.

**Model 1, the triage model** (`src/training/tabular.py`): is this claim high-cost?
- LightGBM, tuned by **Optuna** over 25 trials, scored by PR-AUC with 4-fold cross-validation grouped by policy. PR-AUC suits a rare target (4.6% positive) better than accuracy or ROC-AUC.
- The decision threshold maximises F1 on validation and is stored with the model.
- Exported to **ONNX** with its preprocessing in one graph. The ONNX and Python predictions differ by less than 2e-7.

**Model 2, the notes classifier** (`src/training/nlp.py`): which of 5 categories does this note describe?
- **DistilBERT fine-tuned with LoRA**: only small adapter matrices are trained (r 8, on the attention query and value layers), not the full 66M weights. 4,000 training notes, 2 epochs, on a CPU.
- Exported to ONNX, then **quantized to INT8**. INT8 is kept only if it loses at most 0.005 macro F1.

| Notes model              | Validation macro F1 | Size   |
| ------------------------ | ------------------- | ------ |
| DistilBERT FP32          | 0.9089              | 268 MB |
| DistilBERT INT8 (served) | 0.9088              | 67 MB  |
| TF-IDF + logistic reg.   | 0.9085              | < 1 MB |

The INT8 model agrees with FP32 on 99.97% of validation notes.

**Latency** (`src/training/benchmark.py`): one request at a time, 300 calls each, 2 threads (the API's setting).

| Model                         | p50     | p99     | Within the 50 ms p99 budget |
| ----------------------------- | ------- | ------- | --------------------------- |
| Triage model (LightGBM)       | 0.05 ms | 0.08 ms | Yes                         |
| Notes FP32                    | 31.6 ms | 56.2 ms | No                          |
| Notes INT8                    | 31.4 ms | 39.3 ms | Yes                         |
| Notes INT8, with tokenization | 31.4 ms | 41.0 ms | Yes                         |

INT8's gain here is a 4x smaller model and a shorter tail (p99 30% lower), not a faster median. With one thread, the INT8 p99 sat right at the 50 ms budget, so the benchmark, the gate and the API all use 2 threads (`export.inference_threads`).

### Honest findings

- **The triage model is weak, and that's the data.** Test PR-AUC is 0.077 against a 0.046 base rate: 1.7x better than guessing, which is useful for ranking claims but far from reliable. Policy features (car, driver, area) say little about how expensive a claim will be. The notes would help, but their severity wording is generated from the claim amount, so using them would be leakage, not skill.
- **DistilBERT doesn't beat the TF-IDF baseline** (`make text-baseline`; test macro F1 0.9092 vs 0.9087). The synthetic notes come from templates, so word counts already separate them, and the ceiling is set by the 3% wrong labels and the ambiguous notes. On real adjuster notes the transformer may earn its cost. On this data, the cheap baseline would be the right production choice. The fine-tuning pipeline is kept to show the full LoRA, ONNX and INT8 workflow.

## Step 5: Champion vs challenger model gate

`src/evaluation/gate.py` decides whether `@challenger` replaces `@champion`, scoring both on the untouched test split. All three checks must pass:

| Check          | Rule                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------- |
| Primary metric | At least 0.5% better than the champion (PR-AUC for triage, macro F1 for notes)                    |
| Slices         | No slice loses more than 10% PR-AUC (triage) or 0.02 F1 (notes), so an average can't hide a group |
| Latency        | Single-request p99 at most 50 ms                                                                  |

- **Triage slices:** young drivers, senior drivers, old vehicles, powerful vehicles, dense areas. A slice with fewer than 10 high-cost claims is skipped as too noisy.
- **Notes slices:** each category, and notes with shorthand or typos.
- **First model:** with no champion yet, the challenger must beat a naive baseline instead: 1.5x the positive rate (triage), or macro F1 0.5 (notes).
- **On promotion,** the old champion keeps the alias `@previous_champion`, so rollback is one alias change. A rejected version is tagged `gate_decision=rejected`. Every decision is an MLflow run and a JSON report in `artifacts/gate/`.

**What happened on the real models:**

| Model     | Challenger                   | Result                                                                  |
| --------- | ---------------------------- | ----------------------------------------------------------------------- |
| Triage v1 | First model                  | Promoted: test PR-AUC 0.0774 vs a 0.0695 floor; p99 0.07 ms             |
| Notes v1  | First model                  | Promoted: macro F1 0.9087 vs a 0.5 floor; p99 41.9 ms                   |
| Triage v2 | v1 plus monotone constraints | **Rejected**: PR-AUC 0.0738 vs 0.0774; dense areas slice 0.125 -> 0.100 |

Triage v2 forces two business rules on the model: more vehicle power never lowers the predicted risk, and an older vehicle never raises it. It scored *better* on validation (PR-AUC 0.0912 vs 0.0898) but worse on the untouched test split. That's exactly the regression the gate exists to catch, so v1 stays in production.

The notes "with typos" slice: a bug made "veh" match inside "vehicle", so the first notes gate report counts 1,122 such notes. Fixed and tested (`tests/unit/test_scoring.py`); there are really 118, and the champion scores 0.9033 macro F1 on them vs 0.9087 overall. This first gate didn't check slices (no champion yet), so the decision was unaffected.

## Step 6: Three-tier test suite

`make test` runs all three tiers (104 tests).

| Tier        | Folder               | What it proves                                                                                                                                                      |
| ----------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unit        | `tests/unit/`        | Contracts reject bad data; the split doesn't leak; the gate's rules; the API rejects bad input; training and serving compute identical features and tokens          |
| Behavioral  | `tests/behavioral/`  | Models behave sensibly: with the monotone rules, more power never lowers risk; non-feature columns change nothing; typos and capitals barely move a note's category |
| Integration | `tests/integration/` | 100 claims through the whole pipeline in under 15 s: ingest, validate, split, train, ONNX, the real API, the inference log, with identical training and API results |

The triage behavior tests train a small model on fixture data, so they always run, including in CI. They also run on the trained model when it was trained with the monotone rules. The notes tests need the trained notes model, and skip without it (a fresh clone, or CI).

## Step 7: FastAPI serving in Docker

`src/serving/app.py`:

| Endpoint               | Returns                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `GET /health`          | Model versions, where they were loaded from, uptime                                          |
| `POST /predict`        | High-cost probability and route (`senior_adjuster` or `standard`), plus the note's category  |
| `POST /predict/policy` | The same for a claim sent with only its `policy_id`: features from the online feature store  |
| `GET /metrics`         | Prometheus metrics: requests by status, latency histogram, routes, note categories, versions |

- **Input validation:** request limits mirror the data contracts. Out-of-range or unknown fields get a 422 naming each bad field.
- **Models load once at startup,** from the MLflow registry (`@champion`) or from a bundle folder exported by `make bundle`. Only ONNX Runtime and the `tokenizers` library run them: no PyTorch, scikit-learn or pandas in the image.
- **Every prediction is logged** to a SQLite inference log in a background task, after the response is sent, so logging adds no latency.
- **Docker:** a multi-stage `Dockerfile` installs only the `serve` dependency group, runs as a non-root user, and has a health check. `make up` starts the API, MLflow, Prometheus and Grafana with `docker compose`.

**Verified locally:** `/health` reported both champions loaded from the registry; one claim with a note took 22.9 ms end to end; a bad request returned 422 naming each bad field; `/metrics` counted all of it; the prediction landed in the inference log. Serving from the bundle gave the same prediction to six decimal places.

One bug found this way: MLflow stores the notes model's weights in a separate file beside the `.onnx` graph, and the bundle copied only the graph. The bundle now saves each model as one self-contained file, with a regression test.

## Step 8: Observability, drift monitoring and CI/CD

**Dashboards.** Prometheus scrapes `/metrics`. Grafana (http://localhost:3000 after `make up`) has a provisioned dashboard: request rate, error share, p95/p99 latency, share routed to senior adjusters, note categories and model versions.

**Drift** (`monitoring/drift_detector.py`, `make drift`) compares logged requests with the training reference using **Evidently**:
- each input feature (Wasserstein distance for numbers, Jensen-Shannon for categories on samples this size)
- words per note (the text the notes model sees)
- predicted note categories vs training categories (prediction drift)

It writes `reports/drift_report.html` and `.json`, and exits 1 with a `drift_alert` log line when 30% or more of the monitored columns drift. With fewer than 200 requests it exits 2: too few to judge.

Evidently 0.7 doesn't run on pandas 3, which the rest of the project uses. So the detector has its own pinned environment (declared in the script, locked in `drift_detector.py.lock`), the way a drift job runs as its own scheduled container.

**Drill** (`make simulate`, then `make drift`, 500 requests each):

| Traffic                                                                           | Drifted columns | Result        |
| --------------------------------------------------------------------------------- | --------------- | ------------- |
| Test-split claims, unchanged                                                      | 0 of 12         | No alert      |
| Shifted: vehicles 8 years older, drivers 12 years younger, 6x density, 80% diesel | 4 of 12         | Alert, exit 1 |

The alert named exactly the four shifted columns. A first version alerted on the *unchanged* traffic: it applied a p-value threshold (0.05) to distance tests, which use 0.1. Each test now uses its own default threshold.

**CI/CD** (`.github/workflows/`):

| Workflow            | When                             | What                                                                                                        |
| ------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ci.yml`            | Every pull request, push to main | Lint and types; the drift detector's lockfile and a no-data run; all tests; build the image, import the API |
| `cd-model-gate.yml` | Weekly, or on demand             | Rebuild the data, retrain the triage model (8 trials), run the gate; upload the model, metrics and report   |

Actions are pinned to commit SHAs, the token is read-only, every job has a timeout.

## Step 9 (planned): LLMs vs the fine-tuned notes model

**Question:** should a claim-notes classifier be a fine-tuned small model, a
TF-IDF baseline, or a large language model with no training at all?

- Classify the same test notes with Claude Haiku 5.5, Sonnet 5.5 and
  Opus 5.5 (zero-shot and a few examples in the prompt), next to DistilBERT
  INT8 and TF-IDF
- Compare **quality** (macro F1 with bootstrap confidence intervals, 3
  repetitions per LLM to measure run-to-run variation), **latency** (p50, p99)
  and **cost per 1,000 claims**
- **Label efficiency:** retrain DistilBERT and TF-IDF on 50, 200, 1,000 and
  4,000 labeled notes. How many labels does a trained model need before it
  beats a zero-shot LLM that needed none?
- Honest limit: TF-IDF already matches DistilBERT here (test macro F1 about
  0.909) and the ceiling is set by the 3% wrong labels in the synthetic notes,
  so no model should be expected to win clearly on accuracy. The useful
  answers are cost, latency and labels needed
- No API cost in tests or CI: model calls are recorded once and replayed
- Uses the shared LLM and eval code planned for
  [data-ml-toolkit](https://github.com/armandpl1/data-ml-toolkit) v0.2.0
  (model layer with record/replay, bootstrap statistics), built first in
  [agent-eval-lab](https://github.com/armandpl1/agent-eval-lab)

## Production storage: S3 with Terraform, MLflow on Postgres

Steps 1-8 kept everything on the laptop: the DVC remote in `/tmp` and MLflow in a SQLite file plus a folder. This adds storage shaped the way a team would run it. It runs on **Moto**, an open-source AWS emulator, so it needs no AWS account.

| Piece                          | What it does                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `infra/terraform`              | S3 buckets for the DVC remote and MLflow artifacts, plus one for access logs; a KMS key; IAM roles                  |
| Bucket settings                | Versioning on, KMS encryption with key rotation, public access blocked, access logs, old versions expire in 90 days |
| `training` role                | Push dataset versions (no deletes in the DVC remote), read and write model artifacts                                |
| `serving` role                 | Read model artifacts only                                                                                           |
| DVC remote                     | `s3://claim-triage-dev-dvc`                                                                                         |
| MLflow server (`infra/mlflow`) | Runs, metrics and the registry in **Postgres**; artifacts in **S3**, sent through the server                        |
| `src/infra/storage_check.py`   | Reads the buckets and roles back from AWS and fails on any unsafe setting                                           |
| `src/infra/mlflow_check.py`    | Logs a run, registers a version, downloads it back and finds the file in S3                                         |

```bash
make infra-up       # start Moto, apply Terraform twice, fail if a third plan shows drift
make infra-check    # buckets versioned, encrypted, private, logged; roles least privilege
make push-data      # dataset versions to the S3 bucket
make mlflow-up      # Postgres + MLflow server at http://localhost:5001
make mlflow-check   # smoke test of the server, end to end
MLFLOW_TRACKING_URI=http://localhost:5001 make train-tabular   # train against it
make infra-down     # stop everything; the emulator's and Postgres's data is wiped
```

Design choices:
- **Clients hold no S3 credentials for MLflow.** The server proxies artifacts (`--serve-artifacts`), so training and the API only need the tracking URL. Only the server can reach the bucket.
- **The committed config is the real-account config.** `.dvc/config` holds only the bucket and region. `make infra-up` writes the emulator's endpoint and fake credentials to `.dvc/config.local`, which is not committed.
- **The local SQLite store is still the default.** Training uses `mlflow.db` unless `MLFLOW_TRACKING_URI` is set, so nothing changes for existing runs.

How it was verified:
- Terraform `fmt` and `validate` pass, a Trivy scan finds 0 misconfigurations, and a plan after the second apply shows no changes.
- `make infra-check` passes. Its unit tests run on moto's in-memory AWS and show that each unsafe setting is reported: versioning off, no public access block, no version expiry, wildcard or destructive actions, deletes in the DVC remote, writes in the serving role.
- `make push-data` pushed 12 files. A fresh clone pulled them back byte-identical, and the objects in the bucket are KMS-encrypted.
- `make mlflow-check` passes: the run is stored in Postgres, the artifact lands in the bucket, and the registered version downloads back unchanged.
- **Real training against the server.** `make train-tabular` with `MLFLOW_TRACKING_URI=http://localhost:5001` (5 min 26 s, memory-capped) logged its 25 Optuna trials to Postgres, stored `model.onnx` in the bucket and registered version 1 as `@challenger`. Every metric matched the committed `artifacts/tabular/metrics.json` exactly (validation PR-AUC 0.0912); only the run ID differed.

Limits, stated honestly:
- **Moto doesn't enforce IAM.** The roles are checked by reading their policies back, not by being denied an action. A real account would enforce them.
- **MLflow memory.** MLflow 3 starts background job workers for its GenAI features. They pushed the server past its 768 MB limit, so they are turned off (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`). The server then uses about 400 MB, and Moto, Postgres and MLflow together about 570 MB.
- **Postgres credentials are local only** (`mlflow`/`mlflow` in the compose file). A real deployment reads them from a secret store.
- **The serving stack (`make up`) still uses the local SQLite store.** Moving the API to the server is part of the Kubernetes step.

## Batch scoring and delayed-label monitoring

Drift checks (Step 8) compare inputs. They can't tell whether the model is still *right*: that needs the real outcome, and a claim's final cost is known only when it settles, weeks or months later. This adds the nightly batch path and monitors real quality as outcomes arrive.

The held-out test claims stand in for new claims. `configs/batch_config.yaml` spreads them over 56 days of reports (about 71 a day). Each one settles 14-60 days later, plus 45 more if it is expensive, as in real claims handling: high-cost claims settle in a median of 81 days, the rest in 37.

| Job                        | What it does                                                                                   |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| `src/batch/score.py`       | Scores one day's claims with `@champion`; features as of the report date; idempotent per day   |
| `src/batch/labels.py`      | Loads every outcome known by a date; optional `inflation` scenario                             |
| `src/batch/performance.py` | Joins predictions to outcomes in two-week cohorts and alerts on calibration, recall, precision |
| `src/batch/backfill.py`    | Replays every night from an empty store; records when the first alert fired                    |
| `data/batch/batch.duckdb`  | `predictions`, `outcomes` and an audit table of job runs (git-ignored)                         |

```bash
make batch-score DATE=2025-01-06        # one night's scoring
make batch-backfill                     # the whole period, normal claims (~1.5 min)
make batch-backfill SCENARIO=inflation  # claim costs rise 30% from day 28
```

How a cohort is judged:
- **Only once settled.** A cohort is judged 105 days after its last report date (the slowest settlement), and only if it holds 20+ high-cost claims. A first version judged cohorts at 95% label coverage and raised false alarms: the missing 5% were the expensive claims, so cohorts looked half as costly as they were.
- **Calibration:** is the realized high-cost rate within the 99% Wilson interval of what the model expected (its mean probability)?
- **Recall and precision:** alert only if the whole 99% interval is below the baseline.
- **Baseline:** the champion on the validation split (precision 0.114, recall 0.191, PR-AUC 0.090). The test split is the monitored stream, so it can't be the baseline.
- **99%, not 95%:** each weekly check runs up to 12 tests. At 95%, normal data raised a false alarm (a cohort with 60 high-cost claims where the model expected 44, about a 1-in-80 fluke).

Results (champion v1, `reports/batch/backfill_*.json`):

| Scenario  | Weekly checks | Alerts | First alert | Detail                                                                 |
| --------- | ------------- | ------ | ----------- | ---------------------------------------------------------------------- |
| Normal    | 24            | 0      | none        | Realized high-cost rate within the interval for all 4 cohorts          |
| Inflation | 24            | 1      | 2025-06-15  | Last cohort: high-cost rate 7.2% vs 4.6% expected; 184 → 229 high-cost |

Honest findings:
- **Inflation is invisible to drift checks.** The features don't change, only the costs, so only outcomes can reveal it. 45 more claims crossed the 5,000 EUR threshold.
- **Detection took 132 days** (inflation from 2025-02-03, alert on 2025-06-15). Waiting for labels to settle is slow. The first inflated cohort (6.1% vs 4.5%) missed the 99% interval narrowly. Faster options: judge early outcomes against how early outcomes usually look at the same age, or pool cohorts after a suspected change.
- **Recall fell a little but didn't alert.** In the two inflated cohorts recall went from 0.143 and 0.149 to 0.129 and 0.125. With 40-70 high-cost claims per cohort, a recall interval spans about 0.05-0.28, so only a large drop is provable.

## Orchestration with Dagster

The steps above ran as separate `make` targets. Dagster runs them as one system: assets with lineage, jobs, a nightly schedule, and a sensor that turns an alert into a retraining run. `make orchestrate` opens the UI at http://localhost:3001.

| Piece                         | What it does                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `retrain_tabular` job         | `training_data` (dvc repro) → `tabular_challenger` → `tabular_gate` → `serving_bundle`         |
| `retrain_notes` job           | The same for the notes model; kept apart because fine-tuning takes about an hour on a CPU      |
| `nightly_batch` job           | `batch_predictions` → `claim_outcomes` → `model_performance`, one partition per night          |
| `no_performance_alerts` check | Fails (as a warning) when a settled cohort alerts; `no_feature_drift` does the same for drift  |
| `retrain_on_alert` sensor     | Requests `retrain_tabular` when performance is degraded or features drift, at most once a week |
| `log_failed_runs` sensor      | Logs every failed run as one structured event                                                  |
| Nightly schedule              | 02:00, built from the partitions; stopped by default, like the retrain sensor                  |

```bash
make orchestrate                 # the UI: lineage, runs, backfills, schedule, sensors
make nightly DATE=2025-06-15     # one night of the batch path
make retrain                     # data, train, gate, bundle (~6 min, memory-capped)
```

Design choices:
- **Every step is a subprocess** running the Makefile's command, so the Dagster process stays small and an out-of-memory training run can't take it down. `make retrain` caps training at 2.5 GB (`MEMORY_CAP`), as for every heavy job here.
- **A rejected challenger is an outcome, not a failure.** The gate asset records `promoted: false`; the run succeeds and the old champion keeps serving.
- **Nights run one at a time, in order.** The outcomes table holds everything known *by* a night, so two nights in parallel would race. Batch runs carry a tag limited to one concurrent run (`src/orchestration/dagster.yaml`).
- **One retrain per week at most.** A degraded model alerts every night until it's fixed; one retrain, judged by the gate, is the useful response.
- **The bundle export can't break serving.** The new bundle is built next to the old one and swapped in only when complete. Before this change, a failed export deleted the working bundle first.

How it was verified:
- **Alert to retrain, end to end.** After the inflation backfill, `make nightly DATE=2025-06-15` (scenario `inflation`) recorded `model_performance` as degraded and the `no_performance_alerts` check failed as a warning while the run succeeded. `dagster sensor preview retrain_on_alert` on the same instance then requested one `retrain_tabular` run, reason `model_performance=degraded`.
- **A real retrain through Dagster** (`make retrain` against the Postgres + S3 MLflow server, so the local registry stayed untouched) took 4 minutes: `training_data` 3.4 s (nothing to re-run), `tabular_challenger` 3 min 7 s, then the gate. With no champion on that registry, it promoted the challenger for beating the naive baseline (PR-AUC 0.0738 vs 0.0695) within the latency budget, and `serving_bundle` exported a tabular-only bundle.
- Unit tests run the jobs with a fake command runner (command order, rejection vs failure, memory cap), real nights on fixture claims (scoring, outcomes, the check), and the sensor's rules (trigger, each result used once, cooldown) against a test Dagster instance.

Limits, stated honestly:
- **Retraining on the same data can't fix inflation.** The retrain job trains on the same training split. To learn the new costs, the training data has to include the newly settled claims, which is a data change, not a scheduling one. The sensor and the gate make the loop safe; they don't make it smart.
- **The schedule runs on 2025 dates.** The simulation's nights are in 2025, so the schedule would find nothing to run today. Backfills over the partitions (in the UI) replay the period, like `make batch-backfill`.

## Explainability, fairness and model card

Every routed claim now says why, the routing is audited by driver age and region, and the champion has a model card generated from measured numbers: [docs/model_card_tabular.md](docs/model_card_tabular.md).

```bash
make batch-score DATE=2025-01-06   # predictions plus reason codes in data/batch/batch.duckdb
make fairness                      # audit by driver age and region: reports/fairness/tabular.json
make model-card                    # regenerate docs/model_card_tabular.md from the champion
```

**Reason codes from the served model.** The registry serves ONNX, which can't attribute a score to features. LightGBM can: TreeSHAP is built in (`pred_contrib`), so no new library is needed, and it splits each claim's score (log-odds) exactly into one contribution per feature. `src/explain/champion.py` rebuilds the champion as LightGBM from its logged recipe (hyperparameters, monotone rules, seed) on the training split, and then proves it is the same model: its probabilities must match the served ONNX within `explain.parity_tolerance` (1e-5). The check runs on the validation split at load and again on each night's claims before anything is written. A mismatch raises `ParityError`, because explaining a different model than the one that routed the claim would be worse than no explanation. The serving bundle now carries the recipe, so this works without the registry too.

| Piece           | What it does                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| `src/explain/`  | Rebuild and parity check (`champion.py`); contributions, reason codes, global importance (`reasons.py`) |
| `reasons` table | Each scored claim's top 3 features that raised its risk (above 0.01 log-odds), with the claim's value   |
| Fairness audit  | Recall, routing rate, false-positive rate, precision and calibration per driver-age band and region     |
| Model card      | Intended use, recipe, test performance, importance, fairness and limitations, all measured              |

**Fairness rule.** The judged question is equal opportunity: do high-cost claims from every group reach a senior adjuster at the same rate? A group is flagged when its recall's 95% Wilson interval lies entirely below the overall recall. Groups with fewer than 10 high-cost claims are reported but not judged, and small regions are pooled. Driver age and region are standard motor rating factors; the audit checks the routing outcome, not whether a factor may be used. freMTPL2 has no gender or other protected attributes.

How it was verified (champion v1, test split, 3,969 claims):
- **Parity:** the rebuilt champion matched the served ONNX model to 1.46e-7 on 3,966 validation claims, from both the registry and the bundle. It was trained before the monotone rules, so its run logged none, and a rebuild with today's rules differed by 0.033: the check would have caught it.
- **Exact attributions:** contributions summed back to each claim's probability to 1e-15. Reason codes for all 3,969 claims took 1.5 s.
- **In the batch:** the full backfill took 2 min 8 s. 3,900 claims got reasons (10,688 rows) and every claim routed to a senior adjuster has at least one; the 69 without reasons had no feature raising their risk above the noise floor. The delayed-label performance checks were identical to before. One night through Dagster succeeded (71 claims, 192 reasons).
- **Model card:** two builds were byte-identical; its PR-AUC (0.0774) matches the gate's measurement of this champion.

Honest findings:
- **Routing leans on one feature.** `exposure` (years the policy was in force) is the top reason for 232 of the 301 routed claims (77%). It is also the largest feature globally (mean contribution 0.32 log-odds, twice the next). A short exposure can mean a new customer, or a policy that ended early; the model can't tell which, so this is worth a business review before any rollout.
- **No fairness alert, but little power.** Overall recall is only 17.9% (interval 13.1% to 24.1%), and with 184 high-cost test claims the group intervals are wide (under-25 drivers: 1.6% to 37.7%). No alert is not proof of equal treatment.
- **Young drivers are over-routed.** Under-25 claims are routed at twice the average rate (15.4% vs 7.6%) with 2.2% precision, and the model overstates their risk by half (calibration 1.51). Equal opportunity doesn't flag this, because it looks at missed costly claims, not extra scrutiny. Region R93 is routed at 23.4%, with calibration 0.90.

## Point-in-time feature store

Until now every claim carried its policy's features as one frozen copy. Real policies change: they renew, the driver gets older, the bonus-malus moves. A model has to be trained and scored on the values known *when the claim arrived*. Using today's values for an old claim leaks the future, and a re-run of a past night would no longer match the original one. The feature store keeps every version of each policy's features with the time it took effect, and every consumer reads through it.

| Piece                                | What it does                                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `configs/feature_store.yaml`         | The one definition: entity `policy_id`, source table, stored fields, on-demand `log_density`, renewal settings |
| `src/feature_store/registry.py`      | Loads it and fails fast if it disagrees with the model's inputs or stores an excluded column                   |
| Offline store (`offline.py`, DuckDB) | Every version of every policy (1,356,026 rows, 21 MB); read with an ASOF join, so a row sees only the past     |
| Online store (`online.py`, SQLite)   | Each policy's version in force now, by `policy_id`; standard library only, so the slim API image can read it   |
| Skew check (`materialize.py`)        | After each load, all 678,013 policies online must equal the offline point-in-time read at the same time        |

Who reads it:
- **Training:** a new DVC stage, `feature_store`, runs before `features`, which reads each claim's features as of the source snapshot's date.
- **Nightly batch:** each claim's features as of its report date.
- **The API:** `POST /predict/policy` looks up the online store. The values pass the same contract checks as a hand-sent claim and are scored through the same code. The response says when the features took effect.

```bash
make feature-store                 # rebuild the offline store (also part of make data)
make materialize                   # load the online store as of now, then check for skew
make materialize AS_OF=2025-02-01  # as of a past date
curl -s localhost:8000/predict/policy -H 'content-type: application/json' -d '{"policy_id": 1}'
```

**The renewals are synthetic.** freMTPL2 is one snapshot per policy, with no dates. The snapshot is recorded as taking effect on 2024-01-01. Each policy then renews once, on a stable-hash day in 2025: vehicle and driver one year older, and a claim-free bonus-malus drop of 5% (the French rule, coefficient x 0.95, floor 50). Renewals fall during the batch period, so point-in-time reads matter for real.

**Why not Feast?** Feast 0.68 forces pandas 3 down to 2.3 for the whole project. DuckDB's ASOF join and SQLite give the same guarantees for one entity: one definition, point-in-time training sets and no train/serve skew. Feast would add a registry service, more entities and streaming sources, which this project doesn't need.

How it was verified (real data, champion v1):
- **Training didn't change.** After rewiring the `features` stage through the store, train, validation, test, the drift reference and the preprocessor were byte-identical to before (same md5). The champion needed no retrain.
- **No skew:** materialized as of now (every policy renewed) and as of 2025-02-01 (59,609 renewed). All 678,013 policies matched the offline read both times. Loading took 20 s at 1.2 GB peak.
- **Batch, point-in-time:** in the full backfill, 365 of 3,969 claims (9.2%) were reported after their policy renewed and got the new values. 10 routes changed (2 up, 8 down, mostly from the lower bonus-malus). The parity check held every night. Delayed-label monitoring was unchanged: no alert on normal data, and the inflation alert still fired on 2025-06-15.
- **What a naive join would have done:** scoring the same claims with today's values would have used post-claim values for 3,588 claims and flipped 70 routes (senior-adjuster routes 295 → 273). That is the leak the as-of join prevents.
- **API:** over 300 requests each against the champion bundle, in-handler latency was p50 0.36 ms by `policy_id` vs 0.37 ms with full features. The lookup costs almost nothing.

Limits:
- **One entity, one snapshot.** No claim-history features (such as prior claims), because freMTPL2 has no claim dates. `claim_count` stays excluded: it counts later claims too.
- **The online store is loaded by hand** (`make materialize`), not on a schedule, and a new load needs an API restart to be picked up.

## Running the heavy steps on a laptop

Fine-tuning and Optuna can use several GB of RAM. Running two at once got processes killed by the Linux out-of-memory daemon, including the desktop session. Run one heavy job at a time, ideally in a memory-capped scope:

```bash
systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 nice -n 19 uv run python -m src.training.tabular
```

The notes ONNX evaluation runs in batches, so its memory stays flat.

## Project layout

```
configs/data_config.yaml     sources, paths, contracts, features, tokenizer, monitoring
configs/train_config.yaml    tuning, fine-tuning, export, latency budget, gate, explain, fairness
configs/batch_config.yaml    batch schedule, settlement delays, monitoring rules, scenario
configs/feature_store.yaml   feature view, store paths, synthetic renewal history
src/common/                  config loading, structured JSON logging
src/ingestion/               fetch_data.py (download, clean, join, store), notes.py
src/validation/              contracts.py (Pandera contracts, fail-fast alerts)
src/features/                split, preprocessing, tokenizer, shared row features
src/training/                tabular.py, nlp.py, onnx_export.py, benchmark.py, text_baseline.py
src/evaluation/              scoring.py (scorecards), gate.py (champion vs challenger),
                             fairness.py (audit), model_card.py (docs/model_card_tabular.md)
src/explain/                 champion rebuilt as LightGBM (parity-checked), TreeSHAP reason codes
src/serving/                 FastAPI app, schemas, model loading, tokenizer
src/feature_store/           registry, offline store (DuckDB, as-of reads), online store (SQLite), skew check
src/orchestration/           Dagster assets, jobs, schedule, sensors (dagster.yaml: run limits)
src/batch/                   nightly scoring, outcomes, delayed-label performance, backfill
src/infra/                   storage_check.py (S3 and IAM), mlflow_check.py (server smoke test)
infra/terraform/             S3 buckets, KMS key, IAM roles (applied to Moto locally)
infra/docker-compose.yml     Moto, Postgres and the MLflow server (infra/mlflow)
monitoring/                  drift detector, traffic simulator, Prometheus and Grafana config
tests/                       unit/, behavioral/, integration/
artifacts/                   metrics, gate decisions and benchmark (JSON); models live in MLflow
Dockerfile, docker-compose.yml  serving image; API + MLflow + Prometheus + Grafana
dvc.yaml, dvc.lock           data pipeline and the versions of its outputs
data/                        raw downloads, warehouse, splits (versioned by DVC, not git)
Makefile                     single-command execution (make help)
```
