# Model degraded: settled claims show worse quality

**Signal:** the delayed-label monitoring (`src/batch/performance.py`) finds
a settled cohort whose realized high-cost rate is outside the model's
expected interval (calibration), or whose recall or precision is provably
below the baseline. In Dagster the `no_performance_alerts` check fails as a
warning, and the `retrain_on_alert` sensor requests one retrain (at most one
a week). It is not a Prometheus alert: a cohort's quality is known only once
its claims settle, up to 105 days later.

## 1. Confirm

```bash
cat reports/batch/performance_<date>.json    # which cohort, which check, the numbers
make drift                                   # did the inputs change too?
```

| Feature drift? | Outcomes worse? | Likely cause                                                        |
| -------------- | --------------- | ------------------------------------------------------------------- |
| Yes            | Yes             | New kinds of claims: retraining on recent data should help          |
| No             | Yes             | Concept drift, e.g. costs rising (the inflation scenario)           |
| Yes            | No              | Inputs moved but the model copes: watch, don't act                  |

## 2. Act

The sensor's retrain only reruns training on the same training split. It
can't learn new costs: the training data has to include the newly settled
claims first (a data change). Until then:

- **If routing is too lax** (calibration says more expensive claims than
  expected), lowering the decision threshold sends more claims to senior
  adjusters. That is a model change: it goes through a new model version,
  the gate, `@staging` and a prod canary, never an edit in prod.
- **Retrain** with `make retrain` once the training data includes the new
  outcomes. The gate decides; if it rejects, the champion keeps serving.
- **Promote** a model the gate accepted: `make promote ENV=staging`,
  `make deploy ENV=staging`, a prod canary (`make canary-start`), then
  `make canary-promote`.

## 3. Check

After the next cohorts settle, `reports/batch/` shows them within the
interval again. That takes weeks; the canary and staging are the faster
checks that the new model behaves.

## Drilled

Not re-run in step 7: the full path was exercised in step 2 (orchestration).
After the inflation backfill, one night through Dagster recorded
`model_performance` as degraded, the check failed as a warning, and
`retrain_on_alert` requested one `retrain_tabular` run (reason
`model_performance=degraded`). The inflation alert fired on 2025-06-15 for the
cohort of 2025-02-17, 132 days after costs started rising (README, batch
scoring section).
