# Service-level objectives: claim triage API

What "working" means for the triage API, how it is measured, and what happens
when it isn't met. The alert rules are in [monitoring/alerts.yml](../monitoring/alerts.yml)
(tested by `make alerts-check`); the load test (`make loadtest`) fails if the
latency or error objectives are missed.

## Objectives

Measured over a rolling 28 days, on the prediction endpoints (`/predict` and
`/predict/policy`). `/health` and `/metrics` don't count.

| Objective         | Target                                                   | Error budget (28 days)  | Measured by                                 |
| ----------------- | -------------------------------------------------------- | ----------------------- | ------------------------------------------- |
| Availability      | 99.5% of prediction requests don't fail with a 5xx       | 0.5% of requests        | `claim_api_requests_total` by status        |
| Latency, policy   | 99% of `/predict/policy` requests within 100 ms          | 1% of those requests    | `claim_api_request_seconds` (`le="0.1"`)    |
| Latency, claim    | 99% of `/predict` requests within 250 ms                 | 1% of those requests    | `claim_api_request_seconds` (`le="0.25"`)   |
| Feature freshness | The online store the API serves is under 36 hours old    | none: a ticket at once  | `claim_api_online_store_as_of_seconds`      |
| Model quality     | No settled cohort with a calibration or recall alert     | none: a ticket at once  | delayed-label monitoring (`src/batch/`)     |

Why these numbers:
- **99.5%, not 99.9%.** One node, one region, a laptop. Claims that fail can be
  retried or routed by hand; an adjuster waits minutes, not milliseconds.
  99.5% allows about 3.4 hours of full outage in 28 days.
- **Latency targets have headroom.** The load test at 30 req/s measured
  `/predict/policy` p95 5.5 ms (max 25 ms) and `/predict` with an adjuster
  note p95 61 ms (max 92 ms). The objectives sit well above that, so an
  alert means a real change (a slow disk, a bigger model, overload), not
  noise. Each target is a bucket bound of the API's latency histogram, so
  it is measured exactly.
- **4xx answers don't spend the budget.** A 422 (input outside the data
  contracts) or a 404 (unknown policy) is the API doing its job. A 503
  (online store not loaded) does spend it.

## Alerts

Budget-burn alerts follow the Google SRE workbook's multiwindow pattern: an
alert needs a long window, so the burn is real, and a short one, so it is
still happening (it clears quickly once the cause is fixed).

| Alert                           | Fires when                                                                   | Severity | Runbook                                                   |
| ------------------------------- | ---------------------------------------------------------------------------- | -------- | --------------------------------------------------------- |
| `ClaimApiErrorBudgetFastBurn`   | Errors over 7.2% (14.4x budget) for 1 h and 5 min: 2% of the budget per hour | page     | [bad-release](runbooks/bad-release.md)                    |
| `ClaimApiErrorBudgetSlowBurn`   | Errors over 3% (6x budget) for 6 h and 30 min                                | ticket   | [bad-release](runbooks/bad-release.md)                    |
| `ClaimApiLatencyBudgetFastBurn` | Over 14.4% of an endpoint's requests too slow, 1 h and 5 min                 | page     | [slow-api](runbooks/slow-api.md)                          |
| `ClaimApiLatencyBudgetSlowBurn` | Over 6% too slow, 6 h and 30 min                                             | ticket   | [slow-api](runbooks/slow-api.md)                          |
| `ClaimApiDown`                  | Prometheus can't scrape the API for 2 minutes                                | page     | [bad-release](runbooks/bad-release.md)                    |
| `ClaimApiOnlineStoreStale`      | The served features are over 36 hours old                                    | ticket   | [stale-features](runbooks/stale-features.md)              |
| `ClaimApiInferenceLogDropping`  | Any inference-log row dropped in 15 minutes                                  | ticket   | [stale-features](runbooks/stale-features.md)              |
| Model quality                   | A settled cohort alerts (Dagster check `no_performance_alerts`)              | ticket   | [model-degraded](runbooks/model-degraded.md)              |

Model quality isn't a Prometheus alert. Its signal is the delayed-label job,
which only knows a cohort's quality once its claims settle (105 days). The
Dagster asset check raises it as a warning, and the `retrain_on_alert`
sensor requests one retrain, judged by the gate.

## How the rules were verified

- `make alerts-check`: `promtool` validates the 19 rules and runs
  [monitoring/alerts_test.yml](../monitoring/alerts_test.yml): 10% errors fire
  both availability alerts; 0.1% errors and 4xx answers fire nothing; a burst
  that already stopped doesn't page; 30% slow policy lookups fire the latency
  alerts for that endpoint only; the API down, dropped log rows and a stale
  store fire their alerts, and a fresh store doesn't. A copy of the rules with
  a broken threshold failed the tests.
- Live: Prometheus v3.5.0 with these rules, scraping prod in the kind cluster
  under 10 req/s: all 19 rules healthy, no alert firing, latency ratios 0,
  store age 11.9 hours.
- Found that way: with no 5xx at all, the error ratio had no value instead of
  0 (the error series didn't exist yet). Fixed with `or vector(0)`, and
  covered by a test.

## Limits

- **Prometheus runs in docker compose, not in the cluster.** The rules load
  in the compose stack (`make up`); in the cluster they were checked with a
  Prometheus scraping prod's service, which reaches one pod per scrape. A
  real setup would run the Prometheus Operator and scrape every pod
  (kube-prometheus-stack needs about 1.5 GB, too much for this laptop).
- **No alert routing.** Severities are labels; nothing pages anyone.
  Alertmanager would route `page` to on-call and `ticket` to a queue.
- **Error budgets aren't tracked over 28 days yet.** Nothing runs for 28 days
  here. The burn rates are the actionable part; a budget report would
  compute `1 - claim_api:error_ratio` over `[28d]`.
