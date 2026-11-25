# Runbooks: claim triage API

What to do when something goes wrong. Each runbook starts from the alert or
symptom, gives the commands to confirm it, the fix, and how to check the fix
worked. Every one was drilled on the local kind cluster (step 7): the
results are at the end of each.

| Runbook                                  | When                                                                       |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| [bad-release.md](bad-release.md)         | Errors after a deploy, `ClaimApiErrorBudget*`, `ClaimApiDown`, a bad model |
| [slow-api.md](slow-api.md)               | `ClaimApiLatencyBudget*`: requests slower than their objective             |
| [stale-features.md](stale-features.md)   | `ClaimApiOnlineStoreStale`, `ClaimApiInferenceLogDropping`                 |
| [model-degraded.md](model-degraded.md)   | Model quality: a settled cohort's calibration or recall alert              |

Objectives and alert rules: [docs/slo.md](../slo.md).

## First minute, whatever the alert

```bash
kubectl get pods -n prod                                  # anything not Running 1/1?
curl -s localhost:8080/health                             # version, environment, models, store age
helm history claim-triage-api -n prod --max 5             # what changed, and when
make envs                                                 # which model versions each environment runs
kubectl get events -n prod --sort-by=.lastTimestamp | tail
```

A release changes one of two things, and each has its own rollback:

| What changed                     | How it changed                                  | Roll back with                                            |
| -------------------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| The image or the chart settings  | A pull request edits `environments/prod.yaml`   | `make rollback-release ENV=prod`                          |
| The models                       | `make promote ENV=prod` moves `@prod`           | `make rollback-model ENV=prod` and `make deploy ENV=prod` |
