# Stale features or a dropping inference log

## Online store stale

**Alert:** `ClaimApiOnlineStoreStale`: the features `/predict/policy` serves
are over 36 hours old. Predictions still work, on old policy values (a
renewal since then, with its older driver and lower bonus-malus, isn't seen).

### Confirm

```bash
curl -s localhost:8080/health      # online_store.as_of: when the features were current
make envs                          # (models are unrelated, but rule them out)
```

### Fix

```bash
make materialize                   # reload the store as of now; the skew check must pass
kubectl rollout restart deployment/claim-triage-api-stable -n prod
kubectl rollout status deployment/claim-triage-api-stable -n prod
```

The API holds the store in memory, so a new materialization reaches it only
when the pods restart. `make materialize` writes the file atomically:
running pods are never left with a half-written store. If `make materialize`
fails its skew check, don't restart: the pods keep serving the last good
store. Find out why the offline store changed (`make feature-store`, then
`dvc status`).

### Check

```bash
curl -s localhost:8080/health      # as_of is now
curl -s localhost:8080/metrics | grep claim_api_online_store_as_of_seconds
```

## Inference log dropping rows

**Alert:** `ClaimApiInferenceLogDropping`. Predictions are unaffected; the
drift detector sees an incomplete sample until it stops.

```bash
kubectl logs -n prod deploy/claim-triage-api-stable | grep inference_log_rows_dropped
```

`reason: queue full` means the writer can't keep up (a slow disk under the
pod's volume); any other reason is a SQLite error, usually a full disk
(`df -h`, and the pod's `emptyDir` limit of 100Mi).

## Drilled (2026-11-25, kind)

- **Stale store.** Prod restarted on a store materialized as of 3 days
  earlier: about 89 hours old, well over the 36-hour threshold. The fix (above)
  took 51 seconds, after which the store was 0.01 hours old.
- **Found by the drill:** the store's age read 5 hours too old after a fresh
  materialization. `materialize` wrote its time without a timezone (local
  time, EST), and the pods, in UTC, read it as UTC. Times are now written
  with their UTC offset, so the alert fires on time.
