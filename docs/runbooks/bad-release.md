# Bad release: errors, crashes or a wrong model after a deploy

**Alerts:** `ClaimApiErrorBudgetFastBurn` (page), `ClaimApiErrorBudgetSlowBurn`,
`ClaimApiDown`. **Also:** a deploy that failed or timed out, or routing that
changed after a model promotion.

## 1. Confirm what changed

```bash
kubectl get pods -n prod                         # CrashLoopBackOff, 0/1 Ready?
kubectl logs -n prod <new pod> --previous        # why it crashed
helm history claim-triage-api -n prod --max 5    # a "failed" revision?
curl -s localhost:8080/health                    # version and models now serving
make envs                                        # @prod moved recently?
```

A failed rollout usually costs nothing yet: the Deployment never goes below
its replica count (`maxUnavailable: 0`), and a pod that isn't ready gets no
traffic, so the old pods keep serving. Don't wait for it to fix itself.

## 2. Roll back

**The image or settings** (the last change was a deploy of a new tag):

```bash
make rollback-release ENV=prod     # helm rollback to the previous revision, waits for the rollout
```

Then revert the pull request that changed `infra/k8s/environments/prod.yaml`,
so the next deploy doesn't bring the bad tag back.

**The models** (the last change was `make promote ENV=prod`):

```bash
make rollback-model ENV=prod       # @prod back to @previous_prod
make deploy ENV=prod               # exports the bundle again; the pods restart on the new hash
```

After a model rollback, `@previous_prod` points at the bad version (it's the
undo). Don't run `rollback-model` twice by mistake, and fix or remove the bad
version from `@staging` before the next promotion.

## 3. Check

```bash
curl -s localhost:8080/health      # the old version and models
make k8s-smoke ENV=prod
kubectl get pods -n prod           # all Running 1/1
```

The error-budget alerts clear on their own: the short window (5 or 30
minutes) drops below the threshold soon after the errors stop.

## Drilled (2026-11-25, kind, prod under 15 req/s throughout)

- **Crashing image.** Release 1.0.1 exited on startup. `make deploy ENV=prod`
  failed after its 3-minute wait with one pod in CrashLoopBackOff; the two
  1.0.0 pods kept serving. `make rollback-release ENV=prod` took 5 seconds
  (revision 5 failed, back to 4). **0 of 4,951 requests failed** across the
  whole incident.
- **Bad model.** `@staging` forced to triage v2 (the version the gate
  rejected), promoted to prod and deployed: prod served v2 (threshold 0.0749
  instead of 0.0834). `make rollback-model ENV=prod` and `make deploy ENV=prod`
  took 64 seconds, back to v1. **0 of 4,500 requests failed.**
- `make deploy` also refuses a tag that isn't on GHCR (a typo never reaches
  the cluster), and warns when the image is a local stand-in.
