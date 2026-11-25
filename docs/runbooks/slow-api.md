# Slow API: requests missing their latency objective

**Alerts:** `ClaimApiLatencyBudgetFastBurn` (page), `ClaimApiLatencyBudgetSlowBurn`.
The alert's `endpoint` label says which: `/predict/policy` (objective 100 ms)
or `/predict` (250 ms; it runs the notes model when a note is sent).

## 1. Find the cause

```bash
kubectl get hpa -n prod                          # replicas at 4 (the max) and CPU still over 70%?
kubectl top pods -n prod                         # CPU and memory per pod
helm history claim-triage-api -n prod --max 3    # a release just before it started?
make envs                                        # a model promotion just before it started?
cat /proc/pressure/cpu /proc/pressure/io         # the node itself: CPU or disk stalls
```

| What you see                                              | Likely cause                                  | Do                                                          |
| --------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------- |
| Started right after a deploy or a promotion               | The new image or model is slower              | [bad-release.md](bad-release.md): roll it back              |
| HPA at its maximum, CPU still above target                | More traffic than the pods can serve          | Add capacity, or shed load (below)                          |
| Only `/predict/policy` slow, high I/O pressure            | Lookups waiting on the disk                   | Check the API holds the store in memory (it should)         |
| Pods restarting (`kubectl get pods` RESTARTS)             | OOM kills or failing liveness probes          | `kubectl describe pod`; raise the memory limit if OOMKilled |

## 2. Add capacity or shed load

On a cluster with spare CPU, raise the ceiling for prod:

```bash
helm upgrade claim-triage-api infra/k8s/chart -n prod --reuse-values \
  --set autoscaling.maxReplicas=8 --wait
```

Then make it permanent in `infra/k8s/environments/prod.yaml` through a pull
request. If the nodes have no spare CPU (on this laptop: 4 cores shared by
every pod and the load generator), more pods don't help: shed load at the
source, or move adjuster-note requests to `/predict/policy` (no notes model).

## 3. Check

The fast-burn alert clears once the 5-minute ratio is back under 14.4%;
`kubectl get hpa -n prod` should show CPU back under target.

## Drilled (2026-11-25, kind)

- **Overload.** 50 req/s for 2 minutes, above the load test's 30. The
  autoscaler reached its maximum of 4 pods within about 20 seconds, but CPU
  stayed at 141-226% of target: no room left to scale. `/predict` (with
  notes) missed its objective, p99 837 ms against 250 ms; `/predict/policy`
  stayed at p99 8 ms. No request failed; k6 dropped 24 arrivals it couldn't
  send. Signature of a capacity limit, not a bug.
- **Disk-bound lookups** (step 6): `/predict/policy` p99 875 ms when the
  online store was read from the laptop's spinning disk; holding it in memory
  brought it to 10.5 ms.
