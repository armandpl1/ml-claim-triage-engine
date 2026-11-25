#!/usr/bin/env bash
# Sample the autoscaler every 10 s while a load test runs: time, replicas,
# CPU utilization, in prod. Usage: infra/k8s/watch_pods.sh OUT_FILE (stops when killed)
set -euo pipefail
out="$1"
echo "time,replicas,cpu_utilization" > "$out"
while true; do
  read -r replicas cpu <<< "$(kubectl get hpa claim-triage-api-stable --namespace prod \
    -o jsonpath='{.status.currentReplicas} {.status.currentMetrics[0].resource.current.averageUtilization}')"
  echo "$(date -u +%T),$replicas,${cpu:-}" >> "$out"
  sleep 10
done
