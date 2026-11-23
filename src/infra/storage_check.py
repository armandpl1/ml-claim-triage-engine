"""Check that the storage in infra/terraform exists and is configured safely,
by reading it back with boto3 (from the Moto emulator by default).

Buckets (dvc, mlflow and access-logs):
    exist, versioning on, default encryption with the KMS key, every public
    access block on. dvc and mlflow send access logs to the log bucket and
    expire old object versions.
Training role:
    least privilege: no wildcard actions, every S3 resource inside the dvc
    and mlflow buckets, no destructive or policy actions, no deletes in the
    DVC remote, KMS permissions only on KMS keys.
Serving role:
    read-only: only list and get, only on the mlflow bucket.

Endpoint: AWS_ENDPOINT_URL (boto3 reads it), default the emulator from
infra/docker-compose.yml. Every problem is logged in one structured event
("storage_check_failed") and the process exits 1.

Usage:
    python -m src.infra.storage_check
    python -m src.infra.storage_check --environment prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any
from urllib.parse import unquote

import boto3

from src.common.logging import get_logger, log_event

logger = get_logger("infra.storage_check")

DEFAULT_ENDPOINT = "http://localhost:15100"
FORBIDDEN_ACTIONS = {
    "s3:DeleteBucket",
    "s3:PutBucketPolicy",
    "s3:PutBucketAcl",
    "s3:PutObjectAcl",
    "s3:PutBucketPublicAccessBlock",
    "kms:ScheduleKeyDeletion",
    "kms:PutKeyPolicy",
    "kms:DisableKey",
}
READ_ONLY_ACTIONS = {"s3:ListBucket", "s3:GetBucketLocation", "s3:GetObject", "kms:Decrypt"}


def clients(region: str) -> tuple[Any, Any]:
    endpoint = os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT)
    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint:
        kwargs |= {
            "endpoint_url": endpoint,
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        }
    return boto3.client("s3", **kwargs), boto3.client("iam", **kwargs)


def get_config(call: Callable[..., Any], **kwargs: Any) -> Any:
    """A bucket setting, or None when it was never configured (AWS raises then)."""
    try:
        return call(**kwargs)
    except Exception as err:
        code = getattr(err, "response", {}).get("Error", {}).get("Code", "")
        if code.startswith("NoSuch") or code.endswith("NotFoundError"):
            return None
        raise


def check_bucket(s3: Any, name: str, log_bucket: str | None = None) -> list[str]:
    """Problems with one bucket. Data buckets pass `log_bucket`; they must log
    to it and expire old versions. The log bucket itself passes None."""
    try:
        s3.head_bucket(Bucket=name)
    except Exception as err:
        return [f"{name}: bucket not found ({err})"]

    problems = []
    if s3.get_bucket_versioning(Bucket=name).get("Status") != "Enabled":
        problems.append(f"{name}: versioning is not enabled")

    encryption = get_config(s3.get_bucket_encryption, Bucket=name)
    rules = (encryption or {}).get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    algorithms = {r["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] for r in rules}
    if not algorithms:
        problems.append(f"{name}: no default encryption")
    elif "aws:kms" not in algorithms:
        problems.append(f"{name}: not encrypted with a KMS key ({sorted(algorithms)})")

    found = get_config(s3.get_public_access_block, Bucket=name)
    block = (found or {}).get("PublicAccessBlockConfiguration", {})
    if not block:
        problems.append(f"{name}: no public access block")
    elif off := sorted(k for k, v in block.items() if not v):
        problems.append(f"{name}: public access not fully blocked ({', '.join(off)})")

    if log_bucket is not None:
        logging_cfg = s3.get_bucket_logging(Bucket=name).get("LoggingEnabled") or {}
        if logging_cfg.get("TargetBucket") != log_bucket:
            problems.append(f"{name}: access logs not sent to {log_bucket}")
        config = get_config(s3.get_bucket_lifecycle_configuration, Bucket=name)
        expiring = [
            r
            for r in (config or {}).get("Rules", [])
            if r.get("Status") == "Enabled" and r.get("NoncurrentVersionExpiration")
        ]
        if not expiring:
            problems.append(f"{name}: no rule expiring old object versions")
    return problems


def as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def attached_statements(iam: Any, role: str, policy_name: str) -> list[dict[str, Any]] | str:
    """The statements of the one policy a role should have, or a problem."""
    try:
        iam.get_role(RoleName=role)
    except iam.exceptions.NoSuchEntityException:
        return f"role {role} not found"
    attached = iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]
    names = sorted(p["PolicyName"] for p in attached)
    if names != [policy_name]:
        return f"role {role}: expected only policy {policy_name}, found {names}"
    arn = attached[0]["PolicyArn"]
    version_id = iam.get_policy(PolicyArn=arn)["Policy"]["DefaultVersionId"]
    document = iam.get_policy_version(PolicyArn=arn, VersionId=version_id)["PolicyVersion"][
        "Document"
    ]
    if isinstance(document, str):  # some implementations return it URL-encoded
        document = json.loads(unquote(document))
    statements = document["Statement"]
    return statements if isinstance(statements, list) else [statements]


def least_privilege_problems(
    statements: list[dict[str, Any]],
    buckets: list[str],
    read_only: bool = False,
    no_delete: tuple[str, ...] = (),
) -> list[str]:
    """Problems with a policy that should only touch the given buckets.

    read_only: only list, get and decrypt are allowed.
    no_delete: buckets where s3:DeleteObject must not be granted.
    """
    problems = []
    allowed = {f"arn:aws:s3:::{b}" for b in buckets} | {f"arn:aws:s3:::{b}/*" for b in buckets}
    for st in statements:
        if st.get("Effect") != "Allow":
            continue
        actions = as_list(st.get("Action", []))
        resources = as_list(st.get("Resource", []))
        sid = st.get("Sid", "?")
        if any("*" in a for a in actions):
            problems.append(f"policy {sid}: wildcard action {actions}")
        if bad := sorted(set(actions) & FORBIDDEN_ACTIONS):
            problems.append(f"policy {sid}: destructive or policy actions {bad}")
        if read_only and (writes := sorted(set(actions) - READ_ONLY_ACTIONS)):
            problems.append(f"policy {sid}: write actions in a read-only role {writes}")
        if actions and all(a.startswith("kms:") for a in actions):
            if not all(r.startswith("arn:aws:kms:") for r in resources):
                problems.append(f"policy {sid}: KMS actions on non-key resources")
            continue
        if outside := sorted(r for r in resources if r not in allowed):
            problems.append(f"policy {sid}: resources outside the buckets {outside}")
        if "s3:DeleteObject" in actions:
            for bucket in no_delete:
                if any(r.startswith(f"arn:aws:s3:::{bucket}") for r in resources):
                    problems.append(f"policy {sid}: deletes allowed in {bucket}")
    return problems


def check(region: str, prefix: str, environment: str) -> list[str]:
    s3, iam = clients(region)
    base = f"{prefix}-{environment}"
    dvc, mlflow, logs = f"{base}-dvc", f"{base}-mlflow", f"{base}-access-logs"

    problems = check_bucket(s3, dvc, log_bucket=logs)
    problems += check_bucket(s3, mlflow, log_bucket=logs)
    problems += check_bucket(s3, logs)

    training = attached_statements(iam, f"{base}-training", f"{base}-training-s3")
    if isinstance(training, str):
        problems.append(training)
    else:
        problems += least_privilege_problems(training, [dvc, mlflow], no_delete=(dvc,))

    serving = attached_statements(iam, f"{base}-serving", f"{base}-serving-s3")
    if isinstance(serving, str):
        problems.append(serving)
    else:
        problems += least_privilege_problems(serving, [mlflow], read_only=True)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--region", default="ca-central-1")
    parser.add_argument("--prefix", default="claim-triage")
    parser.add_argument("--environment", default="dev")
    args = parser.parse_args(argv)
    problems = check(args.region, args.prefix, args.environment)
    if problems:
        log_event(logger, logging.ERROR, "storage_check_failed", problems=problems)
        return 1
    log_event(
        logger,
        logging.INFO,
        "storage_check_passed",
        buckets=[f"{args.prefix}-{args.environment}-{b}" for b in ("dvc", "mlflow", "access-logs")],
        detail="versioned, KMS-encrypted, private, logged; training role least privilege; "
        "serving role read-only",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
