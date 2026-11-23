"""Storage check: a safe setup passes, and each unsafe setting is reported.

Runs against moto's in-memory AWS, so no Docker or emulator is needed.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from src.infra import storage_check as sc

REGION = "ca-central-1"
BASE = "claim-triage-dev"
DVC, MLFLOW, LOGS = f"{BASE}-dvc", f"{BASE}-mlflow", f"{BASE}-access-logs"


def bucket_arns(*buckets: str) -> list[str]:
    return [f"arn:aws:s3:::{b}" for b in buckets]


def object_arns(*buckets: str) -> list[str]:
    return [f"arn:aws:s3:::{b}/*" for b in buckets]


TRAINING = [
    {
        "Sid": "List",
        "Effect": "Allow",
        "Action": ["s3:ListBucket"],
        "Resource": bucket_arns(DVC, MLFLOW),
    },
    {
        "Sid": "Dvc",
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject"],
        "Resource": object_arns(DVC),
    },
    {
        "Sid": "Mlflow",
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
        "Resource": object_arns(MLFLOW),
    },
    {
        "Sid": "Key",
        "Effect": "Allow",
        "Action": ["kms:Decrypt"],
        "Resource": "arn:aws:kms:ca-central-1:1:key/k",
    },
]
SERVING = [
    {"Sid": "List", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": bucket_arns(MLFLOW)},
    {"Sid": "Read", "Effect": "Allow", "Action": "s3:GetObject", "Resource": object_arns(MLFLOW)},
]


def make_bucket(s3, name: str, key_arn: str, log_to: str | None) -> None:
    s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION})
    s3.put_bucket_versioning(Bucket=name, VersioningConfiguration={"Status": "Enabled"})
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": key_arn,
                    }
                }
            ]
        },
    )
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    if log_to:
        s3.put_bucket_logging(
            Bucket=name,
            BucketLoggingStatus={"LoggingEnabled": {"TargetBucket": log_to, "TargetPrefix": "x/"}},
        )
        s3.put_bucket_lifecycle_configuration(
            Bucket=name,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "expire",
                        "Status": "Enabled",
                        "Filter": {},
                        "NoncurrentVersionExpiration": {"NoncurrentDays": 90},
                    }
                ]
            },
        )


def make_role(iam, role: str, statements: list[dict]) -> None:
    iam.create_role(
        RoleName=role,
        AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": []}),
    )
    arn = iam.create_policy(
        PolicyName=f"{role}-s3",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": statements}),
    )["Policy"]["Arn"]
    iam.attach_role_policy(RoleName=role, PolicyArn=arn)


@pytest.fixture
def aws(monkeypatch):
    """A safe setup, as infra/terraform creates it, in moto's in-memory AWS."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "")  # no emulator URL: moto intercepts
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        iam = boto3.client("iam", region_name=REGION)
        key = boto3.client("kms", region_name=REGION).create_key()["KeyMetadata"]["Arn"]
        make_bucket(s3, LOGS, key, log_to=None)
        s3.put_bucket_policy(  # lets S3 deliver access logs, as in infra/terraform/s3.tf
            Bucket=LOGS,
            Policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "logging.s3.amazonaws.com"},
                            "Action": "s3:PutObject",
                            "Resource": f"arn:aws:s3:::{LOGS}/*",
                        }
                    ],
                }
            ),
        )
        make_bucket(s3, DVC, key, log_to=LOGS)
        make_bucket(s3, MLFLOW, key, log_to=LOGS)
        make_role(iam, f"{BASE}-training", TRAINING)
        make_role(iam, f"{BASE}-serving", SERVING)
        yield s3, iam


def test_safe_setup_passes(aws):
    assert sc.check(REGION, "claim-triage", "dev") == []
    assert sc.main([]) == 0


def test_unsafe_bucket_settings_are_reported(aws):
    s3, _ = aws
    s3.put_bucket_versioning(Bucket=DVC, VersioningConfiguration={"Status": "Suspended"})
    s3.delete_public_access_block(Bucket=MLFLOW)
    s3.delete_bucket_lifecycle(Bucket=MLFLOW)
    problems = sc.check(REGION, "claim-triage", "dev")
    assert f"{DVC}: versioning is not enabled" in problems
    assert f"{MLFLOW}: no public access block" in problems
    assert f"{MLFLOW}: no rule expiring old object versions" in problems
    assert sc.main([]) == 1


def test_missing_role_is_reported(aws):
    _, iam = aws
    role = f"{BASE}-serving"
    arn = iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"][0]["PolicyArn"]
    iam.detach_role_policy(RoleName=role, PolicyArn=arn)
    iam.delete_role(RoleName=role)
    assert f"role {role} not found" in sc.check(REGION, "claim-triage", "dev")


def test_training_policy_must_be_least_privilege():
    assert sc.least_privilege_problems(TRAINING, [DVC, MLFLOW], no_delete=(DVC,)) == []
    too_wide = [
        {"Sid": "All", "Effect": "Allow", "Action": "s3:*", "Resource": "*"},
        {
            "Sid": "Drop",
            "Effect": "Allow",
            "Action": "s3:DeleteBucket",
            "Resource": bucket_arns(DVC),
        },
        {
            "Sid": "Erase",
            "Effect": "Allow",
            "Action": "s3:DeleteObject",
            "Resource": object_arns(DVC),
        },
        {"Sid": "Key", "Effect": "Allow", "Action": "kms:Decrypt", "Resource": "*"},
    ]
    problems = " | ".join(sc.least_privilege_problems(too_wide, [DVC, MLFLOW], no_delete=(DVC,)))
    assert "policy All: wildcard action" in problems
    assert "policy All: resources outside the buckets ['*']" in problems
    assert "policy Drop: destructive or policy actions ['s3:DeleteBucket']" in problems
    assert f"policy Erase: deletes allowed in {DVC}" in problems
    assert "policy Key: KMS actions on non-key resources" in problems


def test_serving_policy_must_be_read_only():
    assert sc.least_privilege_problems(SERVING, [MLFLOW], read_only=True) == []
    writes = [
        {
            "Sid": "W",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject"],
            "Resource": object_arns(MLFLOW),
        }
    ]
    assert sc.least_privilege_problems(writes, [MLFLOW], read_only=True) == [
        "policy W: write actions in a read-only role ['s3:PutObject']"
    ]
    other_bucket = [
        {"Sid": "R", "Effect": "Allow", "Action": "s3:GetObject", "Resource": object_arns(DVC)}
    ]
    assert sc.least_privilege_problems(other_bucket, [MLFLOW], read_only=True) == [
        f"policy R: resources outside the buckets ['arn:aws:s3:::{DVC}/*']"
    ]
