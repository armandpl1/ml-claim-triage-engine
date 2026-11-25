"""Model promotion between environments, with registry aliases.

    gate     sets @champion: the model passed the offline checks
             (src/evaluation/gate.py)
    staging  @staging <- @champion, for each model (triage and notes)
    prod     @prod    <- @staging: prod only ever runs what staging ran

Each move keeps the version it replaced as @previous_<env>, so a rollback is
one alias move back. A move is refused when there is nothing to promote
(no @champion, or no @staging for prod), and is a no-op when the target
already has that version. The cluster picks a promotion up on the next
`make deploy ENV=...`, which exports the environment's bundle from its
aliases.

Image versions are promoted separately, by pull request: each environment's
values file pins the image tag it runs (infra/k8s/environments/).

Usage:
    python -m src.serving.environments status
    python -m src.serving.environments promote staging
    python -m src.serving.environments promote prod
    python -m src.serving.environments rollback prod
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from src.common.logging import get_logger, log_event
from src.serving import model_store

logger = get_logger("serving.environments")

ENVIRONMENTS = ("staging", "prod")
SOURCE = {"staging": "champion", "prod": "staging"}
MODELS = (model_store.TABULAR_MODEL, model_store.NOTES_MODEL)


class PromotionError(RuntimeError):
    """Nothing to promote, or nothing to roll back to."""


class Registry(Protocol):
    """The MlflowClient methods used here (a fake in tests)."""

    def get_registered_model(self, name: str) -> Any: ...
    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None: ...
    def delete_registered_model_alias(self, name: str, alias: str) -> None: ...


@dataclass(frozen=True)
class Move:
    model: str
    env: str
    old: str | None
    new: str


def aliases(client: Registry, name: str) -> dict[str, str]:
    return {k: str(v) for k, v in client.get_registered_model(name).aliases.items()}


def promote(client: Registry, env: str, models: tuple[str, ...] = MODELS) -> list[Move]:
    """Move @env to the source alias's version for every model that has one.

    The triage model is required; the notes model is optional (the API
    serves without it), so it is skipped when it has no source version.
    """
    if env not in SOURCE:
        raise ValueError(f"unknown environment {env!r}, expected one of {ENVIRONMENTS}")
    source = SOURCE[env]
    moves = []
    for name in models:
        current = aliases(client, name)
        new = current.get(source)
        if new is None:
            if name == model_store.TABULAR_MODEL:
                raise PromotionError(f"{name} has no @{source}: nothing to promote to {env}")
            continue
        old = current.get(env)
        if old == new:
            continue
        if old is not None:
            client.set_registered_model_alias(name, f"previous_{env}", old)
        client.set_registered_model_alias(name, env, new)
        moves.append(Move(name, env, old, new))
        log_event(logger, logging.INFO, "model_promoted", model=name, env=env, old=old, new=new)
    return moves


def rollback(client: Registry, env: str, models: tuple[str, ...] = MODELS) -> list[Move]:
    """Put @env back on @previous_<env> (and remember the version undone)."""
    if env not in SOURCE:
        raise ValueError(f"unknown environment {env!r}, expected one of {ENVIRONMENTS}")
    moves = []
    for name in models:
        current = aliases(client, name)
        previous = current.get(f"previous_{env}")
        if previous is None:
            continue
        undone = current.get(env)
        client.set_registered_model_alias(name, env, previous)
        if undone is not None:
            client.set_registered_model_alias(name, f"previous_{env}", undone)
        else:
            client.delete_registered_model_alias(name, f"previous_{env}")
        moves.append(Move(name, env, undone, previous))
        log_event(
            logger,
            logging.WARNING,
            "model_rolled_back",
            model=name,
            env=env,
            old=undone,
            new=previous,
        )
    if not moves:
        raise PromotionError(f"no @previous_{env} for any model: nothing to roll back to")
    return moves


def status(client: Registry, models: tuple[str, ...] = MODELS) -> dict[str, dict[str, str]]:
    """Each model's aliases, e.g. {'claim-triage-tabular': {'champion': '1', 'prod': '1'}}."""
    return {name: aliases(client, name) for name in models}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote models between environments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    for command in ("promote", "rollback"):
        sub.add_parser(command).add_argument("env", choices=ENVIRONMENTS)
    args = parser.parse_args(argv)

    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        if args.command == "promote":
            moves = promote(client, args.env)
        elif args.command == "rollback":
            moves = rollback(client, args.env)
        else:
            moves = []
    except PromotionError as exc:
        log_event(logger, logging.ERROR, "promotion_refused", reason=str(exc))
        return 1
    for move in moves:
        print(f"{move.model}: @{move.env} {move.old or '-'} -> {move.new}")
    if args.command != "status" and not moves:
        print(f"@{args.env} already has the source versions: nothing to do")
    for name, names in status(client).items():
        print(f"{name}: " + ", ".join(f"@{a}=v{v}" for a, v in sorted(names.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
