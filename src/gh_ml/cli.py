"""Command line runner for the daily GitHub repository registry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import copyfile
from typing import Any

from .classification import classify_repository
from .discovery import discover
from .github import GitHubAPIError, GitHubClient
from .hub import load_checkpoint, publish_run
from .query_catalog import load_queries
from .schema import observation_from_repository, write_jsonl

DEFAULT_REPO = "modelomics/gh-ml"
DISPLAY_NAME = "GitHub ML"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "queries"
DEFAULT_OUTPUT = Path.home() / ".local" / "share" / "modelomics-gh-ml" / "runs"


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _run_id(now: datetime) -> str:
    # Keep UTC time readable while avoiding collisions between rapid retries.
    return f"{now.strftime('%Y%m%dT%H%M%S')}{now.microsecond:06d}Z-{uuid.uuid4().hex[:8]}"


def _read_state(path: Path, default_since: str) -> dict[str, Any]:
    if not path.exists():
        return {"since": default_since, "cursor": None, "checkpoint": None, "initialized": False}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"run state must contain a JSON object: {path}")
    return {
        **state,
        "since": state.get("since", default_since),
        "cursor": state.get("cursor"),
        "checkpoint": state.get("checkpoint"),
        "initialized": True,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _github_token(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _hf_token(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value
    if os.environ.get("HF_OIDC_RESOURCE"):
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            # SDK errors can include response details. Keep credentials and
            # exchange response bodies out of CLI output.
            raise ValueError("Hugging Face OIDC token exchange failed") from None
        if not token:
            raise ValueError("Hugging Face OIDC token exchange returned no token")
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except (ImportError, OSError):
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gh-ml",
        description=f"{DISPLAY_NAME}: discover and classify machine-learning GitHub repositories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a GitHub ML discovery sweep and optionally publish it")
    run.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    run.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="directory of TOML query files")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="local run and checkpoint directory")
    run.add_argument("--max-requests", type=int, default=100, help="maximum GitHub search API requests per invocation")
    run.add_argument("--since-days", type=int, default=1, help="lookback interval when starting a new sweep")
    run.add_argument("--no-publish", action="store_true", help="write local files without publishing to Hugging Face")
    run.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    run.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    backfill = subparsers.add_parser("backfill", help="backfill historical GitHub ML repository candidates")
    backfill.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    backfill.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="directory of TOML query files")
    backfill.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="local run and checkpoint directory")
    backfill.add_argument("--start", default="2008-01-01", help="first GitHub repository created date to search (default: 2008-01-01)")
    backfill.add_argument("--end", help="last GitHub repository created date to search (default: today in UTC)")
    backfill.add_argument("--max-requests", type=int, default=100, help="maximum GitHub search API requests per invocation")
    backfill.add_argument("--no-publish", action="store_true", help="write local files without publishing to Hugging Face")
    backfill.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    backfill.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    return parser


def _run(args: argparse.Namespace) -> int:
    return _collect(args, mode="daily")


def _backfill(args: argparse.Namespace) -> int:
    return _collect(args, mode="backfill")


def _collect(args: argparse.Namespace, *, mode: str) -> int:
    if args.max_requests < 1:
        raise ValueError("--max-requests must be at least 1")
    if mode == "daily" and args.since_days < 1:
        raise ValueError("--since-days must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = _utc_now()
    run_id = _run_id(now)
    state_path = args.output_dir / ("state.json" if mode == "daily" else "backfill-state.json")
    initial_since = (now - timedelta(days=args.since_days)).date().isoformat() if mode == "daily" else args.start
    state = _read_state(state_path, initial_since)
    github_token = _github_token(args.github_token_env)
    hf_token = _hf_token(args.hf_token_env)
    remote_checkpoint: dict[str, Any] | None = None
    if not args.no_publish:
        if not hf_token:
            raise ValueError(
                f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login"
            )
        remote_checkpoint = load_checkpoint(
            args.repo, hf_token, **({"checkpoint_path": "state/backfill.json"} if mode == "backfill" else {})
        )
        if not state["initialized"] and isinstance(remote_checkpoint, dict):
            state.update(remote_checkpoint)
            state["since"] = remote_checkpoint.get("since", initial_since)
            state["cursor"] = remote_checkpoint.get("cursor")
            if mode == "backfill":
                state["start"] = remote_checkpoint.get("start", args.start)
                state["end"] = remote_checkpoint.get("end", args.end)
    specs = load_queries(args.config_dir)
    if not specs:
        raise ValueError(f"no queries found in {args.config_dir}")

    client = GitHubClient(token=github_token)
    backfill_start = state.get("start", args.start) if mode == "backfill" else None
    backfill_end = state.get("end", args.end or now.date().isoformat()) if mode == "backfill" else None
    if mode == "daily":
        # A new sweep gets a fresh upper bound; a truncated sweep retains
        # the exact bound alongside its cursor so it can resume safely.
        until = state.get("until") if state.get("cursor") is not None else now.date().isoformat()
        state["until"] = until
        discover_run = lambda cursor: discover(
            client,
            specs,
            since=state["since"],
            until=until,
            max_requests=args.max_requests,
            cursor=cursor,
        )
    else:
        from .discovery import discover_backfill

        start = backfill_start
        end = backfill_end
        state["start"], state["end"] = start, end
        discover_run = lambda cursor: discover_backfill(
            client,
            specs,
            start=start,
            end=end,
            max_requests=args.max_requests,
            cursor=cursor,
        )

    try:
        try:
            outcome = discover_run(state.get("cursor"))
        except ValueError as exc:
            # A catalog edit invalidates a cursor's query signature. Restart
            # this exact date window; the previous invocation's observations
            # and coverage remain in their already-written run artifacts.
            if str(exc) != "cursor specs does not match this discovery run":
                raise
            print(
                f"Query catalog changed; restarting {mode} window "
                f"from the beginning ({state.get('since') if mode == 'daily' else backfill_start}"
                f"..{until if mode == 'daily' else backfill_end}).",
                file=sys.stderr,
            )
            outcome = discover_run(None)
    except GitHubAPIError as exc:
        coverage_path = args.output_dir / f"coverage-{run_id}.json"
        _write_json(
            coverage_path,
            {
                "run_id": run_id,
                "started_at": now.isoformat().replace("+00:00", "Z"),
                **({"since": state["since"], "until": state.get("until")} if mode == "daily" else {"start": backfill_start, "end": backfill_end}),
                "requests_used": None,
                "complete_sweep": False,
                "queries": [{"status": "error", "error": str(exc)}],
            },
        )
        raise RuntimeError(f"GitHub discovery failed; coverage written to {coverage_path}: {exc}") from None

    specs_by_id = {spec.id: spec for spec in specs}
    observations: list[dict[str, Any]] = []
    observed_at = now.isoformat().replace("+00:00", "Z")
    for repo_id, repository in sorted(outcome.repositories.items()):
        query_ids = sorted(set(outcome.matched_query_ids.get(repo_id, [])))
        matched_specs = [specs_by_id[query_id] for query_id in query_ids if query_id in specs_by_id]
        classification = classify_repository(repository, matched_specs)
        observations.append(
            observation_from_repository(
                repository,
                observed_at=observed_at,
                query_ids=query_ids,
                domains=classification.get("domains", []),
                methods=classification.get("methods", []),
                novelty_signals=classification.get("novelty_signals", []),
            )
        )

    observations_path = args.output_dir / f"observations-{run_id}.jsonl"
    coverage_path = args.output_dir / f"coverage-{run_id}.json"
    card_path = args.output_dir / f"README-{run_id}.md"
    manifest_path = args.output_dir / f"manifest-{run_id}.json"
    write_jsonl(observations, observations_path)
    coverage = {
        "run_id": run_id,
        "started_at": observed_at,
        **({"since": state["since"], "until": state["until"]} if mode == "daily" else {"start": state["start"], "end": state["end"]}),
        "requests_used": outcome.requests_used,
        "complete_sweep": outcome.next_cursor is None,
        "queries": outcome.coverage,
    }
    _write_json(coverage_path, coverage)
    project_card = Path(__file__).resolve().parents[2] / "dataset" / "README.md"
    if project_card.is_file():
        copyfile(project_card, card_path)
    else:
        card_path.write_text(
            "---\npretty_name: GitHub ML\nlicense: other\n---\n\n"
            "# GitHub ML\n\n"
            "A broad index of public GitHub repositories related to machine learning. "
            "Rows are discovery candidates and do not assert verified novelty.\n",
            encoding="utf-8",
        )

    if mode == "daily":
        if outcome.next_cursor is None:
            # Start the next daily pass at the previous upper bound (inclusive
            # overlap) and choose its upper bound when that pass begins.
            next_state = {"since": state.get("until", now.date().isoformat()), "cursor": None}
        else:
            next_state = {"since": state["since"], "until": state["until"], "cursor": outcome.next_cursor}
    else:
        next_state = {"start": state["start"], "end": state["end"], "cursor": outcome.next_cursor}
    failures = [row for row in outcome.coverage if row.get("error") or row.get("status") == "error"]
    if failures:
        print("Search reported failures; retaining state for retry and skipping publish.", file=sys.stderr)
        _write_json(manifest_path, {**coverage, "repo": args.repo, "published": False, "error": True})
        return 2

    # Publish even an empty successful sweep: its coverage and checkpoint are
    # the durable record that lets an ephemeral runner resume past this window.
    # Failed discovery exits above, and publish_run commits the checkpoint in
    # the same Hub commit as the run metadata.
    should_publish = not args.no_publish
    if should_publish:
        # Discovery can take longer than the one-hour lifetime of a GitHub
        # OIDC exchange. Ask huggingface_hub for a fresh token at the point of
        # publication; ordinary local logins and explicit tokens still work.
        hf_token = _hf_token(args.hf_token_env)
        if not hf_token:
            raise ValueError(
                f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login"
            )
        checkpoint_path = "state/checkpoint.json" if mode == "daily" else "state/backfill.json"
        checkpoint = {**(remote_checkpoint or {}), **next_state, "updated_at": observed_at}
        url = publish_run(
            args.repo,
            hf_token,
            run_id=run_id,
            observations_path=observations_path,
            coverage_path=coverage_path,
            checkpoint=checkpoint,
            card_path=card_path,
            **({"checkpoint_path": checkpoint_path} if mode == "backfill" else {}),
        )
        print(f"Published {len(observations)} observations to {url}")

    # A cursor retains both the sweep position and the original `since` value;
    # only a fully completed sweep moves the time window forward.
    next_state["checkpoint"] = checkpoint if should_publish else state.get("checkpoint")
    _write_json(state_path, next_state)
    _write_json(
        manifest_path,
        {
            **coverage,
            "repo": args.repo,
            "observations": observations_path.name,
            "coverage_file": coverage_path.name,
            "published": should_publish,
        },
    )

    print(f"{DISPLAY_NAME} {mode} {run_id}: {len(observations)} candidate repositories, {outcome.requests_used} requests")
    print(f"Coverage: {coverage_path}")
    if not observations and not failures:
        if should_publish:
            print("No candidates found; empty run published with coverage and checkpoint.")
        else:
            print("No candidates found; empty run saved locally.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "backfill":
            return _backfill(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
