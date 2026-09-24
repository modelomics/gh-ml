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

from huggingface_hub.errors import HfHubHTTPError

from .classification import classify_repository
from .discovery import discover, discover_sample
from .historical_ledger import discover_historical_ledger
from .github import GitHubAPIError, GitHubClient, SearchProgress
from .hub import load_checkpoint, publish_run
from .query_catalog import load_queries
from .schema import observation_from_repository, write_jsonl
from .pwc import DEFAULT_DIR as DEFAULT_PWC_DIR, import_pwc
from .current_view import materialize_current_view, export_current_view_parquet
from .census import collect_census
from .snapshot_publish import publish_current_view

DEFAULT_REPO = "modelomics/gh-ml"
DISPLAY_NAME = "GitHub ML"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "queries"
DEFAULT_OUTPUT = Path.home() / ".local" / "share" / "modelomics-gh-ml" / "runs"
DEFAULT_CURRENT_VIEW_OUTPUT = Path.home() / ".local" / "share" / "modelomics-gh-ml" / "current-view.jsonl"
DEFAULT_CENSUS_OUTPUT = Path.home() / ".local" / "share" / "modelomics-gh-ml" / "census"
SEARCH_POLICY_VERSION = 2


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


def _apply_search_policy(state: dict[str, Any]) -> dict[str, Any]:
    """Restart search progress when its filtering policy changes."""
    if state.get("search_policy_version") == SEARCH_POLICY_VERSION:
        return state
    migrated = dict(state)
    # Cursors can encode page offsets, partition positions, or completed-year
    # ledgers. None of those are valid under a changed query/filter policy.
    migrated["cursor"] = None
    migrated.pop("complete", None)
    migrated["search_policy_version"] = SEARCH_POLICY_VERSION
    return migrated


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
    sample = subparsers.add_parser("sample", help="sample newly created GitHub ML repositories")
    sample.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    sample.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="directory of TOML query files")
    sample.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="local run and checkpoint directory")
    sample.add_argument("--max-requests", type=int, default=500, help="maximum GitHub search API requests per invocation")
    sample.add_argument("--since-days", type=int, default=1, help="lookback interval when starting a new sample window")
    sample.add_argument("--no-publish", action="store_true", help="write local files without publishing to Hugging Face")
    sample.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    sample.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
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
    fair_backfill = subparsers.add_parser("backfill-fair", help="fairly backfill historical GitHub ML repository candidates")
    fair_backfill.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    fair_backfill.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="directory of TOML query files")
    fair_backfill.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="local run and checkpoint directory")
    fair_backfill.add_argument("--start", default="2008-01-01", help="first GitHub repository created date to search (default: 2008-01-01)")
    fair_backfill.add_argument("--end", help="last GitHub repository created date to search (default: today in UTC)")
    fair_backfill.add_argument("--max-requests", type=int, default=100, help="maximum GitHub search API requests per invocation")
    fair_backfill.add_argument("--no-publish", action="store_true", help="write local files without publishing to Hugging Face")
    fair_backfill.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    fair_backfill.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    historical = subparsers.add_parser("historical-sample", help="sample repositories across historical creation years")
    historical.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    historical.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG, help="directory of TOML query files")
    historical.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="local run and checkpoint directory")
    historical.add_argument("--max-requests", type=int, default=200, help="maximum GitHub search API requests per invocation")
    historical.add_argument("--start-year", type=int, default=2008, help="first repository creation year to search")
    historical.add_argument("--end", help="last creation date to search (default: today in UTC)")
    historical.add_argument("--no-publish", action="store_true", help="write local files without publishing to Hugging Face")
    historical.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    historical.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    pwc = subparsers.add_parser("pwc-import", help="locally import a bounded batch from the pinned Papers with Code archive")
    pwc.add_argument("--output-dir", type=Path, default=DEFAULT_PWC_DIR, help="local run and checkpoint directory")
    pwc.add_argument("--max-rows", type=int, default=10_000, help="maximum source rows scanned per invocation")
    pwc.add_argument("--max-repos", type=int, default=500, help="maximum GitHub repository resolutions per invocation")
    pwc.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    current_view = subparsers.add_parser("current-view", help="materialize the latest local observation for each GitHub repository")
    current_view.add_argument("dataset_root", type=Path, help="local downloaded dataset root containing data/observations JSONL files")
    current_view.add_argument("--output", type=Path, default=DEFAULT_CURRENT_VIEW_OUTPUT, help="output current-view JSONL path")
    current_view.add_argument("--manifest", type=Path, help="manifest path (default: <output>.manifest.json)")
    current_view.add_argument("--parquet-output", type=Path, help="also export current view to Parquet (requires the parquet extra)")
    census = subparsers.add_parser("census", help="locally collect bounded GitHub Core census pages as candidates")
    census.add_argument("--output-dir", type=Path, default=DEFAULT_CENSUS_OUTPUT, help="local census output directory")
    census.add_argument("--since", type=int, help="GitHub repository ID cursor (default: resume local checkpoint)")
    census.add_argument("--max-pages", type=int, default=1, help="maximum Core pages to collect per invocation")
    census.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    snapshot = subparsers.add_parser(
        "publish-current-view",
        help="publish a current-view Parquet snapshot derived from the Hub observation history",
    )
    snapshot.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    snapshot.add_argument("--work-dir", type=Path, required=True, help="temporary directory for downloaded history and generated snapshot")
    return parser


def _pwc_import(args: argparse.Namespace) -> int:
    token = _github_token(args.github_token_env)
    manifest = import_pwc(output_dir=args.output_dir, max_rows=args.max_rows,
                          max_repos=args.max_repos, client=GitHubClient(token=token))
    print(f"PWC import scanned {manifest['rows_scanned']} rows; resolved {manifest['repositories_resolved']} repositories")
    manifest_path = args.output_dir / f"manifest-{manifest['run_id']}.json"
    print(f"Manifest: {manifest_path}")
    return 2 if manifest.get("resolution_error") else 0


def _current_view(args: argparse.Namespace) -> int:
    sources = sorted(args.dataset_root.glob("data/observations/**/*.jsonl"))
    if not sources:
        raise ValueError(f"no observation JSONL files found under {args.dataset_root / 'data/observations'}")
    manifest = materialize_current_view(sources, args.output, manifest_path=args.manifest)
    print(f"Current view: {manifest['current_view_count']} repositories from {manifest['observation_count']} observations")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest or args.output.with_suffix(args.output.suffix + '.manifest.json')}")
    if args.parquet_output:
        result = export_current_view_parquet(args.output, args.parquet_output)
        print(f"Parquet: {args.parquet_output} ({result['row_count']} rows)")
    return 0


def _census(args: argparse.Namespace) -> int:
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least 1")
    checkpoint = collect_census(
        args.output_dir,
        token=_github_token(args.github_token_env),
        since=args.since,
        max_pages=args.max_pages,
    )
    print(f"Census checkpoint: {args.output_dir / 'checkpoint.json'}")
    print(f"Next GitHub ID cursor: {checkpoint.get('next_since')}")
    return 0


def _publish_current_view(args: argparse.Namespace) -> int:
    # OIDC exchanges are short-lived. Resolve credentials immediately before
    # publication, after collection has completed in the scheduled workflow.
    token = _hf_token("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token missing from HF_TOKEN or Hugging Face CLI login")
    try:
        result = publish_current_view(
            args.repo,
            token,
            work_dir=args.work_dir,
            token_provider=lambda: _hf_token("HF_TOKEN"),
        )
    except HfHubHTTPError as exc:
        # Hub HTTP exception messages may contain response bodies. Report only
        # the status for this expected remote failure, without echoing details.
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        status_text = str(status) if isinstance(status, int) else "unknown"
        raise ValueError(f"Hugging Face snapshot publication failed (HTTP {status_text})") from None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    return _collect(args, mode="daily")


def _backfill(args: argparse.Namespace) -> int:
    return _collect(args, mode="backfill")


def _backfill_fair(args: argparse.Namespace) -> int:
    return _collect(args, mode="backfill-fair")


def _sample(args: argparse.Namespace) -> int:
    return _collect(args, mode="sample")


def _historical_sample(args: argparse.Namespace) -> int:
    return _collect(args, mode="historical-sample")


def _collect(args: argparse.Namespace, *, mode: str) -> int:
    if args.max_requests < 1:
        raise ValueError("--max-requests must be at least 1")
    if mode in {"daily", "sample"} and args.since_days < 1:
        raise ValueError("--since-days must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = _utc_now()
    run_id = _run_id(now)
    state_path = args.output_dir / ({"daily": "state.json", "backfill": "backfill-state.json", "backfill-fair": "backfill-fair-state.json", "sample": "sample-state.json", "historical-sample": "historical-sample-state.json"}[mode])
    initial_since = (now - timedelta(days=args.since_days)).date().isoformat() if mode in {"daily", "sample"} else getattr(args, "start", "")
    if mode == "historical-sample":
        if args.start_year < 2008 or args.start_year > now.year:
            raise ValueError("--start-year must be between 2008 and the current UTC year")
        requested_end = args.end or now.date().isoformat()
        try:
            end_date = datetime.strptime(requested_end, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("--end must be a valid YYYY-MM-DD date") from None
        if end_date.isoformat() != requested_end or end_date > now.date():
            raise ValueError("--end must be a valid date no later than today in UTC")
        state = _read_state(state_path, requested_end)
    else:
        state = _read_state(state_path, initial_since)
    state = _apply_search_policy(state)
    specs = None
    if mode == "historical-sample":
        specs = load_queries(args.config_dir)
        if not specs:
            raise ValueError(f"no queries found in {args.config_dir}")
        current_signature = [{"id": spec.id, "query": spec.q} for spec in specs]
        if (state.get("search_policy_version") == SEARCH_POLICY_VERSION
                and state.get("complete") and state.get("catalog_signature") == current_signature
                and state.get("start_year") == args.start_year):
            print("Historical sample already complete for this query catalog; skipping.")
            return 0
    github_token = _github_token(args.github_token_env)
    hf_token = _hf_token(args.hf_token_env)
    remote_checkpoint: dict[str, Any] | None = None
    if not args.no_publish:
        if not hf_token:
            raise ValueError(
                f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login"
            )
        remote_path = {"daily": "state/checkpoint.json", "backfill": "state/backfill.json", "backfill-fair": "state/backfill-fair.json", "sample": "state/sample.json", "historical-sample": "state/historical-sample.json"}[mode]
        remote_checkpoint = load_checkpoint(args.repo, hf_token, checkpoint_path=remote_path)
        if not state["initialized"] and isinstance(remote_checkpoint, dict):
            remote_checkpoint = _apply_search_policy(remote_checkpoint)
            state.update(remote_checkpoint)
            state["since"] = remote_checkpoint.get("since", initial_since)
            state["cursor"] = remote_checkpoint.get("cursor")
            if mode in {"backfill", "backfill-fair"}:
                state["start"] = remote_checkpoint.get("start", args.start)
                state["end"] = remote_checkpoint.get("end", args.end)
    if specs is None:
        specs = load_queries(args.config_dir)
    if not specs:
        raise ValueError(f"no queries found in {args.config_dir}")

    catalog_signature = [{"id": spec.id, "query": spec.q} for spec in specs]
    if mode == "historical-sample":
        matches = state.get("catalog_signature") == catalog_signature and state.get("start_year") == args.start_year
        if (state.get("search_policy_version") == SEARCH_POLICY_VERSION
                and state.get("complete") and matches):
            print("Historical sample already complete for this query catalog; skipping.")
            return 0
        # Keep the original campaign bounds and cursor when the query catalog
        # changes. The ledger reconciles v1/v2 cursors by query/year identity.
        # An old completed checkpoint with no cursor safely starts a full grid.
        if "end" not in state:
            state["end"] = requested_end
        state["start_year"] = args.start_year
        state["catalog_signature"] = catalog_signature
    if mode == "backfill-fair":
        matches = state.get("catalog_signature") == catalog_signature
        if state.get("complete") and matches:
            print("Fair backfill already complete for this query catalog; skipping.")
            return 0
        if state.get("complete") and not matches:
            state["cursor"] = None
            state.pop("complete", None)
            # A changed catalog starts a new sweep with the requested bounds.
            state["start"] = args.start
            state["end"] = args.end or now.date().isoformat()
        elif state.get("cursor") is None:
            state["start"] = state.get("start", args.start)
            state["end"] = state.get("end") or args.end or now.date().isoformat()
        state["catalog_signature"] = catalog_signature

    client = GitHubClient(token=github_token)

    def report_search_progress(progress: SearchProgress) -> None:
        if progress.completed % 25:
            return
        elapsed = progress.elapsed_seconds
        details = []
        if progress.status is not None:
            details.append(f"HTTP {progress.status}")
        if progress.remaining is not None:
            details.append(f"rate remaining {progress.remaining}")
        suffix = f" ({', '.join(details)})" if details else ""
        print(
            f"GitHub Search: {progress.completed} requests completed in {elapsed:.1f}s{suffix}",
            file=sys.stderr,
            flush=True,
        )

    set_progress_callback = getattr(client, "set_progress_callback", None)
    if callable(set_progress_callback):
        set_progress_callback(report_search_progress)
    backfill_start = state.get("start", args.start) if mode in {"backfill", "backfill-fair"} else None
    backfill_end = state.get("end", args.end or now.date().isoformat()) if mode in {"backfill", "backfill-fair"} else None
    if mode == "historical-sample":
        historical_end = state["end"]
        discover_run = lambda cursor: discover_historical_ledger(
            client, specs, start_year=args.start_year, end=historical_end,
            max_requests=args.max_requests, cursor=cursor,
        )
    elif mode in {"daily", "sample"}:
        # A new sweep gets a fresh upper bound; a truncated sweep retains
        # the exact bound alongside its cursor so it can resume safely.
        until = state.get("until") if state.get("cursor") is not None else now.date().isoformat()
        state["until"] = until
        if mode == "sample":
            discover_run = lambda cursor: discover_sample(
                client, specs, start=state["since"], end=until,
                max_requests=args.max_requests, cursor=cursor,
            )
        else:
            discover_run = lambda cursor: discover(
                client,
                specs,
                since=state["since"],
                until=until,
                max_requests=args.max_requests,
                cursor=cursor,
            )
    elif mode == "backfill-fair":
        from .fair_backfill import discover_fair_backfill

        discover_run = lambda cursor: discover_fair_backfill(
            client, specs, start=backfill_start, end=backfill_end,
            max_requests=args.max_requests, cursor=cursor,
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
            if mode in {"backfill-fair", "historical-sample"} or str(exc) != "cursor specs does not match this discovery run":
                raise
            print(
                f"Query catalog changed; restarting {mode} window "
                f"from the beginning ({state.get('since') if mode in {'daily', 'sample'} else backfill_start}"
                f"..{until if mode in {'daily', 'sample'} else backfill_end}).",
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
                **({"since": state["since"], "until": state.get("until")} if mode in {"daily", "sample"} else {"start": backfill_start, "end": backfill_end}),
                **({"mode": "sample", "date_field": "created"} if mode == "sample" else ({"mode": mode, "date_field": "created", "start_year": args.start_year, "end": state.get("end")} if mode == "historical-sample" else ({"mode": mode, "date_field": "created"} if mode == "backfill-fair" else {}))),
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
        **({"since": state["since"], "until": state["until"]} if mode in {"daily", "sample"} else {"start": state["start"], "end": state["end"]} if mode in {"backfill", "backfill-fair"} else {"end": state["end"]}),
        **({"mode": "sample", "date_field": "created"} if mode == "sample" else ({"mode": mode, "date_field": "created", "start_year": args.start_year} if mode == "historical-sample" else ({"mode": mode, "date_field": "created"} if mode == "backfill-fair" else {}))),
        "requests_used": outcome.requests_used,
        "complete_sweep": (bool(outcome.next_cursor.get("complete")) if mode == "historical-sample" else outcome.next_cursor is None),
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

    if mode in {"daily", "sample"}:
        if outcome.next_cursor is None:
            # Start the next daily pass at the previous upper bound (inclusive
            # overlap) and choose its upper bound when that pass begins.
            next_state = {"since": state.get("until", now.date().isoformat()), "cursor": None}
        else:
            next_state = {"since": state["since"], "until": state["until"], "cursor": outcome.next_cursor}
    elif mode == "historical-sample":
        next_state = {
            "start_year": args.start_year, "end": state["end"],
            "cursor": outcome.next_cursor, "complete": bool(outcome.next_cursor.get("complete")),
            "catalog_signature": catalog_signature,
        }
    elif mode == "backfill-fair":
        next_state = {
            "start": state["start"], "end": state["end"],
            "cursor": outcome.next_cursor, "complete": outcome.next_cursor is None,
            "catalog_signature": catalog_signature,
        }
    else:
        next_state = {"start": state["start"], "end": state["end"], "cursor": outcome.next_cursor}
    next_state["search_policy_version"] = SEARCH_POLICY_VERSION
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
        checkpoint_path = {"daily": "state/checkpoint.json", "backfill": "state/backfill.json", "backfill-fair": "state/backfill-fair.json", "sample": "state/sample.json", "historical-sample": "state/historical-sample.json"}[mode]
        checkpoint = {**(remote_checkpoint or {}), **next_state,
                      "search_policy_version": SEARCH_POLICY_VERSION, "updated_at": observed_at}
        url = publish_run(
            args.repo,
            hf_token,
            run_id=run_id,
            observations_path=observations_path,
            coverage_path=coverage_path,
            checkpoint=checkpoint,
            card_path=card_path,
            checkpoint_path=checkpoint_path,
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
        if args.command == "backfill-fair":
            return _backfill_fair(args)
        if args.command == "sample":
            return _sample(args)
        if args.command == "historical-sample":
            return _historical_sample(args)
        if args.command == "pwc-import":
            return _pwc_import(args)
        if args.command == "current-view":
            return _current_view(args)
        if args.command == "census":
            return _census(args)
        if args.command == "publish-current-view":
            return _publish_current_view(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
