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
from .hub import load_checkpoint, publish_readme_run, publish_run
from .readme_enrichment import enrich_readmes
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


def _is_fair_recent_cursor(cursor: Any) -> bool:
    """Return whether a daily cursor belongs to fair recent discovery."""
    return (
        isinstance(cursor, dict)
        and cursor.get("version") == 1
        and cursor.get("field") == "pushed"
        and "since" in cursor
        and "until" in cursor
        and isinstance(cursor.get("lanes"), dict)
        and isinstance(cursor.get("next_index"), int)
    )


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
    run.add_argument("--max-requests", type=int, default=600, help="maximum GitHub search API requests per invocation")
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
    census_daily = subparsers.add_parser("census-daily", help="collect and publish a bounded daily GitHub census delta")
    census_daily.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    census_daily.add_argument("--work-dir", type=Path, required=True, help="directory for isolated census run files")
    census_daily.add_argument("--max-pages", type=int, default=50, help="maximum Core pages to collect (capped at 100)")
    census_daily.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    census_daily.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    census_daily.add_argument("--no-publish", action="store_true", help="collect locally without downloading or publishing Hub state")
    topic_daily = subparsers.add_parser("topic-breadth-daily", help="collect and publish a bounded daily GitHub topic breadth run")
    topic_daily.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    topic_daily.add_argument("--work-dir", type=Path, required=True, help="directory for isolated topic breadth run files")
    topic_daily.add_argument("--max-pages", type=int, default=68, help="maximum GitHub pages to collect (default: 38 topic heads plus 30 deeper pages; capped at 100)")
    topic_daily.add_argument("--topics-config", type=Path, help="optional TOML topic catalog (default: bundled catalog)")
    topic_daily.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    topic_daily.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    topic_daily.add_argument("--no-publish", action="store_true", help="collect locally without downloading or publishing Hub state")
    papers_daily = subparsers.add_parser("hf-papers-daily", help="collect and publish a bounded Hugging Face Daily Papers run")
    papers_daily.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    papers_daily.add_argument("--work-dir", type=Path, required=True, help="directory for isolated papers run files")
    papers_daily.add_argument("--max-pages", type=int, default=20, help="maximum paper pages to collect (1..100)")
    papers_daily.add_argument("--github-batches", type=int, default=4, help="maximum GitHub lookup batches (1..40)")
    papers_daily.add_argument("--paper-detail-budget", type=int, default=400, help="maximum individual paper detail requests (0..1000)")
    papers_daily.add_argument("--paper-page-size", type=int, default=100, help="papers per API page (1..100)")
    papers_daily.add_argument("--recent-days", type=int, default=3, help="recent papers lookback in days (1..7)")
    papers_daily.add_argument("--recent-page-cap", type=int, default=5, help="maximum pages per recent day (1..20)")
    papers_daily.add_argument("--historical-start", default="2023-01-01", help="first date for historical paper collection")
    papers_daily.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    papers_daily.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
    papers_daily.add_argument("--no-publish", action="store_true", help="collect locally without downloading or publishing Hub state")
    snapshot = subparsers.add_parser(
        "publish-current-view",
        help="publish a current-view Parquet snapshot derived from the Hub observation history",
    )
    snapshot.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    snapshot.add_argument("--work-dir", type=Path, required=True, help="temporary directory for downloaded history and generated snapshot")
    readme = subparsers.add_parser("readme-enrich", help="collect bounded compact README evidence for current repositories")
    readme.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO})")
    readme.add_argument("--max-requests", type=int, default=150, help="maximum GitHub README requests per invocation")
    readme.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT / "readme-enrich", help="temporary download and local output directory")
    readme.add_argument("--no-publish", action="store_true", help="write local compact results without publishing to Hugging Face")
    readme.add_argument("--github-token-env", default="GITHUB_TOKEN", help="environment variable holding GitHub token")
    readme.add_argument("--hf-token-env", default="HF_TOKEN", help="environment variable holding Hugging Face token")
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


def _read_jsonl(path: Path) -> Any:
    """Yield JSONL rows without treating Unicode line separators as newlines."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


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


def _census_daily(args: argparse.Namespace, *, api: Any = None, downloader: Any = None,
                  token_provider: Any = None) -> int:
    """Collect a bounded census delta from the pinned Hub checkpoint and publish it."""
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least 1")
    max_pages = min(args.max_pages, 100)
    github_token = _github_token(args.github_token_env)
    if not github_token:
        raise ValueError(f"GitHub token missing from {args.github_token_env} or gh CLI login")
    initial_hf_token = None if args.no_publish else _hf_token(args.hf_token_env)
    if not args.no_publish and not initial_hf_token:
        raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")

    if not args.no_publish:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=initial_hf_token)
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        try:
            info = api.repo_info(args.repo, repo_type="dataset", token=initial_hf_token)
        except TypeError:
            info = api.repo_info(args.repo, repo_type="dataset")
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face dataset lookup failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        base_revision = getattr(info, "sha", None)
        if not isinstance(base_revision, str) or not base_revision:
            raise ValueError("could not pin Hugging Face dataset revision")

    from .census_state import hydrate_census_state, serialize_census_state

    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(_utc_now())
    run_dir = args.work_dir / run_id
    run_dir.mkdir()
    if not args.no_publish:
        try:
            remote_state = downloader(repo_id=args.repo, filename="state/census.json", repo_type="dataset",
                                      revision=base_revision, token=initial_hf_token,
                                      cache_dir=str(args.work_dir / "hf-cache"))
            payload = Path(remote_state).read_bytes()
        except Exception as exc:
            if (getattr(getattr(exc, "response", None), "status_code", None) == 404
                    or type(exc).__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError"}):
                payload = b""
            elif isinstance(exc, HfHubHTTPError):
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                raise ValueError(f"Hugging Face census state download failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
            else:
                raise
        else:
            hydrate_census_state(payload, run_dir)

    prior_checkpoint_path = run_dir / "checkpoint.json"
    prior_checkpoint = json.loads(prior_checkpoint_path.read_text(encoding="utf-8")) if prior_checkpoint_path.exists() else {}
    prior_cursor = prior_checkpoint.get("next_since", 0)
    coverage_dir = run_dir / "coverage"
    before_coverage = {
        path.name: path.read_bytes() for path in coverage_dir.glob("*.json")
    } if coverage_dir.exists() else {}
    before_state_bytes = serialize_census_state(run_dir)
    checkpoint = collect_census(run_dir, token=github_token, max_pages=max_pages)
    coverage_files = sorted(coverage_dir.glob("*.json"))
    changed_coverage = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in coverage_files if before_coverage.get(path.name) != path.read_bytes()
    ]
    coverage_changes = [
        {
            "since": int(path.stem),
            "before": json.loads(before_coverage[path.name]) if path.name in before_coverage else None,
            "after": json.loads(path.read_text(encoding="utf-8")),
        }
        for path in coverage_files if before_coverage.get(path.name) != path.read_bytes()
    ]
    new_coverage_paths = [path for path in coverage_files if path.name not in before_coverage]
    # The state bundle excludes candidate pages, so every page file in this
    # isolated run directory was written or updated during this invocation.
    delta_page_paths = sorted((run_dir / "pages").glob("*.jsonl"))
    observations: dict[int, dict[str, Any]] = {}
    for candidate_page in delta_page_paths:
        if not candidate_page.exists():
            continue
        for row in _read_jsonl(candidate_page):
            repo_id = row.get("github_id")
            if isinstance(repo_id, int) and not isinstance(repo_id, bool):
                observations[repo_id] = row
    observations_path = run_dir / "observations.jsonl"
    observations_path.write_text("".join(json.dumps(observations[key], sort_keys=True) + "\n" for key in sorted(observations)), encoding="utf-8")
    coverage_path = run_dir / "coverage.json"
    page_reports = [json.loads(path.read_text(encoding="utf-8")) for path in new_coverage_paths]
    aggregate = {
        "run_id": run_id, "base_revision": base_revision if not args.no_publish else None,
        "prior_next_since": prior_cursor, "next_since": checkpoint.get("next_since"),
        "checkpoint": checkpoint,
        "pages_collected": len(new_coverage_paths),
        "retry_pages_updated": max(0, len(changed_coverage) - len(new_coverage_paths)),
        "enumerated": sum(int(row.get("enumerated", 0)) for row in page_reports),
        "candidate_count": len(observations),
        "unknown_count": sum(int(row.get("unknown_count", 0)) for row in page_reports),
        "not_candidate_count": sum(int(row.get("not_candidate_count", 0)) for row in page_reports),
        "changed_coverage": changed_coverage,
        "coverage_changes": coverage_changes,
    }
    _write_json(coverage_path, aggregate)
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        _write_json(checkpoint_path, checkpoint)
    state_bytes = serialize_census_state(run_dir)
    has_delta = bool(new_coverage_paths or changed_coverage or state_bytes != before_state_bytes)
    if not args.no_publish and has_delta:
        fresh_token = (token_provider or (lambda: _hf_token(args.hf_token_env)))()
        if not fresh_token:
            raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")
        from .census_publish import publish_census_run
        try:
            url = publish_census_run(args.repo, fresh_token, base_revision=base_revision,
                                     run_id=run_id, observations_path=observations_path,
                                     coverage_path=coverage_path, state_bytes=state_bytes,
                                     api=api, downloader=downloader)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face census publication failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        print(f"Published {len(observations)} census candidates to {url}")
    elif not args.no_publish:
        print("No new census pages collected; skipping Hub publication.")
    print(f"Census daily {run_id}: {len(observations)} candidate repositories across {len(new_coverage_paths)} new pages")
    print(f"Coverage: {coverage_path}")
    return 0


def _topic_breadth_daily(args: argparse.Namespace, *, api: Any = None, downloader: Any = None,
                         token_provider: Any = None, client_factory: Any = None,
                         collector: Any = None) -> int:
    """Collect a bounded topic sweep from pinned Hub state and optionally publish it."""
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least 1")
    max_pages = min(args.max_pages, 100)
    github_token = _github_token(args.github_token_env)
    if not github_token:
        raise ValueError(f"GitHub token missing from {args.github_token_env} or gh CLI login")
    initial_hf_token = None if args.no_publish else _hf_token(args.hf_token_env)
    if not args.no_publish and not initial_hf_token:
        raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")

    base_revision = None
    if not args.no_publish:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=initial_hf_token)
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        try:
            info = api.repo_info(args.repo, repo_type="dataset", token=initial_hf_token)
        except TypeError:
            info = api.repo_info(args.repo, repo_type="dataset")
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face dataset lookup failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        except Exception:
            raise ValueError("Hugging Face dataset lookup failed") from None
        base_revision = getattr(info, "sha", None)
        if not isinstance(base_revision, str) or not base_revision:
            raise ValueError("could not pin Hugging Face dataset revision")

    from .topic_breadth_state import hydrate_topic_state, serialize_topic_state
    from .topic_catalog import load_topics
    if collector is None:
        from .topic_breadth import collect_topic_breadth
        collector = collect_topic_breadth

    topics = load_topics(args.topics_config)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(_utc_now())
    run_dir = args.work_dir / run_id
    run_dir.mkdir()
    if not args.no_publish:
        try:
            remote_state = downloader(repo_id=args.repo, filename="state/topic-breadth.json", repo_type="dataset",
                                      revision=base_revision, token=initial_hf_token,
                                      cache_dir=str(args.work_dir / "hf-cache"))
            before_state_bytes = Path(remote_state).read_bytes()
        except Exception as exc:
            if (getattr(getattr(exc, "response", None), "status_code", None) == 404
                    or type(exc).__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError"}):
                before_state_bytes = serialize_topic_state(run_dir)
            elif isinstance(exc, HfHubHTTPError):
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                raise ValueError(f"Hugging Face topic state download failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
            else:
                raise ValueError("Hugging Face topic state download failed") from None
        else:
            hydrate_topic_state(before_state_bytes, run_dir)
    else:
        before_state_bytes = serialize_topic_state(run_dir)

    client = (client_factory or GitHubClient)(token=github_token)
    result = collector(run_dir, topics=topics, client=client, max_pages=max_pages)
    observation_paths = [Path(path) for path in result.get("observation_paths", [])]
    coverage_paths = [Path(path) for path in result.get("coverage_paths", [])]
    observations: dict[int, dict[str, Any]] = {}
    for path in observation_paths:
        for row in _read_jsonl(path):
            repo_id = row.get("github_id")
            if isinstance(repo_id, int) and not isinstance(repo_id, bool):
                previous = observations.get(repo_id)
                if previous is None:
                    observations[repo_id] = row
                else:
                    names = set(previous.get("topic_names", [])) | set(row.get("topic_names", []))
                    observations[repo_id] = {**previous, **row, "topic_names": sorted(names)}
    observations_path = run_dir / "observations.jsonl"
    observations_path.write_text("".join(json.dumps(observations[key], sort_keys=True) + "\n" for key in sorted(observations)), encoding="utf-8")
    coverage_rows = []
    for path in coverage_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        coverage_rows.append(value)
    coverage_path = run_dir / "coverage.json"
    aggregate = {
        "run_id": run_id, "base_revision": base_revision,
        "pages_fetched": result.get("pages_fetched", 0),
        "observations_written": len(observations),
        "rate_limit_remaining": result.get("rate_limit_remaining"),
        "rate_limit_remaining_by_page": [row.get("rate_limit_remaining") for row in coverage_rows],
        "coverage": coverage_rows,
        "coverage_paths": [str(path) for path in coverage_paths],
        "source_notes": ["GitHub GraphQL topic repository connections; pages correspond to the configured topic catalog.",
                         *[row["source_notes"] for row in coverage_rows if row.get("source_notes") is not None]],
    }
    _write_json(coverage_path, aggregate)
    state_bytes = serialize_topic_state(run_dir)
    has_pages = bool(result.get("pages_fetched", 0))
    has_delta = has_pages or state_bytes != before_state_bytes
    if not args.no_publish and has_delta:
        fresh_token = (token_provider or (lambda: _hf_token(args.hf_token_env)))()
        if not fresh_token:
            raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")
        from .topic_publish import publish_topic_run
        try:
            url = publish_topic_run(args.repo, fresh_token, base_revision=base_revision, run_id=run_id,
                                    observations_path=observations_path, coverage_path=coverage_path,
                                    state_bytes=state_bytes, api=api, downloader=downloader)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face topic publication failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        except Exception:
            raise ValueError("Hugging Face topic publication failed") from None
        print(f"Published {len(observations)} topic breadth observations to {url}")
    elif not args.no_publish:
        print("No topic pages or state changes; skipping Hub publication.")
    print(f"Topic breadth {run_id}: {len(observations)} repositories across {result.get('pages_fetched', 0)} pages")
    print(f"Coverage: {coverage_path}")
    return 0


def _hf_papers_daily(args: argparse.Namespace, *, api: Any = None, downloader: Any = None,
                     token_provider: Any = None, paper_api: Any = None,
                     client_factory: Any = None, collector: Any = None,
                     publisher: Any = None) -> int:
    """Collect a bounded Hugging Face Daily Papers run and optionally publish it."""
    bounds = (("max_pages", 1, 100), ("github_batches", 1, 40),
              ("paper_detail_budget", 0, 1000),
              ("paper_page_size", 1, 100), ("recent_days", 1, 7),
              ("recent_page_cap", 1, 20))
    for name, low, high in bounds:
        value = getattr(args, name)
        if not low <= value <= high:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be between {low} and {high}")

    github_token = _github_token(args.github_token_env)
    initial_hf_token = None if args.no_publish else _hf_token(args.hf_token_env)
    if not args.no_publish and not initial_hf_token:
        raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")

    if paper_api is None:
        from huggingface_hub import HfApi
        paper_api = HfApi(token=False)
    if not args.no_publish:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=initial_hf_token)
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        try:
            info = api.repo_info(args.repo, repo_type="dataset", token=initial_hf_token)
        except TypeError:
            info = api.repo_info(args.repo, repo_type="dataset")
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face dataset lookup failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        except Exception:
            raise ValueError("Hugging Face dataset lookup failed") from None
        base_revision = getattr(info, "sha", None)
        if not isinstance(base_revision, str) or not base_revision:
            raise ValueError("could not pin Hugging Face dataset revision")
    else:
        base_revision = None

    from .hf_papers_state import hydrate_paper_state, serialize_paper_state
    if collector is None:
        from .hf_papers import collect_paper_run
        collector = collect_paper_run
    if client_factory is None:
        client_factory = GitHubClient

    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(_utc_now())
    run_dir = args.work_dir / run_id
    run_dir.mkdir()
    if not args.no_publish:
        try:
            remote_state = downloader(repo_id=args.repo, filename="state/hf-daily-papers.json", repo_type="dataset",
                                      revision=base_revision, token=initial_hf_token,
                                      cache_dir=str(args.work_dir / "hf-cache"))
            before_state_bytes = Path(remote_state).read_bytes()
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                before_state_bytes = serialize_paper_state(run_dir)
            else:
                raise ValueError("Hugging Face paper state download failed") from None
        else:
            hydrate_paper_state(before_state_bytes, run_dir)
    else:
        before_state_bytes = serialize_paper_state(run_dir)

    github = client_factory(token=github_token)
    result = collector(
        run_dir, paper_api=paper_api, github=github,
        today_utc=_utc_now().date().isoformat(), page_budget=args.max_pages,
        github_batch_budget=args.github_batches, paper_page_size=args.paper_page_size,
        paper_detail_budget=args.paper_detail_budget,
        recent_days=args.recent_days, recent_page_cap=args.recent_page_cap,
        historical_start=args.historical_start,
    )
    observations_path = run_dir / "observations.jsonl"
    paper_links_path = run_dir / "paper-links.jsonl"
    coverage_path = run_dir / "coverage.json"
    # The collector owns these canonical run outputs. Validate them before any
    # publication so a partial or malformed source result cannot be published.
    for path in (observations_path, paper_links_path, coverage_path, run_dir / "checkpoint.json"):
        if not path.is_file():
            raise ValueError(f"Hugging Face paper collection did not produce {path.name}")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not isinstance(coverage, dict):
        raise ValueError("Hugging Face paper coverage must be a JSON object")
    if coverage.get("api_errors"):
        raise ValueError("Hugging Face Daily Papers source collection failed; refusing to publish incomplete results")
    state_bytes = serialize_paper_state(run_dir)
    pages = result.get("pages", result.get("pages_collected", result.get("pages_fetched", 0))) if isinstance(result, dict) else 0
    has_delta = bool(pages) or state_bytes != before_state_bytes
    if not args.no_publish and has_delta:
        fresh_token = (token_provider or (lambda: _hf_token(args.hf_token_env)))()
        if not fresh_token:
            raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")
        if publisher is None:
            from .hf_papers_publish import publish_paper_run
            publisher = publish_paper_run
        try:
            url = publisher(args.repo, fresh_token, base_revision=base_revision, run_id=run_id,
                            observations_path=observations_path if observations_path.stat().st_size else None,
                            paper_links_path=paper_links_path,
                            coverage_path=coverage_path, state_bytes=state_bytes, api=api,
                            downloader=downloader)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face paper publication failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        except Exception:
            raise ValueError("Hugging Face paper publication failed") from None
        print(f"Published Hugging Face Daily Papers run to {url}")
    elif not args.no_publish:
        print("No paper pages or state changes; skipping Hub publication.")
    papers_seen = result.get("papers_collected", coverage.get("papers_seen", 0)) if isinstance(result, dict) else 0
    print(
        f"Hugging Face Daily Papers {run_id}: {papers_seen} papers across {pages} pages; "
        f"detail requests {coverage.get('paper_details_attempted', 0)}/"
        f"{args.paper_detail_budget}, with URL {coverage.get('paper_details_with_url', 0)}, "
        f"queued {coverage.get('detail_pending', 0)}, "
        f"errors {coverage.get('paper_details_errors', 0)}"
    )
    print(f"Coverage: {coverage_path}")
    detail_attempts = coverage.get("paper_details_attempted", 0)
    detail_errors = coverage.get("paper_details_errors", 0)
    if detail_attempts > 0 and detail_errors == detail_attempts:
        return 2
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


def _readme_enrich(args: argparse.Namespace, *, api: Any = None, downloader: Any = None,
                   token_provider: Any = None, client_factory: Any = None) -> int:
    """Download a pinned observation history, enrich its current view, and optionally publish."""
    if args.max_requests < 0:
        raise ValueError("--max-requests must be nonnegative")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    hf_token = None if args.no_publish else _hf_token(args.hf_token_env)
    if not args.no_publish and not hf_token:
        raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    try:
        info = api.repo_info(args.repo, repo_type="dataset", token=hf_token)
    except TypeError:
        info = api.repo_info(args.repo, repo_type="dataset")
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        raise ValueError(f"Hugging Face dataset lookup failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise ValueError("could not pin Hugging Face dataset revision")
    try:
        paths = api.list_repo_files(args.repo, repo_type="dataset", revision=revision, token=hf_token)
    except TypeError:
        paths = api.list_repo_files(args.repo, repo_type="dataset", revision=revision)
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        raise ValueError(f"Hugging Face file listing failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
    observation_files = sorted(path for path in paths if path.startswith("data/observations/") and path.endswith(".jsonl"))
    if not observation_files:
        raise ValueError("no observation JSONL files found in the pinned dataset revision")

    downloaded: list[Path] = []
    for index, filename in enumerate(observation_files):
        try:
            local = downloader(repo_id=args.repo, filename=filename, repo_type="dataset", revision=revision, token=hf_token, cache_dir=str(args.work_dir / "hf-cache"))
        except TypeError:
            local = downloader(repo_id=args.repo, filename=filename, repo_type="dataset", revision=revision, token=hf_token)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face history download failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        source = Path(local)
        target = args.work_dir / "history" / f"{index:06d}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            copyfile(source, target)
        downloaded.append(target)
    current_path = args.work_dir / "current-view.jsonl"
    materialize_current_view(downloaded, current_path, manifest_path=args.work_dir / "current-view.manifest.json")
    with current_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    stored = None
    if "state/readme-evidence.json" in paths:
        try:
            raw_checkpoint = downloader(repo_id=args.repo, filename="state/readme-evidence.json", repo_type="dataset", revision=revision, token=hf_token, cache_dir=str(args.work_dir / "hf-cache"))
        except TypeError:
            raw_checkpoint = downloader(repo_id=args.repo, filename="state/readme-evidence.json", repo_type="dataset", revision=revision, token=hf_token)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face README checkpoint download failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        try:
            stored = json.loads(Path(raw_checkpoint).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Hugging Face README checkpoint is unreadable at pinned revision {revision}: {exc}") from exc
    checkpoint = stored.get("checkpoint", {}) if isinstance(stored, dict) else {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    now = _utc_now()
    client = (client_factory or GitHubClient)(token=_github_token(args.github_token_env))
    records, next_checkpoint, coverage = enrich_readmes(rows, checkpoint, client, now=now, max_requests=args.max_requests)
    records_path = args.work_dir / "readme-evidence.jsonl"
    coverage_path = args.work_dir / "coverage.json"
    checkpoint_path = args.work_dir / "checkpoint.json"
    write_jsonl(records, records_path)
    coverage = {**coverage, "dataset_revision": revision, "current_view_count": len(rows)}
    _write_json(coverage_path, coverage)
    _write_json(checkpoint_path, next_checkpoint)
    if coverage.get("rate_limited"):
        print("GitHub rate limit reached; remaining README targets deferred.", file=sys.stderr)
    if not args.no_publish and coverage.get("attempted", 0) > 0:
        fresh_token = (token_provider or (lambda: _hf_token(args.hf_token_env)))()
        if not fresh_token:
            raise ValueError(f"Hugging Face token missing from {args.hf_token_env} or Hugging Face CLI login")
        try:
            url = publish_readme_run(args.repo, fresh_token, records=records, coverage=coverage,
                                     checkpoint=next_checkpoint, run_date=now)
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            raise ValueError(f"Hugging Face README publication failed (HTTP {status if isinstance(status, int) else 'unknown'})") from None
        print(f"Published {len(records)} compact README evidence records to {url}")
    elif not args.no_publish:
        print("No README targets attempted; skipping Hub publication.")
    print(f"README enrichment used {coverage.get('attempted', 0)} client attempts across {len(rows)} current repositories")
    print(f"Local evidence: {records_path}")
    return 2 if coverage.get("rate_limited") else 0


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
    # Daily used a serial query/page cursor before fair scheduling. Keep its
    # frozen window, but discard that cursor so it cannot be misread as lanes.
    # Remember this across policy migration, which may itself clear old cursors.
    daily_legacy_cursor = (
        mode == "daily" and state.get("cursor") is not None
        and not _is_fair_recent_cursor(state.get("cursor"))
    )
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
            daily_legacy_cursor = (
                mode == "daily" and remote_checkpoint.get("cursor") is not None
                and not _is_fair_recent_cursor(remote_checkpoint.get("cursor"))
            )
            remote_checkpoint = _apply_search_policy(remote_checkpoint)
            state.update(remote_checkpoint)
            state["since"] = remote_checkpoint.get("since", initial_since)
            state["cursor"] = remote_checkpoint.get("cursor")
            if mode in {"backfill", "backfill-fair"}:
                state["start"] = remote_checkpoint.get("start", args.start)
                state["end"] = remote_checkpoint.get("end", args.end)
    if daily_legacy_cursor:
        state["cursor"] = None
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
        until = (
            (state.get("until") or now.date().isoformat())
            if state.get("cursor") is not None or daily_legacy_cursor
            else now.date().isoformat()
        )
        state["until"] = until
        if mode == "sample":
            discover_run = lambda cursor: discover_sample(
                client, specs, start=state["since"], end=until,
                max_requests=args.max_requests, cursor=cursor,
            )
        else:
            from .fair_recent import discover_fair_recent

            discover_run = lambda cursor: discover_fair_recent(
                client, specs, since=state["since"], until=until,
                max_requests=args.max_requests, cursor=cursor,
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
        if args.command == "census-daily":
            return _census_daily(args)
        if args.command == "topic-breadth-daily":
            return _topic_breadth_daily(args)
        if args.command == "hf-papers-daily":
            return _hf_papers_daily(args)
        if args.command == "publish-current-view":
            return _publish_current_view(args)
        if args.command == "readme-enrich":
            return _readme_enrich(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
