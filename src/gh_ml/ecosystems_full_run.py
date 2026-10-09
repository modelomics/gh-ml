"""Resumable, low-priority orchestration for a full ecosyste.ms inventory pass."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .ecosystems import EcosystemsClient
from .ecosystems_collection import run_import

ARCHIVE_FLOOR_GIB = 300
DEFAULT_STORAGE_CAP_GIB = 0
SOURCE_VERSION = "unknown"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = SOURCE_VERSION
    digest = hashlib.sha256()
    package = root / "src" / "gh_ml"
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return f"{head}+src.{digest.hexdigest()}"


def _check_space(min_free_gib: int = ARCHIVE_FLOOR_GIB) -> int:
    free = shutil.disk_usage("/mnt/archive").free
    if free < min_free_gib * 1024**3:
        raise OSError(f"archive free space is below the {min_free_gib} GiB floor")
    return free


def _owned_compressed_bytes(out_dir: Path) -> int:
    return sum(p.stat().st_size for p in out_dir.iterdir()
               if p.is_file() and p.name.startswith("repositories-") and p.name.endswith(".jsonl.gz"))


def _permanent_http_status(report: dict[str, Any]) -> int | None:
    status = report.get("http_status")
    if (isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 500
            and status not in (408, 409, 425, 429)):
        return status
    return None


def _compress_verified(path: Path) -> tuple[Path, int, str]:
    """Gzip a completed JSONL delta and verify bytes and records before unlinking raw."""
    free = shutil.disk_usage("/mnt/archive").free
    if free < ARCHIVE_FLOOR_GIB * 1024**3 + path.stat().st_size:
        raise OSError("insufficient compression headroom above the archive free-space floor")
    digest = hashlib.sha256()
    rows = 0
    compressed = path.with_suffix(path.suffix + ".gz")
    temporary = compressed.with_name(f".{compressed.name}.{os.getpid()}.tmp")
    try:
        with path.open("rb") as source, temporary.open("wb") as raw_out:
            with gzip.GzipFile(fileobj=raw_out, mode="wb", compresslevel=6, mtime=0) as out:
                for line in source:
                    digest.update(line)
                    rows += bool(line.strip())
                    out.write(line)
            raw_out.flush()
            os.fsync(raw_out.fileno())
        verified_digest = hashlib.sha256()
        verified_rows = 0
        with gzip.open(temporary, "rb") as stream:
            for line in stream:
                verified_digest.update(line)
                verified_rows += bool(line.strip())
        if verified_rows != rows or verified_digest.digest() != digest.digest():
            raise OSError("compressed delta verification failed")
        os.replace(temporary, compressed)
        return compressed, rows, digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_report(report: dict[str, Any]) -> tuple[Path, int, str]:
    """Compress a JSONL delta once, verify it, and update its receipt."""
    path = Path(report["export_path"])
    if path.suffix == ".gz":
        return path, int(report.get("export_rows", 0)), str(report.get("export_sha256", ""))
    compressed, rows, digest = _compress_verified(path)
    report.update(export_path=str(compressed), export_compression="gzip",
                  export_rows=rows, export_sha256=digest)
    receipt = Path(report["receipt_path"])
    if receipt.is_file():
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data.update(export_path=str(compressed), export_compression="gzip",
                    export_rows=rows, export_sha256=digest)
        _atomic_json(receipt, data)
    # Keep the source delta until the verified archive path and digest are durable.
    path.unlink()
    return compressed, rows, digest


def run_full(*, state_db: Path, run_dir: Path, chunk_pages: int = 10,
             per_page: int = 1000, chunk_seconds: int = 3300,
             max_runtime_seconds: int = 604800, fallback_requests: int = 100,
             storage_cap_gib: int = DEFAULT_STORAGE_CAP_GIB,
             token_env_names: tuple[str, ...] = ("GITHUB_TOKEN",),
             collector: Callable[..., dict[str, Any]] = run_import,
             ecosystems_client_factory: Callable[[], Any] = EcosystemsClient,
             github_client_factory: Callable[..., Any] | None = None,
             sleeper: Callable[[float], None] = time.sleep,
             monotonic: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run primary pages to exhaustion, then retry the unresolved queue.

    Calls are bounded and durable. Reinvocation resumes from SQLite and the
    pinned source revision; `status.json` is suitable for external monitoring.
    """
    if not 1 <= chunk_pages <= 1000 or not 1 <= per_page <= 1000:
        raise ValueError("chunk-pages and per-page must be from 1 through 1000")
    if chunk_seconds < 1 or max_runtime_seconds < 1 or fallback_requests < 0:
        raise ValueError("time limits must be positive and fallback requests nonnegative")
    if storage_cap_gib < 0:
        raise ValueError("storage cap must be nonnegative")
    state_db, run_dir = Path(state_db).resolve(), Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "deltas"
    out_dir.mkdir(exist_ok=True)
    status_path = run_dir / "status.json"
    revision = _source_revision()
    lock_path = run_dir / "runner.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another full inventory runner holds the run lock") from None
        prior: dict[str, Any] = {}
        if status_path.exists():
            prior = json.loads(status_path.read_text(encoding="utf-8"))
            if prior.get("source_revision") != revision:
                raise RuntimeError("source revision differs from pinned run; resume with the original code revision")
        elif prior.get("source_revision"):
            revision = prior["source_revision"]
        started = monotonic()
        deadline = started + max_runtime_seconds
        phase = prior.get("phase", "primary")
        backoff = max(10, float(prior.get("retry_delay_seconds", 10)))
        status: dict[str, Any] = {
            **prior, "status": "running", "phase": phase, "source_revision": revision,
            "state_db": str(state_db), "run_dir": str(run_dir), "started_at": prior.get("started_at", _now()),
            "updated_at": _now(), "archive_free_bytes": _check_space(),
            "compressed_delta_bytes": _owned_compressed_bytes(out_dir),
        }
        _atomic_json(status_path, status)
        eco_client: Any = None
        github_client: Any = None
        while monotonic() < deadline:
            try:
                free = _check_space()
                used = _owned_compressed_bytes(out_dir)
                if storage_cap_gib and used >= storage_cap_gib * 1024**3:
                    status.update(status="stopped_space_cap", updated_at=_now(), archive_free_bytes=free,
                                  compressed_delta_bytes=used, error="runner storage cap reached")
                    _atomic_json(status_path, status)
                    return status
                if _source_revision() != revision:
                    raise RuntimeError("source revision changed while runner was active")
                if eco_client is None:
                    eco_client = ecosystems_client_factory()
                chunk_deadline = min(deadline, monotonic() + chunk_seconds)
                current_phase = phase
                if phase == "primary":
                    report = collector(state_db=state_db, output_dir=out_dir,
                                       ecosystems_client=eco_client, github_client=None,
                                       max_pages=chunk_pages, per_page=per_page,
                                       max_github_requests=0, deadline=chunk_deadline,
                                       min_free_gib=ARCHIVE_FLOOR_GIB, process_queue=False,
                                       export_batch_limit=chunk_pages * per_page)
                    initial_report = report
                    compressed, _rows, _digest = _finalize_report(report)
                    # A bounded inventory chunk can touch as many as pages × page-size
                    # repositories. Drain all queued deltas before advancing the phase.
                    while report.get("pending_export_queue", 0):
                        if _source_revision() != revision:
                            raise RuntimeError("source revision changed while runner was active")
                        report = collector(state_db=state_db, output_dir=out_dir,
                                           ecosystems_client=eco_client, github_client=None,
                                           max_pages=0, per_page=per_page,
                                           max_github_requests=0, deadline=chunk_deadline,
                                           min_free_gib=ARCHIVE_FLOOR_GIB, process_queue=False,
                                           export_batch_limit=chunk_pages * per_page)
                        compressed, _rows, _digest = _finalize_report(report)
                    if report is not initial_report:
                        report["pages_requested"] = initial_report.get("pages_requested", 0)
                        report["primary_used"] = initial_report.get("primary_used", 0)
                        report["cursor"] = initial_report.get("cursor", report.get("cursor"))
                        report["total_repositories"] = initial_report.get("total_repositories")
                else:
                    if github_client is None and github_client_factory is not None:
                        github_client = github_client_factory(token_env_names, deadline=chunk_deadline)
                    report = collector(state_db=state_db, output_dir=out_dir,
                                       ecosystems_client=eco_client, github_client=github_client,
                                       max_pages=0, per_page=per_page,
                                       max_github_requests=fallback_requests,
                                       deadline=chunk_deadline,
                                       min_free_gib=ARCHIVE_FLOOR_GIB, process_queue=True,
                                       queue_target_limit=5000, export_batch_limit=10_000)
                    compressed, _rows, _digest = _finalize_report(report)
                permanent_status = _permanent_http_status(report)
                if permanent_status is not None:
                    status.update(status="stopped_source_error", phase=phase, updated_at=_now(), retry_at=None,
                                  cursor=report.get("cursor"), pages_requested=report.get("pages_requested"),
                                  last_report_status=report.get("status"),
                                  last_error=f"ecosyste.ms returned permanent HTTP {permanent_status}",
                                  archive_free_bytes=_check_space(),
                                  compressed_delta_bytes=_owned_compressed_bytes(out_dir),
                                  latest_export=str(compressed), latest_receipt=str(report["receipt_path"]))
                    _atomic_json(status_path, status)
                    return status
                receipt = Path(report["receipt_path"])
                cursor = report.get("cursor", {})
                if phase == "primary" and cursor.get("ended"):
                    phase = "fallback"
                status.update(
                    status="running", phase=phase, updated_at=_now(), last_run_id=report.get("run_id"),
                    cursor=cursor, total_repositories=report.get("total_repositories"),
                    remaining_queue=report.get("remaining_queue"), pending_reasons=report.get("pending_reasons", {}),
                    pages_requested=report.get("pages_requested"), last_report_status=report.get("status"),
                    last_error=None if report.get("status") == "complete" else
                    (report.get("error") or report.get("pending_reasons")),
                    retry_delay_seconds=10 if report.get("status") == "complete" else backoff,
                    retry_at=None if report.get("status") == "complete" else _now(),
                    archive_free_bytes=_check_space(), compressed_delta_bytes=_owned_compressed_bytes(out_dir),
                    latest_export=str(compressed), latest_receipt=str(receipt),
                )
                history = list(status.get("phase_runs", []))
                history.append({"phase": current_phase, "run_id": report.get("run_id"),
                                "status": report.get("status"), "cursor": cursor,
                                "remaining_queue": report.get("remaining_queue"),
                                "pending_export_queue": report.get("pending_export_queue"),
                                "export_path": str(compressed), "receipt_path": str(receipt)})
                status["phase_runs"] = history[-20:]
                _atomic_json(status_path, status)
                if (phase == "fallback" and not report.get("remaining_queue")
                        and not report.get("pending_export_queue")):
                    status.update(status="complete", updated_at=_now(), retry_at=None)
                    _atomic_json(status_path, status)
                    return status
                if report.get("status") == "complete":
                    backoff = 10
                else:
                    status["retry_at"] = datetime.fromtimestamp(time.time() + backoff, UTC).isoformat().replace("+00:00", "Z")
                    _atomic_json(status_path, status)
                    sleeper(backoff)
                    backoff = min(3600, backoff * 2)
                    status["retry_delay_seconds"] = backoff
            except Exception as exc:
                message = str(exc)
                if "source revision" in message or "archive free space is below" in message:
                    stop_status = "stopped_space_floor" if "archive free space is below" in message else "stopped_source_changed"
                    status.update(status=stop_status, phase=phase, updated_at=_now(), retry_at=None,
                                  last_error=f"{type(exc).__name__}: {message[:300]}",
                                  archive_free_bytes=shutil.disk_usage("/mnt/archive").free,
                                  compressed_delta_bytes=_owned_compressed_bytes(out_dir))
                    _atomic_json(status_path, status)
                    return status
                backoff = min(3600, backoff * 2)
                status.update(status="retrying", phase=phase, updated_at=_now(),
                              retry_delay_seconds=backoff,
                              retry_at=datetime.fromtimestamp(time.time() + backoff, UTC).isoformat().replace("+00:00", "Z"),
                              last_error=f"{type(exc).__name__}: {str(exc)[:300]}",
                              archive_free_bytes=shutil.disk_usage("/mnt/archive").free,
                              compressed_delta_bytes=_owned_compressed_bytes(out_dir))
                _atomic_json(status_path, status)
                sleeper(backoff)
        status.update(status="stopped_runtime", phase=phase, updated_at=_now(), retry_at=None,
                      archive_free_bytes=shutil.disk_usage("/mnt/archive").free,
                      compressed_delta_bytes=_owned_compressed_bytes(out_dir))
        _atomic_json(status_path, status)
        return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run a resumable ecosyste.ms-first full repository inventory")
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-pages", type=int, default=10)
    parser.add_argument("--per-page", type=int, default=1000)
    parser.add_argument("--chunk-seconds", type=int, default=3300)
    parser.add_argument("--max-runtime-seconds", type=int, default=604800)
    parser.add_argument("--fallback-requests", type=int, default=100)
    parser.add_argument("--storage-cap-gib", type=int, default=DEFAULT_STORAGE_CAP_GIB)
    parser.add_argument("--mailto", default=os.environ.get("ECOSYSTEMS_MAILTO"),
                        help="contact email sent in the ecosyste.ms From header (or ECOSYSTEMS_MAILTO)")
    parser.add_argument("--github-token-env", action="append", default=[])
    args = parser.parse_args(argv)
    from .github_tokens import build_pooled_client

    token_names = tuple(args.github_token_env or ["GITHUB_TOKEN"])

    def github_factory(names: tuple[str, ...], *, deadline: float | None = None) -> Any:
        from .cli import _github_token
        token = next((os.environ[n] for n in names if os.environ.get(n)), None)
        if token is None:
            token = _github_token(names[0])
        return build_pooled_client(names, fallback_token=token, deadline=deadline)

    try:
        def ecosystems_factory() -> EcosystemsClient:
            return EcosystemsClient(mailto=args.mailto)

        result = run_full(state_db=args.state_db, run_dir=args.run_dir,
                          chunk_pages=args.chunk_pages, per_page=args.per_page,
                          chunk_seconds=args.chunk_seconds, max_runtime_seconds=args.max_runtime_seconds,
                          fallback_requests=args.fallback_requests, storage_cap_gib=args.storage_cap_gib,
                          token_env_names=token_names, github_client_factory=github_factory,
                          ecosystems_client_factory=ecosystems_factory)
    except Exception as exc:
        print(f"full inventory runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
