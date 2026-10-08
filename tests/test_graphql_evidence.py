from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import hashlib
import zlib

from gh_ml.graphql_evidence import EvidenceStore, export_run, run_collection
import gh_ml.graphql_evidence as graphql_evidence


def repository(repo_id: int, name: str | None = None, *, fork: bool = False, revision: str = "r1") -> dict[str, Any]:
    return {
        "github_id": repo_id,
        "full_name": name or f"lab/repo-{repo_id}",
        "fork": fork,
        "revision": revision,
    }


def evidence_item(row: dict[str, Any], *, text: str = "# Method\nA model.", status: str = "ok") -> dict[str, Any]:
    return {
        "github_id": row["github_id"],
        "full_name": row["full_name"],
        "status": status,
        "text": text if status == "ok" else None,
        "path": "README.md" if status == "ok" else None,
        "commit_sha": row.get("revision", "r1"),
        "blob_sha": "blob-shared" if status == "ok" else None,
        "canonical_name": row["full_name"],
        "error": None if status == "ok" else "transport failure",
    }


class Fetcher:
    def __init__(self, *, failures: bool = False, delay: float = 0.0) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.failures = failures
        self.delay = delay

    def __call__(self, _client: object, targets: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.calls.append([dict(target) for target in targets])
        items = [
            evidence_item(target, status="error" if self.failures else "ok")
            for target in targets
        ]
        return {
            "items": items,
            "requests": 1,
            "cost": 2,
            "remaining": 4998,
            "reset_at": "2026-10-08T00:00:00Z",
            "rate_limited": False,
        }


class CrashingFetcher(Fetcher):
    def __call__(self, client: object, targets: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        super().__call__(client, targets, **kwargs)
        raise RuntimeError("simulated worker crash")


def _open(path: Path) -> EvidenceStore:
    return EvidenceStore(path, min_free_bytes=0)


class DeferredModel:
    version = "test-triage-v1"
    fingerprint = "model-test-1"
    defer_threshold = 0.01

    def predict(self, row):
        from gh_ml.metadata_triage import metadata_fingerprint

        deferred = row.get("description") == "likely non-ML"
        return {
            "decision": "defer" if deferred else "fetch",
            "predicted_label": "not_ml_relevant" if deferred else "ml_relevant",
            "model_score": 0.001 if deferred else 0.9,
            "reason": "test score",
            "experimental": True,
            "metadata_fingerprint": metadata_fingerprint(row),
        }


def _git_blob_sha(text: str) -> str:
    raw = text.encode()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def test_configured_free_space_floor_blocks_store_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Statvfs:
        f_bavail = 1
        f_frsize = 1

    monkeypatch.setattr("gh_ml.graphql_evidence.os.statvfs", lambda _path: Statvfs())
    path = tmp_path / "evidence.sqlite"

    with pytest.raises((OSError, RuntimeError), match="free|space|capacity"):
        with EvidenceStore(path, min_free_bytes=2):
            pass
    assert not path.exists()


def test_blob_cache_lookup_verifies_git_oid_and_recovers_from_corruption(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    text = "# README\nCached model evidence."
    blob = _git_blob_sha(text)
    with _open(path) as store:
        plan = " ".join(
            row["detail"] for row in store.db.execute(
                "EXPLAIN QUERY PLAN SELECT DISTINCT r.content_sha256 FROM raw_provenance p "
                "JOIN raw_readmes r USING(content_sha256) WHERE lower(p.blob_sha)=lower(?) ORDER BY p.fetched_at DESC",
                (_git_blob_sha(text),),
            )
        )
        assert "SEARCH p USING INDEX raw_provenance_blob_lower" in plan
        rows = [repository(801), repository(802)]
        store.ingest(rows, source="inventory")
        items = [evidence_item(row, text=text) for row in rows]
        for item in items:
            item["blob_sha"] = blob
        run_collection(store, object(), fetcher=lambda *_args, **_kwargs: {
            "items": items, "requests": 1, "unique_blobs_downloaded": 1,
            "download_bytes": len(text.encode()),
        })
        assert store.lookup_blob_text(blob) == text
        assert store.lookup_blob_text("0" * 40) is None
        assert store.db.execute("SELECT COUNT(*) FROM raw_readmes").fetchone()[0] == 1
        assert store.db.execute("SELECT COUNT(*) FROM raw_provenance WHERE blob_sha=?", (blob,)).fetchone()[0] == 2
        store.db.execute("UPDATE raw_readmes SET compressed_text=?", (b"corrupt",))
        store.db.commit()
        assert store.lookup_blob_text(blob) is None


def test_deferred_audit_rotates_by_epoch_and_never_fetches_unselected_rows(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    path = tmp_path / "triage.sqlite"
    model = DeferredModel()
    rows = [
        {"github_id": 10_000 + index, "full_name": f"lab/repo-{index}", "description": "likely non-ML"}
        for index in range(500)
    ]
    with _open(path) as store:
        store.ingest(rows, source="inventory", triage_model=model, audit_rate=0.1, audit_seed="rotate",
                     now=datetime(2026, 9, 8, tzinfo=UTC))
        first_epoch = datetime(2026, 10, 8, tzinfo=UTC)
        first = store.rotate_deferred_audit(model, audit_rate=0.1, audit_seed="rotate", now=first_epoch)
        selected_first = {
            row[0] for row in store.db.execute(
                "SELECT github_id FROM metadata_triage WHERE audit_epoch='2026-10' AND audit_selected=1"
            )
        }
        assert first["updated"] == 500
        assert 20 <= len(selected_first) <= 80
        second_epoch = datetime(2026, 11, 8, tzinfo=UTC)
        second = store.rotate_deferred_audit(model, audit_rate=0.1, audit_seed="rotate", now=second_epoch)
        selected_second = {
            row[0] for row in store.db.execute(
                "SELECT github_id FROM metadata_triage WHERE audit_epoch='2026-11' AND audit_selected=1"
            )
        }
        assert second["updated"] == 500
        assert selected_second - selected_first
        assert store.triage_scope_counts(model.fingerprint) == {
            "assessed": 500, "fetch": 0, "deferred": 500,
            "audit_selected": len(selected_second),
        }
        # Every unselected deferral remains outside the due collection queue,
        # even after its 30-day reconsideration date has arrived.
        future = datetime(2026, 12, 9, tzinfo=UTC)
        third = store.rotate_deferred_audit(model, audit_rate=0.1, audit_seed="rotate", now=future)
        assert third["updated"] == 500
        selected_third = {
            row[0] for row in store.db.execute(
                "SELECT github_id FROM metadata_triage WHERE audit_epoch='2026-12' AND audit_selected=1"
            )
        }
        fetched: list[int] = []

        def fake_fetcher(_client, targets, **_kwargs):
            fetched.extend(target["github_id"] for target in targets)
            return {"items": [evidence_item(target) for target in targets]}

        summary = run_collection(store, object(), fetcher=fake_fetcher, triage_model=model,
                                 audit_epoch="2026-12", max_repositories=500,
                                 max_batches=20, now=future)
        assert set(fetched).issubset(selected_third)
        assert summary["attempted"] == len(selected_third)


def test_metadata_change_and_model_change_rescore_and_requeue(tmp_path: Path) -> None:
    path = tmp_path / "rescore.sqlite"
    model = DeferredModel()
    row = {"github_id": 90_001, "full_name": "lab/repo", "description": "likely non-ML"}
    with _open(path) as store:
        first = store.ingest([row], source="inventory", triage_model=model)
        assert first["triage_scored"] == 1
        assert store.pending_count() == 0
        store.db.execute("UPDATE repositories SET last_run_id='finished'")
        changed = store.ingest([{**row, "description": "known ML method"}], source="inventory", triage_model=model)
        assert changed["triage_scored"] == 1
        assert store.pending_count() == 1
        model.fingerprint = "model-test-2"
        rescore = store.retriage_existing(model, max_repositories=10)
        assert rescore["scored"] == 1
        assert store.db.execute("SELECT model_fingerprint FROM metadata_triage WHERE github_id=?", (90_001,)).fetchone()[0] == "model-test-2"


def test_new_contribution_evidence_overrides_reused_metadata_prediction(tmp_path: Path) -> None:
    path = tmp_path / "contribution.sqlite"
    model = DeferredModel()
    row = {"github_id": 90_002, "full_name": "lab/repo", "description": "likely non-ML"}
    with _open(path) as store:
        store.ingest([row], source="inventory", triage_model=model)
        assessment = store.db.execute("SELECT decision FROM metadata_triage WHERE github_id=?", (90_002,)).fetchone()
        assert assessment[0] == "defer"
        assert store.pending_count() == 0
        store.ingest([{**row, "selection_status": "include"}], source="inventory", triage_model=model)
        assessment = store.db.execute("SELECT decision,reason,audit_selected FROM metadata_triage WHERE github_id=?", (90_002,)).fetchone()
        assert tuple(assessment) == ("fetch", "existing_contribution_evidence", 0)
        assert store.pending_count() == 1
        store.ingest([row], source="inventory", triage_model=model)
        assessment = store.db.execute("SELECT decision,reason FROM metadata_triage WHERE github_id=?", (90_002,)).fetchone()
        assert tuple(assessment) == ("defer", "test score")


def test_ingest_is_resumable_and_reports_insert_update_unchanged_counts(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(1), repository(2)]

    with _open(path) as store:
        first = store.ingest(rows, source="inventory")
        assert first["seen"] == 2
        assert first["inserted"] == 2
        assert first["updated"] == 0
        assert first["unchanged"] == 0

        same = store.ingest(rows, source="inventory")
        assert same["seen"] == 2
        assert same["inserted"] == 0
        assert same["updated"] == 0
        assert same["unchanged"] == 2

        changed = store.ingest([repository(1, revision="r2")], source="inventory")
        assert changed["seen"] == 1
        assert changed["inserted"] == 0
        assert changed["updated"] == 1

    with _open(path) as reopened:
        assert reopened.ingest([], source="inventory")["seen"] == 0


def test_partial_stream_ingest_resumes_from_committed_offset(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(index) for index in range(1, 5)]
    with _open(path) as store:
        first_rows = ((row, index + 1) for index, row in enumerate(rows))
        first = store.ingest(first_rows, source="inventory", source_revision="snapshot", max_rows=2)
        assert first["seen"] == 2 and first["complete"] == 0
        cursor = store.ingest_cursor("inventory", "snapshot")
        assert cursor == 2
        resumed_rows = ((row, index + 1) for index, row in enumerate(rows[2:], start=2))
        resumed = store.ingest(resumed_rows, source="inventory", source_revision="snapshot", resume=True)
        assert resumed["seen"] == 2 and resumed["complete"] == 1
        assert resumed["offset"] == 4
        assert store.pending_count() == 4


def test_collection_crash_resume_is_idempotent_and_changed_inputs_requeue(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(1), repository(2)]
    first_fetcher = Fetcher()

    with _open(path) as store:
        store.ingest(rows, source="inventory")
        first = run_collection(store, object(), batch_size=25, fetcher=first_fetcher)
        assert first["attempted"] == 2
        assert first["batches"] == 1

        resumed_fetcher = Fetcher()
        resumed = run_collection(store, object(), batch_size=25, fetcher=resumed_fetcher)
        assert resumed["attempted"] == 0
        assert resumed["batches"] == 0
        assert resumed_fetcher.calls == []

        # A new revision is a new evidence request for the same GitHub ID.
        store.ingest([repository(1, revision="r2")], source="inventory")
        changed_fetcher = Fetcher()
        changed = run_collection(store, object(), batch_size=25, fetcher=changed_fetcher)
        assert changed["attempted"] == 1
        assert [target["github_id"] for target in changed_fetcher.calls[0]] == [1]


def test_collection_replays_uncommitted_batch_after_worker_crash(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"

    with _open(path) as store:
        store.ingest([repository(3)], source="inventory")
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            run_collection(store, object(), fetcher=CrashingFetcher())

        retry = Fetcher()
        summary = run_collection(store, object(), fetcher=retry)
        assert summary["attempted"] == 1
        assert [target["github_id"] for target in retry.calls[0]] == [3]


def test_staged_response_resumes_after_processing_crash_without_refetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "evidence.sqlite"
    fetcher = Fetcher()
    run_id = "stable-run"
    with _open(path) as store:
        store.ingest([repository(4), repository(5)], source="inventory")
        original = graphql_evidence._store_item
        monkeypatch.setattr(graphql_evidence, "_store_item", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError, match="crash"):
            run_collection(store, object(), run_id=run_id, fetcher=fetcher)
        assert store.db.execute("SELECT COUNT(*) FROM pending_fetches").fetchone()[0] == 2
        monkeypatch.setattr(graphql_evidence, "_store_item", original)
        retry_fetcher = Fetcher()
        resumed = run_collection(store, object(), run_id=run_id, fetcher=retry_fetcher)
        assert resumed["attempted"] == 2
        assert retry_fetcher.calls == []
        assert store.db.execute("SELECT COUNT(*) FROM pending_fetches").fetchone()[0] == 0


def test_staged_response_is_discarded_when_repository_revision_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "evidence.sqlite"
    run_id = "stable-run"
    with _open(path) as store:
        store.ingest([repository(6, revision="r1")], source="inventory")
        original = graphql_evidence._store_item
        monkeypatch.setattr(graphql_evidence, "_store_item", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError, match="crash"):
            run_collection(store, object(), run_id=run_id, fetcher=Fetcher())
        monkeypatch.setattr(graphql_evidence, "_store_item", original)
        store.ingest([repository(6, revision="r2")], source="inventory")
        fresh = Fetcher()
        summary = run_collection(store, object(), run_id=run_id, fetcher=fresh)
        assert summary["attempted"] == 1
        assert len(fresh.calls) == 1
        assert fresh.calls[0][0]["revision"] == "r2"
        provenance = store.db.execute(
            "SELECT pushed_at FROM raw_provenance WHERE github_id=6 ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        assert provenance[0] == "r2"


def test_collection_keeps_previous_raw_evidence_when_a_refresh_fails(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    row = repository(7)

    with _open(path) as store:
        store.ingest([row], source="paper-links")
        successful = run_collection(store, object(), fetcher=Fetcher())
        assert successful["attempted"] == 1

        store.ingest([repository(7, revision="r2")], source="paper-links")
        failed = run_collection(store, object(), fetcher=Fetcher(failures=True))
        assert failed["attempted"] == 1
        assert failed["errors"] == 1

        # The store must expose compact latest evidence while retaining the
        # successful raw body after a failed refresh.  The export is deliberately
        # tested through the public API so its on-disk representation can evolve.
        compact = store.export_compact()
        record = next(item for item in compact if item["github_id"] == 7)
        assert record["status"] == "ok"
        assert record["text"] == "# Method\nA model."
        assert record["last_error"] == "transport failure"


def test_forks_are_collected_and_preserve_source_provenance(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    fork = repository(11, "lab/forked-project", fork=True)
    fetcher = Fetcher()

    with _open(path) as store:
        counts = store.ingest([fork], source="paper-links")
        assert counts["inserted"] == 1
        summary = run_collection(store, object(), fetcher=fetcher)
        assert summary["attempted"] == 1
        assert fetcher.calls[0][0]["fork"] is True

        compact = store.export_compact()
        record = next(item for item in compact if item["github_id"] == 11)
        assert "paper-links" in record["sources"]


def test_collection_obeys_repository_batch_and_wall_budgets_without_attempts(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(index) for index in range(1, 4)]

    with _open(path) as store:
        store.ingest(rows, source="inventory")

        no_repositories = Fetcher()
        summary = run_collection(store, object(), max_repositories=0, fetcher=no_repositories)
        assert summary["attempted"] == 0
        assert summary["batches"] == 0
        assert no_repositories.calls == []

        no_batches = Fetcher()
        summary = run_collection(store, object(), max_batches=0, fetcher=no_batches)
        assert summary["attempted"] == 0
        assert summary["batches"] == 0
        assert no_batches.calls == []

        expired = Fetcher()
        summary = run_collection(store, object(), max_seconds=0, fetcher=expired)
        assert summary["attempted"] == 0
        assert summary["batches"] == 0
        assert expired.calls == []


def test_collection_batches_targets_and_accumulates_fetcher_rate_metadata(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(index) for index in range(1, 5)]
    fetcher = Fetcher()

    with _open(path) as store:
        store.ingest(rows, source="inventory")
        summary = run_collection(store, object(), batch_size=2, fetcher=fetcher)

    assert [len(batch) for batch in fetcher.calls] == [2, 2]
    assert summary["attempted"] == 4
    assert summary["batches"] == 2
    assert summary["requests"] == 2
    assert summary["cost"] == 4
    assert summary["remaining"] == 4998


def test_two_consecutive_all_error_batches_stop_with_explicit_reason(tmp_path: Path) -> None:
    with _open(tmp_path / "evidence.sqlite") as store:
        store.ingest([repository(index) for index in range(1, 4)], source="inventory")
        summary = run_collection(store, object(), batch_size=1, fetcher=Fetcher(failures=True))
        assert summary["attempted"] == 2
        assert summary["stop_reason"] == "repeated_batch_failure"
        assert summary["error_counts"] == {"error": 2}
        assert store.pending_count() == 1


def test_raw_content_hash_deduplicates_body_but_keeps_each_repository_provenance(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    rows = [repository(21, "lab/one"), repository(22, "lab/two")]

    with _open(path) as store:
        store.ingest(rows, source="inventory")
        assert run_collection(store, object(), batch_size=2, fetcher=Fetcher())["attempted"] == 2
        compact = store.export_compact()
        assert {item["github_id"] for item in compact} == {21, 22}
        assert {item["content_hash"] for item in compact} == {compact[0]["content_hash"]}
        assert {item["full_name"] for item in compact} == {"lab/one", "lab/two"}


def test_truncation_is_marked_and_cached_text_can_be_reextracted(tmp_path: Path) -> None:
    with _open(tmp_path / "evidence.sqlite") as store:
        store.ingest([repository(23)], source="inventory")

        class LongFetcher(Fetcher):
            def __call__(self, _client: object, targets: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
                self.calls.append([dict(target) for target in targets])
                return {"items": [evidence_item(target, text="# Method\n" + "x" * 210_000) for target in targets],
                        "requests": 1, "remaining": 100}

        summary = run_collection(store, object(), fetcher=LongFetcher())
        assert summary["truncated_records"] == 1
        compact = store.export_compact()
        assert compact[0]["extractor_truncated"] is True
        store.db.execute("UPDATE repositories SET evidence_version='gh-ml-readme-evidence-v2' WHERE github_id=23")
        reextracted = store.reextract_cached(max_repositories=1)
        assert reextracted["updated"] == 1
        assert reextracted["remaining"] == 0


def test_export_run_accepts_recovered_run_ids_and_deduplicates_latest_record(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite"
    with _open(path) as store:
        store.ingest([repository(41)], source="inventory")
        run_collection(store, object(), fetcher=Fetcher(), run_id="run-one")
        store.ingest([repository(41, revision="r2")], source="inventory")
        run_collection(store, object(), fetcher=Fetcher(), run_id="run-two")
        output = tmp_path / "readme.jsonl"
        assert export_run(store, ["run-one", "run-two"], output) == 1
        row = __import__("json").loads(output.read_text(encoding="utf-8"))
        assert row["github_id"] == 41
        assert row["readme_status"] == "ok"


@pytest.mark.parametrize("limit_name", ["max_repositories", "max_batches"])
def test_zero_limits_do_not_consume_pending_work(tmp_path: Path, limit_name: str) -> None:
    path = tmp_path / "evidence.sqlite"
    with _open(path) as store:
        store.ingest([repository(31)], source="inventory")
        kwargs = {limit_name: 0}
        assert run_collection(store, object(), fetcher=Fetcher(), **kwargs)["attempted"] == 0
        fetcher = Fetcher()
        assert run_collection(store, object(), fetcher=fetcher)["attempted"] == 1
        assert len(fetcher.calls) == 1
