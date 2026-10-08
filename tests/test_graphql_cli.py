from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gh_ml import cli
from gh_ml.graphql_evidence import EvidenceStore


def _args(tmp_path: Path, *inputs: Path):
    return SimpleNamespace(
        input=list(inputs), state_db=tmp_path / "state.sqlite",
        output_dir=tmp_path / "runs", batch_size=25, max_seconds=3300,
        max_repositories=10_000, max_batches=1000, min_free_gib=300,
        github_token_env="GH_TOKEN",
    )


def _summary(**overrides):
    return {
        "run_id": "run-test", "attempted": 0, "records": 0, "failed": 0,
        "errors": 0, "deferred": 0, "rate_limited": False, "requests": 0,
        "cost": None, "remaining": None, "reset_at": None,
        "elapsed_seconds": 0.0, "budget_exhausted": False, "target_count": 0,
        "source_complete": False, "batches": 0,
        **overrides,
    }


def test_graphql_parser_sets_bounded_resumable_defaults():
    args = cli._parser().parse_args([
        "readme-graphql", "--state-db", "/tmp/state.sqlite", "--output-dir", "/tmp/run",
    ])
    assert args.input == []
    assert args.batch_size == 25
    assert args.max_seconds == 3300
    assert args.max_repositories == 10_000
    assert args.max_batches == 1000
    assert args.min_free_gib == 300


def test_jsonl_reader_seeks_to_saved_byte_cursor_and_hashes_only_read_rows(tmp_path: Path):
    first = b'{"github_id":31,"full_name":"owner/first"}\n'
    second = b'{"github_id":32,"full_name":"owner/second"}\n'
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_bytes(first + second)
    digest = hashlib.sha256()
    rows = list(cli._graphql_rows(inventory, digest, cursor=len(first)))
    assert len(rows) == 1
    assert rows[0][0]["github_id"] == 32
    assert rows[0][1] == len(first + second)
    assert digest.hexdigest() == hashlib.sha256(second).hexdigest()


def test_jsonl_ingest_cursor_advances_without_replaying_prefix(tmp_path: Path):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        '{"github_id":35,"full_name":"owner/first"}\n'
        '{"github_id":36,"full_name":"owner/second"}\n',
        encoding="utf-8",
    )
    state = tmp_path / "state.sqlite"
    with EvidenceStore(state, min_free_bytes=0) as store:
        first = store.ingest(
            cli._graphql_rows(inventory, hashlib.sha256()), source=str(inventory),
            source_revision="test-v1", max_rows=1,
        )
        assert first["complete"] == 0
        cursor = store.ingest_cursor(str(inventory), "test-v1")
        assert cursor == len('{"github_id":35,"full_name":"owner/first"}\n'.encode())
        second = store.ingest(
            cli._graphql_rows(inventory, hashlib.sha256(), cursor=cursor), source=str(inventory),
            source_revision="test-v1", resume=True,
        )
        assert second["complete"] == 1
        assert second["inserted"] == 1
        assert store.db.execute("SELECT COUNT(*) FROM repositories").fetchone()[0] == 2


def test_parquet_reader_resumes_at_saved_row_group_cursor(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    inventory = tmp_path / "inventory.parquet"
    table = pa.table({"github_id": [41, 42], "full_name": ["owner/a", "owner/b"]})
    pq.write_table(table, inventory, row_group_size=1)
    rows = list(cli._graphql_rows(inventory, hashlib.sha256(), cursor={"row_group": 1, "row_offset": 0}))
    assert len(rows) == 1
    assert rows[0][0]["github_id"] == 42
    assert rows[0][1] == {"row_group": 2, "row_offset": 0}


def test_graphql_cli_streams_jsonl_and_writes_receipt(tmp_path: Path, monkeypatch, capsys):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_bytes(
        b'{"github_id":17,"full_name":"owner/repo","pushed_at":"2026-10-07T00:00:00Z"}\n'
        b'{"github_id":"bad","full_name":"owner/bad"}\n'
    )
    observed = {}

    def collect(store, client, **kwargs):
        observed.update(pending=store.pending_count(), **kwargs)
        return _summary(**{"run_id": kwargs["run_id"], "target_count": observed["pending"]})

    monkeypatch.setattr(cli, "_github_token", lambda _: "test-token")
    result = cli._readme_graphql(
        _args(tmp_path, inventory), client_factory=lambda token: object(), collection=collect,
    )
    assert result == 0
    assert observed["pending"] == 1
    assert observed["batch_size"] == 25
    assert callable(observed["fetcher"])
    receipt_path = next((tmp_path / "runs").glob("receipt-*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["input_scope"] == "supplied-inputs-scanned"
    assert receipt["inputs"][0]["complete"] is True
    assert receipt["inputs"][0]["read_row_stream_sha256"]
    assert receipt["ingestion"]["inserted"] == 1
    assert receipt["ingestion"]["invalid"] == 1
    assert receipt["stop_reason"] == "pending_work_deferred"
    assert receipt["source_complete"] is False
    output = capsys.readouterr().out
    assert "pending_work_deferred" in output
    assert str(receipt_path) in output


def test_graphql_cli_can_resume_without_an_input(tmp_path: Path, monkeypatch):
    state = tmp_path / "state.sqlite"
    with EvidenceStore(state, min_free_bytes=0) as store:
        store.ingest([{"github_id": 18, "full_name": "owner/resume"}], source="fixture")
    observed = {}

    def collect(store, client, **kwargs):
        observed["pending"] = store.pending_count()
        return _summary(**{"run_id": kwargs["run_id"]})

    args = _args(tmp_path)
    args.state_db = state
    monkeypatch.setattr(cli, "_github_token", lambda _: None)
    assert cli._readme_graphql(args, client_factory=lambda token: object(), collection=collect) == 0
    assert observed["pending"] == 1
    receipt = json.loads(next((tmp_path / "runs").glob("receipt-*.json")).read_text())
    assert receipt["input_scope"] == "resume-pending-only"
    assert receipt["inputs"] == []


@pytest.mark.parametrize("field,value", [("batch_size", 51), ("max_seconds", 604801), ("min_free_gib", -1)])
def test_graphql_cli_rejects_out_of_policy_budgets(tmp_path: Path, field: str, value: int):
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        cli._readme_graphql(args, client_factory=lambda token: object())


def test_graphql_cli_rejects_state_inside_repository(tmp_path: Path):
    args = _args(tmp_path)
    args.state_db = Path(cli.__file__).resolve().parents[2] / "state.sqlite"
    with pytest.raises(ValueError, match="outside the source repository"):
        cli._readme_graphql(args, client_factory=lambda token: object())


def test_graphql_cli_preserves_archive_free_space_floor(tmp_path: Path):
    args = _args(tmp_path)
    args.state_db = Path("/mnt/archive/runs/gh-ml-test/state.sqlite")
    args.output_dir = Path("/mnt/archive/runs/gh-ml-test/output")
    args.min_free_gib = 299
    with pytest.raises(ValueError, match="300"):
        cli._readme_graphql(args, client_factory=lambda token: object())
