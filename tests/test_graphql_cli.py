from __future__ import annotations

import hashlib
import json
import time
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
    assert args.triage_model is None
    assert args.deferred_audit_rate == 0.05
    assert args.max_triage_repositories == 10_000


def test_triage_loader_dispatches_by_schema_without_loading_unused_models(tmp_path, monkeypatch):
    import importlib
    from gh_ml.metadata_triage import ALLOWED_FEATURES, ARTIFACT_SCHEMA, MODEL_VERSION, _canonical_hash

    artifact = {
        "schema": ARTIFACT_SCHEMA, "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES), "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant", "vocabulary": {}, "idf": [], "weights": [],
        "intercept": 0.0,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = []
    original = importlib.import_module

    def tracked(name, package=None):
        loaded.append(name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked)
    schema, model = cli._load_graphql_triage_model(path)
    assert schema == ARTIFACT_SCHEMA
    assert model.fingerprint == artifact["artifact_sha256"]
    assert loaded == [".metadata_triage"]

    unsupported = tmp_path / "unknown.json"
    unsupported.write_text('{"schema":"unregistered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported.*schema"):
        cli._load_graphql_triage_model(unsupported)


def test_triage_loader_routes_lexical_schema_and_lazy_semantic_backend(tmp_path, monkeypatch):
    import importlib
    from gh_ml.lexical_triage import (
        ALLOWED_FEATURES as LEXICAL_FEATURES,
        ARTIFACT_SCHEMA as LEXICAL_SCHEMA,
        MODEL_VERSION as LEXICAL_VERSION,
        _canonical_hash as lexical_hash,
    )

    lexical_artifact = {
        "schema": LEXICAL_SCHEMA, "model_version": LEXICAL_VERSION,
        "features": list(LEXICAL_FEATURES), "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant", "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5], "blocks": {
            "word": {"vocabulary": {}, "idf": [], "weights": [], "norm": "l2", "input_weight": 1.0},
            "char_wb": {"vocabulary": {}, "idf": [], "weights": [], "norm": "l2", "input_weight": 1.0},
        }, "intercept": 0.0, "defer_threshold": 0.0,
    }
    lexical_artifact["artifact_sha256"] = lexical_hash(lexical_artifact)
    lexical_path = tmp_path / "lexical.json"
    lexical_path.write_text(json.dumps(lexical_artifact), encoding="utf-8")
    schema, model = cli._load_graphql_triage_model(lexical_path)
    assert schema == LEXICAL_SCHEMA
    assert model.fingerprint == lexical_artifact["artifact_sha256"]

    semantic_path = tmp_path / "semantic.json"
    semantic_path.write_text('{"schema":"gh-ml-semantic-triage-v1"}', encoding="utf-8")
    fake_model = SimpleNamespace(schema="gh-ml-semantic-triage-v1")
    loaded = []
    original = importlib.import_module

    class FakeSemanticModule:
        @staticmethod
        def load_model(path):
            assert path == str(semantic_path)
            return fake_model

    def tracked(name, package=None):
        loaded.append(name)
        if name == ".semantic_triage":
            return FakeSemanticModule
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked)
    schema, model = cli._load_graphql_triage_model(semantic_path)
    assert schema == fake_model.schema
    assert model is fake_model
    assert loaded == [".semantic_triage"]


def test_graphql_triage_rows_preserve_cursor_and_batch_predictions():
    class BatchModel:
        def predict_batch(self, rows, *, deadline_monotonic=None):
            assert len(rows) <= 64
            return [{"decision": "fetch", "github_id": row["github_id"]} for row in rows]

    inputs = [({"github_id": index}, index + 1) for index in range(3)]
    scored = list(cli._predict_graphql_rows(inputs, BatchModel(), deadline=time.monotonic() + 10, batch_size=2))
    assert [(row["github_id"], cursor, prediction["github_id"])
            for row, cursor, prediction in scored] == [(0, 1, 0), (1, 2, 1), (2, 3, 2)]


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


def test_graphql_cli_optional_model_is_persisted_as_experimental_triage(tmp_path: Path, monkeypatch):
    from gh_ml.metadata_triage import ALLOWED_FEATURES, ARTIFACT_SCHEMA, MODEL_VERSION, _canonical_hash, write_artifact

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "features": list(ALLOWED_FEATURES),
        "positive_label": "ml_relevant",
        "negative_label": "not_ml_relevant",
        "vocabulary": {"garden": 0},
        "idf": [1.0], "weights": [-2.0], "intercept": -4.0,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    model_path = tmp_path / "triage.json"
    write_artifact(str(model_path), artifact)
    observed = {}

    def collect(store, client, **kwargs):
        observed.update(kwargs)
        return _summary(**{"run_id": kwargs["run_id"]})

    args = _args(tmp_path)
    args.triage_model = model_path
    monkeypatch.setattr(cli, "_github_token", lambda _: None)
    assert cli._readme_graphql(args, client_factory=lambda token: object(), collection=collect) == 0
    assert observed["triage_model"].fingerprint == artifact["artifact_sha256"]
    receipt = json.loads(next((tmp_path / "runs").glob("receipt-*.json")).read_text())
    assert receipt["metadata_triage"]["enabled"] is True
    assert receipt["metadata_triage"]["experimental"] is True
    assert receipt["metadata_triage"]["model_fingerprint"] == artifact["artifact_sha256"]
    assert receipt["metadata_triage"]["scope"]["assessed"] == 0


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
