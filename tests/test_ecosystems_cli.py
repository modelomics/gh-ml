from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from gh_ml import cli


def test_ecosystems_parser_exposes_bounded_primary_import_options():
    args = cli._parser().parse_args([
        "ecosystems-import", "--state-db", "/tmp/state.sqlite", "--output-dir", "/tmp/out",
        "--max-pages", "4", "--per-page", "750", "--input", "/tmp/discovery.jsonl",
        "--github-token-env", "GH_TOKEN_A", "--github-token-env", "GH_TOKEN_B",
        "--updated-after", "2026-10-01",
    ])
    assert args.max_pages == 4
    assert args.per_page == 750
    assert args.max_github_requests == 100
    assert args.max_seconds == 3300
    assert args.input == [Path("/tmp/discovery.jsonl")]
    assert args.github_token_env == ["GH_TOKEN_A", "GH_TOKEN_B"]
    assert args.updated_after == "2026-10-01"
    assert args.no_github_fallback is False
    zero_pages = cli._parser().parse_args([
        "ecosystems-import", "--state-db", "/tmp/state.sqlite", "--output-dir", "/tmp/out",
        "--max-pages", "0",
    ])
    assert zero_pages.max_pages == 0


def test_ecosystems_main_dispatches_to_command_handler(monkeypatch):
    seen = {}
    def handle(args):
        seen["command"] = args.command
        return 0

    monkeypatch.setattr(cli, "_ecosystems_import", handle)
    assert cli.main(["ecosystems-import", "--state-db", "/tmp/state.sqlite", "--output-dir", "/tmp/out"]) == 0
    assert seen["command"] == "ecosystems-import"


def test_ecosystems_command_dispatches_primary_and_discovery_sources(tmp_path, monkeypatch, capsys):
    discovery = tmp_path / "discovery.jsonl"
    discovery.write_text('{"github_id":4,"full_name":"owner/repo"}\n', encoding="utf-8")
    observed = {}

    def collector(state_db, output_dir, **kwargs):
        observed.update(state_db=state_db, output_dir=output_dir, **kwargs)
        return {"paths": [str(output_dir / "metadata.jsonl")], "cursor": 8,
                "pending": 2, "stop_reason": "quota_deferred"}

    monkeypatch.setattr(cli, "_github_token", lambda _name: None)
    # Call the injectable runner directly so this verifies the full argument contract
    # without network access or a live SQLite collector.
    args = cli._parser().parse_args([
        "ecosystems-import", "--state-db", str(tmp_path / "state.sqlite"),
        "--output-dir", str(tmp_path / "out"), "--input", str(discovery), "--no-github-fallback",
    ])
    result = cli._ecosystems_import(args, collector=collector)
    assert result == 2
    assert observed["discovery_paths"] == [discovery.resolve()]
    assert observed["github_client"] is None
    assert observed["max_pages"] == 1
    assert observed["per_page"] == 1000
    assert observed["max_github_requests"] == 100
    output = json.loads(capsys.readouterr().out)
    assert output["report"]["cursor"] == 8
    assert output["report"]["pending"] == 2
    assert output["report"]["paths"] == [str(tmp_path / "out" / "metadata.jsonl")]
    assert output["token_pool"] is None


def test_ecosystems_bad_bounds_fail_before_client_or_collector(tmp_path):
    args = SimpleNamespace(
        max_pages=-1, per_page=1000, max_github_requests=100, max_seconds=3300,
        min_free_gib=300, updated_after=None, state_db=tmp_path / "state.sqlite",
        output_dir=tmp_path / "out", input=[], github_token_env=[], no_github_fallback=True,
    )
    called = []
    try:
        cli._ecosystems_import(args, ecosystems_client_factory=lambda: called.append("client"),
                              collector=lambda *a, **k: called.append("collector"))
    except ValueError as exc:
        assert "--max-pages" in str(exc)
    else:
        raise AssertionError("invalid page bound was accepted")
    assert called == []


def test_zero_page_mode_skips_primary_listing_but_imports_discovery_and_labels_receipt(tmp_path, monkeypatch, capsys):
    discovery = tmp_path / "discovery.jsonl"
    discovery.write_text('{"github_id":51,"full_name":"owner/repo"}\n', encoding="utf-8")
    calls = {"list": 0, "detail": 0}

    class PublicClient:
        def list_repositories(self, **_kwargs):
            calls["list"] += 1
            return []

        def get_repository(self, _name, **_kwargs):
            calls["detail"] += 1
            return None

    from gh_ml import ecosystems
    monkeypatch.setattr(ecosystems, "EcosystemsClient", PublicClient)
    monkeypatch.setattr(cli, "_github_token", lambda _name: None)
    result = cli.main([
        "ecosystems-import", "--state-db", str(tmp_path / "state.sqlite"),
        "--output-dir", str(tmp_path / "out"), "--input", str(discovery),
        "--max-pages", "0", "--no-github-fallback",
    ])
    assert result == 2
    assert calls == {"list": 0, "detail": 1}
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["pages_requested"] == 0
    assert summary["report"]["remaining_queue"] == 1
    assert summary["github_request_accounting"]["github_budget_unit"] == "github_client.get_repository invocations"
    receipt = json.loads(Path(summary["report"]["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["github_request_accounting"] == summary["github_request_accounting"]


def test_ecosystems_delta_loads_through_graphql_inventory_reader(tmp_path, monkeypatch, capsys):
    from gh_ml import ecosystems
    from gh_ml.graphql_evidence import EvidenceStore

    raw = {
        "id": 501, "uuid": 501, "full_name": "owner/imported", "html_url": "https://github.com/owner/imported",
        "description": "A local metadata fixture", "topics": ["machine-learning"], "language": "Python",
        "fork": False, "archived": False, "created_at": "2024-01-01T00:00:00Z",
        "pushed_at": "2026-10-01T00:00:00Z", "last_synced_at": "2026-10-02T00:00:00Z",
        "stargazers_count": 3, "forks_count": 1,
    }

    class PublicClient:
        def list_repositories(self, **_kwargs):
            return [raw]

        def get_repository(self, _name, **_kwargs):
            return raw

    monkeypatch.setattr(ecosystems, "EcosystemsClient", PublicClient)
    monkeypatch.setattr(cli, "_github_token", lambda _name: None)
    result = cli.main([
        "ecosystems-import", "--state-db", str(tmp_path / "state.sqlite"),
        "--output-dir", str(tmp_path / "out"), "--max-pages", "1", "--no-github-fallback",
    ])
    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    delta_path = Path(summary["report"]["export_path"])
    rows = list(cli._graphql_rows(delta_path, hashlib.sha256()))
    assert len(rows) == 1
    imported, _cursor = rows[0]
    assert imported["github_id"] == 501
    assert imported["full_name"] == "owner/imported"
    with EvidenceStore(tmp_path / "consumer.sqlite", min_free_bytes=0) as store:
        counts = store.ingest([imported], source=str(delta_path))
        assert counts["inserted"] == 1
        assert store.db.execute("SELECT full_name FROM repositories").fetchone()[0] == "owner/imported"


def test_ecosystems_fallback_pool_uses_env_names_and_never_prints_credentials(tmp_path, monkeypatch, capsys):
    from gh_ml import github_tokens

    monkeypatch.setenv("GH_IMPORT_A", "secret-a")
    monkeypatch.setenv("GH_IMPORT_B", "secret-b")
    args = cli._parser().parse_args([
        "ecosystems-import", "--state-db", str(tmp_path / "state.sqlite"),
        "--output-dir", str(tmp_path / "out"), "--github-token-env", "GH_IMPORT_A",
        "--github-token-env", "GH_IMPORT_B",
    ])
    observed = {}
    pool_state = {
        "credential_count": 2, "account_count": 1, "validation_attempts": 2,
        "api_transport_attempts": 0, "quota": {"core": {"known_remaining": 80}},
    }
    fake_client = SimpleNamespace(token_pool=SimpleNamespace(summary=lambda: dict(pool_state)))

    def build(names, **kwargs):
        observed.update(names=names, **kwargs)
        return fake_client

    monkeypatch.setattr(github_tokens, "build_pooled_client", build)
    monkeypatch.setattr(cli, "_github_token", lambda _name: "fallback-secret")
    def collect(**_kwargs):
        pool_state["api_transport_attempts"] = 7
        receipt_path = tmp_path / "out" / "receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        return {
            "status": "complete", "export_path": str(tmp_path / "out" / "metadata.jsonl"),
            "receipt_path": str(receipt_path), "cursor": {"next_page": 2},
            "remaining_queue": 3, "deferred": 3,
        }

    result = cli._ecosystems_import(args, collector=collect)
    output = capsys.readouterr().out
    assert result == 2
    assert observed["names"] == ["GH_IMPORT_A", "GH_IMPORT_B"]
    assert observed["fallback_token"] == "secret-a"
    assert observed["deadline"] > 0
    assert "secret" not in output
    assert '"account_count": 1' in output
    assert '"remaining_queue": 3' in output
    summary = json.loads(output)
    assert summary["token_pool"]["validation_attempts"] == 2
    assert summary["token_pool"]["api_transport_attempts"] == 7
    assert summary["report"]["token_pool"] == summary["token_pool"]
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["token_pool"] == summary["token_pool"]


def test_graphql_pool_flag_uses_global_deadline_and_keeps_default_factory(monkeypatch, tmp_path):
    from gh_ml import github_tokens

    args = SimpleNamespace(
        input=[], input_revision=[], state_db=tmp_path / "state.sqlite", output_dir=tmp_path / "runs",
        batch_size=25, max_seconds=30, max_repositories=0, max_batches=0, min_free_gib=300,
        github_token_env="GH_TOKEN", github_pool_token_env=["GH_TOKEN_1", "GH_TOKEN_2"],
        triage_model=None, deferred_audit_rate=0.05, max_triage_repositories=10, audit_seed="test",
    )
    observed = {}

    class Store:
        def __init__(self, *_args, **_kwargs):
            self.db = SimpleNamespace(execute=lambda *_a: SimpleNamespace(fetchone=lambda: None))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def pending_count(self):
            return 0

        def export_run(self, *_args, **_kwargs):
            return 0

    pooled = SimpleNamespace(token_pool=SimpleNamespace(summary=lambda: {"account_count": 2}))

    def build(names, **kwargs):
        observed.update(names=names, **kwargs)
        return pooled

    monkeypatch.setattr(github_tokens, "build_pooled_client", build)
    monkeypatch.setattr(cli, "_github_token", lambda _name: "fallback")
    from gh_ml import graphql_evidence
    monkeypatch.setattr(graphql_evidence, "export_run", lambda *_args, **_kwargs: 0)

    def collect(store, client, **kwargs):
        observed["client"] = client
        observed["deadline_budget"] = kwargs["max_seconds"]
        return {"run_id": "run", "attempted": 0, "target_count": 0, "requests": 0}

    cli._readme_graphql(args, collection=collect, store_factory=Store)
    assert observed["names"] == ["GH_TOKEN_1", "GH_TOKEN_2"]
    assert observed["fallback_token"] == "fallback"
    assert observed["deadline"] > 0
    assert observed["client"] is pooled
