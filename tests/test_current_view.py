from __future__ import annotations

import json

import pytest

import gh_ml.current_view as current_view
from gh_ml.current_view import export_current_view_parquet, export_observations_parquet, materialize_current_view


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(github_id, observed_at, **fields):
    return {"github_id": github_id, "observed_at": observed_at, **fields}


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_newest_observation_wins_over_later_old_backfill_and_output_is_sorted(tmp_path):
    recent = _write(tmp_path / "recent.jsonl", [
        _row(20, "2026-09-24T12:00:00Z", stars=50, extra={"kept": True},
             query_ids=["new.query"], domains=["Vision"], methods=["K means"],
             novelty_signals=["new-signal"]),
        _row(5, "2026-09-24T12:00:00+00:00", stars=3),
    ])
    old_backfill = _write(tmp_path / "backfill.jsonl", [
        _row(20, "2020-01-01T00:00:00Z", stars=1,
             query_ids=["old.query"], domains=["health"], methods=["K-Means", "Transformer"],
             novelty_signals=["paper-reference"]),
        _row(5, "2025-01-01T00:00:00Z", stars=2),
    ])
    output = tmp_path / "view.jsonl"
    repeat = tmp_path / "view-repeat.jsonl"

    report = materialize_current_view([recent, old_backfill], output)
    materialize_current_view([old_backfill, recent], repeat)

    assert _read(output) == [
        {**_row(5, "2026-09-24T12:00:00+00:00", stars=3),
         "observation_count": 2, "first_observed_at": "2025-01-01T00:00:00Z",
         "all_query_ids": [], "all_domains": [], "all_methods": [], "all_novelty_signals": [],
         "evidence_version": "gh-ml-relevance-v1", "evidence_tier": "no_text_signal",
             "evidence_signals": [], "selection_version": current_view.SELECTION_VERSION,
         "selection_status": "review", "selection_reason": "insufficient-repository-evidence",
         "selection_signals": [], "candidate_rule_version": "ml-candidate-v2",
         "candidate_eligible": False, "candidate_reason": "insufficient-repository-evidence"},
        {**_row(20, "2026-09-24T12:00:00Z", stars=50, extra={"kept": True},
                query_ids=["new.query"], domains=["Vision"], methods=["K means"],
                novelty_signals=["new-signal"]),
         "observation_count": 2, "first_observed_at": "2020-01-01T00:00:00Z",
         "all_query_ids": ["new.query", "old.query"], "all_domains": ["Vision", "health"],
         "all_methods": ["k-means", "transformer"],
         "all_novelty_signals": ["new-signal", "paper-reference"],
         "evidence_version": "gh-ml-relevance-v1", "evidence_tier": "no_text_signal",
             "evidence_signals": [], "selection_version": current_view.SELECTION_VERSION,
         "selection_status": "review", "selection_reason": "insufficient-repository-evidence",
         "selection_signals": [], "candidate_rule_version": "ml-candidate-v2",
         "candidate_eligible": False, "candidate_reason": "insufficient-repository-evidence"},
    ]
    assert report["observation_count"] == 4
    assert report["current_view_count"] == 2
    assert report["version"] == current_view.CURRENT_VIEW_PROJECTION_VERSION == 4
    manifest = json.loads((tmp_path / "view.jsonl.manifest.json").read_text())
    assert manifest["candidate_rule_version"] == "ml-candidate-v2"
    assert manifest["input_files"][1]["observations"] == 2
    assert repeat.read_bytes() == output.read_bytes()


def test_equal_timestamp_tie_break_does_not_depend_on_input_order(tmp_path):
    left = _write(tmp_path / "left.jsonl", [_row(1, "2026-01-01T00:00:00Z", name="alpha")])
    right = _write(tmp_path / "right.jsonl", [_row(1, "2026-01-01T00:00:00+00:00", name="zeta")])
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"

    materialize_current_view([left, right], first)
    materialize_current_view([right, left], second)

    assert _read(first) == _read(second)
    assert _read(first)[0]["name"] == "zeta"


def test_empty_input_files_create_empty_view_and_manifest(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    output = tmp_path / "empty-view.jsonl"

    report = materialize_current_view([empty], output)

    assert output.read_text() == ""
    assert report["observation_count"] == report["current_view_count"] == 0
    assert (tmp_path / "empty-view.jsonl.manifest.json").is_file()


@pytest.mark.parametrize("line", [
    '{"github_id": true, "observed_at": "2026-01-01T00:00:00Z"}',
    '{"github_id": 0, "observed_at": "2026-01-01T00:00:00Z"}',
    '{"github_id": 3, "observed_at": "not a timestamp"}',
    '{"github_id": 3, "observed_at": "2026-01-01T00:00:00"}',
    '{"github_id": 3, "observed_at": "2026-01-01T00:00:00Z"',
    '{"github_id": 3, "observed_at": "2026-01-01T00:00:00Z", "domains": ["ok", 3]}',
])
def test_corrupt_input_fails_without_replacing_existing_output_or_manifest(tmp_path, line):
    source = tmp_path / "bad.jsonl"
    source.write_text(line + "\n", encoding="utf-8")
    output = tmp_path / "view.jsonl"
    output.write_text("existing output\n", encoding="utf-8")
    manifest = tmp_path / "view.manifest.json"
    manifest.write_text("existing manifest\n", encoding="utf-8")

    with pytest.raises(ValueError):
        materialize_current_view([source], output, manifest_path=manifest)

    assert output.read_text() == "existing output\n"
    assert manifest.read_text() == "existing manifest\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bad.jsonl", "view.jsonl", "view.manifest.json"]


def test_rejects_output_as_input(tmp_path):
    source = _write(tmp_path / "observations.jsonl", [_row(1, "2026-01-01T00:00:00Z")])
    with pytest.raises(ValueError, match="must not also be"):
        materialize_current_view([source], source)


def test_parquet_export_preserves_numeric_ids_nested_lists_nulls_and_late_extra_fields(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "view.jsonl"
    rows = [
        {"github_id": 1, "name": "one", "observed_at": "2026-01-01T00:00:00Z",
         "stars": 3, "topics": ["vision", "learning"], "domains": [], "description": None},
        {"github_id": 2, "name": "two", "observed_at": "2026-01-02T00:00:00Z",
         "stars": 0, "topics": [], "domains": ["biology"], "description": "example",
         "candidate_evidence": [{"kind": "paper", "signals": ["citation", "abstract"]}]},
        {"github_id": 3, "name": "three", "observed_at": "2026-01-03T00:00:00Z",
         "stars": 8, "topics": None, "domains": ["biology", "chemistry"], "description": None,
         "pwc_assertions": {"paper_id": "P42", "models": ["model-a", "model-b"]}},
    ]
    _write(source, rows)
    output = tmp_path / "view.parquet"
    monkeypatch.setattr(current_view, "_PARQUET_BATCH_ROWS", 1)

    result = export_current_view_parquet(source, output)

    table = pq.read_table(output)
    assert result == {"row_count": 3, "size_bytes": output.stat().st_size, "compression": "zstd"}
    assert pa.types.is_int64(table.schema.field("github_id").type)
    assert pa.types.is_list(table.schema.field("topics").type)
    assert pa.types.is_string(table.schema.field("observed_at").type)
    assert pa.types.is_string(table.schema.field("extra_json").type)
    assert table.to_pylist()[0]["github_id"] == 1
    assert table.to_pylist()[0]["topics"] == ["vision", "learning"]
    assert table.to_pylist()[0]["domains"] == []
    assert table.to_pylist()[0]["description"] is None
    assert table.to_pylist()[1]["topics"] == []
    assert table.to_pylist()[2]["topics"] is None
    for actual, expected in zip(table.to_pylist(), rows, strict=True):
        extras = {key: expected[key] for key in ("candidate_evidence", "pwc_assertions") if key in expected}
        if not extras:
            assert actual["extra_json"] is None
        else:
            assert json.loads(actual["extra_json"]) == extras
        expected_core = {key: value for key, value in expected.items() if key not in extras}
        assert {key: actual[key] for key in expected_core} == expected_core


def test_parquet_export_writes_valid_empty_file_and_keeps_output_on_bad_input(tmp_path):
    pytest.importorskip("pyarrow.parquet")
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    output = tmp_path / "empty.parquet"
    report = export_current_view_parquet(empty, output)
    import pyarrow.parquet as pq
    assert report["row_count"] == 0
    assert pq.read_table(output).num_rows == 0

    bad = _write(tmp_path / "bad.jsonl", [{"github_id": None, "name": "bad"}])
    output.write_bytes(b"old output")
    with pytest.raises(ValueError, match="positive numeric"):
        export_current_view_parquet(bad, output)
    assert output.read_bytes() == b"old output"


def test_raw_observation_parquet_keeps_every_row_in_file_order_and_preserves_extras(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    first = _write(tmp_path / "part-1.jsonl", [
        {"github_id": 12, "name": "owner/first", "observed_at": "2026-01-01T00:00:00Z",
         "query_ids": ["first.query"], "domains": ["vision"], "methods": ["transformer"],
         "novelty_signals": ["paper"], "selection_status": "exclude",
         "candidate_evidence": {"source": "readme", "signals": ["algorithm"]}},
        {"github_id": 12, "name": "owner/first", "observed_at": "2026-01-02T00:00:00Z",
         "query_ids": ["second.query"], "domains": [], "methods": [], "novelty_signals": [],
         "selection_status": "include", "batch_marker": 2},
    ])
    second = _write(tmp_path / "part-2.jsonl", [
        {"github_id": 3, "name": "owner/third", "observed_at": "2026-01-03T00:00:00Z",
         "selection_status": "review", "pwc_assertions": [{"paper_id": "P7"}]},
    ])
    output = tmp_path / "raw.parquet"
    monkeypatch.setattr(current_view, "_PARQUET_BATCH_ROWS", 1)

    result = export_observations_parquet((path for path in (first, second)), output)

    table = pq.read_table(output)
    rows = table.to_pylist()
    assert result == {"row_count": 3, "size_bytes": output.stat().st_size, "compression": "zstd"}
    assert [row["github_id"] for row in rows] == [12, 12, 3]
    assert [row["observed_at"] for row in rows] == [
        "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z",
    ]
    assert pa.types.is_int64(table.schema.field("github_id").type)
    assert pa.types.is_list(table.schema.field("query_ids").type)
    assert pa.types.is_string(table.schema.field("selection_status").type)
    assert json.loads(rows[0]["extra_json"]) == {
        "candidate_evidence": {"source": "readme", "signals": ["algorithm"]},
    }
    assert json.loads(rows[1]["extra_json"]) == {"batch_marker": 2}
    assert json.loads(rows[2]["extra_json"]) == {"pwc_assertions": [{"paper_id": "P7"}]}
    assert rows[0]["query_ids"] == ["first.query"]
    assert rows[1]["query_ids"] == ["second.query"]
    assert [row["selection_status"] for row in rows] == ["exclude", "include", "review"]
    assert output.parent == tmp_path


def test_raw_observation_parquet_empty_inputs_and_source_collision(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    output = tmp_path / "raw.parquet"

    result = export_observations_parquet([empty], output, compression="snappy")

    assert result["row_count"] == 0
    assert result["compression"] == "snappy"
    assert pq.read_table(output).num_rows == 0
    with pytest.raises(ValueError, match="distinct"):
        export_observations_parquet([output], output)


def test_parquet_extra_json_round_trips_aggregated_evidence(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    older = _write(tmp_path / "older.jsonl", [
        _row(7, "2024-01-01T00:00:00Z", query_ids=["old"], domains=["bio"],
             methods=["K means"], novelty_signals=["paper"]),
    ])
    newer = _write(tmp_path / "newer.jsonl", [
        _row(7, "2026-01-01T00:00:00Z", query_ids=["new"], domains=["ml"],
             methods=["Transformer"], novelty_signals=["topics"], candidate_evidence=[{"source": "census"}]),
    ])
    view = tmp_path / "view.jsonl"
    parquet = tmp_path / "view.parquet"
    materialize_current_view([older, newer], view)
    export_current_view_parquet(view, parquet)

    row = pq.read_table(parquet).to_pylist()[0]
    extra = json.loads(row["extra_json"])
    assert row["all_query_ids"] == ["new", "old"]
    assert row["all_domains"] == ["bio", "ml"]
    assert row["all_methods"] == ["k-means", "transformer"]
    assert row["all_novelty_signals"] == ["paper", "topics"]
    assert row["observation_count"] == 2
    assert row["first_observed_at"] == "2024-01-01T00:00:00Z"
    assert extra["candidate_evidence"] == [{"source": "census"}]


def test_current_view_adds_evidence_only_from_latest_repository_text(tmp_path):
    old = _write(tmp_path / "old.jsonl", [
        _row(31, "2025-01-01T00:00:00Z", description="machine learning project",
             domains=["finance"], query_ids=["finance.ml"]),
    ])
    latest = _write(tmp_path / "latest.jsonl", [
        _row(31, "2026-01-01T00:00:00Z", name="trading-strategy",
             description="Financial backtesting framework", domains=["finance"],
             methods=["machine-learning"], query_ids=["finance.ml"]),
    ])
    output = tmp_path / "view.jsonl"

    materialize_current_view([old, latest], output)
    row = _read(output)[0]

    assert row["evidence_version"] == "gh-ml-relevance-v1"
    assert row["evidence_tier"] == "no_text_signal"
    assert row["evidence_signals"] == []
    assert row["all_methods"] == ["machine-learning"]


def test_parquet_exposes_evidence_and_accumulated_fields_as_typed_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = _write(tmp_path / "view.jsonl", [{
        "github_id": 8, "name": "ml-transformer", "description": "",
        "observed_at": "2026-01-01T00:00:00Z", "first_observed_at": "2025-01-01T00:00:00Z",
        "observation_count": 2, "all_query_ids": ["one"], "all_domains": ["vision"],
        "all_methods": ["transformer"], "all_novelty_signals": [],
        "evidence_version": "gh-ml-relevance-v1", "evidence_tier": "direct_ml_text",
        "evidence_signals": ["machine-learning", "transformer"],
    }])
    output = tmp_path / "view.parquet"

    export_current_view_parquet(source, output)
    table = pq.read_table(output)
    row = table.to_pylist()[0]

    assert pa.types.is_list(table.schema.field("all_methods").type)
    assert pa.types.is_list(table.schema.field("evidence_signals").type)
    assert pa.types.is_int64(table.schema.field("observation_count").type)
    assert row["all_methods"] == ["transformer"]
    assert row["evidence_tier"] == "direct_ml_text"
    assert row["evidence_signals"] == ["machine-learning", "transformer"]
    assert row["extra_json"] is None


def test_parquet_selection_columns_are_typed_and_optional_filter_preserves_local_export(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = _write(tmp_path / "view.jsonl", [
        {"github_id": 1, "observed_at": "2026-01-01T00:00:00Z", "selection_version": "v1",
         "selection_status": "include", "selection_reason": "strong", "selection_signals": ["paper"]},
        {"github_id": 2, "observed_at": "2026-01-02T00:00:00Z", "selection_version": "v1",
         "selection_status": "exclude", "selection_reason": "fork", "selection_signals": ["fork"]},
    ])
    all_output, included_output = tmp_path / "all.parquet", tmp_path / "included.parquet"

    assert export_current_view_parquet(source, all_output)["row_count"] == 2
    assert export_current_view_parquet(source, included_output, selection_status="include")["row_count"] == 1
    table = pq.read_table(included_output)
    assert pa.types.is_string(table.schema.field("selection_version").type)
    assert pa.types.is_string(table.schema.field("selection_status").type)
    assert pa.types.is_string(table.schema.field("selection_reason").type)
    assert pa.types.is_list(table.schema.field("selection_signals").type)
    assert table.to_pylist()[0]["github_id"] == 1


def test_candidate_assessment_is_applied_to_latest_deduped_observation(tmp_path):
    eligible_old_and_latest = _write(tmp_path / "eligible.jsonl", [
        _row(41, "2025-01-01T00:00:00Z", name="transformer-project",
             description="Awesome list of transformer projects"),
        _row(41, "2026-01-01T00:00:00Z", name="transformer-project",
             description="Transformer model implementation; paper and code are available for this machine learning project."),
    ])
    hard_negative = _write(tmp_path / "negative.jsonl", [
        _row(42, "2026-01-01T00:00:00Z", name="awesome-transformers",
             description="Awesome list of transformer projects"),
    ])
    output = tmp_path / "current.jsonl"

    materialize_current_view([eligible_old_and_latest, hard_negative], output)
    rows = _read(output)

    assert rows[0]["github_id"] == 41
    assert rows[0]["observation_count"] == 2
    assert rows[0]["candidate_eligible"] is True
    assert rows[0]["candidate_reason"] == "review-with-repository-evidence"
    assert rows[1]["github_id"] == 42
    assert rows[1]["selection_status"] == "exclude"
    assert rows[1]["candidate_eligible"] is False
    assert rows[1]["candidate_reason"] == "not-selected-or-reviewable"


def test_candidate_parquet_columns_are_typed_and_filter_is_independent_of_selection(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = _write(tmp_path / "view.jsonl", [
        {"github_id": 1, "observed_at": "2026-01-01T00:00:00Z", "selection_status": "review",
         "candidate_rule_version": "ml-candidate-v2", "candidate_eligible": True,
         "candidate_reason": "review-with-repository-evidence"},
        {"github_id": 2, "observed_at": "2026-01-02T00:00:00Z", "selection_status": "review",
         "candidate_rule_version": "ml-candidate-v2", "candidate_eligible": False,
         "candidate_reason": "insufficient-repository-evidence"},
        {"github_id": 3, "observed_at": "2026-01-03T00:00:00Z", "selection_status": "include",
         "candidate_rule_version": "ml-candidate-v2", "candidate_eligible": True,
         "candidate_reason": "selected-by-current-rule"},
    ])
    output = tmp_path / "eligible.parquet"

    result = export_current_view_parquet(source, output, candidate_eligible=True,
                                         selection_status="review")
    table = pq.read_table(output)

    assert result["row_count"] == 1
    assert [row["github_id"] for row in table.to_pylist()] == [1]
    assert pa.types.is_string(table.schema.field("candidate_rule_version").type)
    assert pa.types.is_boolean(table.schema.field("candidate_eligible").type)
    assert pa.types.is_string(table.schema.field("candidate_reason").type)
