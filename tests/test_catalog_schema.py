from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml.query_catalog import load_queries
from gh_ml.schema import QuerySpec, observation_from_repository, write_jsonl


def test_load_queries_normalizes_lists_and_sorts_ids(tmp_path: Path) -> None:
    (tmp_path / "z.toml").write_text(
        '[[queries]]\nid = "z-query"\nq = "  topic:machine-learning  "\n'
        'domains = ["vision", "nlp"]\nmethods = ["transformer", "cnn"]\n'
    )
    (tmp_path / "a.toml").write_text(
        '[[queries]]\nid = "a-query"\nq = "repo:example/model"\n'
    )

    queries = load_queries(tmp_path)

    assert [query.id for query in queries] == ["a-query", "z-query"]
    assert queries[0].domains == ()
    assert queries[0].methods == ()
    assert queries[1].domains == ("nlp", "vision")
    assert queries[1].methods == ("cnn", "transformer")
    assert queries[1].q == "topic:machine-learning"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[[queries]]\nid = "Bad"\nq = "x"\n', "id"),
        ('[[queries]]\nid = "ok"\nq = "  "\n', "q"),
        ('[[queries]]\nid = "ok"\nq = "x\\ny"\n', "control"),
        ('[[queries]]\nid = "ok"\nq = "x"\ndomains = "vision"\n', "domains"),
        ('[[queries]]\nid = "ok"\nq = "x"\nmethods = ["Bad"]\n', "methods"),
        ('[[queries]]\nid = "ok"\nq = "x"\nextra = true\n', "unknown fields"),
        ('[[queries]]\nid = "ok"\nq = "x"\ndomains = ["vision", "vision"]\n', "duplicate"),
    ],
)
def test_load_queries_rejects_invalid_config(tmp_path: Path, body: str, message: str) -> None:
    (tmp_path / "queries.toml").write_text(body)

    with pytest.raises(ValueError, match=message):
        load_queries(tmp_path)


def test_load_queries_rejects_duplicate_ids_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text('[[queries]]\nid = "same"\nq = "x"\n')
    (tmp_path / "b.toml").write_text('[[queries]]\nid = "same"\nq = "y"\n')

    with pytest.raises(ValueError, match="duplicate query id"):
        load_queries(tmp_path)


def test_load_queries_requires_config_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_queries(tmp_path / "missing")


def test_applied_application_labels_are_domains_not_methods() -> None:
    config_dir = Path(__file__).parents[1] / "config" / "queries"
    by_id = {query.id: query for query in load_queries(config_dir)}

    for query_id, domain in [
        ("applied.algorithmic-trading", "algorithmic-trading"),
        ("applied.fraud-detection", "fraud-detection"),
        ("applied.credit-scoring", "credit-scoring"),
    ]:
        assert domain in by_id[query_id].domains
        assert by_id[query_id].methods == ()

    assert by_id["applied.algorithmic-trading"].q == (
        '"algorithmic trading" "machine learning" in:readme'
    )


def _repository(**overrides: object) -> dict[str, object]:
    repo: dict[str, object] = {
        "id": 42,
        "full_name": "  Org/Novel-Model  ",
        "html_url": " https://github.com/Org/Novel-Model ",
        "description": "  A new method  ",
        "topics": ["vision", "ML", "vision"],
        "homepage": " ",
        "language": "Python",
        "license": {"spdx_id": "MIT"},
        "stargazers_count": 17,
        "forks_count": 2,
        "created_at": "2024-01-01T00:00:00Z",
        "pushed_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-02T00:00:00Z",
        "archived": False,
        "fork": False,
    }
    repo.update(overrides)
    return repo


def test_query_spec_is_frozen_and_requires_string_tuples() -> None:
    spec = QuerySpec(id="vision", q="topic:vision", domains=("vision",), methods=())
    with pytest.raises(AttributeError):
        spec.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="domains"):
        QuerySpec(id="vision", q="topic:vision", domains=["vision"], methods=())  # type: ignore[arg-type]


def test_observation_normalizes_repository_and_discovery_metadata() -> None:
    row = observation_from_repository(
        _repository(),
        observed_at="  2026-09-24T12:00:00Z  ",
        query_ids=["vision", "baseline", "vision"],
        domains=["Vision", " vision "],
        methods=["new-method"],
        novelty_signals=["paper-link", "readme-claim", "paper-link"],
    )

    assert row["github_id"] == 42
    assert row["name"] == "Org/Novel-Model"
    assert row["url"] == "https://github.com/Org/Novel-Model"
    assert row["description"] == "A new method"
    assert row["topics"] == ["ML", "vision"]
    assert row["homepage"] is None
    assert row["license"] == "MIT"
    assert row["stars"] == 17
    assert row["candidate_status"] == "candidate"
    assert row["observed_at"] == "2026-09-24T12:00:00Z"
    assert row["query_ids"] == ["baseline", "vision"]
    assert row["domains"] == ["vision"]
    assert row["methods"] == ["new-method"]
    assert row["novelty_signals"] == ["paper-link", "readme-claim"]


@pytest.mark.parametrize("repo", [{}, {"id": True}, {"id": 0}, {"id": 1, "full_name": "x"}])
def test_observation_requires_stable_github_identity(repo: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        observation_from_repository(
            repo,
            observed_at="2026-09-24T12:00:00Z",
            query_ids=[],
            domains=[],
            methods=[],
            novelty_signals=[],
        )


def test_observation_rejects_bad_counts_and_labels() -> None:
    for repo, kwargs in [
        (_repository(stargazers_count=-1), {}),
        (_repository(topics="vision"), {}),
        (_repository(), {"domains": [""]}),
    ]:
        with pytest.raises((TypeError, ValueError)):
            observation_from_repository(
                repo,
                observed_at="2026-09-24T12:00:00Z",
                query_ids=[],
                domains=kwargs.get("domains", []),
                methods=[],
                novelty_signals=[],
            )


def test_write_jsonl_has_stable_order_encoding_and_trailing_newline(tmp_path: Path) -> None:
    first = {"github_id": 2, "name": "café", "nested": {"z": 1, "a": 2}}
    second = {"name": "alpha", "github_id": 1}
    left, right = tmp_path / "left" / "out.jsonl", tmp_path / "right" / "out.jsonl"

    write_jsonl([first, second], left)
    write_jsonl([second, first], right)

    assert left.read_bytes() == right.read_bytes()
    assert left.read_text(encoding="utf-8").splitlines() == [
        '{"github_id":1,"name":"alpha"}',
        '{"github_id":2,"name":"café","nested":{"a":2,"z":1}}',
    ]
    assert left.read_bytes().endswith(b"\n")
    assert [json.loads(line)["github_id"] for line in left.read_text().splitlines()] == [1, 2]


def test_write_jsonl_rejects_rows_without_integer_github_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="github_id"):
        write_jsonl([{"github_id": True}], tmp_path / "out.jsonl")
