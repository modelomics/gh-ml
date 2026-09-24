from __future__ import annotations

import json
from pathlib import Path

import pytest

from gh_ml.schema import observation_from_repository, write_jsonl


def _repository(**overrides: object) -> dict[str, object]:
    repo: dict[str, object] = {
        "id": 123,
        "full_name": "org/模型",
        "html_url": "https://github.com/org/模型",
        "topics": ["科学", "Cafe\u0301", "科学"],
        "license": {"spdx_id": "NOASSERTION", "name": "Custom License"},
        "stargazers_count": 0,
        "forks_count": 0,
        "archived": False,
        "fork": False,
    }
    repo.update(overrides)
    return repo


def test_observation_preserves_unicode_and_optional_github_metadata() -> None:
    row = observation_from_repository(
        _repository(),
        observed_at="2026-09-24T12:00:00Z",
        query_ids=["q-模型"],
        domains=["科学"],
        methods=[],
        novelty_signals=[],
    )

    assert row["name"] == "org/模型"
    assert row["topics"] == ["Cafe\u0301", "科学"]
    assert row["license"] == "NOASSERTION"
    assert row["description"] is None
    assert row["created_at"] is None
    assert row["query_ids"] == ["q-模型"]
    assert row["domains"] == ["科学"]


def test_observation_uses_license_key_or_name_when_spdx_is_missing() -> None:
    for license_info, expected in [
        ({"spdx_id": None, "key": "other"}, "other"),
        ({"name": "License with Δ"}, "License with Δ"),
        ({}, None),
    ]:
        row = observation_from_repository(
            _repository(license=license_info),
            observed_at="observed",
            query_ids=[],
            domains=[],
            methods=[],
            novelty_signals=[],
        )
        assert row["license"] == expected


def test_observation_normalizes_and_deduplicates_method_slugs() -> None:
    row = observation_from_repository(
        _repository(),
        observed_at="observed",
        query_ids=[],
        domains=[],
        methods=[
            " Reinforcement Learning ",
            "reinforcement-learning",
            "Graph Neural Network",
            "Café Method",
            "α-β",
        ],
        novelty_signals=[],
    )

    assert row["methods"] == [
        "café-method",
        "graph-neural-network",
        "reinforcement-learning",
        "α-β",
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_write_jsonl_rejects_nonstandard_json_numbers(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError):
        write_jsonl([{"github_id": 1, "score": value}], tmp_path / "out.jsonl")


@pytest.mark.parametrize("github_id", [0, -1, True, "1"])
def test_write_jsonl_requires_positive_numeric_identity(
    tmp_path: Path, github_id: object
) -> None:
    with pytest.raises(ValueError, match="github_id"):
        write_jsonl([{"github_id": github_id}], tmp_path / "out.jsonl")


def test_write_jsonl_keeps_unicode_unescaped_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "out.jsonl"
    write_jsonl([{"github_id": 1, "name": "模型 café"}], destination)

    contents = destination.read_text(encoding="utf-8")
    assert "模型 café" in contents
    assert json.loads(contents) == {"github_id": 1, "name": "模型 café"}
