from gh_ml import ecosystems_collection as collection


def _ecosystems(*, synced, description="provider description", language="Python"):
    return collection._validated_eco_row(
        {
            "id": 9001,
            "uuid": 7,
            "full_name": "owner/repo",
            "last_synced_at": synced,
            "description": description,
            "topics": ["ml"],
            "language": language,
            "fork": False,
            "archived": False,
            "created_at": "2020-01-01T00:00:00Z",
            "pushed_at": "2026-09-01T00:00:00Z",
        },
        observed_at="2026-10-08T00:00:00Z",
    )


def _github(*, event_time, observed_at, description="GitHub description", **overrides):
    row = {
        "id": 7,
        "full_name": "owner/repo",
        "description": description,
        "topics": [],
        "language": None,
        "homepage": None,
        "stargazers_count": 0,
        "forks_count": 0,
        "fork": False,
        "archived": False,
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2026-09-01T00:00:00Z",
        "updated_at": event_time,
    }
    row.update(overrides)
    return collection._git_metadata(row, observed_at=observed_at)


def test_github_updated_at_is_event_time_and_not_a_source_sync_clock():
    row = _github(
        event_time="2020-04-05T12:00:00-07:00",
        observed_at="2026-10-09T10:30:00-07:00",
    )

    assert row["updated_at"] == "2020-04-05T19:00:00.000000Z"
    assert row["observed_at"] == "2026-10-09T17:30:00.000000Z"
    assert row["last_synced_at"] is None
    assert row["source_last_synced_at"] is None
    assert "last_synced_at" not in row["known_fields"]
    assert row["field_provenance"]["description"]["source_last_synced_at"] is None
    assert row["field_provenance"]["description"]["observed_at"] == row["observed_at"]
    assert not collection._is_incomplete(row)


def test_fresh_github_observation_can_replace_old_provider_clock_despite_old_event_time():
    provider = _ecosystems(synced="2026-10-02T00:00:00Z")
    github = _github(event_time="2020-01-01T00:00:00Z", observed_at="2026-10-09T00:00:00Z")

    merged = collection._merge(provider, github)

    assert merged["description"] == "GitHub description"
    assert merged["field_provenance"]["description"]["source"] == "github"
    assert merged["field_provenance"]["description"]["observed_at"] == "2026-10-09T00:00:00.000000Z"
    assert merged["field_provenance"]["description"]["source_last_synced_at"] is None


def test_later_provider_sync_beats_older_github_observation_and_retained_fields_keep_clocks():
    old_provider = _ecosystems(synced="2026-10-02T00:00:00Z", language="Rust")
    github = _github(
        event_time="2020-01-01T00:00:00Z",
        observed_at="2026-10-03T00:00:00Z",
        language="Go",
    )
    github_update = collection._merge(old_provider, github)
    assert github_update["language"] == "Go"

    later_provider = _ecosystems(synced="2026-10-04T00:00:00Z", language="Julia")
    merged = collection._merge(github_update, later_provider)

    assert merged["language"] == "Julia"
    assert merged["field_provenance"]["language"]["source"] == "ecosyste.ms"
    assert merged["field_provenance"]["language"]["source_last_synced_at"] == "2026-10-04T00:00:00.000000Z"
    # The provider's old known topic value remains attached to its original
    # source clock when the GitHub candidate lacks a topic sync clock.
    assert merged["topics"] == ["ml"]
    assert merged["field_provenance"]["topics"]["source"] == "ecosyste.ms"
    assert merged["field_provenance"]["topics"]["source_last_synced_at"] == "2026-10-04T00:00:00.000000Z"


def test_github_known_nulls_false_values_and_empty_topics_remain_known():
    row = _github(
        event_time="not-a-timestamp",
        observed_at="2026-10-09T00:00:00Z",
        description=None,
    )

    assert row["updated_at"] is None
    assert row["description"] is None
    assert row["topics"] == []
    assert row["fork"] is False
    assert row["archived"] is False
    assert {"description", "topics", "fork", "archived"}.issubset(row["known_fields"])
    assert "last_synced_at" in row["missing_required_fields"]
    assert not collection._is_incomplete(row)


def test_invalid_github_event_timestamp_does_not_invalidate_observation_evidence():
    provider = _ecosystems(synced="2026-10-02T00:00:00Z")
    github = _github(
        event_time="2026-99-99T99:99:99Z",
        observed_at="2026-10-03T00:00:00Z",
        description="Observed despite invalid event clock",
    )

    merged = collection._merge(provider, github)

    assert github["updated_at"] is None
    assert merged["description"] == "Observed despite invalid event clock"
    assert merged["field_provenance"]["description"]["observed_at"] == "2026-10-03T00:00:00.000000Z"

