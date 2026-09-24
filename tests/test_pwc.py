from __future__ import annotations

import json

import pytest

from gh_ml.github import GitHubAPIError, RepositoryBatchResult
from gh_ml.pwc import DATASET_REVISION, import_pwc, normalize_github_repo


@pytest.mark.parametrize(("value", "expected"), [
    ("https://github.com/Owner/Repo/tree/main/deep?x=1#frag", "Owner/Repo"),
    ("https://www.github.com/Owner/Repo.git", "Owner/Repo"),
    ("git@github.com:Owner/Repo.git", "Owner/Repo"),
    ("ssh://git@github.com/Owner/Repo", "Owner/Repo"),
    ("https://example.com/Owner/Repo", None),
    ("https://github.com/Owner", None),
    ("https://github.com/Owner/%2Fbad", None),
    (None, None),
])
def test_normalize_github_repo(value, expected):
    assert normalize_github_repo(value) == expected


def _row(url: str, **extra):
    return {"repo_url": url, "paper_url": "https://arxiv.org/abs/1234.5678",
            "paper_arxiv_id": "1234.5678", "is_official": True, **extra}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.batch_sizes = []

    def get_repositories_batch(self, names):
        self.calls.extend(names)
        self.batch_sizes.append(len(names))
        return RepositoryBatchResult(tuple(
            {"id": 17 + index, "full_name": name, "html_url": f"https://github.com/{name}",
             "description": "Vision transformer research implementation",
             "topics": [], "license": None, "stargazers_count": 2, "forks_count": 0,
             "archived": False, "fork": False}
            for index, name in enumerate(names)
        ), (None,) * len(names))


def test_bounded_import_stops_at_queue_limit_and_resumes(tmp_path):
    client = FakeClient()
    first = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client, rows=[
        _row("https://github.com/Owner/Repo/tree/main/x"),
        _row("https://github.com/owner/repo.git"),
        _row("https://github.com/Unscanned/Project"),
    ])
    assert first["rows_scanned"] == 1
    assert first["scan_offset_end"] == 1
    assert first["duplicate_links"] == 0
    assert first["repositories_resolved"] == 1
    assert first["graphql_batches"] == 1
    assert first["repositories_attempted"] == 1
    assert client.calls == ["Owner/Repo"]
    output = json.loads((tmp_path / first["observations_file"]).read_text())
    assert output["github_id"] == 17
    assert output["source_revision"] == DATASET_REVISION
    assert output["source_license"] == "CC-BY-SA-4.0"
    assert output["pwc_assertions"][0]["is_official"] is True
    assert output["pwc_assertions"][0]["paper_arxiv_id"] == "1234.5678"
    assert "computer-vision" in output["domains"]
    assert "transformer" in output["methods"]
    assert "paper-reference" in output["novelty_signals"]
    first_manifest = json.loads((tmp_path / f"manifest-{first['run_id']}.json").read_text())
    assert first_manifest["links_count"] == 1
    assert first_manifest["links_file"] == f"links-{first['run_id']}.jsonl"
    first_link = json.loads((tmp_path / first_manifest["links_file"]).read_text())
    assert first_link["source_row_offset"] == 0
    assert first_link["normalized_repo_name"] == "Owner/Repo"

    second = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client, rows=[
        _row("https://github.com/Owner/Repo"),
        _row("https://github.com/Other/Project/deep"),
        _row("https://github.com/Still/Unscanned"),
    ])
    assert second["scan_offset_start"] == 1
    assert second["rows_scanned"] == 2
    assert second["scan_offset_end"] == 3
    assert client.calls == ["Owner/Repo", "Other/Project"]
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["scan_offset"] == 3
    assert state["pending"] == []


def test_deduplicates_normalized_names_within_candidate_batch(tmp_path):
    client = FakeClient()
    result = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=2, client=client, rows=[
        _row("https://github.com/Owner/Repo/tree/main/a"),
        _row("https://github.com/owner/repo.git"),
        _row("https://github.com/Other/Project"),
        _row("https://github.com/NotScanned/Repo"),
    ])
    assert result["rows_scanned"] == 3
    assert result["duplicate_links"] == 1
    assert result["repositories_resolved"] == 2
    assert client.calls == ["Owner/Repo", "Other/Project"]


def test_import_batches_up_to_50_and_leaves_graphql_nulls_pending(tmp_path):
    class PartialClient(FakeClient):
        def get_repositories_batch(self, names):
            self.calls.extend(names)
            self.batch_sizes.append(len(names))
            repositories = [
                {"id": index + 1, "full_name": name, "html_url": f"https://github.com/{name}",
                 "topics": [], "license": None, "stargazers_count": 0, "forks_count": 0,
                 "archived": False, "fork": False}
                for index, name in enumerate(names)
            ]
            errors = [None] * len(names)
            if len(names) > 4:
                repositories[3] = None
                errors[3] = "GraphQL field error"
                repositories[4] = None
            return RepositoryBatchResult(tuple(repositories), tuple(errors))

    client = PartialClient()
    rows = [_row(f"https://github.com/owner-{index}/repo-{index}") for index in range(52)]
    result = import_pwc(output_dir=tmp_path, max_rows=100, max_repos=52,
                        client=client, rows=rows)
    assert client.batch_sizes == [50, 2]
    assert result["graphql_batches"] == 2
    assert result["repositories_attempted"] == 52
    assert result["repositories_resolved"] == 50
    assert result["repositories_unresolved"] == 2
    assert result["repositories_not_found"] == 0
    assert result["pending_repositories"] == 2
    assert len(json.loads((tmp_path / "checkpoint.json").read_text())["pending"]) == 2


def test_pending_candidate_retries_on_resume_without_scanning_or_marking_404(tmp_path):
    class NullThenSuccess(FakeClient):
        def __init__(self):
            super().__init__()
            self.null = True

        def get_repositories_batch(self, names):
            if self.null:
                self.null = False
                self.calls.extend(names)
                return RepositoryBatchResult((None,), (None,))
            return super().get_repositories_batch(names)

    client = NullThenSuccess()
    first = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client,
                       rows=[_row("https://github.com/owner/repo")])
    assert first["repositories_unresolved"] == 1
    assert first["repositories_not_found"] == 0
    assert first["pending_repositories"] == 1
    second = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client, rows=[])
    assert second["scan_offset_start"] == second["scan_offset_end"] == 1
    assert second["repositories_resolved"] == 1
    assert second["repositories_not_found"] == 0
    assert second["pending_repositories"] == 0
    assert client.calls == ["owner/repo", "owner/repo"]


def test_resolved_repositories_deduplicate_by_numeric_github_id_across_runs(tmp_path):
    class RenamedAliasClient(FakeClient):
        def get_repositories_batch(self, names):
            self.calls.extend(names)
            return RepositoryBatchResult(tuple(
                {"id": 4242, "full_name": "canonical/new-name",
                 "html_url": "https://github.com/canonical/new-name", "topics": [],
                 "license": None, "stargazers_count": 0, "forks_count": 0,
                 "archived": False, "fork": False}
                for _ in names
            ), (None,) * len(names))

    client = RenamedAliasClient()
    first = import_pwc(output_dir=tmp_path, max_rows=1, max_repos=1, client=client,
                       rows=[_row("https://github.com/old-owner/old-name")])
    second = import_pwc(output_dir=tmp_path, max_rows=1, max_repos=1, client=client,
                        rows=[_row("https://github.com/other-owner/other-name")])

    first_rows = (tmp_path / first["observations_file"]).read_text().splitlines()
    second_rows = (tmp_path / second["observations_file"]).read_text().splitlines()
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert len(first_rows) == 1
    assert second_rows == []
    assert second["repositories_resolved"] == 1
    assert second["duplicate_repository_ids"] == 1
    assert second["observations_written"] == 0
    assert state["seen_github_ids"] == [4242]


def test_resume_reconciles_durable_observations_with_stale_checkpoint(tmp_path):
    class RenamedAliasClient(FakeClient):
        def get_repositories_batch(self, names):
            self.calls.extend(names)
            return RepositoryBatchResult(tuple(
                {"id": 4242, "full_name": "canonical/new-name",
                 "html_url": "https://github.com/canonical/new-name", "topics": [],
                 "license": None, "stargazers_count": 0, "forks_count": 0,
                 "archived": False, "fork": False}
                for _ in names
            ), (None,) * len(names))

    client = RenamedAliasClient()
    first = import_pwc(output_dir=tmp_path, max_rows=1, max_repos=1, client=client,
                       rows=[_row("https://github.com/old-owner/old-name")])
    checkpoint_path = tmp_path / "checkpoint.json"
    stale_state = json.loads(checkpoint_path.read_text())
    stale_state.update({"scan_offset": 0, "seen_names": [], "seen_github_ids": [],
                        "pending": []})
    checkpoint_path.write_text(json.dumps(stale_state), encoding="utf-8")

    resumed = import_pwc(output_dir=tmp_path, max_rows=1, max_repos=1, client=client,
                         rows=[_row("https://github.com/new-owner/new-alias",
                                    paper_url="https://paperswithcode.com/paper/replayed")])

    observations = [
        json.loads(line)
        for path in tmp_path.glob("observations-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    manifest = json.loads((tmp_path / f"manifest-{resumed['run_id']}.json").read_text())
    replayed_links = [
        json.loads(line)
        for line in (tmp_path / manifest["links_file"]).read_text().splitlines()
    ]
    repaired_state = json.loads(checkpoint_path.read_text())
    assert len(observations) == 1
    assert resumed["observations_written"] == 0
    assert resumed["duplicate_repository_ids"] == 1
    assert repaired_state["seen_github_ids"] == [4242]
    assert len(replayed_links) == 1
    assert replayed_links[0]["normalized_repo_name"] == "new-owner/new-alias"
    assert replayed_links[0]["paper_url"].endswith("/replayed")


def test_link_sidecars_preserve_duplicate_assertions_across_resume_runs(tmp_path):
    client = FakeClient()
    first = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client, rows=[
        _row("https://github.com/Owner/Repo", is_official=True,
             paper_url="https://paperswithcode.com/paper/first"),
    ])
    first_observation = json.loads((tmp_path / first["observations_file"]).read_text())
    assert first_observation["pwc_assertions"][0]["is_official"] is True

    second = import_pwc(output_dir=tmp_path, max_rows=10, max_repos=1, client=client, rows=[
        _row("https://github.com/owner/repo.git", is_official=False,
             paper_url="https://paperswithcode.com/paper/second"),
        _row("https://github.com/Other/Project"),
    ])
    manifest = json.loads((tmp_path / f"manifest-{second['run_id']}.json").read_text())
    links = [json.loads(line) for line in (tmp_path / manifest["links_file"]).read_text().splitlines()]
    assert manifest["links_count"] == 2
    assert [link["source_row_offset"] for link in links] == [1, 2]
    assert [link["normalized_repo_name"].casefold() for link in links] == [
        "owner/repo", "other/project",
    ]
    assert links[0]["is_official"] is False
    assert links[0]["paper_url"].endswith("/second")
    assert links[0]["source_license"] == "CC-BY-SA-4.0"
    assert client.calls == ["Owner/Repo", "Other/Project"]


def test_links_are_written_before_checkpoint_advances(tmp_path, monkeypatch):
    import gh_ml.pwc as pwc

    original_write_json = pwc._write_json

    def fail_checkpoint(path, value):
        if path.name == "checkpoint.json":
            raise OSError("simulated checkpoint failure")
        original_write_json(path, value)

    monkeypatch.setattr(pwc, "_write_json", fail_checkpoint)
    with pytest.raises(OSError, match="simulated checkpoint failure"):
        import_pwc(output_dir=tmp_path, max_rows=1, client=FakeClient(), rows=[
            _row("https://github.com/owner/repo"),
        ])
    sidecars = list(tmp_path.glob("links-*.jsonl"))
    assert len(sidecars) == 1
    assert json.loads(sidecars[0].read_text())["normalized_repo_name"] == "owner/repo"
    assert not (tmp_path / "checkpoint.json").exists()
    assert not list(tmp_path.glob("links-*.jsonl.tmp"))


def test_non_404_lookup_failure_preserves_pending_and_scan_cursor(tmp_path):
    class FailingClient:
        def get_repositories_batch(self, names):
            raise GitHubAPIError(503, "temporary")

    manifest = import_pwc(output_dir=tmp_path, max_rows=1, client=FailingClient(),
                          rows=[_row("https://github.com/owner/repo")])
    assert manifest["resolution_error"]
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["scan_offset"] == 1
    assert state["pending"][0]["name"] == "owner/repo"
