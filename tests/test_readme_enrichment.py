from datetime import datetime, timedelta, timezone

import gh_ml.readme_enrichment as readme_enrichment
from gh_ml.github import GitHubAPIError, GitHubReadmeResult
from gh_ml.readme_enrichment import enrich_readmes, select_readme_targets
from gh_ml.readme_signals import README_EVIDENCE_VERSION


NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def row(repo_id, name, *, status="review", signals=(), description="transformer method", topics=(), methods=(), evidence_tier=None):
    return {
        "github_id": repo_id, "full_name": name, "name": name.split("/")[-1],
        "description": description, "selection_status": status,
        "selection_signals": list(signals), "candidate_eligible": False,
        "topics": list(topics), "methods": list(methods), "evidence_tier": evidence_tier,
    }


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def get_readme(self, name, etag=None):
        self.calls.append((name, etag))
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def result(status, text=None, etag='"e1"', sha="blob1"):
    return GitHubReadmeResult(status, etag, sha if status == 200 else None, text)


def test_selector_uses_generic_signals_and_rejects_applied_classes_profiles_and_utilities():
    rows = [
        row(1, "research/sparse", signals=["paper-and-code-cue"], description="A specialized method."),
        row(2, "research/method", signals=["ml-method-cue"], description="A sparse research project."),
        row(3, "apps/classifier", description="Applied transformer model for customer support."),
        row(4, "school/course-project", signals=["official-paper-implementation-cue"], description="Course project."),
        row(5, "alice/alice", signals=["official-paper-implementation-cue"], description="Research implementation."),
        row(6, "quant/zipline", signals=["backtesting-utility-cue", "paper-and-code-cue"], description="Algorithmic trading backtester."),
        row(7, "tools/tutorial", signals=["ml-method-cue"], description="Transformer tutorial."),
    ]
    chosen = select_readme_targets(rows, {}, now=NOW, max_requests=10)
    assert [r["github_id"] for r in chosen] == [1, 2]


def test_candidate_eligibility_is_not_required_for_known_research_metadata_patterns():
    rows = [
        row(101, "lab/drag-edit", signals=["ml-method-cue"], description="Interactive image manipulation method."),
        row(102, "lab/language-model", signals=["paper-and-code-cue"], description="Autoregressive language model."),
        row(103, "lab/video-diffusion", signals=["official-paper-implementation-cue"], description="Video generation method."),
        row(104, "lab/self-supervised", signals=["ml-method-cue"], description="Self-supervised representation method."),
    ]
    assert [r["github_id"] for r in select_readme_targets(rows, {}, now=NOW, max_requests=10)] == [102, 101, 103, 104]


def test_low_priority_official_code_route_uses_generic_ml_context_or_method_labels():
    rows = [
        row(201, "XingangPan/DragGAN", signals=["ml-context-only"],
            description="Official Code for DragGAN (SIGGRAPH 2023)", topics=["generative-models", "GAN"]),
        row(202, "guoyww/AnimateDiff", description="Official implementation of AnimateDiff.",
            methods=["diffusion", "generative-modeling"]),
        row(203, "tools/official-cli", description="Official implementation of a command line tool."),
        row(204, "school/course-code", description="Official implementation of course materials.", methods=["diffusion"]),
        row(205, "apps/applied", description="Official code for an applied customer classifier.", topics=["customer-support"]),
    ]
    # These are inspection targets in the lowest queue tier, not selector promotions.
    assert [r["github_id"] for r in select_readme_targets(rows, {}, now=NOW, max_requests=10)] == [201, 202]


def test_tiered_round_robin_and_checkpoint_cursor_fairness():
    rows = [row(i, f"lab/r{i}", status="include", signals=["ml-method-cue"]) for i in range(1, 5)]
    rows += [row(10, "lab/high", signals=["official-paper-implementation-cue"])]
    first = select_readme_targets(rows, {}, now=NOW, max_requests=3)
    assert [r["github_id"] for r in first] == [10, 1, 2]
    checkpoint = {"cursors": {"0": 10, "1": 2}, "repositories": {}}
    second = select_readme_targets(rows, checkpoint, now=NOW, max_requests=3)
    assert [r["github_id"] for r in second] == [10, 3, 4]


def test_200_persists_only_enums_and_304_reuses_compact_prior_evidence():
    candidate = row(12, "lab/research", signals=["paper-and-code-cue"])
    body = "# Method\nWe propose a novel transformer model. Official implementation and paper."
    client = FakeClient([result(200, body), result(304, etag='"e2"')])
    records, checkpoint, coverage = enrich_readmes([candidate], {}, client, now=NOW, max_requests=1)
    assert records[0]["readme_status"] == "ok"
    assert "text" not in records[0] and "body" not in str(checkpoint)
    assert records[0]["readme_signals"] == [
        "method-contribution", "ml-method-context", "official-implementation-claim",
        "paper-code-relationship", "paper-reference",
    ]
    updated = NOW + timedelta(days=365)
    records2, checkpoint2, _ = enrich_readmes([candidate], checkpoint, client, now=updated, max_requests=1)
    assert client.calls == [("lab/research", None), ("lab/research", '"e1"')]
    assert records2[0]["readme_status"] == "unchanged"
    assert records2[0]["readme_signals"] == records[0]["readme_signals"]


def test_rename_drops_etag_and_404_emits_empty_latest_evidence_with_cooldown():
    candidate = row(20, "lab/old", signals=["paper-and-code-cue"])
    checkpoint = {"repositories": {"20": {
        "repository_name_at_fetch": "lab/old", "readme_etag": '"old"',
        "readme_signals": ["code-for-paper"], "readme_sections": ["method"],
        "readme_evidence_version": "v1", "readme_blob_sha": "sha",
        "due_at": "2027-09-24T12:00:00Z",
    }}}
    renamed = dict(candidate, full_name="lab/new", name="new")
    client = FakeClient([result(404)])
    records, cp, _ = enrich_readmes([renamed], checkpoint, client, now=NOW, max_requests=1)
    assert client.calls == [("lab/new", None)]
    assert records[0]["readme_status"] == "missing" and records[0]["readme_signals"] == []
    assert cp["repositories"]["20"]["readme_signals"] == ["code-for-paper"]
    assert select_readme_targets([renamed], cp, now=NOW + timedelta(days=1), max_requests=1) == []


def test_stale_evidence_reenriches_before_due_but_current_and_excluded_rows_do_not():
    candidate = row(30, "lab/research", signals=["paper-and-code-cue"])
    future = "2027-09-24T12:00:00Z"
    checkpoint = {"repositories": {
        "30": {
            "repository_name_at_fetch": "lab/research", "readme_etag": '"old"',
            "readme_evidence_version": "gh-ml-readme-evidence-v1",
            "readme_signals": ["old-signal"], "due_at": future,
        },
        "31": {
            "repository_name_at_fetch": "lab/current", "readme_etag": '"current"',
            "readme_evidence_version": README_EVIDENCE_VERSION, "due_at": future,
        },
        "32": {
            "repository_name_at_fetch": "school/course", "readme_etag": '"excluded"',
            "readme_evidence_version": "gh-ml-readme-evidence-v1", "due_at": future,
        },
    }}
    current = row(31, "lab/current", signals=["paper-and-code-cue"])
    excluded = row(32, "school/course", signals=["paper-and-code-cue"], description="Course project.")
    assert [item["github_id"] for item in select_readme_targets(
        [candidate, current, excluded], checkpoint, now=NOW, max_requests=10,
    )] == [30]

    client = FakeClient([result(200, "# Method\nWe propose a transformer method.")])
    records, updated, coverage = enrich_readmes(
        [candidate, current, excluded], checkpoint, client, now=NOW, max_requests=1,
    )
    assert client.calls == [("lab/research", None)]
    assert coverage["attempted"] == 1
    assert records[0]["readme_evidence_version"] == README_EVIDENCE_VERSION
    assert updated["repositories"]["30"]["readme_evidence_version"] == README_EVIDENCE_VERSION


def test_stale_404_refresh_obeys_cooldown_and_retriggers_after_version_bump(monkeypatch):
    candidate = row(40, "lab/missing", signals=["paper-and-code-cue"])
    checkpoint = {"repositories": {"40": {
        "repository_name_at_fetch": "lab/missing", "readme_etag": '"old"',
        "readme_evidence_version": "gh-ml-readme-evidence-v1",
        "readme_signals": ["old-signal"], "due_at": "2027-09-24T12:00:00Z",
    }}}
    client = FakeClient([result(404)])
    _, refreshed, _ = enrich_readmes([candidate], checkpoint, client, now=NOW, max_requests=1)
    prior = refreshed["repositories"]["40"]
    assert prior["readme_refresh_attempted_version"] == README_EVIDENCE_VERSION
    assert select_readme_targets([candidate], refreshed, now=NOW + timedelta(days=1), max_requests=1) == []

    monkeypatch.setattr(readme_enrichment, "README_EVIDENCE_VERSION", "gh-ml-readme-evidence-v3")
    assert select_readme_targets([candidate], refreshed, now=NOW + timedelta(days=1), max_requests=1) == [candidate]


def test_stale_transient_error_refresh_obeys_error_cooldown():
    candidate = row(41, "lab/timeout", signals=["paper-and-code-cue"])
    checkpoint = {"repositories": {"41": {
        "repository_name_at_fetch": "lab/timeout", "readme_etag": '"old"',
        "readme_evidence_version": "gh-ml-readme-evidence-v1",
        "readme_signals": ["old-signal"], "due_at": "2027-09-24T12:00:00Z",
    }}}
    client = FakeClient([GitHubAPIError(None, "timeout")])
    _, refreshed, coverage = enrich_readmes([candidate], checkpoint, client, now=NOW, max_requests=1)
    assert coverage["deferred"] == 1
    assert refreshed["repositories"]["41"]["readme_refresh_attempted_version"] == README_EVIDENCE_VERSION
    assert select_readme_targets([candidate], refreshed, now=NOW + timedelta(hours=12), max_requests=1) == []


def test_request_budget_is_exact_and_rate_limit_stops_without_advancing_failing_id():
    candidates = [row(i, f"lab/r{i}", status="include") for i in range(1, 5)]
    budget_client = FakeClient([result(200, "# Overview\nTransformer model.") for _ in range(4)])
    records, _, coverage = enrich_readmes(candidates, {}, budget_client, now=NOW, max_requests=2)
    assert len(records) == len(budget_client.calls) == coverage["attempted"] == 2

    rate_client = FakeClient([result(200, "# Overview\nTransformer model."), GitHubAPIError(429, "rate")])
    records, cp, coverage = enrich_readmes(candidates, {}, rate_client, now=NOW, max_requests=4)
    assert len(records) == 1 and coverage["attempted"] == 2 and coverage["rate_limited"] == 1
    assert cp["cursors"]["1"] == 1
    assert 2 not in [int(k) for k in cp["repositories"]]

    forbidden_client = FakeClient([GitHubAPIError(403, "denied")])
    _, forbidden_cp, forbidden_coverage = enrich_readmes(candidates[:1], {}, forbidden_client, now=NOW, max_requests=3)
    assert forbidden_coverage["attempted"] == 1 and forbidden_coverage["rate_limited"] == 1
    assert "1" not in forbidden_cp["cursors"] and "1" not in forbidden_cp["repositories"]

    trans_client = FakeClient([GitHubAPIError(None, "timeout")])
    _, deferred_cp, _ = enrich_readmes(candidates[:1], {}, trans_client, now=NOW, max_requests=1)
    assert deferred_cp["repositories"]["1"]["due_at"] == "2026-09-25T12:00:00Z"


def test_successful_readme_waits_a_year_while_unprocessed_row_advances_next_day():
    first = row(1, "lab/first", status="include")
    next_row = row(2, "lab/next", status="include")
    client = FakeClient([result(200, "# Overview\nTransformer model.")])
    _, checkpoint, _ = enrich_readmes([first, next_row], {}, client, now=NOW, max_requests=1)

    tomorrow = NOW + timedelta(days=1)
    selected = select_readme_targets([first, next_row], checkpoint, now=tomorrow, max_requests=1)
    assert [item["github_id"] for item in selected] == [2]
    assert checkpoint["repositories"]["1"]["due_at"] == "2027-09-24T12:00:00Z"
