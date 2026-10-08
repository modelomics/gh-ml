"""Broad, auditable eligibility rule for probable original ML repositories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .probable_content import PROBABLE_CONTENT_SIGNALS, assess_probable_content


CANDIDATE_RULE_VERSION = "ml-candidate-v4"
_REVIEW_REASON = "ml-relevance-without-clear-contribution"
_ALLOWED_EVIDENCE_TIERS = {"direct_ml_text", "ml_related_text"}
_REVIEWABLE_REASONS = {
    _REVIEW_REASON,
    "insufficient-repository-evidence",
    "overview-reproduction-or-dataset",
    "course-readme-without-official-paper-evidence",
    "course-or-utility-with-novel-method-cue",
    "non-ml-utility-with-ml-contribution-cue",
}
_RESCUABLE_EXCLUSION_REASONS = {
    "course-or-utility-repository",
    "tutorial-repository",
    "explicit-noncontribution",
    "non-ml-utility",
}
_HARD_EXCLUSION_REASONS = {
    "fork",
    "owner-profile-repository",
    "survey-or-paper-list-repository",
}
_LEGACY_PAPER_SIGNALS = {"ml-method-cue", "paper-and-code-cue"}


def assess_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Assess repository-owned text without using query or popularity metadata."""
    evidence: set[str] = set()
    reason = "insufficient-repository-evidence"
    status = row.get("selection_status") if isinstance(row, Mapping) else None
    raw_reason = row.get("selection_reason") if isinstance(row, Mapping) else None
    selection_reason = raw_reason if isinstance(raw_reason, str) else None

    if not isinstance(row, Mapping):
        eligible = False
    elif row.get("fork") is True or selection_reason == "fork":
        eligible, reason = False, "fork"
    elif selection_reason in _HARD_EXCLUSION_REASONS:
        eligible, reason = False, "not-selected-or-reviewable"
    elif status == "include":
        eligible, reason = True, "selected-by-current-rule"
    elif status not in ("review", "exclude"):
        eligible = False
    else:
        description = row.get("description")
        if isinstance(description, str) and description.strip():
            evidence.update(f"description:{signal}" for signal in assess_probable_content(description))

        if row.get("readme_status") in ("ok", "unchanged") and row.get("readme_evidence_version") == "gh-ml-readme-evidence-v3":
            readme_signals = row.get("readme_signals")
            if isinstance(readme_signals, (list, tuple, set, frozenset)):
                evidence.update(
                    f"readme:{signal}"
                    for signal in readme_signals
                    if isinstance(signal, str) and signal in PROBABLE_CONTENT_SIGNALS
                )

        active_readme = row.get("readme_status") in ("ok", "unchanged") and row.get("readme_evidence_version") == "gh-ml-readme-evidence-v3"
        readme_sections = row.get("readme_sections")
        substantive_readme_section = isinstance(readme_sections, (list, tuple, set, frozenset)) and bool(
            {"method", "results"} & {section for section in readme_sections if isinstance(section, str)}
        )
        review_route = status == "review" and selection_reason in _REVIEWABLE_REASONS
        exclusion_route = (
            status == "exclude"
            and selection_reason in _RESCUABLE_EXCLUSION_REASONS
            and (
                (
                    selection_reason == "non-ml-utility"
                    and "description:substantive-application-or-experiments" in evidence
                )
                or (
                    selection_reason != "non-ml-utility"
                    and active_readme
                    and substantive_readme_section
                    and any(item.startswith("readme:") for item in evidence)
                )
            )
        )
        eligible = bool(evidence) and (review_route or exclusion_route)
        if eligible:
            # Categories are fixed enum strings; source order makes the reason deterministic.
            first = sorted(evidence)[0]
            reason = f"probable-original-content:{first}"
        elif status == "review" and not evidence and selection_reason == _REVIEW_REASON:
            # Preserve the established paper-linked review path for rows whose
            # candidate status is available but whose description has no new cue.
            signals = row.get("selection_signals")
            signal_set = (
                {value for value in signals if isinstance(value, str)}
                if isinstance(signals, (list, tuple, set, frozenset))
                else set()
            )
            has_description = isinstance(description, str) and bool(description.strip())
            paper_ids = row.get("paper_ids")
            has_paper_ids = (
                isinstance(paper_ids, list)
                and bool(paper_ids)
                and all(isinstance(value, str) and bool(value.strip()) for value in paper_ids)
            )
            tier = row.get("evidence_tier")
            legacy_eligible = (
                has_description
                and "ml-method-cue" in signal_set
                and (("paper-and-code-cue" in signal_set) or has_paper_ids)
                and isinstance(tier, str)
                and tier in _ALLOWED_EVIDENCE_TIERS
            )
            eligible = legacy_eligible
            if legacy_eligible:
                reason = (
                    "review-with-repository-evidence"
                    if "paper-and-code-cue" in signal_set
                    else "review-with-unverified-paper-link"
                )
                evidence.update(f"selection:{signal}" for signal in sorted(signal_set & _LEGACY_PAPER_SIGNALS))
        else:
            eligible = False
            if status == "exclude":
                reason = "not-selected-or-reviewable"

    return {
        "candidate_rule_version": CANDIDATE_RULE_VERSION,
        "candidate_eligible": eligible,
        "candidate_reason": reason,
        "candidate_evidence": sorted(evidence),
    }
