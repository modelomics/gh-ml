"""Conservative eligibility rule for materialized ML repository candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CANDIDATE_RULE_VERSION = "ml-candidate-v2"
_REVIEW_REASON = "ml-relevance-without-clear-contribution"
_ALLOWED_EVIDENCE_TIERS = {"direct_ml_text", "ml_related_text"}
_REQUIRED_METHOD_SIGNAL = "ml-method-cue"
_REQUIRED_PAPER_CODE_SIGNAL = "paper-and-code-cue"


def assess_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Assess a materialized row without using discovery or popularity metadata."""
    status = row.get("selection_status") if isinstance(row, Mapping) else None

    if status == "include":
        eligible, reason = True, "selected-by-current-rule"
    elif status != "review":
        eligible, reason = False, "not-selected-or-reviewable"
    else:
        signals = row.get("selection_signals")
        signal_set = (
            {value for value in signals if isinstance(value, str)}
            if isinstance(signals, (list, tuple, set, frozenset))
            else set()
        )
        description = row.get("description")
        has_description = isinstance(description, str) and bool(description.strip())
        evidence_tier = row.get("evidence_tier")
        eligible = (
            row.get("selection_reason") == _REVIEW_REASON
            and has_description
            and _REQUIRED_METHOD_SIGNAL in signal_set
            and _REQUIRED_PAPER_CODE_SIGNAL in signal_set
            and isinstance(evidence_tier, str)
            and evidence_tier in _ALLOWED_EVIDENCE_TIERS
        )
        reason = "review-with-repository-evidence" if eligible else "insufficient-repository-evidence"

    return {
        "candidate_rule_version": CANDIDATE_RULE_VERSION,
        "candidate_eligible": eligible,
        "candidate_reason": reason,
    }
