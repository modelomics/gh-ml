from gh_ml.evidence import EVIDENCE_VERSION, classify_repository_text


def test_explicit_ml_text_gets_direct_tier():
    result = classify_repository_text({
        "name": "new method",
        "description": "A machine learning approach for tabular data",
        "topics": ["research"],
    })

    assert result == {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_tier": "direct_ml_text",
        "evidence_signals": ["machine-learning"],
    }


def test_related_artifact_text_gets_intermediate_tier():
    result = classify_repository_text({"name": "fast-transformers", "description": None})

    assert result["evidence_tier"] == "ml_related_text"
    assert result["evidence_signals"] == ["transformer"]


def test_query_and_domain_labels_cannot_raise_text_evidence():
    finance_candidate = {
        "name": "algorithmic-trading-backtest",
        "description": "Backtesting strategies for financial markets",
        "topics": ["finance", "trading"],
        "domains": ["applied.algorithmic-trading"],
        "methods": ["reinforcement-learning"],
        "query_ids": ["finance.reinforcement-learning"],
    }

    assert classify_repository_text(finance_candidate) == {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_tier": "no_text_signal",
        "evidence_signals": [],
    }


def test_missing_or_unexpected_repository_fields_are_safe():
    assert classify_repository_text({})["evidence_tier"] == "no_text_signal"
    assert classify_repository_text({"topics": [None, 4, "LLM"]}) == {
        "evidence_version": EVIDENCE_VERSION,
        "evidence_tier": "direct_ml_text",
        "evidence_signals": ["large-language-model"],
    }


def test_homepage_url_alone_does_not_create_evidence():
    result = classify_repository_text({
        "homepage": "https://ml.example.org/models/transformer",
    })

    assert result["evidence_tier"] == "no_text_signal"
    assert result["evidence_signals"] == []


def test_signal_order_is_stable():
    row = {"description": "Transformer and neural networks for machine learning"}
    expected = classify_repository_text(row)

    for _ in range(5):
        assert classify_repository_text(row) == expected
    assert expected["evidence_signals"] == ["machine-learning", "neural-network", "transformer"]
