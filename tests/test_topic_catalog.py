from pathlib import Path

import pytest

from gh_ml.topic_catalog import load_topics


EXPECTED_TOPICS = [
    "deep-learning",
    "machine-learning",
    "computer-vision",
    "natural-language-processing",
    "reinforcement-learning",
    "generative-ai",
    "diffusion-models",
    "diffusion-model",
    "stable-diffusion",
    "text-to-image",
    "video-generation",
    "large-language-models",
    "llm",
    "transformer",
    "transformers",
    "graph-neural-networks",
    "graph-neural-network",
    "time-series",
    "time-series-forecasting",
    "robotics",
    "knowledge-distillation",
    "self-supervised-learning",
    "meta-learning",
    "mixture-of-experts",
    "offline-reinforcement-learning",
    "world-models",
    "flow-matching",
    "multimodal-deep-learning",
    "vision-language-model",
    "federated-learning",
]


def write_catalog(tmp_path: Path, topics: str, extra: str = "") -> Path:
    path = tmp_path / "topics.toml"
    path.write_text(f"[catalog]\ntopics = {topics}\n{extra}", encoding="utf-8")
    return path


def test_default_catalog_has_expected_topics_in_stable_order() -> None:
    assert load_topics() == EXPECTED_TOPICS
    assert len(load_topics()) == 30


def test_load_topics_preserves_order(tmp_path: Path) -> None:
    path = write_catalog(tmp_path, '["second-topic", "first-topic"]')
    assert load_topics(path) == ["second-topic", "first-topic"]


@pytest.mark.parametrize(
    "topics",
    [
        "[]",
        "[" + ", ".join(f'"topic-{index}"' for index in range(101)) + "]",
        '["valid-topic", "valid-topic"]',
        '["Uppercase"]',
        '["leading-hyphen-"]',
        '[1]',
    ],
)
def test_rejects_invalid_catalogs(tmp_path: Path, topics: str) -> None:
    with pytest.raises(ValueError):
        load_topics(write_catalog(tmp_path, topics))


def test_rejects_unexpected_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_topics(write_catalog(tmp_path, '["valid-topic"]', "unexpected = true\n"))
