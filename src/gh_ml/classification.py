"""Lightweight, open-vocabulary labels for candidate ML repositories.

These labels describe retrieval evidence and likely subject matter. They are not
an assertion that a repository is novel, correct, or peer reviewed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("computer-vision", (r"computer vision", r"image classification", r"object detection", r"segmentation", r"image generation", r"vision transformer", r"\b(?:cv|vision)\b")),
    ("natural-language-processing", (r"natural language", r"\bnlp\b", r"language model", r"text generation", r"question answering", r"machine translation", r"\bllm\b")),
    ("speech-and-audio", (r"speech recognition", r"speech synthesis", r"text.to.speech", r"\basr\b", r"audio processing", r"\baudio\b", r"\bvoice\b")),
    ("reinforcement-learning", (r"reinforcement learning", r"\brl\b", r"policy gradient", r"\bq.learning\b", r"\bppo\b", r"\bsac\b")),
    ("robotics-and-control", (r"robotics?", r"robot manipulation", r"embodied ai", r"\bcontrol\b", r"autonomous driving", r"\bvla\b")),
    ("graph-learning", (r"graph neural", r"\bgnn\b", r"graph learning", r"knowledge graph", r"geometric deep learning")),
    ("time-series-and-forecasting", (r"time.series", r"forecast(?:ing)?", r"temporal model", r"sequence prediction")),
    ("recommender-systems", (r"recommend(?:er|ation|ing)", r"collaborative filtering", r"ranking model")),
    ("health-and-biomedicine", (r"biomedical", r"bioinformatics", r"medical imaging", r"drug discovery", r"protein", r"genomics", r"healthcare", r"\bclinical\b")),
    ("science-and-engineering", (r"scientific machine learning", r"\bphysics.informed\b", r"\bchemistry\b", r"\bclimate\b", r"\bearth observation\b", r"\bremote sensing\b")),
    ("tabular-and-structured-data", (r"tabular data", r"tabular learning", r"structured data", r"gradient boosting", r"\bxgboost\b", r"\bcatboost\b")),
    ("multimodal-learning", (r"multimodal", r"multi.modal", r"vision.language", r"image.text", r"audio.text")),
    ("generative-modeling", (r"generative model", r"diffusion model", r"\bdiffusion\b", r"\bgan\b", r"flow matching", r"autoregressive generation")),
    ("machine-learning-systems", (r"distributed training", r"model serving", r"inference engine", r"kernel fusion", r"\bquantization\b", r"\bpruning\b", r"\bcompiler\b")),
    ("interpretability-and-safety", (r"interpretability", r"explainable ai", r"\bxai\b", r"alignment", r"ai safety", r"adversarial robustness")),
    ("privacy-and-federated-learning", (r"federated learning", r"differential privacy", r"privacy preserving", r"secure aggregation")),
    ("optimization", (r"optimization algorithm", r"\bmetaheuristic\b", r"\boptimizer\b", r"hyperparameter optimization")),
)

_METHOD_TERMS = (
    "transformer", "diffusion", "flow matching", "mixture of experts", "state space model",
    "selective scan", "mamba", "retrieval augmented generation", "rag", "contrastive learning",
    "self supervised learning", "graph neural network", "neural operator", "neural network",
    "reinforcement learning", "federated learning", "quantization", "distillation", "pruning",
    "low rank adaptation", "lora", "test time adaptation", "meta learning", "causal inference",
    "bayesian optimization", "gradient boosting", "policy gradient", "world model",
)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    return []


def _spec_terms(spec: Any) -> list[str]:
    terms: list[str] = []
    for key in ("domain", "field", "method", "technique", "category", "query", "search_term", "name", "keywords"):
        terms.extend(_strings(_value(spec, key)))
    return terms


def classify_repository(repo: Mapping[str, Any], matched_specs: Sequence[Any]) -> dict[str, list[str]]:
    """Return reproducible domain/method labels and explicit evidence signals.

    Search query matches can contribute useful labels, while repository text is
    checked against narrow phrase rules to avoid labeling on generic words.
    """
    topics = _strings(repo.get("topics"))
    text_parts: list[str] = []
    for key in ("name", "full_name", "description", "readme", "readme_text", "homepage"):
        text_parts.extend(_strings(repo.get(key)))
    corpus = " ".join([*topics, *text_parts]).casefold().replace("_", " ").replace("-", " ")

    domains: set[str] = set()
    for domain, patterns in _DOMAIN_RULES:
        if any(re.search(pattern, corpus, re.IGNORECASE) for pattern in patterns):
            domains.add(domain)

    methods: set[str] = set()
    for method in _METHOD_TERMS:
        # Avoid very short abbreviations matching as arbitrary substrings.
        pattern = rf"(?<![a-z0-9]){re.escape(method).replace(r'\ ', r'\s+')}(?![a-z0-9])"
        if re.search(pattern, corpus, re.IGNORECASE):
            methods.add(method)
    for spec in matched_specs:
        for key in ("method", "methods", "technique"):
            for label in _strings(_value(spec, key)):
                if label.strip():
                    methods.add(label.strip().casefold())
        for key in ("domain", "domains", "field", "category"):
            for label in _strings(_value(spec, key)):
                if label.strip():
                    domains.add(label.strip().casefold())

    signals: set[str] = set()
    if matched_specs:
        signals.add("query-match")
    if repo.get("description"):
        signals.add("description")
    if topics:
        signals.add("github-topics")
    if repo.get("readme") or repo.get("readme_text"):
        signals.add("readme")
    if re.search(r"(?:10\.[0-9]{4,9}/|arxiv\.org/(?:abs|pdf)/|\barxiv\s*:\s*\d{4}\.\d{4,5})", corpus, re.I):
        signals.add("paper-reference")
    if repo.get("license"):
        signals.add("license-metadata")
    if repo.get("default_branch") or repo.get("pushed_at") or repo.get("updated_at"):
        signals.add("repository-metadata")
    if re.search(r"\.(?:safetensors|ckpt|pt|pth|bin|onnx)(?:\b|$)", corpus, re.I) or re.search(r"model weights|pretrained weights|checkpoint", corpus, re.I):
        signals.add("model-weights")

    return {
        "domains": sorted(domains),
        "methods": sorted(methods),
        "novelty_signals": sorted(signals),
    }
