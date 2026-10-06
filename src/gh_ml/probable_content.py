"""Repository-text cues for probable original ML content.

These bounded signals identify contribution-shaped descriptions. They do not
verify that a repository's claims are true or that its implementation works.
"""

from __future__ import annotations

import re

PROBABLE_CONTENT_SIGNALS = frozenset({
    "original-implementation",
    "adaptation-or-fine-tuning",
    "substantive-application-or-experiments",
    "original-dataset-or-benchmark",
    "original-tooling",
})

_ML = re.compile(
    r"\b(?:machine[ -]learning|deep[ -]learning|artificial intelligence|\bml\b|\bai\b|"
    r"neural(?: network| net)?s?|transformers?|diffusion(?: models?)?|gans?|"
    r"large language models?|\bllms?\b|\b(?:bert|gpt|llama|vit)\b|"
    r"random forests?|gradient boosting|\b(?:xgboost|lightgbm|catboost|svm)\b|"
    r"naive bayes|decision trees?|k[ -]?nearest neighbors?|support vector machines?|"
    r"classifier|regression model|reinforcement learning|embedding model|"
    r"speech recognition model|speech models?|computer vision model|"
    r"image synthesis|image generation|super[ -]resolution|growing neural networks?)\b", re.I,
)
_MODEL_NOUN = re.compile(
    r"\b(?:models?|methods?|architectures?|algorithms?|classifiers?|neural networks?|"
    r"predictors?|polic(?:y|ies)|estimators?|embeddings?|adapters?|agents?)\b", re.I,
)
_ML_MODEL_TARGET = re.compile(r"\b(?:model|network|weights?|parameters?|neural)\b", re.I)
_CREATE = re.compile(
    r"\b(?:implement(?:s|ed|ing|ation)?|build(?:s|ing)?|train(?:s|ed|ing)?|"
    r"develop(?:s|ed|ing|ment)?|design(?:s|ed|ing)?|introduc(?:e|es|ed|ing)|"
    r"propos(?:e|es|ed|ing)|construct(?:s|ed|ing)?|creat(?:e|es|ed|ing))\b", re.I,
)
_ADAPT = re.compile(
    r"\b(?:fine[ -]?tun(?:e|es|ed|ing)|adapt(?:s|ed|ation|ing)?|"
    r"distill(?:s|ed|ation|ing)?|prun(?:e|es|ed|ing)|quantiz(?:e|es|ed|ation|ing)|"
    r"domain[ -]adaptation|transfer learning)\b", re.I,
)
_APPLICATION = re.compile(
    r"\b(?:appl(?:y|ies|ied|ying)|detect(?:s|ed|ing)?|classif(?:y|ies|ied|ying)|"
    r"predict(?:s|ed|ing|ion)?|forecast(?:s|ed|ing)?|segment(?:s|ed|ation)?|"
    r"evaluat(?:e|es|ed|ing|ion)|benchmark(?:s|ed|ing)?|experiment(?:s|ed|ing|ation)?|"
    r"measure(?:s|d|ment|ing)?|compar(?:e|es|ed|ing|ison|isons)|recogniz(?:e|es|ed|ing)|"
    r"ablat(?:e|es|ed|ing)|ablation|ablations|estimate(?:s|d|ing)?|generat(?:e|es|ed|ing))\b", re.I,
)
_DATASET_CREATE = re.compile(
    r"\b(?:creat(?:e|es|ed|ing)|collect(?:s|ed|ing)?|curat(?:e|es|ed|ing)|"
    r"annotat(?:e|es|ed|ing)|label(?:s|ed|ing)?|build(?:s|ing)?|"
    r"introduc(?:e|es|ed|ing)|releas(?:e|es|ed|ing))\b.{0,100}"
    r"\b(?:dataset|data set|corpus|benchmark|test set)\b|"
    r"\b(?:dataset|data set|corpus|benchmark|test set)\b.{0,100}"
    r"\b(?:creat(?:e|es|ed|ing)|collect(?:s|ed|ing)?|curat(?:e|es|ed|ing)|"
    r"annotat(?:e|es|ed|ing)|label(?:s|ed|ing)?|introduc(?:e|es|ed|ing))\b", re.I,
)
_DATA_PROTOCOL = re.compile(
    r"\b(?:splits?|annotations?|labels?|tasks?|metrics?|evaluation protocol|leaderboard)\b", re.I,
)
_TOOLING = re.compile(
    r"\b(?:library|toolkit|framework|pipeline|training system|evaluation suite|"
    r"benchmark harness|data pipeline|compiler|runtime|infrastructure|package)\b", re.I,
)
_TOOL_PURPOSE = re.compile(
    r"\b(?:train(?:ing)?|evaluat(?:e|ion)|benchmark(?:ing)?|fine[ -]?tun(?:e|ing)|"
    r"preprocess(?:ing)?|label(?:ing)?|annotat(?:ion|ing)|inference|data set|dataset|"
    r"model)\b", re.I,
)
_SUBSTANTIVE_EVIDENCE = re.compile(
    r"\b(?:train(?:s|ed|ing)?|fine[ -]?tun(?:e|es|ed|ing)|experiment(?:s|ed|ing)?|"
    r"evaluat(?:e|es|ed|ing|ion)|benchmark(?:s|ed|ing)?|experimentation|dataset|data set|corpus|"
    r"measurements?|annotations?|labels?|metrics?|accuracy|results?|cross[ -]validation|"
    r"ablation|collected|curated|annotated|labeled|training data)\b", re.I,
)
_NEGATIVE_ONLY = re.compile(
    r"\b(?:tutorial|walkthrough|coursework|course project|homework|assignment|"
    r"faithful reproduction|reimplementation|replication|mirror|fork of|"
    r"reading list|awesome list|paper list)\b", re.I,
)
_NONASSERTION = re.compile(
    r"\b(?:does not|do not|doesn't|don't|will(?: not)?|won't|planned? to|plans to|"
    r"roadmap|future work|coming soon|list of|collection of|directory of|"
    r"links to|projects that|repositories that|aims? to|intends? to|"
    r"they (?:propose|present|introduce|develop|design|create|construct))\b", re.I,
)
_ADDITION = re.compile(
    r"\b(?:extend(?:s|ed|ing)?|add(?:s|ed|ing)?|improv(?:e|es|ed|ing)|"
    r"introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|develop(?:s|ed|ing)|"
    r"collect(?:s|ed|ing)?|annotat(?:e|es|ed|ing)|design(?:s|ed|ing)|"
    r"additional|extension|extensions|ablation|ablations)\b", re.I,
)
_NEW_MODEL_CLAIM = re.compile(
    r"\b(?:new|novel)\b.{0,40}\b(?:model|method|architecture|algorithm|network)\b", re.I,
)
_API_WRAPPER = re.compile(
    r"\b(?:reference implementation|starter|provider[ -]agnostic|cloud api|"
    r"token[ -]auth|api agent|api wrapper|powered by (?:an? )?api)\b", re.I,
)
_TUTORIAL_CONTEXT = re.compile(r"\b(?:tutorial|walkthrough)\b", re.I)
_COURSE_CONTEXT = re.compile(r"\b(?:course|coursework|homework|assignment)\b", re.I)
_AUTHORED_IMPLEMENTATION = re.compile(
    r"\b(?:we|our project|our repository|this repository)\b.{0,80}"
    r"\b(?:implement(?:ed|s|ing)|train(?:ed|s|ing)|fine[ -]?tun(?:ed|es|ing)|"
    r"collect(?:ed|s|ing)|annotat(?:ed|es|ing)|curat(?:ed|es|ing)|"
    r"introduc(?:e|es|ed|ing)|propos(?:e|es|ed|ing)|develop(?:s|ed|ing)|"
    r"design(?:s|ed|ing)|build(?:s|ing))\b", re.I,
)
_AUTHORED_TUTORIAL_EXTENSION = re.compile(
    r"\b(?:we|our project|our repository|this repository)\b.{0,80}"
    r"\b(?:implement(?:ed|s|ing)|train(?:ed|s|ing)|fine[ -]?tun(?:ed|es|ing)|"
    r"collect(?:ed|s|ing)|annotat(?:ed|es|ing)|curat(?:ed|es|ing))\b", re.I,
)
_THIRD_PARTY_PROFILE = re.compile(
    r"\b(?:independent third[ -]party profile|profile of a public api|"
    r"third[ -]party profile of|unofficial profile of)\b", re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|[\n\r]+")


def assess_probable_content(text: str) -> list[str]:
    """Return sorted fixed-enum contribution cues found in repository text.

    Sentences are assessed independently so generic ML mentions, course labels,
    and unchanged reproduction claims cannot lend support to unrelated text.
    A separate sentence describing a substantive extension may still qualify.
    """
    if not isinstance(text, str):
        raise TypeError("repository text must be a string")
    if _THIRD_PARTY_PROFILE.search(text[:200_000]):
        return []

    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text[:200_000]):
        if _THIRD_PARTY_PROFILE.search(sentence):
            continue
        if not _ML.search(sentence):
            continue
        if _API_WRAPPER.search(sentence) and not _SUBSTANTIVE_EVIDENCE.search(sentence):
            continue
        negative_only = (
            bool(_NONASSERTION.search(sentence))
            or (bool(_NEGATIVE_ONLY.search(sentence)) and not _ADDITION.search(sentence))
            or (bool(_TUTORIAL_CONTEXT.search(sentence)) and not _AUTHORED_TUTORIAL_EXTENSION.search(sentence))
            or (bool(_COURSE_CONTEXT.search(sentence)) and not _AUTHORED_IMPLEMENTATION.search(sentence))
        )
        if negative_only:
            continue

        adaptation = _ADAPT.search(sentence)
        model_specific_adaptation = adaptation and (
            not re.search(r"\b(?:prun|quantiz|distill)", adaptation.group(0), re.I)
            or bool(_ML_MODEL_TARGET.search(sentence))
        )
        if model_specific_adaptation and (_MODEL_NOUN.search(sentence) or _ML.search(sentence)):
            found.add("adaptation-or-fine-tuning")
        if _CREATE.search(sentence) and _MODEL_NOUN.search(sentence):
            found.add("original-implementation")
        if _NEW_MODEL_CLAIM.search(sentence):
            found.add("original-implementation")
        if re.search(r"\bextensions?\b", sentence, re.I) and _MODEL_NOUN.search(sentence):
            found.add("original-implementation")
        if _APPLICATION.search(sentence) and _SUBSTANTIVE_EVIDENCE.search(sentence) and re.search(
            r"\b(?:for|on|using|with|from|against|across|to|of|under)\b", sentence, re.I
        ):
            found.add("substantive-application-or-experiments")
        if _DATASET_CREATE.search(sentence) and _DATA_PROTOCOL.search(sentence):
            found.add("original-dataset-or-benchmark")
        if _TOOLING.search(sentence) and _TOOL_PURPOSE.search(sentence) and _CREATE.search(sentence):
            found.add("original-tooling")

    return sorted(found)
