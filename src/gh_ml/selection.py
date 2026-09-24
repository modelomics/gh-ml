"""Conservative, deterministic selection of repositories with ML contributions.

This uses only repository-owned name, description, and topics. It is a
heuristic triage rule, not a claim that a contribution is genuinely novel.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SELECTION_VERSION = "ml-contribution-v2"

_METHOD_PATTERNS = (
    r"\btransformers?\b", r"\bdiffusion(?: models?)?\b", r"\bflow matching\b",
    r"\bmixture of experts\b", r"\bstate.space models?\b", r"\bmamba\b",
    r"\bretrieval.augmented generation\b", r"\brag\b",
    r"\bcontrastive learning\b", r"\bself.supervised learning\b",
    r"\bgraph neural networks?\b", r"\bneural operators?\b",
    r"\breinforcement learning\b", r"\bfederated learning\b",
    r"\bdistillation\b", r"\bpruning\b", r"\btest.time adaptation\b",
    r"\bmeta.learning\b",
    r"\bgradient boosting\b", r"\bpolicy gradients?\b", r"\bworld models?\b",
    r"\b(?:xgboost|catboost|lightgbm)\b", r"\b(?:bert|gpt|llama|stable diffusion)\b",
    r"\b(?:svm|support vector machines?|random forests?|k.means)\b",
    r"\blime\b",
    r"\bprotein(?: language| folding)? models?\b", r"\bmolecular (?:graph )?(?:neural )?networks?\b",
    r"\bmolecular (?:property prediction|docking|dynamics)\b", r"\bdrug discovery models?\b",
    r"\brobot(?:ic)? (?:polic(?:y|ies)|control|manipulation)\b", r"\bvision language action\b",
    r"\b(?:visual )?autoregressive (?:modeling|models?|generation)\b",
    r"\btime.series (?:imputation|forecasting|prediction|models?)\b",
    r"\b(?:long.term|multivariate) time.series (?:imputation|forecasting|prediction)\b",
    r"\b(?:imputation|forecasting|prediction) for (?:multivariate )?time.series\b",
    r"\bvideo (?:creation|editing|generation|synthesis|generative modeling)\b",
    r"\bmultimodal (?:large language )?models?\b", r"\bany.to.any multimodal\b",
    r"\bagentic reinforcement learning\b", r"\bpolicy optimization\b",
)
_AMBIGUOUS_METHOD = re.compile(
    r"\b(?:lora|quantization|causal inference|(?:bayesian )?optimization)\b", re.I,
)
_DOMAIN_METHOD = re.compile(
    r"\b(?:protein folding|protein design|molecular modeling|molecular docking|"
    r"computational chemistry|robotics?)\b", re.I,
)
_ML_CONTEXT = re.compile(
    r"\b(?:machine learning|deep learning|artificial intelligence|\bml\b|\bai\b|"
    r"neural|model training|model inference|\bllms?\b)\b", re.I,
)
_METHOD = re.compile("|".join(f"(?:{p})" for p in _METHOD_PATTERNS), re.I)
_NOVEL_METHOD_PHRASE = re.compile(
    r"\b(?:new|novel)(?:[\s,/-]+[a-z0-9]+){0,3}[\s,/-]+"
    r"(?:architectures?|methods?|models?|techniques?|algorithms?|polic(?:y|ies)|"
    r"networks?|operators?|optimizers?)\b", re.I,
)
_CONTRIBUTION = re.compile(
    r"\b(?:implement(?:s|ed|ing|ation)?|train(?:s|ed|ing)?|fine.?tun(?:e|es|ed|ing)|"
    r"evaluat(?:e|es|ed|ing|ion)|reproduc(?:e|es|ed|ing|tion))\b", re.I,
)
_PAPER = re.compile(r"\b(?:paper|arxiv|doi)\b|10\.\d{4,9}/", re.I)
_CODE = re.compile(r"\b(?:official )?(?:source )?code\b|\bcode for\b|\bimpl(?:ementation)?s?\b|\bgithub repo(?:sitory)?\b", re.I)
_OFFICIAL_PAPER = re.compile(
    r"\bour\s+(?:[a-z0-9]+\s+){0,5}(?:paper|work|method|approach)\b|"
    r"\bofficial\s+paper\b|\bpaper\s+(?:introduces?|proposes?|presents?)\b|"
    r"\bofficial(?:\s+[a-z]+){0,3}\s+(?:implementation|impl|code)(?:s)?\b|"
    r"\bcode(?:\s+and\s+models?)?\s+for\s+(?:(?:icml|neurips|iclr|cvpr|iccv|aaai|acl)\s+20\d{2}\s+)?paper\b",
    re.I,
)
_VENUE_YEAR = re.compile(r"\b(?:icml|neurips|nips|iclr|cvpr|iccv|eccv|acl|emnlp|naacl|aaai|ijcai|kdd|www|sigir|interspeech|icassp|miccai|eccv|acm mm|ieee t[op]ami)\s*['’]?(?:19|20)?\d{2}\b", re.I)
_EXPLICIT_EXCLUDE = re.compile(
    r"\b(?:homework\d*|assignments?|lab assignments?|textbook|book|awesome list|curated list|portfolio)\b|"
    r"\b(?:cs|eecs)\s?\d{3,4}[a-z]?\b|\bntu[- ]?ee\d{4}\b|\bllm ?book\b", re.I,
)
_COURSE_CUE = re.compile(
    r"\b(?:course|coursework|course work|course projects?|course materials?|class projects?|class materials?|education(?:al)?|"
    r"coursera|specialization|training course|workshop|bootcamp|student project|lab assignments?|"
    r"\d+[- ]?day)\b", re.I,
)
_UTILITY_CUE = re.compile(
    r"\b(?:codex skill|claude skills?|skills? trees?|agent skills?|agentic skills?|"
    r"skills for|workflows?|index(?:es|ing)?|"
    r"catalog(?:ue)?|directory of|list of projects|lectures?|slides?|presentations?)\b", re.I,
)
_SURVEY_CUE = re.compile(
    r"\b(?:surveys?|collecting (?:awesome )?papers|"
    r"literature review|reading list|bibliography|awesome papers)\b", re.I,
)
_NON_ML_SIGNAL = re.compile(r"\bdigital filter design\b", re.I)
_TUTORIAL = re.compile(r"\btutorial\b", re.I)
_REVIEW_ONLY_CUE = re.compile(
    r"\b(?:overviews?|reflections?|replications?|"
    r"reproduc(?:e|es|ed|ing|tion)(?:s)?|reimplement(?:s|ed|ing|ation)(?:s)?|"
    r"codes and reflections)\b", re.I,
)
_DATASET_CUE = re.compile(r"\bdatasets?\b", re.I)
_BACKTESTING = re.compile(r"\b(?:backtest(?:ing)?|algorithmic trading|trading library)\b", re.I)
_GENERIC_AI = re.compile(r"\b(?:ai|artificial intelligence|machine learning|deep learning|ml)\b", re.I)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in value if isinstance(item, str)]
    return []


def _has_method_specific_novelty(sentence: str) -> bool:
    """Require new/novel to modify a method noun near an ML method cue."""
    claim = _NOVEL_METHOD_PHRASE.search(sentence)
    if claim is None:
        return False
    method_matches = list(_METHOD.finditer(sentence))
    if _ML_CONTEXT.search(sentence):
        method_matches.extend(_AMBIGUOUS_METHOD.finditer(sentence))
        method_matches.extend(_DOMAIN_METHOD.finditer(sentence))
    return any(
        method.start() <= claim.end() + 48 and method.end() >= claim.start() - 48
        for method in method_matches
    )


def _normalized_text(row: Mapping[str, Any]) -> tuple[str, str]:
    # API rows often have `name` as the bare repository and `full_name` as
    # owner/repository. Prefer the latter for profile-repository detection.
    name = " ".join(_strings(row.get("full_name")) or _strings(row.get("name")))
    description = " ".join(_strings(row.get("description")))
    topics = " ".join(_strings(row.get("topics")))
    owned_text = " ".join((name, description, topics)).casefold().replace("_", " ").replace("-", " ")
    return name, owned_text


def assess_repository(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable include/review/exclude assessment for a repository row."""
    name, text = _normalized_text(row)
    signals: set[str] = set()

    def result(status: str, reason: str) -> dict[str, Any]:
        return {
            "selection_version": SELECTION_VERSION,
            "selection_status": status,
            "selection_reason": reason,
            "selection_signals": sorted(signals),
        }

    if row.get("fork") is True:
        signals.add("fork")
        return result("exclude", "fork")

    parts = name.strip().split("/")
    profile_repository = (len(parts) >= 2 and parts[-1].casefold() in {parts[-2].casefold(), ".github"}) or (len(parts) == 1 and parts[0].casefold() == ".github")

    fields = [
        " ".join(_strings(row.get(key)))
        for key in ("name", "full_name", "description", "topics")
        if _strings(row.get(key))
    ]
    novelty_fields = [
        " ".join(_strings(row.get(key)))
        for key in ("name", "full_name", "description")
        if _strings(row.get(key))
    ]
    sentences = [
        sentence
        for field in fields
        for sentence in re.split(r"[.!?;\n]+", field.casefold().replace("-", " ").replace("_", " "))
    ]
    method = any(
        _METHOD.search(sentence)
        or (_AMBIGUOUS_METHOD.search(sentence) and _ML_CONTEXT.search(sentence))
        or (_DOMAIN_METHOD.search(sentence) and _ML_CONTEXT.search(sentence))
        for sentence in sentences
    )
    ml_context = bool(_ML_CONTEXT.search(text))
    substantive = bool(_CONTRIBUTION.search(text))
    # Require the novelty claim and method cue in the same field and sentence;
    # a method topic cannot combine with an unrelated description such as CLI.
    method_novelty = any(
        _has_method_specific_novelty(sentence)
        for field in novelty_fields
        for sentence in re.split(r"[.!?;\n]+", field.casefold().replace("-", " ").replace("_", " "))
    )
    description = " ".join(_strings(row.get("description"))).strip()
    description_sentences = re.split(
        r"(?<!\d)\.(?!\d)|[!?;\n]+",
        re.sub(r"\b(impl|e\.g|i\.e)\.", r"\1", description.casefold().replace("-", " ").replace("_", " ")),
    )
    official_paper_description = any(
        (claim := _OFFICIAL_PAPER.search(sentence))
        and (paper := (_PAPER.search(sentence) or _VENUE_YEAR.search(sentence)))
        and (code := _CODE.search(sentence))
        and max(claim.start(), paper.start(), code.start()) - min(claim.start(), paper.start(), code.start()) <= 180
        and (_METHOD.search(sentence) or (_AMBIGUOUS_METHOD.search(sentence) and _ML_CONTEXT.search(sentence)) or (_DOMAIN_METHOD.search(sentence) and _ML_CONTEXT.search(sentence)) or (_ML_CONTEXT.search(sentence) and re.search(r"\b(?:model|modeling|method|approach|optimization|generation|imputation|forecasting)\b", sentence, re.I)))
        for sentence in description_sentences
    )
    research_name_topics = " ".join(_strings(row.get("name")) + _strings(row.get("full_name")) + _strings(row.get("topics")))
    official_paper_code = official_paper_description or (any(
        (claim := _OFFICIAL_PAPER.search(sentence))
        and (paper := (_PAPER.search(sentence) or _VENUE_YEAR.search(sentence)))
        and (code := _CODE.search(sentence))
        and max(claim.start(), paper.start(), code.start()) - min(claim.start(), paper.start(), code.start()) <= 180
        for sentence in description_sentences
    ) and bool(_METHOD.search(research_name_topics)))
    paper_code = bool(_PAPER.search(text) and _CODE.search(text))
    tutorial = bool(
        _TUTORIAL.search(" ".join(_strings(row.get("name")) + _strings(row.get("full_name")) + _strings(row.get("topics"))))
        or _TUTORIAL.search(description)
    )
    course_cue = bool(_COURSE_CUE.search(text))
    utility_cue = bool(_UTILITY_CUE.search(text))
    survey_cue = bool(_SURVEY_CUE.search(text))
    if _EXPLICIT_EXCLUDE.search(text):
        signals.add("explicit-noncontribution")
        return result("exclude", "explicit-noncontribution")
    if survey_cue:
        signals.add("survey-or-taxonomy-cue")
        return result("exclude", "survey-or-paper-list-repository")
    if tutorial:
        signals.add("tutorial-cue")
        return result("exclude", "tutorial-repository")
    # Dataset topics often describe the application domain of an original model.
    # Only repository name/description can indicate that this repository is a dataset.
    review_only_cue = bool(_REVIEW_ONLY_CUE.search(text) or _DATASET_CUE.search(" ".join(_strings(row.get("name")) + _strings(row.get("full_name")) + _strings(row.get("description")))))
    if method:
        signals.add("ml-method-cue")
    elif ml_context:
        signals.add("ml-context-only")
    if substantive:
        signals.add("contribution-language")
    if method_novelty:
        signals.add("method-tied-novelty-claim")
    if paper_code:
        signals.add("paper-and-code-cue")
    if official_paper_code:
        signals.add("official-paper-implementation-cue")
    if tutorial:
        signals.add("tutorial-cue")
    if review_only_cue:
        signals.add("overview-reproduction-or-dataset-cue")
    if course_cue:
        signals.add("course-cue")
    if utility_cue:
        signals.add("workflow-or-index-cue")
    if _BACKTESTING.search(text):
        signals.add("backtesting-utility-cue")
    if _NON_ML_SIGNAL.search(text):
        signals.add("non-ml-utility-cue")
        return result("exclude", "non-ml-utility")

    if profile_repository and not official_paper_code:
        signals.add("owner-profile-repository")
        return result("exclude", "owner-profile-repository")

    strong_contribution = method_novelty or official_paper_code
    if _BACKTESTING.search(text):
        return result("review", "non-ml-utility-with-ml-contribution-cue") if strong_contribution else result("exclude", "non-ml-utility")
    if (course_cue or utility_cue) and not method_novelty:
        return result("exclude", "course-or-utility-repository")
    if (course_cue or utility_cue) and method_novelty:
        return result("review", "course-or-utility-with-novel-method-cue")
    if review_only_cue:
        return result("review", "overview-reproduction-or-dataset")
    if strong_contribution:
        return result("include", "official-paper-method-implementation" if official_paper_code else "specific-method-with-novelty-claim")
    if method or ml_context or _GENERIC_AI.search(text):
        return result("review", "ml-relevance-without-clear-contribution")
    return result("review", "insufficient-repository-evidence")
