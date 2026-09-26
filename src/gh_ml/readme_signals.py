"""Bounded, deterministic signals extracted from a repository README.

These signals describe README text only. They do not verify a paper's claims,
implementation quality, or novelty.
"""

from __future__ import annotations

import re
import textwrap


README_EVIDENCE_VERSION = "gh-ml-readme-evidence-v2"
SUPPORTED_README_EVIDENCE_VERSIONS = frozenset({"gh-ml-readme-evidence-v1", README_EVIDENCE_VERSION})
_MAX_INPUT_CHARS = 200_000

# All labels in the returned schema are fixed enums. Keep patterns broad enough
# to cover methods, rather than repositories or project names.
_ML_CONTEXT = re.compile(
    r"\b(?:machine[ -]learning|deep[ -]learning|neural[ -](?:network|nets?)|"
    r"transformers?|diffusion(?:[ -]models?)?|gans?|generative adversarial networks?|"
    r"autoregressive(?:[ -](?:language )?models?)?|large[ -]language[ -]models?|llms?|"
    r"reinforcement learning|\brl\b|vision[ -]language models?|\bvlms?\b|"
    r"self[ -]supervised|representation learning|graph neural networks?|"
    r"convolutional neural networks?|recurrent neural networks?|embeddings?|"
    r"classifiers?|language models?|text[ -]to[ -]image|"
    r"joint[ -]embedding predictive architecture|\b(?:bert|gpt|vit)\b)\b",
    re.I,
)
_PYTORCH_METHOD_CONTEXT = re.compile(
    r"\bpytorch\b.*\b(?:architecture|architectures|method|methods|network|networks|"
    r"transformer|diffusion|encoder|decoder|joint[ -]embedding|self[ -]supervised)\b|"
    r"\b(?:architecture|architectures|method|methods|network|networks|transformer|"
    r"diffusion|encoder|decoder|joint[ -]embedding|self[ -]supervised)\b.*\bpytorch\b",
    re.I,
)
_PAPER = re.compile(
    r"\b(?:paper|preprint|publication|published in|proceedings|arxiv|doi:)\b|"
    r"arxiv\.org/(?:abs|pdf)/|openreview\.net/|doi\.org/|\b\d{4}\.\d{4,5}(?:v\d+)?\b",
    re.I,
)
_CODE = re.compile(
    r"\b(?:code|codebase|implementation|source code|repository|repo|software|official implementation)\b|"
    r"github\.com/",
    re.I,
)
_OFFICIAL = re.compile(
    r"\b(?:official|authors?'|author[- ]maintained|maintained by the authors|"
    r"reference implementation)\b",
    re.I,
)
_CONTRIBUTION = re.compile(
    r"\b(?:we (?:propose|present|introduce|develop|design|create|construct)|"
    r"(?:we )?(?:introduced|presented|proposed|designed|developed)|"
    r"our (?:method|approach|model|framework|architecture|network|module)|"
    r"(?:proposed|novel|new) (?:method|approach|model|framework|architecture|network|module)|"
    r"introducing a (?:method|model|framework|module)|"
    r"module\b.{0,80}\bthat (?:turns|transforms|converts|enables|allows)\b)",
    re.I,
)
_COURSE = re.compile(
    r"(?<!of )\bcourses?\b|\bcoursework\b|\b(?:class|course) projects?\b|"
    r"\b(?:homework|assignments?|"
    r"curricul(?:um|a)|bootcamps?)\b|"
    r"\b(?:acknowledg(?:e?ments?|ed)|thanks?)\b.{0,120}\bengineering studies\b|"
    r"\bengineering studies\b.{0,120}\b(?:acknowledg(?:e?ments?|ed)|thanks?)\b|"
    r"\bconducted as part of engineering studies\b",
    re.I,
)
_REPRODUCTION = re.compile(
    r"\b(?:faithful )?(?:re-?implementation|reproduction|replication)\b|"
    r"\b(?:reproduce|reproducing|replicate|replicating) (?:the |this )?(?:paper|work|results)\b",
    re.I,
)
_SURVEY = re.compile(r"\b(?:survey|literature review|reading list|bibliography|awesome list)\b", re.I)
_DATASET = re.compile(r"\b(?:dataset|data set|benchmark|corpus|data collection)\b", re.I)
_ARTIFACT = re.compile(
    r"\b(?:pretrained|pre-trained|trained) (?:model|weights?)\b|"
    r"\b(?:model )?(?:weights?|checkpoints?|training code|training scripts?)\b|"
    r"\btrain(?:ing)? (?:the |a |our )?model\b",
    re.I,
)

_SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("overview", re.compile(r"\b(?:overview|introduction|about|description)\b", re.I)),
    ("method", re.compile(r"\b(?:method|approach|model|methodology|architecture)\b", re.I)),
    ("results", re.compile(r"\b(?:result|experiment|evaluation|performance)\b", re.I)),
    ("installation", re.compile(r"\b(?:install|installation|setup|requirements)\b", re.I)),
    ("usage", re.compile(r"\b(?:usage|quickstart|getting started|example|demo)\b", re.I)),
    ("citation", re.compile(r"\b(?:citation|cite|bibtex)\b", re.I)),
    ("references", re.compile(r"\b(?:references|related work|further reading)\b", re.I)),
    ("course", re.compile(r"\b(?:course|lectures?|assignments?|homework)\b", re.I)),
    ("dataset", re.compile(r"\b(?:dataset|data set|benchmark|data collection)\b", re.I)),
)


def _clean_markdown(text: str) -> str:
    """Remove high-noise markup while preserving useful link labels/targets."""
    text = textwrap.dedent(text[:_MAX_INPUT_CHARS])
    text = re.sub(r"(?s)```.*?```|~~~.*?~~~", "\n", text)
    # README introductions often use HTML for typography. Preserve link targets
    # and turn block boundaries into paragraph boundaries before stripping tags.
    text = re.sub(
        r"(?is)<a\b[^>]*href\s*=\s*(['\"])(.*?)\1[^>]*>(.*?)</a>",
        r"\3 \2",
        text,
    )
    text = re.sub(r"(?i)</?(?:p|div|h[1-6]|br|li|tr)\b[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"(?m)^\s*<[^>]+>\s*$", "\n", text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 \2", text)
    text = re.sub(r"<((?:https?://)[^>]+)>", r"\1", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    return text


def _section_enum(heading: str) -> str:
    for label, pattern in _SECTION_RULES:
        if pattern.search(heading):
            return label
    return "other"


def _readme_blocks(text: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Return section/paragraph pairs and only fixed, recognized section labels."""
    current_section = "other"
    sections: set[str] = set()
    blocks: list[tuple[str, str]] = []
    paragraph_lines: list[str] = []

    def flush() -> None:
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
        if paragraph:
            blocks.append((current_section, paragraph))
        paragraph_lines.clear()

    for line in text.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            current_section = _section_enum(heading.group(1))
            sections.add(current_section)
        elif not line.strip():
            flush()
        else:
            paragraph_lines.append(line)
    flush()
    return blocks, sections


def extract_readme_evidence(text: str) -> dict[str, object]:
    """Extract versioned, enum-only evidence while preserving local context.

    Positive relations require their cues to share a paragraph. Generic ML
    mentions elsewhere in a README therefore cannot turn an unrelated paper
    mention into a code-for-paper or method-contribution signal.
    """
    if not isinstance(text, str):
        raise TypeError("README text must be a string")

    cleaned = _clean_markdown(text)
    blocks, sections = _readme_blocks(cleaned)
    signals: set[str] = set()
    dataset_only = bool(blocks) and all(_DATASET.search(p) for _, p in blocks)

    for _, paragraph in blocks:
        has_paper = bool(_PAPER.search(paragraph))
        has_code = bool(_CODE.search(paragraph))
        has_ml = bool(_ML_CONTEXT.search(paragraph) or _PYTORCH_METHOD_CONTEXT.search(paragraph))
        has_reproduction = bool(_REPRODUCTION.search(paragraph))
        has_course = bool(_COURSE.search(paragraph))
        has_survey = bool(_SURVEY.search(paragraph))

        if has_paper:
            signals.add("paper-reference")
        if has_ml:
            signals.add("ml-method-context")
        if has_course:
            signals.add("course-cue")
        if has_reproduction:
            signals.add("reproduction-cue")
        if has_survey:
            signals.add("survey-cue")
        if _ARTIFACT.search(paragraph) and has_ml:
            signals.add("model-training-artifact")

        if has_ml and not (has_course or has_reproduction or has_survey):
            if has_paper and has_code:
                signals.add("paper-code-relationship")
            if _CONTRIBUTION.search(paragraph):
                signals.add("method-contribution")
            if has_code and _OFFICIAL.search(paragraph):
                signals.add("official-implementation-claim")

    if dataset_only:
        signals.add("dataset-only-cue")

    return {
        "readme_evidence_version": README_EVIDENCE_VERSION,
        "readme_signals": sorted(signals),
        "readme_sections": sorted(sections),
    }
