"""Experimental, evidence-scoped retrieval and novelty-review prioritization.

This module embeds repository-owned text with the project's pinned MiniLM
encoder, retrieves likely related records through FAISS, and applies an
explicit uncalibrated review rule. It never proves scientific novelty. A
``probable_original_content`` result is a human-review hypothesis relative to
the declared, incomplete corpus; missing evidence or scope yields ``uncertain``.

NumPy, FAISS, and sentence-transformers are supplied by the optional
``semantic`` extra plus the local FAISS runtime. The index uses exact search for
small corpora and HNSW for larger ones. Both use bounded top-k retrieval, not an
all-pairs similarity matrix.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


ASSESSMENT_VERSION = "gh-ml-novelty-review-v1"
INDEX_VERSION = "gh-ml-faiss-retrieval-v1"
ENCODER_VERSION = "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SMALL_CORPUS_EXACT_LIMIT = 5_000
HNSW_LINKS = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 96
MAX_DOCUMENT_CHARS = 1_800
MAX_SEQUENCE_TOKENS = 256
DOCUMENT_SELECTION_VERSION = "contribution-and-method-passages-v1"
_CONTRIBUTION_SIGNALS = frozenset(
    {
        "original-implementation",
        "adaptation-or-fine-tuning",
        "substantive-application-or-experiments",
        "original-dataset-or-benchmark",
        "original-tooling",
    }
)


def _dependencies():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("novelty retrieval requires NumPy and faiss-cpu in the semantic environment") from exc
    return faiss, np


def _readme_passages(text: str, *, char_limit: int = 1_100) -> str:
    """Prefer contribution and method text over install, badge, and code sections."""
    import re

    from .probable_content import assess_probable_content

    section = "overview"
    blocks: list[tuple[int, int, str]] = []
    buffer: list[str] = []
    order = 0
    in_fence = False

    def flush() -> None:
        nonlocal order
        content = " ".join(line.strip() for line in buffer if line.strip()).strip()
        buffer.clear()
        if not content:
            return
        if section in {"installation", "usage", "requirements", "citation", "references", "acknowledgments", "contributing", "license", "table of contents"}:
            return
        cues = assess_probable_content(content)
        score = (8 if cues else 0) + (3 if section in {"abstract", "overview", "introduction", "method", "approach", "model", "architecture", "results", "experiments", "contribution"} else 0)
        if len(content) >= 60:
            score += 1
        if not re.search(r"[.!?]", content):
            score -= 1
        blocks.append((score, order, content))
        order += 1

    for line in text[:80_000].splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            section = heading.group(1).casefold().strip()
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    if not blocks:
        return ""
    # Highest evidence first; preserve source order among equally scored blocks.
    blocks.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    length = 0
    for _, _, content in blocks:
        remaining = char_limit - length
        if remaining <= 0:
            break
        clipped = content[:remaining].strip()
        if clipped:
            selected.append(clipped)
            length += len(clipped) + 1
    return "\n".join(selected)


def document_text(row: Mapping[str, Any]) -> str:
    """Create a short embedding view; exclude discovery, labels, and popularity."""
    header: list[str] = []
    for key, limit in (("name", 220), ("description", 500), ("topics", 220), ("paper_title", 200), ("model_card_excerpt", 280)):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            header.append(value.strip()[:limit])
        elif key == "topics" and isinstance(value, (list, tuple)):
            header.extend(item.strip()[:80] for item in value if isinstance(item, str) and item.strip())
    abstract = row.get("paper_abstract")
    if isinstance(abstract, str) and abstract.strip():
        header.append(abstract.strip()[:350])
    readme = row.get("readme_excerpt")
    selected = _readme_passages(readme) if isinstance(readme, str) and readme.strip() else ""
    parts = ["\n".join(header)] if header else []
    if selected:
        parts.append(selected)
    return "\n\n".join(parts)[:MAX_DOCUMENT_CHARS]


def encode_documents(
    rows: Sequence[Mapping[str, Any]],
    encoder: Any,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    return_provenance: bool = False,
):
    """Embed records in bounded batches using a sentence-transformers encoder."""
    _, np = _dependencies()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    texts = [document_text(row) for row in rows]
    if any(not text for text in texts):
        raise ValueError("every indexed source must have repository-owned text")
    tokenizer = getattr(encoder, "tokenizer", None)
    max_tokens = int(getattr(encoder, "max_seq_length", MAX_SEQUENCE_TOKENS))
    provenance: list[dict[str, Any]] = []
    for text in texts:
        token_count = None
        if tokenizer is not None:
            encoded_text = tokenizer(text, add_special_tokens=True, truncation=False)
            token_ids = encoded_text.get("input_ids") if isinstance(encoded_text, Mapping) else None
            if isinstance(token_ids, list):
                token_count = len(token_ids)
        provenance.append({
            "embedding_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding_text_chars": len(text),
            "token_count_before_truncation": token_count,
            "token_count_used": min(token_count, max_tokens) if token_count is not None else None,
            "tokens_truncated": max(0, token_count - max_tokens) if token_count is not None else None,
            "truncated": token_count > max_tokens if token_count is not None else None,
        })
    chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = encoder.encode(
            texts[start : start + batch_size],
            batch_size=min(batch_size, len(texts) - start),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        chunks.append(np.asarray(encoded, dtype=np.float32))
    vectors = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("encoder returned malformed or non-finite embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise ValueError("encoder returned a zero-length embedding")
    vectors = np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
    return (vectors, provenance) if return_provenance else vectors


def load_pinned_encoder(*, device: str = "cpu") -> Any:
    """Load and verify the cached project encoder without network access."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install the project's semantic extra to load the novelty encoder") from exc
    from .semantic_triage import ENCODER_FILES_SHA256, ENCODER_SNAPSHOT_PATH, _encoder_file_hashes

    path = Path(ENCODER_SNAPSHOT_PATH)
    if not path.is_dir():
        raise FileNotFoundError(f"pinned local encoder snapshot is unavailable: {path}")
    if _encoder_file_hashes(path) != ENCODER_FILES_SHA256:
        raise ValueError("cached encoder files do not match the pinned manifest")
    model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 256
    return model


def _unit_matrix(vectors: Any):
    _, np = _dependencies()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("vectors must be a non-empty finite two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise ValueError("vectors must be nonzero")
    return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)


@dataclass(frozen=True)
class Neighbor:
    source_id: str
    full_name: str | None
    similarity: float | None
    source_date: str | None
    date_kind: str | None
    source_locator: str | None
    source_version: str | None
    family_id: str | None
    readme_blob_sha: str | None
    readme_excerpt: str | None
    local_evidence_locator: str | None
    retrieval_channel: str
    source_excerpt: str
    evidence: tuple[str, ...]


class NoveltyIndex:
    """FAISS cosine index with stable source IDs and reviewable source metadata."""

    def __init__(
        self,
        vectors: Any,
        records: Sequence[Mapping[str, Any]],
        *,
        corpus_scope: str,
        built_at: str | None = None,
        hnsw_ef_search: int = HNSW_EF_SEARCH,
        retrieval_channel: str = "metadata",
    ) -> None:
        faiss, _ = _dependencies()
        # Keep index builds from competing with the active shared data workers.
        faiss.omp_set_num_threads(2)
        if not isinstance(corpus_scope, str) or not corpus_scope.strip():
            raise ValueError("corpus_scope must identify the searched corpus")
        self.vectors = _unit_matrix(vectors)
        self.dimension = int(self.vectors.shape[1])
        self.records = tuple(dict(row) for row in records)
        if len(self.vectors) != len(self.records):
            raise ValueError("one source record is required per embedding")
        if hnsw_ef_search < 1:
            raise ValueError("hnsw_ef_search must be positive")
        ids: set[str] = set()
        for row in self.records:
            source_id = row.get("source_id", row.get("github_id"))
            if source_id is None or not str(source_id).strip():
                raise ValueError("every source record needs source_id or github_id")
            if str(source_id) in ids:
                raise ValueError(f"duplicate source ID in index: {source_id}")
            ids.add(str(source_id))
            if not document_text(row):
                raise ValueError(f"source {source_id} has no repository-owned text")
        self.corpus_scope = corpus_scope.strip()
        self.retrieval_channel = retrieval_channel
        self.built_at = built_at or datetime.now(timezone.utc).isoformat()
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.source_digest = self._digest_records(self.records)
        dimension = int(self.vectors.shape[1])
        if len(self.records) <= SMALL_CORPUS_EXACT_LIMIT:
            self.backend = "faiss-flat-ip"
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.backend = "faiss-hnsw-flat-ip"
            self.index = faiss.IndexHNSWFlat(dimension, HNSW_LINKS, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch = self.hnsw_ef_search
        self.index.add(self.vectors)

    @staticmethod
    def _digest_records(records: Sequence[Mapping[str, Any]]) -> str:
        clean = []
        for row in records:
            clean.append({
                "source_id": str(row.get("source_id", row.get("github_id"))),
                "source_version": row.get("source_version"),
                "source_date": row.get("source_date"),
                "date_kind": row.get("date_kind"),
                "source_locator": row.get("source_locator"),
                "family_id": row.get("family_id"),
                "text_sha256": hashlib.sha256(document_text(row).encode("utf-8")).hexdigest(),
            })
        raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "index_version": INDEX_VERSION,
            "embedding_model": ENCODER_VERSION,
            "embedding_dimension": self.dimension,
            "embedding_max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "document_selection_version": DOCUMENT_SELECTION_VERSION,
            "backend": self.backend,
            "source_count": len(self.records),
            "source_digest_sha256": self.source_digest,
            "corpus_scope": self.corpus_scope,
            "retrieval_channel": self.retrieval_channel,
            "built_at": self.built_at,
            "hnsw_links": HNSW_LINKS if self.backend.startswith("faiss-hnsw") else None,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION if self.backend.startswith("faiss-hnsw") else None,
            "hnsw_ef_search": self.hnsw_ef_search if self.backend.startswith("faiss-hnsw") else None,
        }

    def query(
        self,
        vector: Any,
        *,
        k: int = 50,
        exclude_source_id: str | None = None,
        exclude_family_id: str | None = None,
        prior_to: str | None = None,
        search_budget: int | None = None,
    ) -> list[Neighbor]:
        _, np = _dependencies()
        if k < 1:
            raise ValueError("k must be positive")
        query = _unit_matrix(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self.dimension:
            raise ValueError("query embedding dimension does not match index")
        budget = min(len(self.records), search_budget or max(k * 8, 128))
        scores, ids = self.index.search(query, budget)
        matches: list[Neighbor] = []
        for score, raw_id in zip(scores[0], ids[0], strict=True):
            if raw_id < 0:
                continue
            row = self.records[int(raw_id)]
            source_id = str(row.get("source_id", row.get("github_id")))
            family_id = row.get("family_id")
            if exclude_source_id is not None and source_id == str(exclude_source_id):
                continue
            if exclude_family_id and family_id is not None and str(family_id) == str(exclude_family_id):
                continue
            source_date = row.get("source_date")
            # Only publication, release, or commit dates support a prior-work
            # ordering claim. Repo-created/observed times are context only.
            if prior_to and (
                row.get("date_kind") not in {"publication", "release", "commit"}
                or not isinstance(source_date, str)
                or source_date >= prior_to
            ):
                continue
            raw_evidence = row.get("evidence", ())
            evidence = tuple(sorted(str(item) for item in raw_evidence if isinstance(item, str))) if isinstance(raw_evidence, (list, tuple, set, frozenset)) else ()
            matches.append(
                Neighbor(
                    source_id=source_id,
                    full_name=row.get("full_name") if isinstance(row.get("full_name"), str) else None,
                    similarity=float(score),
                    source_date=source_date if isinstance(source_date, str) else None,
                    date_kind=row.get("date_kind") if isinstance(row.get("date_kind"), str) else None,
                    source_locator=row.get("source_locator") if isinstance(row.get("source_locator"), str) else None,
                    source_version=row.get("source_version") if isinstance(row.get("source_version"), str) else None,
                    family_id=str(family_id) if family_id is not None else None,
                    readme_blob_sha=row.get("readme_blob_sha") if isinstance(row.get("readme_blob_sha"), str) else None,
                    readme_excerpt=row.get("readme_excerpt") if isinstance(row.get("readme_excerpt"), str) else None,
                    local_evidence_locator=row.get("local_evidence_locator") if isinstance(row.get("local_evidence_locator"), str) else None,
                    retrieval_channel=str(row.get("retrieval_channel") or self.retrieval_channel),
                    source_excerpt=document_text(row),
                    evidence=evidence,
                )
            )
            if len(matches) >= k:
                break
        return matches

    def exact_readme_matches(
        self,
        readme_blob_sha: str | None,
        *,
        exclude_source_id: str | None = None,
    ) -> list[Neighbor]:
        """Return exact-content duplicates as a separate, provenance-rich channel."""
        if not isinstance(readme_blob_sha, str) or not readme_blob_sha:
            return []
        matches = []
        for row in self.records:
            source_id = str(row.get("source_id", row.get("github_id")))
            if source_id == str(exclude_source_id) or row.get("readme_blob_sha") != readme_blob_sha:
                continue
            raw_evidence = row.get("evidence", ())
            evidence = tuple(sorted(str(item) for item in raw_evidence if isinstance(item, str))) if isinstance(raw_evidence, (list, tuple, set, frozenset)) else ()
            matches.append(
                Neighbor(
                    source_id=source_id,
                    full_name=row.get("full_name") if isinstance(row.get("full_name"), str) else None,
                    similarity=1.0,
                    source_date=row.get("source_date") if isinstance(row.get("source_date"), str) else None,
                    date_kind=row.get("date_kind") if isinstance(row.get("date_kind"), str) else None,
                    source_locator=row.get("source_locator") if isinstance(row.get("source_locator"), str) else None,
                    source_version=row.get("source_version") if isinstance(row.get("source_version"), str) else None,
                    family_id=str(row["family_id"]) if row.get("family_id") is not None else None,
                    readme_blob_sha=readme_blob_sha,
                    readme_excerpt=row.get("readme_excerpt") if isinstance(row.get("readme_excerpt"), str) else None,
                    local_evidence_locator=row.get("local_evidence_locator") if isinstance(row.get("local_evidence_locator"), str) else None,
                    retrieval_channel="exact-readme-blob-sha256",
                    source_excerpt=document_text(row),
                    evidence=evidence + (f"exact-readme-blob-sha256:{readme_blob_sha}",),
                )
            )
        return matches

    def save(self, directory: str | Path) -> None:
        """Persist FAISS index, source metadata, and checksummed manifest."""
        faiss, _ = _dependencies()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with (path / "sources.json").open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        index_sha = hashlib.sha256((path / "index.faiss").read_bytes()).hexdigest()
        manifest = {**self.manifest, "index_sha256": index_sha}
        with (path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, directory: str | Path) -> "NoveltyIndex":
        faiss, np = _dependencies()
        path = Path(directory)
        with (path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("index_version") != INDEX_VERSION or manifest.get("embedding_model") != ENCODER_VERSION:
            raise ValueError("unsupported novelty index or encoder version")
        digest = hashlib.sha256((path / "index.faiss").read_bytes()).hexdigest()
        if digest != manifest.get("index_sha256"):
            raise ValueError("novelty FAISS index digest mismatch")
        with (path / "sources.json").open(encoding="utf-8") as handle:
            records = json.load(handle)
        if len(records) != manifest.get("source_count"):
            raise ValueError("novelty index and manifest have different source counts")
        index = faiss.read_index(str(path / "index.faiss"))
        if index.ntotal != len(records):
            raise ValueError("novelty index and source manifest have different record counts")
        obj = cls.__new__(cls)
        obj.vectors = np.empty((0, int(manifest["embedding_dimension"])), dtype=np.float32)
        obj.dimension = int(manifest["embedding_dimension"])
        obj.records = tuple(dict(row) for row in records)
        obj.corpus_scope = str(manifest["corpus_scope"])
        obj.retrieval_channel = str(manifest.get("retrieval_channel", "metadata"))
        obj.built_at = str(manifest["built_at"])
        obj.hnsw_ef_search = int(manifest.get("hnsw_ef_search") or HNSW_EF_SEARCH)
        obj.source_digest = cls._digest_records(obj.records)
        if obj.source_digest != manifest.get("source_digest_sha256"):
            raise ValueError("novelty source manifest digest mismatch")
        obj.index = index
        obj.backend = str(manifest["backend"])
        if obj.backend.startswith("faiss-hnsw"):
            obj.index.hnsw.efSearch = obj.hnsw_ef_search
        return obj


def _specific_terms(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "into", "using", "use", "based", "model", "models", "learning", "machine", "deep", "system", "systems", "method", "methods", "project", "code", "repository", "repo", "implementation"}
    import re

    return {token.casefold() for token in re.findall(r"(?u)\b[a-z][a-z0-9+#.-]{2,}\b", text, re.I) if token.casefold() not in stop}


def assess_novelty(
    candidate: Mapping[str, Any],
    neighbors: Sequence[Neighbor],
    *,
    corpus_scope: str | None,
    prior_work_cutoff: str | None,
    assessed_at: str | None = None,
    relatedness_threshold: float = 0.78,
) -> dict[str, Any]:
    """Create an experimental review tag; similarity is never a probability."""
    if not 0.0 <= relatedness_threshold <= 1.0:
        raise ValueError("relatedness_threshold must be in [0, 1]")
    assessed_at = assessed_at or datetime.now(timezone.utc).isoformat()
    text = document_text(candidate)
    raw_signals = candidate.get("probable_content_signals", ())
    signals = {value for value in raw_signals if isinstance(value, str)} if isinstance(raw_signals, (list, tuple, set, frozenset)) else set()
    if isinstance(candidate.get("readme_excerpt"), str):
        from .probable_content import assess_probable_content

        signals.update(assess_probable_content(candidate["readme_excerpt"]))
    contribution = sorted(signals & _CONTRIBUTION_SIGNALS)
    substantive = any(isinstance(candidate.get(field), str) and candidate[field].strip() for field in ("readme_excerpt", "paper_abstract", "model_card_excerpt"))
    # Content relation and chronology are independent: similarity may flag a
    # copy/derivative for review without establishing which source came first.
    strong = [
        neighbor for neighbor in neighbors
        if neighbor.retrieval_channel == "exact-readme-blob-sha256"
        or (neighbor.similarity is not None and neighbor.similarity >= relatedness_threshold)
    ]
    candidate_terms = _specific_terms(text)
    overlap_neighbors = []
    for neighbor in strong:
        prior_terms = _specific_terms(neighbor.source_excerpt)
        overlap = len(candidate_terms & prior_terms) / max(1, min(len(candidate_terms), len(prior_terms)))
        overlap_neighbors.append((neighbor, overlap))
    linked = [neighbor for neighbor in strong if any(item.startswith(("shared-paper:", "same-method:", "derived-from:")) for item in neighbor.evidence)]
    if not text or not substantive or not corpus_scope:
        tag, reason = "uncertain", "insufficient-text-or-declared-corpus-scope"
    elif linked or any(overlap >= 0.35 for _, overlap in overlap_neighbors):
        tag, reason = "possible_derivative", "high-similarity-neighbor-with-independent-overlap-evidence"
    elif contribution and not strong:
        tag, reason = "probable_original_content", "explicit-contribution-evidence-with-no-close-match-in-declared-corpus"
    else:
        tag, reason = "uncertain", "no-adjudicated-prior-work-relation"
    return {
        "assessment_version": ASSESSMENT_VERSION,
        "tag": tag,
        "reason": reason,
        "experimental": True,
        "verified_novelty": False,
        "confidence": None,
        "confidence_status": "uncalibrated",
        "candidate_id": str(candidate.get("source_id", candidate.get("github_id", ""))),
        "candidate_source_version": candidate.get("source_version"),
        "corpus_scope": corpus_scope,
        "prior_work_cutoff": prior_work_cutoff,
        "chronology_status": "prior-work-date-evidence-present" if any(
            item.date_kind in {"publication", "release", "commit"}
            and item.source_date is not None
            and prior_work_cutoff is not None
            and item.source_date < prior_work_cutoff
            for item in neighbors
        ) else "unknown",
        "scientific_novelty_status": "undetermined",
        "assessed_at": assessed_at,
        "embedding_model": ENCODER_VERSION,
        "index_version": INDEX_VERSION,
        "assessment_rule_version": "evidence-neighbor-rules-v1",
        "contribution_signals": contribution,
        "compared_sources": [asdict(item) for item in neighbors],
        "matched_source_ids": [item.source_id for item in neighbors],
        "novelty_claim_scope": "repository-content-and-retrieval-evidence-only",
    }


def build_blinded_pair_bundles(
    records: Sequence[Mapping[str, Any]],
    retrieved_pairs: Sequence[Mapping[str, Any]],
    *,
    split_assignments: Mapping[str, Mapping[str, str]] | None = None,
    limit: int = 120,
    split_seed: str = "gh-ml-novelty-pairs-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze README-supported pairs and separate blinded bundles from provenance.

    ``retrieved_pairs`` is an already ordered, frozen retrieval roster. Pair
    selection never consults assessment tags or labels. Every emitted pair has
    a readable README on both sides. Scores and split assignments are returned
    only in the provenance sidecar, never in the annotation bundles.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in records:
        raw_id = row.get("github_id", row.get("source_id"))
        if raw_id is None:
            continue
        source_id = str(raw_id)
        if source_id in by_id:
            raise ValueError(f"duplicate README source ID: {source_id}")
        by_id[source_id] = row

    def usable(row: Mapping[str, Any] | None) -> bool:
        return bool(
            row
            and row.get("readme_status", "ok") == "ok"
            and isinstance(row.get("readme_text", row.get("readme_excerpt")), str)
            and row.get("readme_text", row.get("readme_excerpt")).strip()
        )

    assignments = split_assignments or {}
    selected: list[tuple[str, str, Mapping[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for edge in retrieved_pairs:
        left = str(edge.get("candidate_id", ""))
        right = str(edge.get("neighbor_id", ""))
        if not left or not right or left == right or left not in by_id or right not in by_id:
            continue
        if not usable(by_id[left]) or not usable(by_id[right]):
            continue
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        selected.append((left, right, edge))
        if len(selected) == limit:
            break

    # A connected component joins every pair touching the same repository or
    # preassigned content family. Splitting components prevents family leakage.
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    def family_for(source_id: str) -> str:
        row = by_id[source_id]
        assigned = assignments.get(source_id, {})
        family = assigned.get("split_group") or row.get("family_id")
        if isinstance(family, str) and family:
            return family
        owner = str(row.get("full_name", source_id)).split("/", 1)[0].casefold()
        return f"owner-content-family:{owner}"

    for left, right, _ in selected:
        for source_id in (left, right):
            union(f"repo:{source_id}", f"family:{family_for(source_id)}")
        union(f"repo:{left}", f"repo:{right}")

    component_pairs: dict[str, list[int]] = {}
    for index, (left, _, _) in enumerate(selected):
        component_pairs.setdefault(find(f"repo:{left}"), []).append(index)
    components = list(component_pairs.items())
    components.sort(
        key=lambda item: hashlib.sha256(f"{split_seed}:{item[0]}".encode()).hexdigest()
    )
    targets = {"train": 72, "validation": 24, "test": 24}
    split_counts = {key: 0 for key in targets}
    pair_split: dict[int, str] = {}
    for component, indices in components:
        split = min(
            targets,
            key=lambda key: (
                split_counts[key] / targets[key],
                hashlib.sha256(f"{split_seed}:{component}:{key}".encode()).hexdigest(),
            ),
        )
        for index in indices:
            pair_split[index] = split
        split_counts[split] += len(indices)

    def side(source_id: str) -> dict[str, Any]:
        row = by_id[source_id]
        full_name = str(row.get("full_name") or row.get("name") or source_id)
        path = str(row.get("readme_path") or "README.md")
        commit = row.get("commit_sha") or row.get("source_version")
        ref = str(commit) if isinstance(commit, str) and commit else "HEAD"
        locator = f"https://github.com/{full_name}/blob/{ref}/{quote(path, safe='/')}"
        text = row.get("readme_text", row.get("readme_excerpt"))
        raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "repository_id": source_id,
            "full_name": full_name,
            "readme_status": "ok",
            "readme_path": path,
            "readme_commit_sha": commit,
            "readme_locator": locator,
            "readme_blob_sha": row.get("blob_sha", row.get("readme_blob_sha")),
            "readme_text_sha256": raw_sha,
            "readme_text": text,
            "source_date": row.get("source_date", row.get("repository_created_at")),
            "date_kind": row.get("date_kind", "repository_created" if row.get("repository_created_at") else None),
            "local_evidence_locator": row.get("local_evidence_locator"),
        }

    bundles: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, (left, right, edge) in enumerate(selected):
        pair_id = hashlib.sha256(f"{min(left, right)}\0{max(left, right)}".encode()).hexdigest()
        bundles.append({
            "pair_id": pair_id,
            "candidate": side(left),
            "neighbor": side(right),
        })
        provenance.append({
            "pair_id": pair_id,
            "candidate_id": left,
            "neighbor_id": right,
            "retrieval_channel": edge.get("retrieval_channel"),
            "retrieval_rank": edge.get("retrieval_rank"),
            "similarity": edge.get("similarity"),
            "candidate_family_id": family_for(left),
            "neighbor_family_id": family_for(right),
            "component_id": find(f"repo:{left}"),
            "split": pair_split[index],
        })
    return bundles, provenance
