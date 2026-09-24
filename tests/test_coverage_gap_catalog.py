from __future__ import annotations

from pathlib import Path

from gh_ml.query_catalog import load_queries


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "config" / "queries"
GAP_FILE = CATALOG / "coverage_gaps_2026.toml"


def test_gap_queries_expand_audited_low_incidence_areas() -> None:
    queries = {query.id: query for query in load_queries(CATALOG)}
    added = {query_id: queries[query_id] for query_id in queries if query_id.startswith("gap2026.")}

    assert len(added) == 29
    required_domains = {
        "neuroimaging",
        "clinical-research",
        "green-ai",
        "econometrics",
        "physics",
        "mathematics",
        "forecasting",
        "climate-science",
        "archaeology",
    }
    covered_domains = {domain for query in added.values() for domain in query.domains}
    assert required_domains <= covered_domains

    # These terms map to audit-identified gaps or to specific methods in fields
    # that the existing catalog covered only with broad queries.
    search_text = "\n".join(query.q.lower() for query in added.values())
    for phrase in (
        "fMRI",
        "electronic health record",
        "carbon emissions",
        "double machine learning",
        "neural PDE solver",
        "learned optimizer",
        "intermittent demand forecasting",
    ):
        assert phrase.lower() in search_text


def test_gap_queries_are_specific_and_do_not_duplicate_existing_searches() -> None:
    all_queries = load_queries(CATALOG)
    added = [query for query in all_queries if query.id.startswith("gap2026.")]
    existing = [query for query in all_queries if not query.id.startswith("gap2026.")]

    # Exact query duplication spends a second Search request without adding a
    # new retrieval surface. Require explicit GitHub text-field scope too.
    assert len({query.q.casefold() for query in all_queries}) == len(all_queries)
    assert all(query.q.endswith("in:description,readme") for query in added)
    assert all(len(query.domains) >= 2 for query in added)
    assert all(query.methods for query in added)
    assert not ({query.q.casefold() for query in added} & {query.q.casefold() for query in existing})


def test_gap_catalog_file_is_included_by_standard_catalog_loader() -> None:
    assert GAP_FILE.is_file()
    loaded = load_queries(CATALOG)
    assert any(query.id == "gap2026.neuroimaging-fmri" for query in loaded)
    assert any(query.id == "gap2026.archaeology-ml" for query in loaded)
