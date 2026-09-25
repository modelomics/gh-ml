import pytest

from gh_ml.github_links import canonical_github_url, normalize_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Owner/Repo", "Owner/Repo"),
        ("https://www.github.com/Owner/Repo/", "Owner/Repo"),
        ("https://github.com/Owner/Repo.git", "Owner/Repo"),
        ("https://github.com/Owner/Repo.git/", "Owner/Repo"),
        ("https://github.com/Owner/Repo/tree/main", "Owner/Repo"),
        ("https://github.com/Owner/Repo/tree/feature/branch/subdir", "Owner/Repo"),
        ("https://github.com/Owner/Repo/blob/main/path/file.py", "Owner/Repo"),
    ],
)
def test_normalize_github_url(url: str, expected: str) -> None:
    assert normalize_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/repo",
        "ftp://github.com/owner/repo",
        "https://github.com.evil.test/owner/repo",
        "https://evilgithub.com/owner/repo",
        "https://github.com@evil.test/owner/repo",
        "https://user@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com/owner/repo?",
        "https://github.com/owner/repo#",
        "https://github.com/owner%2Fother/repo",
        "https://github.com/owner/repo/tree/main?plain=1",
        "https://github.com/owner",
        "https://github.com/owner/repo/issues/1",
        "https://github.com/owner/repo/tree",
        "https://github.com/owner/repo/tree/../other",
        "https://github.com/owner/repo//",
        "https://github.com/-owner/repo",
        "https://github.com/owner-/repo",
        "https://github.com/owner/repo.",
        "https://github.com/owner/...",
        "https://github.com/owner/repo\\other",
    ],
)
def test_rejects_unsupported_or_malformed_urls(url: str) -> None:
    assert normalize_github_url(url) is None


def test_canonical_url() -> None:
    assert canonical_github_url("SomeOwner/Some.Repo_2") == (
        "https://github.com/SomeOwner/Some.Repo_2"
    )


@pytest.mark.parametrize("name", ["owner", "owner/repo/extra", "../repo", "owner/..", "-owner/repo", "owner/repo."])
def test_canonical_url_rejects_invalid_name(name: str) -> None:
    with pytest.raises(ValueError):
        canonical_github_url(name)
