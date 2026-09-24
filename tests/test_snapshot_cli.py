from __future__ import annotations

import json
from pathlib import Path

import httpx
from huggingface_hub.errors import HfHubHTTPError

from gh_ml import cli


def test_publish_current_view_cli_resolves_token_and_prints_result_without_token(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    secret = "hf-secret-test-value"
    calls: dict[str, object] = {}
    token_values = iter([secret, "hf-fresh-test-value"])

    monkeypatch.setattr(cli, "_hf_token", lambda env_name: next(token_values))

    def publish(
        repo_id: str,
        token: str | None,
        *,
        work_dir: Path,
        token_provider,
        **kwargs,
    ) -> dict[str, object]:
        calls.update(
            repo_id=repo_id,
            token=token,
            fresh_token=token_provider(),
            work_dir=work_dir,
            kwargs=kwargs,
        )
        return {"repo_id": repo_id, "path": "data/current/repositories.parquet", "rows": 12}

    monkeypatch.setattr(cli, "publish_current_view", publish)

    assert cli.main([
        "publish-current-view", "--repo", "modelomics/example", "--work-dir", str(tmp_path)
    ]) == 0

    assert calls == {
        "repo_id": "modelomics/example",
        "token": secret,
        "fresh_token": "hf-fresh-test-value",
        "work_dir": tmp_path,
        "kwargs": {},
    }
    output = capsys.readouterr().out
    assert secret not in output
    assert json.loads(output) == {
        "path": "data/current/repositories.parquet",
        "repo_id": "modelomics/example",
        "rows": 12,
    }


def test_publish_current_view_cli_sanitizes_hub_http_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    secret = "hf-secret-in-response-body"
    monkeypatch.setattr(cli, "_hf_token", lambda _env_name: "hf-initial-token")

    def publish(*_args, **_kwargs):
        response = httpx.Response(
            403, text=secret, request=httpx.Request("POST", "https://huggingface.co")
        )
        raise HfHubHTTPError(secret, response=response, server_message=secret)

    monkeypatch.setattr(cli, "publish_current_view", publish)

    assert cli.main([
        "publish-current-view", "--repo", "modelomics/example", "--work-dir", str(tmp_path)
    ]) == 2

    captured = capsys.readouterr()
    assert "HTTP 403" in captured.err
    assert secret not in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
