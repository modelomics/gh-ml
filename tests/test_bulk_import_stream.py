from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import gh_ml.bulk_import_stream as bulk_stream


def test_cli_passes_bounded_binary_stdin_to_importer(tmp_path: Path, monkeypatch):
    raw_input = BytesIO("COPY public.hosts (id) FROM stdin;\n".encode("utf-8"))
    captured: dict[str, object] = {}

    def import_stream(stream, **kwargs):
        captured["stream"] = stream
        captured["first_byte"] = stream.readline(1)
        captured["kwargs"] = kwargs
        return {"row_counts": {"github_rows": 0}, "shards": []}

    monkeypatch.setattr(bulk_stream.sys, "stdin", SimpleNamespace(buffer=raw_input))
    monkeypatch.setattr(bulk_stream.sys, "argv", [
        "bulk_import_stream", "--output-dir", str(tmp_path),
        "--source-fingerprint", "sha256:fixture", "--observed-at", "2026-10-09T00:00:00Z",
    ])
    monkeypatch.setattr(bulk_stream, "import_pg_restore_stream", import_stream)

    assert bulk_stream.main() == 0
    assert captured["stream"] is raw_input
    assert captured["first_byte"] == b"C"
    assert captured["kwargs"]["output_dir"] == tmp_path
