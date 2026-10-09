import argparse
import json
import subprocess

from gh_ml import ecosystems_download as download


def test_unknown_partial_provenance_is_preserved(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    dataset.mkdir()
    run_dir.mkdir()
    partial = dataset / "repos-2023-08-30.tar.gz.partial"
    control = dataset / "repos-2023-08-30.tar.gz.partial.aria2"
    source = dataset / "repos-2023-08-30.tar.gz.partial.source.json"
    partial.write_bytes(b"keep this unknown partial")
    control.write_bytes(b"aria2 state")
    source.write_text(json.dumps({"url": "https://other.example/archive", "etag": "other"}))
    monkeypatch.setattr(download.shutil, "which", lambda _: "/usr/bin/aria2c")

    result = download._run(argparse.Namespace(dataset_dir=dataset, run_dir=run_dir))

    assert result == 6
    assert partial.read_bytes() == b"keep this unknown partial"
    assert control.read_bytes() == b"aria2 state"
    assert json.loads((run_dir / "status.json").read_text())["state"] == (
        "aria2_source_provenance_review_required"
    )


def test_complete_rpc_transfer_is_verified_and_promoted(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    monkeypatch.setattr(download, "EXPECTED_BYTES", 100)
    monkeypatch.setattr(download, "RESERVE_BYTES", 0)
    monkeypatch.setattr(download, "GUARD_HEADROOM_BYTES", 0)
    monkeypatch.setattr(download.shutil, "which", lambda _: "/usr/bin/aria2c")
    monkeypatch.setattr(download, "_free_bytes", lambda _: 10**12)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def bind(self, _address):
            return None

        def getsockname(self):
            return ("127.0.0.1", 12345)

    monkeypatch.setattr(download.socket, "socket", FakeSocket)

    class FakeProcess:
        def __init__(self):
            self.poll_calls = 0

        def poll(self):
            self.poll_calls += 1
            return None

        def wait(self):
            return 0

        def send_signal(self, _signal):
            raise AssertionError("unexpected stop signal")

    process = FakeProcess()

    def fake_popen(cmd, **_kwargs):
        assert "--header=If-Match: " + download.EXPECTED_ETAG in cmd
        (dataset / "repos-2023-08-30.tar.gz.partial").write_bytes(b"x" * 100)
        return process

    monkeypatch.setattr(download.subprocess, "Popen", fake_popen)
    responses = iter(
        [
            [{"completedLength": "60", "totalLength": "100", "downloadSpeed": "10"}],
            [],
            [{"completedLength": "100", "totalLength": "100", "status": "complete"}],
        ]
    )

    def fake_rpc(_port, _token, method, _params):
        if method == "aria2.tellActive":
            return next(responses)
        if method == "aria2.tellStopped":
            return next(responses)
        if method == "aria2.shutdown":
            return None
        raise AssertionError(f"unexpected RPC method: {method}")

    monkeypatch.setattr(download, "_rpc", fake_rpc)
    status_updates = []
    write_status = download._write_status

    def capture_status(path, data):
        status_updates.append(data.copy())
        write_status(path, data)

    monkeypatch.setattr(download, "_write_status", capture_status)
    monkeypatch.setattr(download.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0))
    monkeypatch.setattr(download.subprocess, "check_output", lambda *a, **k: "a" * 64 + "  archive\n")

    result = download._run(argparse.Namespace(dataset_dir=dataset, run_dir=run_dir))

    archive = dataset / "repos-2023-08-30.tar.gz"
    status = json.loads((run_dir / "status.json").read_text())
    assert result == 0
    assert archive.read_bytes() == b"x" * 100
    assert status["state"] == "complete"
    assert status["bytes"] == 100
    assert status["gzip_integrity"] == "passed"
    assert status["sha256"] == "a" * 64
    assert any(update.get("downloaded_bytes") == 60 for update in status_updates)
