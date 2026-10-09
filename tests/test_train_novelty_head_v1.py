from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_novelty_head_v1.py"
SPEC = importlib.util.spec_from_file_location("train_novelty_head_v1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def test_output_lock_rejects_competing_writer_and_cleans_up_after_failure(tmp_path):
    output = tmp_path / "model-v1"

    with pytest.raises(RuntimeError, match="simulated failure"):
        with trainer._exclusive_output_lock(output):
            lock_path = tmp_path / ".model-v1.lock"
            assert lock_path.exists()
            with pytest.raises(FileExistsError):
                with trainer._exclusive_output_lock(output):
                    pass
            raise RuntimeError("simulated failure")

    assert not (tmp_path / ".model-v1.lock").exists()


def test_atomic_publication_never_replaces_concurrently_created_destination(tmp_path):
    staging = tmp_path / ".staging"
    output = tmp_path / "model-v1"
    staging.mkdir()
    (staging / "artifact.json").write_text("from-staging", encoding="utf-8")

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def create_destination() -> None:
        barrier.wait()
        try:
            output.mkdir()
            (output / "sentinel").write_text("preexisting", encoding="utf-8")
            outcomes.append("creator-published")
        except FileExistsError:
            outcomes.append("creator-destination-exists")

    def publish_staging() -> None:
        barrier.wait()
        try:
            trainer._publish_directory_noreplace(staging, output)
            outcomes.append("publisher-published")
        except FileExistsError:
            outcomes.append("publisher-destination-exists")

    creator = threading.Thread(target=create_destination)
    publisher = threading.Thread(target=publish_staging)
    creator.start()
    publisher.start()
    creator.join()
    publisher.join()

    assert len(outcomes) == 2
    assert ("creator-published" in outcomes) != ("publisher-published" in outcomes)
    if (output / "sentinel").exists():
        assert (output / "sentinel").read_text(encoding="utf-8") == "preexisting"
        assert staging.exists()
    else:
        assert (output / "artifact.json").read_text(encoding="utf-8") == "from-staging"
        assert not staging.exists()
