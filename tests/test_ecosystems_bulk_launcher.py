import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_ecosystems_bulk_import.py"
SPEC = importlib.util.spec_from_file_location("ecosystems_bulk_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_run_directory_must_be_absent_or_empty(tmp_path):
    run_dir = tmp_path / "run"
    assert LAUNCHER.is_empty_run_dir(run_dir)
    run_dir.mkdir()
    assert LAUNCHER.is_empty_run_dir(run_dir)
    (run_dir / "receipt.json").write_text("preserve")
    assert not LAUNCHER.is_empty_run_dir(run_dir)


def test_dataset_allows_only_empty_quarantine_file(tmp_path):
    dataset = tmp_path / "dataset"
    assert LAUNCHER.is_pristine_dataset_dir(dataset)
    dataset.mkdir()
    (dataset / "quarantine.jsonl").touch()
    assert LAUNCHER.is_pristine_dataset_dir(dataset)
    (dataset / "quarantine.jsonl").write_text("existing row\n")
    assert not LAUNCHER.is_pristine_dataset_dir(dataset)


def test_dataset_rejects_any_other_existing_file(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    assert not LAUNCHER.is_pristine_dataset_dir(dataset)
