from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "daily.yml"


def test_daily_workflow_configures_bounded_hf_daily_papers_collection() -> None:
    source = WORKFLOW.read_text()
    workflow = yaml.safe_load(source)

    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["hf_papers_max_pages"]["type"] == "number"
    assert inputs["hf_papers_max_pages"]["default"] == 20
    assert "1–100" in inputs["hf_papers_max_pages"]["description"]

    env = workflow["jobs"]["collect"]["env"]
    assert env["HF_PAPERS_MAX_PAGES"] == (
        "${{ github.event_name == 'schedule' && 20 || inputs.hf_papers_max_pages }}"
    )
    assert '"HF_PAPERS_MAX_PAGES": (1, 100)' in source

    steps = workflow["jobs"]["collect"]["steps"]
    paper_step = next(step for step in steps if step.get("id") == "hf_papers")
    snapshot_index = next(i for i, step in enumerate(steps) if step.get("id") == "snapshot")
    paper_index = steps.index(paper_step)
    assert paper_index < snapshot_index
    assert paper_step["if"] == "${{ !inputs.snapshot_only }}"
    assert paper_step["continue-on-error"] is True
    assert paper_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert paper_step["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    assert paper_step["env"]["HF_OIDC_RESOURCE"] == "datasets/modelomics/gh-ml"
    assert paper_step["run"] == (
        'uv run --frozen gh-ml hf-papers-daily --work-dir '
        '"$RUNNER_TEMP/gh-ml-hf-papers" --max-pages "$HF_PAPERS_MAX_PAGES"'
    )
    assert "steps.hf_papers.outcome == 'failure'" in source
