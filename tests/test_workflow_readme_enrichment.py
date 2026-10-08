from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "daily.yml"


def test_daily_workflow_exposes_bounded_readme_enrichment_budget() -> None:
    source = WORKFLOW.read_text()
    workflow = yaml.safe_load(source)

    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["readme_enrich_max_requests"]["type"] == "number"
    assert inputs["readme_enrich_max_requests"]["default"] == 500
    assert "1–1000" in inputs["readme_enrich_max_requests"]["description"]

    env = workflow["jobs"]["collect"]["env"]
    assert env["README_ENRICH_MAX_REQUESTS"] == (
        "${{ github.event_name == 'schedule' && 500 || inputs.readme_enrich_max_requests }}"
    )
    assert '"README_ENRICH_MAX_REQUESTS": (1, 1000)' in source

    steps = workflow["jobs"]["collect"]["steps"]
    readme_step = next(step for step in steps if step.get("id") == "readme")
    assert readme_step["if"] == "${{ !inputs.snapshot_only }}"
    assert readme_step["continue-on-error"] is True
    assert readme_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert readme_step["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    assert readme_step["env"]["HF_OIDC_RESOURCE"] == "datasets/modelomics/gh-ml"
    assert readme_step["run"] == (
        'uv run --frozen gh-ml readme-enrich --max-requests "$README_ENRICH_MAX_REQUESTS"'
    )
    assert "steps.readme.outcome == 'failure'" in source
