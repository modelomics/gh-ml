from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "daily.yml"


def test_daily_workflow_configures_and_runs_topic_breadth_collector() -> None:
    workflow = WORKFLOW.read_text()

    assert "topic_max_pages:" in workflow
    assert 'description: "Maximum GitHub topic pages to collect (1–100)"' in workflow
    assert "type: number\n        default: 60" in workflow
    assert "TOPIC_MAX_PAGES: ${{ github.event_name == 'schedule' && 60 || inputs.topic_max_pages }}" in workflow
    assert '"TOPIC_MAX_PAGES": (1, 100)' in workflow

    census_step = workflow.index("id: census")
    topic_step = workflow.index("id: topic_breadth")
    readme_step = workflow.index("id: readme")
    assert census_step < topic_step < readme_step

    topic_block = workflow[topic_step:readme_step]
    assert "if: ${{ !inputs.snapshot_only }}" in topic_block
    assert "continue-on-error: true" in topic_block
    assert "GITHUB_TOKEN: ${{ github.token }}" in topic_block
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in topic_block
    assert "HF_OIDC_RESOURCE: datasets/modelomics/gh-ml" in topic_block
    assert (
        'gh-ml topic-breadth-daily --work-dir "$RUNNER_TEMP/gh-ml-topic-breadth" '
        '--max-pages "$TOPIC_MAX_PAGES"'
    ) in topic_block
    assert "steps.topic_breadth.outcome == 'failure'" in workflow
