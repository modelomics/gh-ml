# GitHub ML

Use `xonsh` and `uv` for project commands. Keep the collector modular, use
established libraries where they fit, and keep generated datasets and run
outputs outside the source repository. Store the maintained registry on the
Modelomics Hugging Face organization and keep this repository focused on the
collection and publishing code.

Set up the full test environment with `uv sync --locked --extra test`. Run the offline suite with `xonsh --no-rc -c '$HF_HUB_OFFLINE = "1"; $TRANSFORMERS_OFFLINE = "1"; uv run --frozen pytest'`.
