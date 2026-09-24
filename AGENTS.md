# GitHub ML

Use `xonsh` and `uv` for project commands. Keep the collector modular, use
established libraries where they fit, and keep generated datasets and run
outputs outside the source repository. Store the maintained registry on the
Modelomics Hugging Face organization and keep this repository focused on the
collection and publishing code.

Set up the environment with `uv sync --extra dev`. Run tests with `uv run
pytest`.
