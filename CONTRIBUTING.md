# Contributing

Thank you for contributing to `topsbot_centerpoint`.

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Install ROS 2 Humble and TOPSBOT board dependencies (`ta_runtime`, dmabuf).
3. Build the required packages first:

```bash
colcon build --packages-select tb_det_msgs topsbot_ai_infer topsbot_centerpoint
source install/setup.bash
```

4. For board validation, test on MES20 (or document host-only limitations if any).

## Model and dataset files

- Do **not** commit `.nb` model binaries, `models.zip`, or test datasets.
- Update `config/models_manifest.yaml` / `config/dataset_manifest.yaml` only when the download URLs change.
- Test the download flow with `./scripts/download_models.sh` and `./scripts/download_dataset.sh` when touching model/dataset-related scripts.

## Code Style

- Use English for code comments, log messages, and documentation.
- Match existing naming and module layout under `topsbot_centerpoint/` and `src/`.
- Prefer focused changes; avoid unrelated refactors in the same MR.
- Default build type is Release (`-O3`) for performance-sensitive paths.

## Pull Requests

- Describe the problem, approach, and test method (board / file CLI / launch demo).
- Update `README.md` / `README_cn.md` when user-facing behavior or launch parameters change.
- Add an entry under `CHANGELOG.md` (Unreleased section).
- Do not commit models, captured images, or local build artifacts.

## Reporting Issues

Please include:

- Board model (e.g. MES20) and ROS 2 distro
- Launch command and config yaml
- Model path and whether auto-download was enabled
- Relevant log excerpts (model load, dmabuf, inference errors)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.