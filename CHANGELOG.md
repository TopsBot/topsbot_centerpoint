# Changelog

All notable changes to this package are documented here.

## 0.1.0 (2026-09-24)

### Added

- CenterPoint-PointPillars (nuScenes official, 10-sweep) inference backend for `topsbot_ai_infer` on MES20 (offline point-cloud file mode, `input_msg: centerpoint_file`).
- `centerpoint_infer` node publishing BEV 2D detections (`/detections`) and color-coded BEV render (`/image_jpeg`).
- ROS-free offline CLI `centerpoint_file_cli` and `file_cli.launch.py` for single-frame / batch inference.
- `centerpoint.launch.py` with optional WebViz browser preview (`http_port`, `enable_webviz`).
- Model auto-download via `config/models_manifest.yaml` and test dataset auto-download via `config/dataset_manifest.yaml` (`~/topsbot/models`, `~/topsbot/data/seq_test`).
- Multi-frame replay via `centerpoint.frames_list`, 10-sweep fusion mode and experimental single-frame mode (`centerpoint.single_frame_mode`).
- Dual-core NPU worker mode with load balancer (`centerpoint.device_ids`, `centerpoint.load_balancer.strategy`).
- Bilingual README files and contribution guide.