# topsbot_centerpoint

[中文文档](README_cn.md)

# Overview

`topsbot_centerpoint` runs **CenterPoint-PointPillars** (the official nuScenes 10-sweep model) on the TOPSBOT MES20 board.

- **Images**: offline detection on local PCD.BIN files, generates images with overlaid boxes
- **Browser preview**: paired with [`topsbot_webviz`](../topsbot_webviz) to view the BEV and detection boxes in a web browser

# Bill of Materials

| # | Name | Description |
| ---- | ---- | ---- |
| 1 | TOPSBOT MES20 board | NPU (`ta_runtime`) and dmabuf environment |
| 2 | CenterPoint models | dual-NB: `centerpoint_pfe_p5268_float16.nb` + `centerpoint_rpn_head_float16.nb`; single-NB: `centerpoint_p5268_float16.nb` |
| 3 | nuScenes point-cloud data | `.bin` files + 10-sweep manifest (`sweeps.txt`) |
| 4 | `topsbot_webviz` | Browser preview of the BEV video stream (optional) |
| 5 | `topsbot_ai_infer` | Plugin interface and runtime integration |

# Build and Install

## PC (cross-compile)

Build inside the `topsbot_cc` toolchain environment to produce `install/` for the MES20:

```bash
# PC: topsbot_cc toolchain
cd topsbot_cc/topsbot_ws
./tools/build.sh -s topsbot_centerpoint topsbot_ai_infer
```

Sync `install/` to the board, or build directly on the board if a full build environment is available.

## MES20 board (install and source)

**Option A: DEB packages (recommended when available)**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**Option B: workspace overlay**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

# Model Files and Dataset

Models (`.nb`) and the test dataset are **not** shipped with the source. On first launch, `~/topsbot/models/` and `~/topsbot/data/` are checked; missing files trigger automatic download per `config/models_manifest.yaml` / `config/dataset_manifest.yaml`.

| Item | Default path |
| ---- | -------- |
| PFE model (dual-NB) | `~/topsbot/models/centerpoint_pfe_p5268_float16.nb` |
| RPN + CenterHead model (dual-NB) | `~/topsbot/models/centerpoint_rpn_head_float16.nb` |
| Merged single-graph model (single-NB) | `~/topsbot/models/centerpoint_p5268_float16.nb` |
| Dataset | `~/topsbot/data/` |

Both `float16` and `int8` `.nb` files are supported; int8 variants are optional entries in the model manifest.

For offline environments or when models/data are already placed manually:

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint.launch.py \
  auto_download_models:=false auto_download_dataset:=false
```

Manual download scripts (maintainers must configure URLs in `models_manifest.yaml` first):

```bash
# PC or MES20 (source tree)
./scripts/download_models.sh
./scripts/download_dataset.sh
```

# Usage

## file mode

Launch: `file_cli.launch.py`.

**Default launch**

```bash
# MES20
ros2 launch topsbot_centerpoint file_cli.launch.py \
  input:=/root/topsbot/data/sweeps.txt \
  output:=/tmp/centerpoint_bev.jpg
```

| Launch argument | Purpose |
|-------------|------|
| `input` | Required: frame dir (with `sweeps.txt`) |
| `output` | Output image path (default `/tmp/centerpoint_bev.jpg`) |
| `pfe_model` / `rpn_model` | Override model paths; empty resolves `models_manifest.yaml` |
| `single_frame` | `true` reads one `.bin` directly (default `false`) |
| `core` | NPU core id |
| `score_threshold` / `bev_size` | Override detection threshold / BEV render resolution |

## Joint Testing Mode — Single-frame Streaming (single-NB + Web Preview)

Launch: `centerpoint_single_frame.launch.py`.

**Default launch**

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint_single_frame.launch.py
```

| Launch argument | Purpose |
|-------------|------|
| `bin_files_dir` / `bin_list_file` | Input data source: directory or file list (default `/root/topsbot/data/bins.txt`) |
| `merged_model` | Override merged NB path; default uses `centerpoint_p5268_float16.nb` |
| `frame_interval_ms` | Frame interval (default `100`) |
| `loop` | Loop playback after the last frame (default `true`) |

## Joint Testing Mode (dual-NB + Web Preview)

Launch: `centerpoint.launch.py` (starts `centerpoint_infer`, optionally with `topsbot_webviz`).

**Default launch**

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint.launch.py
```

Open `http://<board-ip>:8000/` in a browser to view the BEV video stream.

**Common adjustments**

```bash
# MES20 — replay interval (0 = run once; 200 ≈ 5fps video stream)
ros2 launch topsbot_centerpoint centerpoint.launch.py run_interval_ms:=200

# MES20 — inference only, no Web preview
ros2 launch topsbot_centerpoint centerpoint.launch.py enable_webviz:=false

# MES20 — explicit models / manifest
ros2 launch topsbot_centerpoint centerpoint.launch.py \
  pfe_model:=/root/topsbot/models/centerpoint_pfe_p5268_float16.nb \
  rpn_model:=/root/topsbot/models/centerpoint_rpn_head_float16.nb \
  frames_list:=/root/topsbot/data/seq_test/board_manifest.txt
```

**Verify the pipeline**

```bash
# MES20
ros2 topic hz /detections
ros2 topic echo /detections --once
ros2 topic echo /image_jpeg --once --field data | base64 -d > /tmp/bev.jpg
```

# Interface Reference

## ROS Topics

| Direction | Topic | Message type | Description |
|------|------|----------|------|
| Publish | `/detections` | `tb_det_msgs/TbPerceptionTargets` | BEV 2D detection boxes |
| Publish | `/image_jpeg` | `sensor_msgs/CompressedImage` | BEV rendered image  |

Input is offline point-cloud file mode (`input_msg: centerpoint_file`); no image subscription required.

## Node Parameters

Scene configs: `config/ai_infer_centerpoint.yaml` (used by `centerpoint.launch.py`), `config/centerpoint_frame_stream.yaml` (used by `centerpoint_single_frame.launch.py`).

| Parameter | Default | Description |
|------|--------|------|
| `input_msg` | `centerpoint_file` | Offline point-cloud file mode |
| `detections_topic` | `/detections` | BEV detection topic |
| `centerpoint.image_topic` | `/image_jpeg` | BEV render topic |
| `centerpoint.frame_id` | `centerpoint` | Output `header.frame_id` |
| `centerpoint.pfe_model` | — | PFE model path (required) |
| `centerpoint.rpn_model` | — | RPN + CenterHead model path (required) |
| `centerpoint.sweeps_manifest` | `""` | 10-line manifest, or single `.bin` in single-frame mode |
| `centerpoint.frames_list` | `""` | Multi-frame list; overrides `sweeps_manifest` |
| `centerpoint.single_frame_mode` | `false` | `true` reads one `.bin` directly |
| `centerpoint.run_interval_ms` | `800` | `0` run once; `>0` replay interval (ms) |
| `centerpoint.score_threshold` | `0.2` | Final detection filter  |
| `centerpoint.bev_size` | `1024` | BEV render / projection resolution (px) |
| `centerpoint.device_ids` | `[0, 1]` | NPU cores; dual-core worker mode |
| `centerpoint.load_balancer.strategy` | `least_loaded` | Task assignment across cores (`least_loaded` / `round_robin`) |

The BEV projection range is fixed at `[-61.2, 61.2]` m (model `post_center_limit_range`), matching the voxelization range.

## Launch Files

Two deployment modes: `centerpoint.launch.py` = multi-frame + dual-NB (pfe + rpn); `centerpoint_single_frame.launch.py` = single-frame + single-NB (merged graph).

| Launch | Purpose |
|--------|------|
| `centerpoint.launch.py` | Multi-frame inference node (dual-NB) (optional WebViz) |
| `file_cli.launch.py` | Offline single-BEV-image CLI (no ROS) |
| `centerpoint_single_frame.launch.py` | Single-frame streaming detection (single-NB merged) |

List arguments: `ros2 launch topsbot_centerpoint <launch> -s`

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).

Contributions welcome; see [CONTRIBUTING.md](./CONTRIBUTING.md).
