# topsbot_centerpoint

[English](README.md)

# 功能介绍

`topsbot_centerpoint` 在 TOPSBOT MES20 板卡上运行 **CenterPoint-PointPillars**（nuScenes 官方 10-sweep 模型）。

- **图片**：对本地 PCD.BIN 做离线检测，生成带框图片
- **浏览器预览**：配合 [`topsbot_webviz`](../topsbot_webviz) 在网页上查看画面与检测框

# 物料清单

| 序号 | 名称 | 说明 |
| ---- | ---- | ---- |
| 1 | TOPSBOT MES20 板卡 | NPU（`ta_runtime`）与 dmabuf 环境 |
| 2 | CenterPoint 模型 | 双NB：`centerpoint_pfe_p5268_float16.nb` + `centerpoint_rpn_head_float16.nb`；单NB：`centerpoint_p5268_float16.nb` |
| 3 | nuScenes 点云数据 | `.bin` 文件 + 10-sweep manifest（`sweeps.txt`） |
| 4 | `topsbot_webviz` | 浏览器查看 BEV 视频流（可选） |
| 5 | `topsbot_ai_infer` | 插件接口与运行时集成 |

# 编译与安装

## PC 端（交叉编译）

在 `topsbot_cc` 工具链环境中编译，生成供 MES20 安装的 `install/`：

```bash
# PC：topsbot_cc 工具链环境
cd topsbot_cc/topsbot_ws
./tools/build.sh -s topsbot_centerpoint topsbot_ai_infer
```

将 `install/` 同步到板卡，或直接在板端工作空间中编译（若板端具备完整编译环境）。

## MES20 板端（安装与加载）

**方式 A：DEB 包（推荐，源可用时）**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**方式 B：加载工作空间**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
```

# 模型文件与数据集

模型（`.nb`）与测试数据集不随源码分发。首次在板端启动时，launch 会检查 `~/topsbot/models/` 与 `~/topsbot/data/`，缺失则按 `config/models_manifest.yaml` / `config/dataset_manifest.yaml` 自动下载。

| 项 | 默认路径 |
| ---- | -------- |
| PFE 模型（双NB） | `~/topsbot/models/centerpoint_pfe_p5268_float16.nb` |
| RPN + CenterHead 模型（双NB） | `~/topsbot/models/centerpoint_rpn_head_float16.nb` |
| 单图融合模型（单NB） | `~/topsbot/models/centerpoint_p5268_float16.nb` |
| 数据集 | `~/topsbot/data/` |

`float16` 与 `int8` 两种 `.nb` 均支持；int8 变体为模型清单中的可选条目。

离线环境或已手动放置模型 / 数据时：

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint.launch.py \
  auto_download_models:=false auto_download_dataset:=false
```

手动下载脚本（（维护者需先配置 `models_manifest.yaml` 中的 URL）：

```bash
# PC 或 MES20（源码树）
./scripts/download_models.sh
./scripts/download_dataset.sh
```

# 使用方式

## file 模式

Launch：`file_cli.launch.py` 。

**默认启动**

```bash
# MES20
ros2 launch topsbot_centerpoint file_cli.launch.py \
  input:=/root/topsbot/data/sweeps.txt \
  output:=/tmp/centerpoint_bev.jpg
```

| launch 参数 | 作用 |
|-------------|------|
| `input` | 必填：帧目录（含 `sweeps.txt`） |
| `output` | 输出图片路径（默认 `/tmp/centerpoint_bev.jpg`） |
| `pfe_model` / `rpn_model` | 覆盖模型路径；留空按 `models_manifest.yaml` 解析 |
| `single_frame` | `true` 直接读取单个 `.bin`（默认 `false`） |
| `core` | NPU core 编号 |
| `score_threshold` / `bev_size` | 覆盖检测阈值 / BEV 渲染分辨率 |

## 联合测试模式 单帧流式模式（单NB+ Web 预览）

Launch：`centerpoint_single_frame.launch.py`。

**默认启动**

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint_single_frame.launch.py
```

| launch 参数 | 作用 |
|-------------|------|
| `bin_files_dir` / `bin_list_file` | 输入数据源：目录或文件列表（默认 `/root/topsbot/data/bins.txt`） |
| `merged_model` | 覆盖融合 NB 路径；默认使用 `centerpoint_p5268_float16.nb` |
| `frame_interval_ms` | 帧间隔（默认 `100`） |
| `loop` | 播完后是否循环（默认 `true`） |

## 联合测试模式（双NB + Web 预览）

Launch：`centerpoint.launch.py`（启动 `centerpoint_infer`，默认附带 `topsbot_webviz`）。

**默认启动**

```bash
# MES20
ros2 launch topsbot_centerpoint centerpoint.launch.py
```

浏览器访问 `http://<板卡IP>:8000/` 查看 BEV 视频流。

**常用调整**

```bash
# MES20 — 回灌间隔（0 = 只跑一次；200 ≈ 5fps 视频流）
ros2 launch topsbot_centerpoint centerpoint.launch.py run_interval_ms:=200

# MES20 — 仅推理，不要 Web 预览
ros2 launch topsbot_centerpoint centerpoint.launch.py enable_webviz:=false

# MES20 — 显式指定模型 / manifest
ros2 launch topsbot_centerpoint centerpoint.launch.py \
  pfe_model:=/root/topsbot/models/centerpoint_pfe_p5268_float16.nb \
  rpn_model:=/root/topsbot/models/centerpoint_rpn_head_float16.nb \
  frames_list:=/root/topsbot/data/seq_test/board_manifest.txt

```

**确认链路正常**

```bash
# MES20
ros2 topic hz /detections
ros2 topic echo /detections --once
ros2 topic echo /image_jpeg --once --field data | base64 -d > /tmp/bev.jpg
```


# 接口说明

## ROS 话题

| 方向 | 话题 | 消息类型 | 说明 |
|------|------|----------|------|
| 发布 | `/detections` | `tb_det_msgs/TbPerceptionTargets` | BEV 2D 检测框 |
| 发布 | `/image_jpeg` | `sensor_msgs/CompressedImage` | BEV 渲染图 |

输入为离线点云文件模式（`input_msg: centerpoint_file`），无需订阅图像话题。

## 节点参数

场景配置：`config/ai_infer_centerpoint.yaml`（`centerpoint.launch.py` 使用）、`config/centerpoint_frame_stream.yaml`（`centerpoint_single_frame.launch.py` 使用）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_msg` | `centerpoint_file` | 离线点云文件模式 |
| `detections_topic` | `/detections` | BEV 检测话题 |
| `centerpoint.image_topic` | `/image_jpeg` | BEV 渲染话题 |
| `centerpoint.frame_id` | `centerpoint` | 输出 `header.frame_id` |
| `centerpoint.pfe_model` | — | PFE 模型路径（必填） |
| `centerpoint.rpn_model` | — | RPN + CenterHead 模型路径（必填） |
| `centerpoint.sweeps_manifest` | `""` | 10 行 manifest，或单帧模式下的单个 `.bin` |
| `centerpoint.frames_list` | `""` | 多帧列表；指定后覆盖 `sweeps_manifest` |
| `centerpoint.single_frame_mode` | `false` | `true` 直接读取单个 `.bin` |
| `centerpoint.run_interval_ms` | `800` | `0` 只跑一次；`>0` 按间隔回放（毫秒） |
| `centerpoint.score_threshold` | `0.2` | 输出最终过滤 |
| `centerpoint.bev_size` | `1024` | BEV 渲染 / 投影分辨率（像素） |
| `centerpoint.device_ids` | `[0, 1]` | NPU core；双核 worker 模式 |
| `centerpoint.load_balancer.strategy` | `least_loaded` | 多核任务分配（`least_loaded` / `round_robin`） |


## Launch 文件

两种部署模式：`centerpoint.launch.py` = 多帧 + 双NB（pfe + rpn）；`centerpoint_single_frame.launch.py` = 单帧 + 单NB（融合图）。

| Launch | 用途 |
|--------|------|
| `centerpoint.launch.py` | 多帧推理节点（双NB）（可选 WebViz） |
| `file_cli.launch.py` | 离线单张 BEV 图 CLI（无 ROS） |
| `centerpoint_single_frame.launch.py` | 单帧流式检测（单NB 融合） |

各 launch 参数：`ros2 launch topsbot_centerpoint <launch名> -s`

---

## 许可

Apache License 2.0 — 见 [LICENSE](./LICENSE)。

欢迎贡献，请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。