# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0
#
# CenterPoint via unified ai_infer_node framework (uses CenterPointBackend plugin)

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import ensure_centerpoint_models  # noqa: E402


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_centerpoint")

    # Ensure models are available
    merged_model = LaunchConfiguration("merged_model").perform(context)

    # Build model_path for ai_infer: single path = merged graph; otherwise
    # semicolon-separated pfe;rpn.
    if merged_model:
        model_path = os.path.expanduser(merged_model)
    else:
        pfe_model, rpn_model = ensure_centerpoint_models(
            context, {},
            LaunchConfiguration("pfe_model").perform(context),
            LaunchConfiguration("rpn_model").perform(context))
        model_path = f"{pfe_model};{rpn_model}"

    device_id = int(LaunchConfiguration("device_id").perform(context) or "0")
    score_threshold = float(LaunchConfiguration("score_threshold").perform(context) or "0.2")
    bev_size = int(LaunchConfiguration("bev_size").perform(context) or "1024")
    render_enabled_str = LaunchConfiguration("render_enabled").perform(context)
    render_enabled = render_enabled_str.lower() in ("true", "1", "yes")

    # Get file input parameters
    frames_list = LaunchConfiguration("frames_list").perform(context)
    sweeps_manifest = LaunchConfiguration("sweeps_manifest").perform(context)
    run_interval_ms = LaunchConfiguration("run_interval_ms").perform(context) or "800"

    # Determine input mode
    input_msg = "centerpoint_file" if (frames_list or sweeps_manifest) else "pointcloud"
    input_topic = "" if input_msg == "centerpoint_file" else (
        LaunchConfiguration("input_topic").perform(context) or "/points")

    # ai_infer_node parameters - match ai_infer_centerpoint.yaml structure
    ai_infer_params = {
        "backend": "centerpoint",
        "model_name": "centerpoint",
        "model_path": model_path,
        "device_id": device_id,
        "confidence_threshold": score_threshold,
        "input_msg": input_msg,
        "input_topic": input_topic,
        "image_topic": LaunchConfiguration("image_topic").perform(context) or "/centerpoint/image",
        "detections_topic": LaunchConfiguration("detections_topic").perform(context) or "/detections",
        "centerpoint.frame_id": "centerpoint",
        "centerpoint.image_topic": LaunchConfiguration("image_topic").perform(context) or "/image_jpeg",
        "centerpoint.pfe_model": pfe_model,
        "centerpoint.rpn_model": rpn_model,
        "centerpoint.score_threshold": score_threshold,
        "centerpoint.sweeps_manifest": sweeps_manifest,
        "centerpoint.frames_list": frames_list,
        "centerpoint.single_frame_mode": False,
        "centerpoint.run_interval_ms": int(run_interval_ms),
        "centerpoint.bev_size": bev_size,
        "render": {
            "enabled": render_enabled,
            "size": bev_size,
        }
    }

    print(f"\n{'='*60}")
    print(f"[centerpoint_ai_infer] Launching with configuration:")
    print(f"  backend: {ai_infer_params['backend']}")
    print(f"  model_name: {ai_infer_params['model_name']}")
    print(f"  model_path: {ai_infer_params['model_path']}")
    print(f"  device_id: {ai_infer_params['device_id']}")
    print(f"  input_msg: {input_msg}")
    if input_msg == "centerpoint_file":
        print(f"  frames_list: {frames_list or '(none)'}")
        print(f"  sweeps_manifest: {sweeps_manifest or '(none)'}")
        print(f"  run_interval_ms: {run_interval_ms}")
    else:
        print(f"  input_topic: {input_topic}")
    print(f"  confidence_threshold: {ai_infer_params['confidence_threshold']}")
    print(f"  render.enabled: {render_enabled}")
    print(f"  render.size: {bev_size}")
    print(f"{'='*60}\n")

    return [
        Node(
            package="topsbot_ai_infer",
            executable="ai_infer_node",
            name="centerpoint_ai_infer",
            output="screen",
            parameters=[ai_infer_params],
            # Add arguments to prevent loading default config
            arguments=['--ros-args', '--log-level', 'INFO'],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "pfe_model", default_value="",
            description="PFE model path (auto-resolves from manifest if empty)"),
        DeclareLaunchArgument(
            "rpn_model", default_value="",
            description="RPN model path (auto-resolves from manifest if empty)"),
        DeclareLaunchArgument(
            "merged_model", default_value="",
            description="Single-graph NB (PFE+ScatterND+RPN); overrides pfe/rpn"),
        DeclareLaunchArgument(
            "frames_list", default_value="",
            description="Path to frames list file (board_manifest.txt)"),
        DeclareLaunchArgument(
            "sweeps_manifest", default_value="",
            description="Path to sweeps manifest file (sweeps.txt)"),
        DeclareLaunchArgument(
            "run_interval_ms", default_value="800",
            description="Frame interval in milliseconds (0 = run once)"),
        DeclareLaunchArgument(
            "device_id", default_value="0",
            description="NPU core ID"),
        DeclareLaunchArgument(
            "score_threshold", default_value="0.2",
            description="Detection confidence threshold"),
        DeclareLaunchArgument(
            "bev_size", default_value="1024",
            description="BEV render size (pixels)"),
        DeclareLaunchArgument(
            "input_topic", default_value="/points",
            description="Input PointCloud2 topic (used only if no file input specified)"),
        DeclareLaunchArgument(
            "image_topic", default_value="/centerpoint/image",
            description="Output BEV image topic"),
        DeclareLaunchArgument(
            "detections_topic", default_value="/detections",
            description="Output detections topic"),
        DeclareLaunchArgument(
            "render_enabled", default_value="true",
            description="Enable BEV image rendering"),
        DeclareLaunchArgument(
            "models_dir", default_value="~/topsbot/models"),
        DeclareLaunchArgument(
            "auto_download_models", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
