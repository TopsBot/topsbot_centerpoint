# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0
#
# Offline single-image BEV detection via launch (counterpart of
# topsbot_mobilenet_cls/launch/file_cli.launch.py). Wraps the ROS-free
# centerpoint_file_cli executable; models are optional args and fall back to
# config/models_manifest.yaml with auto-download, exactly like centerpoint_infer
# launch.
#
# Usage:
#   ros2 launch topsbot_centerpoint file_cli.launch.py \
#     input:=/root/topsbot/data/sweeps.txt output:=/root/topsbot/data/out.jpg
#   ros2 launch topsbot_centerpoint file_cli.launch.py \
#     input:=... output:=... pfe_model:=... rpn_model:=... core:=1

import os
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import ensure_centerpoint_models  # noqa: E402


def _true(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _launch_setup(context):
    input_path = LaunchConfiguration("input").perform(context)
    if not input_path:
        raise RuntimeError(
            "input is required: a frame dir (with sweeps.txt), one .bin file "
            "or a 10-line manifest (input:=/path/...)")
    output_path = LaunchConfiguration("output").perform(context) or "/tmp/centerpoint_bev.jpg"

    merged_model = LaunchConfiguration("merged_model").perform(context)
    command = [
        os.path.join(
            get_package_prefix("topsbot_centerpoint"),
            "lib", "topsbot_centerpoint", "centerpoint_file_cli"),
        input_path,
        output_path,
    ]
    if merged_model:
        command += ["--merged", os.path.expanduser(merged_model)]
    else:
        pfe_model, rpn_model = ensure_centerpoint_models(
            context, {},
            LaunchConfiguration("pfe_model").perform(context),
            LaunchConfiguration("rpn_model").perform(context))
        command += ["--pfe", pfe_model, "--rpn", rpn_model]
    if _true(LaunchConfiguration("single_frame").perform(context)):
        command += ["--single-frame"]
    core = LaunchConfiguration("core").perform(context)
    if core:
        command += ["--core", core]
    score = LaunchConfiguration("score_threshold").perform(context)
    if score:
        command += ["--score-threshold", score]
    bev = LaunchConfiguration("bev_size").perform(context)
    if bev:
        command += ["--bev-size", bev]

    return [
        LogInfo(msg=f"[file_cli] input={input_path} output={output_path}"),
        ExecuteProcess(cmd=command, output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "input", default_value="",
            description="Frame dir / single .bin / 10-line sweeps manifest."),
        DeclareLaunchArgument(
            "output", default_value="",
            description="Output image (.jpg/.png); default /tmp/centerpoint_bev.jpg"),
        DeclareLaunchArgument(
            "pfe_model", default_value="",
            description="Override PFE .nb; empty resolves models_manifest.yaml."),
        DeclareLaunchArgument(
            "rpn_model", default_value="",
            description="Override RPN .nb; empty resolves models_manifest.yaml."),
        DeclareLaunchArgument(
            "merged_model", default_value="",
            description="Single-graph NB (PFE+ScatterND+RPN); overrides pfe/rpn."),
        DeclareLaunchArgument("models_dir", default_value="~/topsbot/models"),
        DeclareLaunchArgument("auto_download_models", default_value="true"),
        DeclareLaunchArgument("single_frame", default_value="false"),
        DeclareLaunchArgument("core", default_value=""),
        DeclareLaunchArgument("score_threshold", default_value=""),
        DeclareLaunchArgument("bev_size", default_value=""),
        OpaqueFunction(function=_launch_setup),
    ])
