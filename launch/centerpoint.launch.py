# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import (  # noqa: E402
    ensure_centerpoint_dataset,
    ensure_centerpoint_models,
    load_params,
    resolve_params_file,
)

# Import webviz utilities if available
try:
    webviz_share = get_package_share_directory('topsbot_webviz')
    _webviz_launch_dir = os.path.join(webviz_share, 'launch')
    if _webviz_launch_dir not in sys.path:
        sys.path.insert(0, _webviz_launch_dir)
    from web_http_utils import load_ros_params, prepare_web_http_root
    _has_webviz = True
except Exception:
    _has_webviz = False


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_centerpoint")
    params_file = resolve_params_file(
        package_share, LaunchConfiguration("params_file").perform(context))
    params = load_params(params_file)
    overrides = {}
    pfe_model, rpn_model = ensure_centerpoint_models(
        context, params,
        LaunchConfiguration("pfe_model").perform(context),
        LaunchConfiguration("rpn_model").perform(context))
    overrides["centerpoint.pfe_model"] = pfe_model
    overrides["centerpoint.rpn_model"] = rpn_model
    frames_list, sweeps_manifest = ensure_centerpoint_dataset(
        context, params,
        LaunchConfiguration("frames_list").perform(context),
        LaunchConfiguration("sweeps_manifest").perform(context))
    if frames_list:
        overrides["centerpoint.frames_list"] = frames_list
    if sweeps_manifest:
        overrides["centerpoint.sweeps_manifest"] = sweeps_manifest
    for launch_name, parameter_name in (
        ("run_interval_ms", "centerpoint.run_interval_ms"),
    ):
        value = LaunchConfiguration(launch_name).perform(context)
        if value:
            overrides[parameter_name] = value

    # Check if webviz should be enabled
    enable_webviz = LaunchConfiguration("enable_webviz").perform(context).lower() in ('true', '1', 'yes')

    nodes = [
        Node(
            package="topsbot_centerpoint",
            executable="centerpoint_infer",
            name="centerpoint_infer",
            output="screen",
            parameters=[params_file, overrides],
        )
    ]

    # Add webviz if enabled and available
    if enable_webviz:
        if not _has_webviz:
            print("WARNING: topsbot_webviz not found, webviz disabled")
        else:
            webviz_params = os.path.join(package_share, "config", "centerpoint_webviz.yaml")
            http_port = LaunchConfiguration("http_port").perform(context)

            try:
                yaml_params = load_ros_params(webviz_params)
                http_root = prepare_web_http_root(webviz_share, yaml_params)

                nodes.extend([
                    Node(
                        package='topsbot_webviz',
                        executable='webviz_node',
                        name='centerpoint_webviz',
                        output='screen',
                        parameters=[webviz_params],
                    ),
                    ExecuteProcess(
                        cmd=['python3', '-m', 'http.server', http_port],
                        cwd=http_root,
                        output='screen',
                    ),
                ])
                print(f"INFO: Webviz enabled, browser: http://<board-ip>:{http_port}/")
            except Exception as e:
                print(f"WARNING: Failed to setup webviz: {e}")

    return nodes


def generate_launch_description():
    package_share = get_package_share_directory("topsbot_centerpoint")
    default_params = os.path.join(package_share, "config", "ai_infer_centerpoint.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="Path to the centerpoint_infer parameter file."),
        DeclareLaunchArgument(
            "pfe_model", default_value="",
            description="Override centerpoint.pfe_model."),
        DeclareLaunchArgument(
            "rpn_model", default_value="",
            description="Override centerpoint.rpn_model."),
        DeclareLaunchArgument(
            "sweeps_manifest", default_value="",
            description="Override centerpoint.sweeps_manifest."),
        DeclareLaunchArgument(
            "frames_list", default_value="",
            description="Override centerpoint.frames_list (path to board_manifest.txt for video sequence)."),
        DeclareLaunchArgument(
            "run_interval_ms", default_value="",
            description="Override centerpoint.run_interval_ms (frame interval in ms, e.g. 100 for 10fps)."),
        DeclareLaunchArgument(
            "models_dir", default_value="~/topsbot/models",
            description="Directory the models manifest is installed into."),
        DeclareLaunchArgument(
            "auto_download_models", default_value="true",
            description="Download missing manifest models from the share link."),
        DeclareLaunchArgument(
            "datasets_dir", default_value="~/topsbot/data",
            description="Directory dataset_manifest.yaml installs into."),
        DeclareLaunchArgument(
            "auto_download_dataset", default_value="true",
            description="Download the seq_test dataset when not specified and missing."),
        DeclareLaunchArgument(
            "enable_webviz", default_value="true",
            description="Enable webviz for browser visualization (true/false)."),
        DeclareLaunchArgument(
            "http_port", default_value="8000",
            description="HTTP server port for webviz browser UI."),
        OpaqueFunction(function=_launch_setup),
    ])