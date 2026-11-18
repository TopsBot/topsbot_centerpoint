#!/usr/bin/env python3
# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

"""Launch single-frame streaming CenterPoint inference node with WebViz."""

import os
import sys
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

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

# Import dataset ensure helper (same mechanism as centerpoint.launch.py)
_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
try:
    from model_utils import ensure_dataset_present, ensure_models_present  # noqa: E402
    _has_dataset_utils = True
except Exception:
    _has_dataset_utils = False


def _launch_setup(context):
    """Setup launch nodes based on configuration."""
    pkg_dir = get_package_share_directory('topsbot_centerpoint')
    default_config = os.path.join(pkg_dir, 'config', 'centerpoint_frame_stream.yaml')

    config = LaunchConfiguration('config').perform(context)
    bin_files_dir = LaunchConfiguration('bin_files_dir').perform(context)
    bin_list_file = LaunchConfiguration('bin_list_file').perform(context)
    merged_model = LaunchConfiguration('merged_model').perform(context)
    frame_interval_ms = LaunchConfiguration('frame_interval_ms').perform(context)
    loop = LaunchConfiguration('loop').perform(context)
    device_id = LaunchConfiguration('device_id').perform(context)
    enable_webviz = LaunchConfiguration('enable_webviz').perform(context).lower() in ('true', '1', 'yes')

    # Ensure models and dataset exist before the node starts; auto-download when
    # missing. The merged model path has a default in centerpoint_frame_stream.yaml.
    models_dir = LaunchConfiguration('models_dir').perform(context)
    auto_download_models = LaunchConfiguration('auto_download_models').perform(context)
    datasets_dir = LaunchConfiguration('datasets_dir').perform(context)
    auto_download = LaunchConfiguration('auto_download_dataset').perform(context)
    if _has_dataset_utils:
        if not ensure_models_present(models_dir, auto_download_models):
            raise RuntimeError(
                f"CenterPoint models missing under {models_dir}; run "
                "scripts/download_models.sh or use auto_download_models:=true")
        if not ensure_dataset_present(datasets_dir, auto_download):
            raise RuntimeError(
                f"CenterPoint dataset missing under {datasets_dir}; run "
                "scripts/download_dataset.sh (or pass --archive /path/data.zip), "
                "or provide bin_files_dir/bin_list_file explicitly")

    # Build parameters dict (only non-empty values, so they override config
    # instead of clobbering its defaults with empty strings)
    params = {}
    if bin_files_dir:
        params['bin_files_dir'] = bin_files_dir
    if bin_list_file:
        params['bin_list_file'] = bin_list_file
    if merged_model:
        params['merged_model'] = merged_model

    params['frame_interval_ms'] = frame_interval_ms
    params['loop'] = loop
    params['device_id'] = device_id

    nodes = [
        Node(
            package='topsbot_centerpoint',
            executable='centerpoint_single_frame_stream',
            name='centerpoint_single_frame_stream',
            output='screen',
            parameters=[config, params],
        ),
    ]

    # Add webviz if enabled and available
    if enable_webviz:
        if not _has_webviz:
            print("WARNING: topsbot_webviz not found, webviz disabled")
        else:
            webviz_params = os.path.join(pkg_dir, "config", "centerpoint_webviz.yaml")
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
    """Generate launch description for single-frame streaming node."""
    pkg_dir = get_package_share_directory('topsbot_centerpoint')
    default_config = os.path.join(pkg_dir, 'config', 'centerpoint_frame_stream.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=default_config,
            description='Path to configuration YAML file'
        ),
        DeclareLaunchArgument(
            'merged_model',
            default_value='',
            description='Single-graph NB (PFE+ScatterND+RPN); overrides config'
        ),
        DeclareLaunchArgument(
            'bin_files_dir',
            default_value='',
            description='Directory containing .bin files (overrides config)'
        ),
        DeclareLaunchArgument(
            'bin_list_file',
            default_value='',
            description='Text file listing .bin file paths (overrides config)'
        ),
        DeclareLaunchArgument(
            'frame_interval_ms',
            default_value='100',
            description='Time between frames in milliseconds'
        ),
        DeclareLaunchArgument(
            'loop',
            default_value='true',
            description='Loop playback (true/false)'
        ),
        DeclareLaunchArgument(
            'device_id',
            default_value='0',
            description='NPU core ID'
        ),
        DeclareLaunchArgument(
            'enable_webviz',
            default_value='true',
            description='Enable webviz for browser visualization (true/false)'
        ),
        DeclareLaunchArgument(
            'http_port',
            default_value='8000',
            description='HTTP server port for webviz browser UI'
        ),
        DeclareLaunchArgument(
            'datasets_dir',
            default_value='~/topsbot/data',
            description='Directory dataset_manifest.yaml installs into.'
        ),
        DeclareLaunchArgument(
            'auto_download_dataset',
            default_value='true',
            description='Download data.zip when the dataset is missing.'
        ),
        DeclareLaunchArgument(
            'models_dir',
            default_value='~/topsbot/models',
            description='Directory models_manifest.yaml installs into.'
        ),
        DeclareLaunchArgument(
            'auto_download_models',
            default_value='true',
            description='Download missing manifest models from the share link.'
        ),
        OpaqueFunction(function=_launch_setup),
    ])