# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def resolve_params_file(package_share: str, params_file: str) -> str:
    path = params_file if os.path.isabs(params_file) else os.path.join(
        package_share, "config", params_file)
    if not os.path.isfile(path):
        raise RuntimeError(f"topsbot_centerpoint params file not found: {path}")
    return path


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("/**", {}).get("ros__parameters", data)


def param_value(params: dict, key: str, default=None):
    if key in params:
        return params[key]
    value = params
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _load_manifest(package_share: str) -> dict:
    with open(os.path.join(package_share, "config", "models_manifest.yaml"),
              "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_dataset_manifest(package_share: str) -> dict:
    with open(os.path.join(package_share, "config", "dataset_manifest.yaml"),
              "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_models_present(models_dir: str, auto_download: str) -> bool:
    """Ensure every manifest ``required_models`` file exists (and is non-empty)
    under ``models_dir``; auto-downloads when missing. Returns readiness."""
    package_share = get_package_share_directory("topsbot_centerpoint")
    manifest = _load_manifest(package_share)
    required = manifest.get("required_models") or []
    models_dir = os.path.expanduser(models_dir)

    def all_present():
        return bool(required) and all(
            os.path.isfile(os.path.join(models_dir, name)) and
            os.path.getsize(os.path.join(models_dir, name)) > 0
            for name in required)

    if all_present():
        return True
    script = os.path.join(package_share, "scripts", "ensure_models.sh")
    result = subprocess.run(
        [script, "--models-dir", models_dir, "--auto-download", auto_download],
        check=False)
    return result.returncode == 0 and all_present()


def ensure_centerpoint_models(context, params: dict, explicit_pfe: str,
                              explicit_rpn: str):
    """Resolve (pfe_path, rpn_path); auto-download missing manifest models.

    Precedence per stage: explicit launch arg -> yaml centerpoint.*_model ->
    manifest required_models order ([0]=PFE, [1]=RPN) under models_dir.
    When both candidates already exist on disk nothing is downloaded.
    """
    package_share = get_package_share_directory("topsbot_centerpoint")
    manifest = _load_manifest(package_share)
    required = manifest.get("required_models") or []
    if len(required) < 2:
        raise RuntimeError("models_manifest.yaml must list [pfe, rpn] in required_models")
    models_dir = os.path.expanduser(LaunchConfiguration("models_dir").perform(context))
    auto_download = LaunchConfiguration("auto_download_models").perform(context)

    def candidate(explicit: str, yaml_key: str, fallback: str) -> str:
        value = (explicit or str(param_value(params, yaml_key, "")) or
                 os.path.join(models_dir, fallback))
        return os.path.expanduser(value)

    pfe = candidate(explicit_pfe, "centerpoint.pfe_model", required[0])
    rpn = candidate(explicit_rpn, "centerpoint.rpn_model", required[1])
    if os.path.isfile(pfe) and os.path.isfile(rpn):
        return pfe, rpn

    script = os.path.join(package_share, "scripts", "ensure_models.sh")
    result = subprocess.run(
        [script, "--models-dir", models_dir, "--auto-download", auto_download],
        check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"CenterPoint models missing under {models_dir}; "
            "run scripts/download_models.sh or use auto_download_models:=true")

    def finalize(path: str) -> str:
        if os.path.isfile(path):
            return path
        alt = os.path.join(models_dir, os.path.basename(path))
        if os.path.isfile(alt):
            return alt
        raise RuntimeError(f"CenterPoint model file not found after model check: {path}")

    return finalize(pfe), finalize(rpn)


def ensure_dataset_present(datasets_dir: str, auto_download: str) -> bool:
    """Ensure the dataset tree declared by dataset_manifest.yaml (its
    ``required_paths``) is present under ``datasets_dir``.

    Auto-downloads ``data.zip`` when any entry is missing and ``auto_download``
    is enabled. Returns True when the dataset is present after the check.
    """
    package_share = get_package_share_directory("topsbot_centerpoint")
    manifest = _load_dataset_manifest(package_share)
    required = manifest.get("required_paths") or []
    datasets_dir = os.path.expanduser(datasets_dir)

    def all_present():
        return bool(required) and all(
            os.path.isfile(os.path.join(datasets_dir, *rel.split("/")))
            for rel in required)

    if all_present():
        return True
    script = os.path.join(package_share, "scripts", "ensure_dataset.sh")
    result = subprocess.run(
        [script, "--datasets-dir", datasets_dir, "--auto-download", auto_download],
        check=False)
    return result.returncode == 0 and all_present()


def ensure_centerpoint_dataset(context, params: dict, explicit_frames: str,
                               explicit_sweeps: str):
    """Resolve (frames_list, sweeps_manifest); auto-download when missing.

    Precedence: explicit launch arg -> yaml centerpoint.frames_list /
    centerpoint.sweeps_manifest (when the file exists) -> manifest
    default_frames_list under datasets_dir. Returns at most one non-empty
    pair; downloads only when the chosen entry is not on disk.
    """
    package_share = get_package_share_directory("topsbot_centerpoint")
    manifest = _load_manifest(package_share)
    dataset_manifest = _load_dataset_manifest(package_share)

    frames = os.path.expanduser(explicit_frames or
                                str(param_value(params, "centerpoint.frames_list", "")))
    sweeps = os.path.expanduser(explicit_sweeps or
                                str(param_value(params, "centerpoint.sweeps_manifest", "")))
    if (frames and os.path.isfile(frames)) or (sweeps and os.path.isfile(sweeps)):
        return frames, sweeps

    datasets_dir = os.path.expanduser(
        LaunchConfiguration("datasets_dir").perform(context))
    default_rel = dataset_manifest.get("default_frames_list") or \
        "seq_test/board_manifest.txt"
    default_path = os.path.join(datasets_dir, *default_rel.split("/"))

    auto_download = LaunchConfiguration("auto_download_dataset").perform(context)
    if not os.path.isfile(default_path):
        ensure_dataset_present(datasets_dir, auto_download)
        if not os.path.isfile(default_path):
            raise RuntimeError(
                f"CenterPoint dataset missing under {datasets_dir}; run "
                "scripts/download_dataset.sh (or pass --archive /path/data.zip), "
                "or provide frames_list/sweeps_manifest explicitly")

    return default_path, ""
