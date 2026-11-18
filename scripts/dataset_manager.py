#!/usr/bin/env python3
"""Check and download CenterPoint test datasets declared by dataset_manifest.yaml.

Shares the Aliyun-drive resolution code with model_manager.py but extracts the
archive STRUCTURE-PRESERVING: a dataset is a directory tree whose files repeat
basenames (frame_XXXX/sweeps.txt), so every member keeps its relative path.
``--archive`` supports offline installs from a local zip.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_manager import aliyun_download_url  # noqa: E402


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    required = data.get("required_paths") or []
    if not isinstance(required, list) or not required:
        raise RuntimeError("manifest required_paths must be a non-empty list")
    return data


def data_dir(manifest, override):
    path = override or os.environ.get("TOPSBOT_DATA_DIR") or manifest.get("install_dir")
    if not path:
        raise RuntimeError("dataset install directory is not configured")
    return os.path.abspath(os.path.expanduser(path))


def missing_paths(manifest, destination):
    return [
        rel for rel in manifest["required_paths"]
        if not os.path.isfile(os.path.join(destination, *rel.split("/")))
    ]


def _safe_parts(name):
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return parts


def extract_dataset(archive_path, destination, manifest):
    """Extract the whole archive structure-preserving into ``destination``.

    ``data.zip`` may ship its contents either directly (``seq_test/...``) or
    wrapped in a single top-level ``data/`` directory. The strip decision is
    anchored on the manifest ``required_paths``: if the required files only
    appear after dropping one leading component, that component is dropped, so
    the contents land directly under ``destination``. Every file is written via
    a ``.part`` file then ``os.replace``; paths are sanitized with
    ``_safe_parts`` so nothing escapes ``destination``.
    """
    required = [tuple(r.split("/")) for r in (manifest.get("required_paths") or [])]

    def present(rels):
        rel_set = {tuple(r) for r in rels if r}
        return required and all(r in rel_set for r in required)

    with zipfile.ZipFile(archive_path) as archive:
        members = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = _safe_parts(info.filename)
            if parts:
                members.append((info, parts))
        if not members:
            return 0

        rels = [parts for _, parts in members]
        strip = 0
        if required and not present(rels):
            # Only strip a wrapper when every member shares the same single
            # top-level directory and the required paths match without it.
            tops = {p[0] for p in rels}
            if len(tops) == 1 and all(len(p) >= 2 for p in rels):
                stripped = [p[1:] for p in rels]
                if present(stripped):
                    strip = 1

        written = 0
        for info, parts in members:
            rel = parts[strip:] if strip else parts
            if not rel:
                continue
            target = os.path.join(destination, *rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = target + ".part"
            with archive.open(info) as source, open(tmp, "wb") as output:
                shutil.copyfileobj(source, output)
            os.replace(tmp, target)
            written += 1
    return written


def download_dataset(args):
    manifest = load_manifest(args.manifest)
    destination = data_dir(manifest, args.install_dir)
    os.makedirs(destination, exist_ok=True)

    if args.archive:
        archive_path = os.path.abspath(os.path.expanduser(args.archive))
        if not os.path.isfile(archive_path):
            raise RuntimeError("archive not found: " + archive_path)
    else:
        share_url = args.url or manifest.get("models_zip_url")
        if not share_url or "REPLACE_WITH" in share_url:
            raise RuntimeError(
                "dataset link not configured: set models_zip_url in dataset_manifest.yaml "
                "or pass --archive /path/to/dataset.zip")
        print("[INFO] Downloading CenterPoint dataset to " + destination, file=sys.stderr)
        url = aliyun_download_url(share_url, args.debug)
        fd, archive_path = tempfile.mkstemp(prefix="topsbot_centerpoint_ds_", suffix=".zip")
        os.close(fd)
        import urllib.request
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "topsbot-centerpoint-model-manager/1.0"})
            with urllib.request.urlopen(request, timeout=1800) as response, \
                    open(archive_path, "wb") as output:
                shutil.copyfileobj(response, output)
        except Exception:
            os.unlink(archive_path)
            raise

    try:
        count = extract_dataset(archive_path, destination, manifest)
    finally:
        if not args.archive:
            try:
                os.unlink(archive_path)
            except FileNotFoundError:
                pass
    print("[INFO] Extracted %d dataset files under %s" % (count, destination),
          file=sys.stderr)

    missing = missing_paths(manifest, destination)
    if missing:
        raise RuntimeError("dataset archive is missing required paths: " + ", ".join(missing))
    return 0


def ensure_dataset(args):
    manifest = load_manifest(args.manifest)
    destination = data_dir(manifest, args.install_dir)
    missing = missing_paths(manifest, destination)
    if not missing:
        print("[INFO] Dataset ready under " + destination)
        return 0
    print("[INFO] Missing dataset paths under %s: %s" % (destination, " ".join(missing)),
          file=sys.stderr)
    enabled = str(args.auto_download).lower() in ("true", "1", "yes") and not args.no_download
    if not enabled:
        raise RuntimeError("dataset files are missing and auto download is disabled")
    return download_dataset(argparse.Namespace(
        manifest=args.manifest, install_dir=destination,
        url=None, archive=args.archive, debug=args.debug))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--install-dir", "--datasets-dir", dest="install_dir")
    download.add_argument("--url")
    download.add_argument("--archive", help="local zip; skips share download")
    download.add_argument("--debug", action="store_true")
    download.set_defaults(handler=download_dataset)
    ensure = sub.add_parser("ensure")
    ensure.add_argument("--manifest", required=True)
    ensure.add_argument("--install-dir", "--datasets-dir", dest="install_dir")
    ensure.add_argument("--archive")
    ensure.add_argument("--auto-download", default="true")
    ensure.add_argument("--no-download", action="store_true")
    ensure.add_argument("--debug", action="store_true")
    ensure.set_defaults(handler=ensure_dataset)
    return parser


def main():
    args = make_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print("[ERROR] " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
