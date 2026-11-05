#!/usr/bin/env python3
"""Copy only this file, edit MODEL_NAME, then run: python3 test.py.

All evaluator code is embedded. Folders, a private Python environment and the
fixed COCO subset are prepared for the current workspace/SSH account.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit
import venv
import zlib

# -------- 이 부분만 변경하세요 --------
MODEL_NAME = "user1_v1.onnx"
GITHUB_REPOSITORY = "foxyaef/rtdetr-raspberrypi-eval"
UPLOAD_TO_GITHUB = True
CPU_THREADS = 4
LATENCY_REPEATS = 1
# ------------------------------------
BUNDLE_SHA256 = "826c6cd08479b99f7aac7925d5daaf69ef41cd472f61e957807e616cb70a3937"
BUNDLE_DATA = (
    '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'
)
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
EXPECTED_FILES = frozenset({
    "code/benchmark.py", "code/coco_data.py", "code/coco_metrics.py",
    "code/github_upload.py", "protocol.json", "requirements.txt",
})
ENVIRONMENT_MARKER = ".rtdetr_bootstrap_environment.json"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _is_link(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def workspace_root(script_path=None):
    script = Path(script_path if script_path is not None else __file__).resolve()
    return script.parent.parent if script.parent.name == "code" else script.parent


def _private_directory(path):
    if _is_link(path):
        raise RuntimeError(f"Refusing a symlink/junction for managed directory: {path}")
    path.mkdir(mode=0o700, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Expected a directory: {path}")


def decode_bundle():
    try:
        compressed = base64.b64decode(BUNDLE_DATA, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, MAX_BUNDLE_BYTES + 1)
        if len(raw) > MAX_BUNDLE_BYTES or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Oversized, incomplete or trailing compressed payload")
        if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
            raise ValueError("Embedded bundle SHA256 mismatch")
        bundle = json.loads(raw)
        if bundle.get("format") != 1 or set(bundle.get("files", {})) != EXPECTED_FILES:
            raise ValueError("Unexpected embedded file list")
        for name, entry in bundle["files"].items():
            if not isinstance(entry.get("content"), str):
                raise ValueError("Embedded file must contain UTF-8 text")
            if hashlib.sha256(entry["content"].encode("utf-8")).hexdigest() != entry.get("sha256"):
                raise ValueError(f"Embedded file hash mismatch: {name}")
        return bundle
    except (ValueError, TypeError, KeyError, zlib.error) as error:
        raise RuntimeError("This test.py is incomplete or damaged; copy the original file again.") from error


def _atomic_bytes(path, payload):
    if _is_link(path):
        raise RuntimeError(f"Refusing to replace a symlink/junction: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".rtdetr_", dir=str(path.parent))
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def materialize_bundle(root):
    root = Path(root)
    bundle = decode_bundle()
    private = root / ".rtdetr_runtime"
    _private_directory(private)
    destination = private / BUNDLE_SHA256
    _private_directory(destination)
    _private_directory(destination / "code")
    for name, entry in bundle["files"].items():
        target = destination.joinpath(*name.split("/"))
        if _is_link(target):
            raise RuntimeError(f"Managed runtime file is a symlink/junction: {target}")
        payload = entry["content"].encode("utf-8")
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Managed runtime changed: {target}. Preserve your edits and use a fresh workspace.")
        else:
            _atomic_bytes(target, payload)
    _write_json(destination / "bundle_manifest.json", {
        "format": 1, "bundle_sha256": BUNDLE_SHA256,
        "files": {name: item["sha256"] for name, item in bundle["files"].items()},
    })
    return destination


def parse_requirements(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match: raise RuntimeError("Only pinned package==version requirements are allowed")
        result[match[1]] = match[2]
    return result


def _child_environment():
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONUSERBASE", None)
    # Ambient pip config can redirect installs outside a virtual environment.
    for name in list(env):
        if name.startswith("PIP_"): env.pop(name)
    env.update({"PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
                "PIP_CONFIG_FILE": os.devnull, "PIP_REQUIRE_VIRTUALENV": "true"})
    return env


def environment_versions(python, expected):
    source = (
        "import importlib.metadata as m,json,sys; names=json.loads(sys.argv[1]); out={}; "
        "exec('for n in names:\\n try: out[n]=m.version(n)\\n except m.PackageNotFoundError: out[n]=None'); "
        "print(json.dumps(out))"
    )
    result = subprocess.run([str(python), "-c", source, json.dumps(list(expected))],
                            env=_child_environment(), capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Could not inspect the selected Python environment: " + result.stderr[-1500:])
    return json.loads(result.stdout)


def _download_get_pip(path):
    request = urllib.request.Request(GET_PIP_URL, headers={"User-Agent": "RT-DETR-standalone-setup/1"})
    descriptor, temporary = tempfile.mkstemp(prefix=".get_pip_", dir=str(path.parent))
    temporary = Path(temporary)
    try:
        size = 0
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=90) as response:
            endpoint = urlsplit(response.geturl())
            if endpoint.scheme != "https" or endpoint.hostname != "bootstrap.pypa.io":
                raise RuntimeError("Unexpected redirect for pip bootstrap")
            while block := response.read(256 * 1024):
                size += len(block)
                if size > 8 * 1024 * 1024: raise RuntimeError("Unexpected pip bootstrap download size")
                output.write(block)
        if size < 1000: raise RuntimeError("Incomplete pip bootstrap download")
        if _is_link(path): raise RuntimeError("pip bootstrap destination cannot be a symlink/junction")
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()
    return {"url": GET_PIP_URL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _verify_private_environment(python, directory):
    source = "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:2]),'bits':64 if sys.maxsize>2**32 else 32}))"
    result = subprocess.run([str(python), "-c", source], env=_child_environment(),
                            capture_output=True, text=True, timeout=60)
    if result.returncode: raise RuntimeError("Private Python interpreter did not start")
    info = json.loads(result.stdout)
    if (Path(info["prefix"]).resolve() != directory.resolve()
            or Path(info["prefix"]).resolve() == Path(info["base_prefix"]).resolve()
            or info["bits"] != 64 or not (3, 11) <= tuple(info["version"]) <= (3, 13)):
        raise RuntimeError("Private Python does not identify this folder as its virtual environment; no packages were installed")


def ensure_environment(root, runtime, use_current=False):
    root, runtime = Path(root), Path(runtime)
    expected = parse_requirements(runtime / "requirements.txt")
    if use_current:
        actual = environment_versions(Path(sys.executable), expected)
        if actual != expected:
            raise RuntimeError("--use-current-python requires all exact package versions. Remove the option for automatic private-environment setup.")
        _write_json(runtime / "installer_provenance.json", {
            "mode": "explicit-current-python", "python": sys.executable,
            "versions": actual, "system_python_modified": False,
        })
        return Path(sys.executable)
    directory = root / ".rtdetr_venv"
    marker = directory / ENVIRONMENT_MARKER
    owned = {}
    if _is_link(directory): raise RuntimeError("Private environment directory cannot be a symlink/junction")
    if directory.exists():
        if not directory.is_dir(): raise RuntimeError("Private environment path is not a directory")
        if any(directory.iterdir()):
            if _is_link(marker) or not marker.is_file():
                raise RuntimeError("Existing .rtdetr_venv is not owned by this bootstrap; it will not be changed")
            try: owned = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error: raise RuntimeError("Invalid private environment marker") from error
            if owned.get("purpose") != "rtdetr-standalone" or owned.get("format") != 1:
                raise RuntimeError("Existing .rtdetr_venv has an unrecognized owner marker")
    else:
        directory.mkdir(mode=0o700)
    sources = [item for item in owned.get("bootstrap_downloads", [])
               if isinstance(item, dict) and item.get("url") == GET_PIP_URL
               and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))]
    _write_json(marker, {"format": 1, "purpose": "rtdetr-standalone", "bootstrap_downloads": sources})
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Preparing private Python environment:", directory, flush=True)
        try: venv.EnvBuilder(with_pip=False, clear=False).create(directory)
        except Exception as error:
            raise RuntimeError("Python venv support is unavailable. An administrator must install python3-venv once for the device; user files were preserved.") from error
    _verify_private_environment(python, directory)
    actual = environment_versions(python, expected)
    if actual != expected:
        available = subprocess.run([str(python), "-m", "pip", "--version"],
                                   env=_child_environment(), capture_output=True, text=True)
        if available.returncode:
            seeded = subprocess.run([str(python), "-m", "ensurepip", "--upgrade"],
                                    env=_child_environment(), capture_output=True, text=True)
            if seeded.returncode:
                print("Installing pip inside the private environment using official PyPA bootstrap.", flush=True)
                get_pip = runtime / "get-pip.py"
                sources.append(_download_get_pip(get_pip))
                subprocess.run([str(python), str(get_pip), "--index-url", "https://pypi.org/simple"],
                               env=_child_environment(), check=True)
        print("Installing the pinned evaluation packages (first setup/version change only).", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                        "--index-url", "https://pypi.org/simple", "-r", str(runtime / "requirements.txt")],
                       env=_child_environment(), check=True)
        actual = environment_versions(python, expected)
        if actual != expected: raise RuntimeError("Installed packages do not match the pinned versions")
    subprocess.run([str(python), "-m", "pip", "check"], env=_child_environment(), check=True)
    _write_json(marker, {"format": 1, "purpose": "rtdetr-standalone", "bootstrap_downloads": sources})
    _write_json(runtime / "installer_provenance.json", {
        "mode": "private-venv", "python": str(python), "versions": actual,
        "system_python_modified": False, "bootstrap_downloads": sources,
    })
    return python


@contextmanager
def setup_lock(root):
    lock = Path(root) / ".rtdetr_setup.lock"
    try: descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Setup is already running in this workspace. If an earlier setup was interrupted, verify no setup is active before removing .rtdetr_setup.lock.") from error
    try:
        with os.fdopen(descriptor, "w") as stream: stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


HELP = """RT-DETR standalone evaluator - copy only this test.py.

python3 test.py                         prepare folders/packages/COCO, then evaluate
python3 test.py --setup-only            prepare environment and images only
python3 test.py --model user2_v1.onnx    select another model in models/
python3 test.py --limit 3 --no-upload    first smoke check without GitHub upload
python3 test.py --repeats 3             repeat inference timing
python3 test.py --upload-only results/<run>  retry an existing upload

Optional: --data-dir PATH, --no-download, --threads N, --repo owner/repository.
--use-current-python skips private-venv creation only if all pinned packages match.

The workspace is the script's folder, or the parent folder when the script is
inside code/. Each SSH account/workspace has independent data/results/environment.
Git/Git LFS are device-level tools; GitHub write authentication remains per account.
Without a power meter electrical power is recorded as null.
"""


def launch(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        print(HELP)
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use-current-python", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    options, forwarded = parser.parse_known_args(arguments)
    if sys.maxsize <= 2**32 or not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Use 64-bit Python 3.11, 3.12 or 3.13")
    root = workspace_root()
    for name in ("code", "models", "data", "results"):
        _private_directory(root / name)
    with setup_lock(root):
        runtime = materialize_bundle(root)
        python = ensure_environment(root, runtime, options.use_current_python)
    if options.setup_only and "--prepare-only" not in forwarded: forwarded.append("--prepare-only")
    environment = _child_environment()
    environment.update({
        "RTDETR_WORKSPACE_ROOT": str(root), "RTDETR_PROTOCOL_PATH": str(runtime / "protocol.json"),
        "RTDETR_MODEL_NAME": MODEL_NAME, "RTDETR_REPO": GITHUB_REPOSITORY,
        "RTDETR_UPLOAD": "1" if UPLOAD_TO_GITHUB else "0", "RTDETR_THREADS": str(CPU_THREADS),
        "RTDETR_REPEATS": str(LATENCY_REPEATS), "RTDETR_BUNDLE_SHA256": BUNDLE_SHA256,
        "RTDETR_LAUNCHER_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print("Workspace:", root, flush=True)
    return subprocess.call([str(python), "-u", str(runtime / "code/benchmark.py"), *forwarded], env=environment)


if __name__ == "__main__":
    try:
        raise SystemExit(launch())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print("Setup/evaluation could not complete:", str(error), file=sys.stderr)
        raise SystemExit(1)
