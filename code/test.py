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
BUNDLE_SHA256 = "f2baa8f2e752cdeeedd6a816e2019e87917fc53da04baa1ef69f565a18838958"
BUNDLE_DATA = (
    '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'
)
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
EXPECTED_FILES = frozenset({
    "code/benchmark.py", "code/coco_data.py", "code/coco_metrics.py",
    "code/github_upload.py", "code/model_queue.py", "protocol.json", "requirements.txt",
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
python3 test.py --queue                 evaluate queue/inbox/ sequentially
python3 test.py --queue --no-upload     queue with local ZIP output only
python3 test.py --queue --models user2_v1.onnx user1_v1.onnx  explicit order
python3 test.py --queue --retry-failed  retry failed/interrupted jobs
python3 test.py --queue-status         inspect saved queue without setup
python3 test.py --watch                keep watching queue/inbox/ for new models

Optional: --data-dir PATH, --no-download, --threads N, --repo owner/repository.
--use-current-python skips private-venv creation only if all pinned packages match.

The workspace is the script's folder, or the parent folder when the script is
inside code/. All queued models in one workspace share data and environment.
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
    for name in ("code", "models", "data", "results", "queue"):
        _private_directory(root / name)
    _private_directory(root / "queue/inbox")
    if "--queue-status" in forwarded:
        state = root / "queue/state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "Queue has not been started.")
        return 0
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
