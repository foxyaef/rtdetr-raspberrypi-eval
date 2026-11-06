"""Upload one self-contained ONNX model and its benchmark run to its user branch.

Git authentication must already be configured outside this program. Models are
always stored with Git LFS; result files are normal Git objects. The caller's
checkout and original result directory are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from urllib.parse import quote

DEFAULT_REPOSITORY = "foxyaef/rtdetr-raspberrypi-eval"
USER_PREFIX = re.compile(r"^(user[0-9]+)_")
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
RESULT_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".log", ".md", ".jpg", ".jpeg", ".png", ".webp", ".zip"}
PRIVATE_NAMES = {".git", ".env", ".ssh", ".netrc", "_netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
GIT_TIMEOUT_SECONDS = 1800


class UploadError(RuntimeError):
    """An upload failed; the original local results remain available."""


def _sanitize(value: str) -> str:
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", str(value))
    text = re.sub(r"(authorization:)\s*[^\r\n]+", r"\1 [redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"([?&](?:access_token|token)=)[^\s&]+", r"\1[redacted]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[redacted]", text)
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GIT_HTTP_PASSWORD"):
        secret = os.environ.get(name)
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[redacted]")
    return text[-6000:]


def infer_user(model_path: Path | str) -> str:
    """Route optional userN_ prefix; unrestricted model names use shared."""
    name = Path(model_path).name
    if (name.startswith('.') or Path(name).suffix.lower() not in ('.onnx','.onix')
            or len(name.encode('utf-8')) > 200
            or any(ord(char)<32 or ord(char)==127 or char in '\\/:*?"<>|' for char in name)):
        raise UploadError("Use a regular ONNX filename (no hidden name, control/path characters; max 200 UTF-8 bytes). A user prefix is optional.")
    match = USER_PREFIX.match(name)
    return match.group(1) if match else 'shared'


def _is_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _validate_onnx_file(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise UploadError("Install the project requirements (onnx is required for safe model validation).") from exc
    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise UploadError("Model is not a readable ONNX file: " + _sanitize(str(exc))) from exc

    def visit(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise UploadError("External-weight ONNX is not supported. Export one self-contained ONNX file first.")
            return
        for descriptor, value in message.ListFields():
            if descriptor.message_type is None:
                continue
            if descriptor.is_repeated:
                for child in value:
                    visit(child)
            else:
                visit(value)
    visit(model)
    del model


def _validate_inputs(model_path: Path | str, result_dir: Path | str, repo: str, branch: str | None):
    model = Path(model_path).expanduser().absolute()
    results = Path(result_dir).expanduser().absolute()
    user = infer_user(model)
    if branch is not None and branch != user:
        raise UploadError("Branch must match the inferred userN prefix or shared for a model without that prefix.")
    if not REPO_PATTERN.fullmatch(repo) or ".." in repo:
        raise UploadError("Repository must be a GitHub owner/repository name, without credentials or a URL.")
    if _is_link(model) or not model.is_file():
        raise UploadError("Model must be an existing regular file, not a symlink or junction.")
    if _is_link(results) or not results.is_dir():
        raise UploadError("Results must be an existing directory, not a symlink or junction.")
    if not SLUG_PATTERN.fullmatch(results.name) or ".." in results.name:
        raise UploadError("Run directory name must contain only letters, digits, underscore, dot or hyphen.")
    files = []
    for folder, directories, names in os.walk(results, followlinks=False):
        current = Path(folder)
        for name in directories + names:
            item = current / name
            if _is_link(item):
                raise UploadError("Result tree contains a symlink or junction; upload refused.")
            if name.lower() in PRIVATE_NAMES or name.lower().startswith(".env"):
                raise UploadError("Result tree contains a credential/configuration path; upload refused.")
        for name in names:
            path = current / name
            if path == results / "upload_status.json":
                # The caller writes this receipt after the remote commit.
                # Excluding it keeps an upload-only retry idempotent.
                continue
            if not path.is_file() or path.suffix.lower() not in RESULT_EXTENSIONS:
                raise UploadError("Unsupported result file type: " + _sanitize(path.name))
            files.append(path.relative_to(results))
    if not files:
        raise UploadError("Result directory is empty.")
    _validate_onnx_file(model)
    return model, results, user, sorted(files)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Remove inherited tracing rather than setting "0": some clients treat
    # a nonempty verbose variable as enabled and could print HTTP headers.
    for name in ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE", "GIT_TRACE_PACKET",
                 "GIT_TRACE2", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF", "GIT_TRACE_SETUP",
                 "GIT_TRACE_PERFORMANCE", "GIT_TRANSFER_TRACE"):
        environment.pop(name, None)
    environment.update({"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    return environment


def _run(args: list[str], cwd: Path | None = None, *, check: bool = True):
    try:
        result = subprocess.run(["git", *args], cwd=cwd, env=_environment(), text=True,
                                encoding="utf-8", errors="replace", capture_output=True,
                                timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UploadError("Git command could not complete: " + _sanitize(str(exc))) from exc
    if check and result.returncode:
        raise UploadError("Git operation failed. Check GitHub write access and Git LFS setup.\n" + _sanitize(result.stderr or result.stdout))
    return result


def _remote_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _add_lfs_attributes(checkout: Path, model_extension: str = "onnx") -> None:
    attributes = checkout / ".gitattributes"
    if _is_link(attributes):
        raise UploadError("Remote .gitattributes must not be a symlink.")
    content = attributes.read_text(encoding="utf-8") if attributes.exists() else ""
    for extension in dict.fromkeys(("onnx", "onix", model_extension)):
        rule = f"models/*.{extension} filter=lfs diff=lfs merge=lfs -text"
        if rule not in content.splitlines():
            content = content.rstrip("\n") + ("\n" if content else "") + rule + "\n"
    attributes.write_text(content, encoding="utf-8", newline="\n")


def _assert_safe_destination(path: Path, root: Path) -> None:
    for part in [path, *path.parents]:
        if part == root:
            return
        if part.exists() and _is_link(part):
            raise UploadError("Remote upload destination contains a symlink or junction.")
    raise UploadError("Upload destination escapes its isolated checkout.")


def _non_fast_forward(result) -> bool:
    text = (result.stderr + "\n" + result.stdout).lower()
    return "non-fast-forward" in text or "fetch first" in text or ("[rejected]" in text and "failed to push" in text)


def _receipt(repo: str, user: str, commit: str, model: Path, model_sha: str,
             run_id: str, *, already_present: bool = False) -> dict:
    base = f"https://github.com/{repo}"
    result_path = f"results/{user}/{run_id}"
    return {"status": "uploaded", "repository": repo, "branch": user,
            "commit_sha": commit, "model_sha256": model_sha, "run_id": run_id,
            "already_present": already_present, "result_path": result_path,
            "github_branch_url": f"{base}/tree/{user}",
            "github_results_url": f"{base}/tree/{commit}/{quote(result_path, safe='/')}",
            "github_model_url": f"{base}/blob/{commit}/models/{quote(model.name)}"}


def _same_existing_run(checkout: Path, destination_run: Path, result_hashes: dict,
                       model_relative: str, model_sha: str) -> bool:
    existing = {}
    for folder, directories, names in os.walk(destination_run, followlinks=False):
        current = Path(folder)
        if any(_is_link(current / name) for name in directories + names):
            return False
        for name in names:
            path = current / name
            if not path.is_file():
                return False
            existing[str(path.relative_to(destination_run)).replace("\\", "/")] = _sha256(path)
    if existing != result_hashes:
        return False
    pointer = _model_pointer(checkout, model_relative)
    return pointer is not None and f"oid sha256:{model_sha}" in pointer.splitlines()


def _model_pointer(checkout: Path, model_relative: str) -> str | None:
    size = _run(["cat-file", "-s", f"HEAD:{model_relative}"], checkout, check=False)
    if size.returncode or not size.stdout.strip().isdigit() or int(size.stdout.strip()) > 1024:
        return None
    content = _run(["show", f"HEAD:{model_relative}"], checkout, check=False)
    if content.returncode or not content.stdout.startswith("version https://git-lfs.github.com/spec/v1\n"):
        return None
    return content.stdout


def upload_run(model_path: Path | str, result_dir: Path | str,
               repo: str = DEFAULT_REPOSITORY, branch: str | None = None,
               archive_path: Path | str | None = None) -> dict:
    """Upload one completed run and model to its inferred user branch via Git LFS.

    Authentication is supplied by the caller's existing Git credential helper.
    A failed upload raises UploadError and leaves model/results intact locally.
    Concurrent branch changes receive at most three normal push attempts.
    """
    model, results, user, files = _validate_inputs(model_path, result_dir, repo, branch)
    sources = {str(relative).replace("\\", "/"): results / relative for relative in files}
    if archive_path is not None:
        archive = Path(archive_path).absolute()
        if (_is_link(archive) or not archive.is_file() or archive.suffix != ".zip"
                or not zipfile.is_zipfile(archive)):
            raise UploadError("Result archive must be an existing regular ZIP file.")
        if "results.zip" in sources: raise UploadError("Reserved remote filename results.zip already exists")
        # Only accept the sibling produced by the evaluator, never an arbitrary file.
        if archive.resolve() != Path(str(results.resolve()) + ".zip"):
            raise UploadError("Result archive must be the benchmark directory's sibling ZIP")
        with zipfile.ZipFile(archive) as bundle:
            expected = {name: _sha256(path) for name, path in sources.items()}
            infos = bundle.infolist()
            if len(infos) != len(expected) or {item.filename for item in infos} != set(expected):
                raise UploadError("Result ZIP contents do not match the result directory")
            for item in infos:
                digest = hashlib.sha256()
                with bundle.open(item) as stream:
                    for part in iter(lambda: stream.read(1024 * 1024), b""): digest.update(part)
                if digest.hexdigest() != expected[item.filename]:
                    raise UploadError("Result ZIP contents differ from the result directory")
        sources["results.zip"] = archive
    model_sha = _sha256(model)
    try:
        provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise UploadError("A readable provenance.json is required before uploading benchmark results.") from exc
    if provenance.get("model_name") != model.name or provenance.get("model_sha256") != model_sha:
        raise UploadError("provenance.json model_name/model_sha256 does not match the selected model.")
    _run(["--version"])
    _run(["lfs", "version"])
    result_hashes = {name: _sha256(path) for name, path in sources.items()}
    remote = _remote_url(repo)
    try:
        with tempfile.TemporaryDirectory(prefix="rtdetr_upload_") as temporary:
            checkout = Path(temporary) / "checkout"
            _run(["clone", "--no-checkout", "--", remote, str(checkout)])
            _run(["config", "--local", "core.autocrlf", "false"], checkout)
            _run(["config", "--local", "user.name", "RT-DETR benchmark uploader"], checkout)
            _run(["config", "--local", "user.email", "rtdetr-benchmark@users.noreply.github.com"], checkout)
            exists = _run(["show-ref", "--verify", "--quiet", f"refs/remotes/origin/{user}"], checkout, check=False)
            if exists.returncode == 0:
                _run(["checkout", "-b", user, "--track", f"origin/{user}"], checkout)
            elif exists.returncode == 1:
                default = _run(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], checkout).stdout.strip()
                _run(["checkout", "-b", user, default], checkout)
            else:
                raise UploadError("Could not inspect the remote user branch.")
            _run(["lfs", "install", "--local"], checkout)
            # A repository's .lfsconfig must not redirect model uploads elsewhere.
            if remote.startswith("https://github.com/"):
                endpoint = remote + "/info/lfs"
                _run(["config", "--local", "lfs.url", endpoint], checkout)
                _run(["config", "--local", "lfs.pushurl", endpoint], checkout)
            _add_lfs_attributes(checkout, model.suffix[1:])
            destination_model = checkout / "models" / model.name
            destination_run = checkout / "results" / user / results.name
            model_relative = f"models/{model.name}"
            run_relative = f"results/{user}/{results.name}"
            _assert_safe_destination(destination_model, checkout)
            _assert_safe_destination(destination_run, checkout)
            if destination_run.exists():
                if _same_existing_run(checkout, destination_run, result_hashes, model_relative, model_sha):
                    commit = _run(["rev-parse", "HEAD"], checkout).stdout.strip()
                    return _receipt(repo, user, commit, model, model_sha, results.name, already_present=True)
                raise UploadError("This run directory already exists with different contents; use a new run ID.")
            destination_model.parent.mkdir(parents=True, exist_ok=True)
            destination_run.mkdir(parents=True)
            shutil.copyfile(model, destination_model)
            if _sha256(destination_model) != model_sha or _sha256(model) != model_sha:
                raise UploadError("Model changed during upload preparation; upload refused.")
            for relative, source in sources.items():
                if _is_link(source) or not source.is_file():
                    raise UploadError("Results changed during upload preparation.")
                target = destination_run / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if _sha256(target) != result_hashes[relative]:
                    raise UploadError("Results changed during upload preparation; upload refused.")
            _run(["add", "--", ".gitattributes"], checkout)
            # Main ignores local weights/results. Only these explicit paths are
            # deliberately included on the selected user branch.
            _run(["add", "--force", "--", model_relative, run_relative], checkout)
            _run(["commit", "-m", f"Add {user} benchmark {results.name}"], checkout)
            pointer = _model_pointer(checkout, model_relative)
            if (pointer is None or f"oid sha256:{model_sha}" not in pointer.splitlines() or
                    f"size {model.stat().st_size}" not in pointer.splitlines()):
                raise UploadError("Git LFS did not create the expected model pointer; nothing was pushed.")
            for attempt in range(3):
                pushed = _run(["push", "origin", f"HEAD:refs/heads/{user}"], checkout, check=False)
                if pushed.returncode == 0:
                    break
                if attempt == 2 or not _non_fast_forward(pushed):
                    raise UploadError("GitHub upload failed; local results are retained.\n" + _sanitize(pushed.stderr or pushed.stdout))
                _run(["fetch", "origin", f"refs/heads/{user}:refs/remotes/origin/{user}"], checkout)
                rebased = _run(["rebase", f"origin/{user}"], checkout, check=False)
                if rebased.returncode:
                    _run(["rebase", "--abort"], checkout, check=False)
                    raise UploadError("Concurrent upload caused a conflict. Nothing was overwritten; retry with a new model version/run ID.")
            commit = _run(["rev-parse", "HEAD"], checkout).stdout.strip()
    except UploadError:
        raise
    except OSError as exc:
        raise UploadError("Upload preparation failed; local results are retained: " + _sanitize(str(exc))) from exc
    return _receipt(repo, user, commit, model, model_sha, results.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    args = parser.parse_args()
    try:
        result = upload_run(args.model, args.results, args.repo)
    except UploadError as exc:
        print(json.dumps({"status": "upload_failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
