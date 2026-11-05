"""Persistent sequential model queue. Standard library only; no model copies."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

from github_upload import infer_user, _is_link, _sha256


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_state(directory, state):
    """Atomic state replacement; CSV is a convenient, rebuildable view."""
    directory = Path(directory)
    state["updated_at_utc"] = now()
    temporary = directory / (".state_" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, directory / "state.json")
    finally:
        temporary.unlink(missing_ok=True)
    fields = ["model_name", "status", "upload_status", "result_dir", "archive_path", "error", "model_sha256"]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(state["jobs"])


def load_state(directory):
    path = Path(directory) / "state.json"
    if not path.exists(): return {"format": 1, "jobs": []}
    if _is_link(path): raise RuntimeError("Queue state must not be a symlink")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("format") != 1 or not isinstance(state.get("jobs"), list):
        raise RuntimeError("Unrecognized queue state; existing state was preserved")
    return state


@contextmanager
def evaluation_lock(root):
    """OS-held lock shared by single evaluations and queues; auto-release on exit."""
    path = Path(root) / ".rtdetr_evaluation.lock"
    if _is_link(path): raise RuntimeError("Evaluation lock cannot be a symlink")
    with path.open("a+b") as stream:
        if stream.seek(0, 2) == 0: stream.write(b"\0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another evaluation/queue is already active in this workspace") from error
        try: yield
        finally:
            stream.seek(0)
            if os.name == "nt": msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def interrupt_signals():
    """Let ordinary service/SSH shutdown use the same cleanup as Ctrl+C."""
    previous = {}
    def stop(signum, frame): raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous[number] = signal.signal(number, stop)
    try: yield
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)


def scan_models(inbox, names=None):
    inbox = Path(inbox)
    selected = list(names) if names else sorted(
        p.name for p in inbox.iterdir() if p.suffix.lower() in (".onnx", ".onix"))
    snapshots, rejected = [], []
    for name in dict.fromkeys(selected):
        path = inbox / name
        try:
            if Path(name).name != name: raise ValueError("Expected a filename only")
            infer_user(path)
            if _is_link(path) or not path.is_file(): raise ValueError("Expected a regular model file")
            before = path.stat()
            digest = _sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Model is still changing; upload as .part and rename when finished")
            snapshots.append({"model_name": name, "model_sha256": digest, "model_file_bytes": after.st_size})
        except (ValueError, RuntimeError, OSError) as error:
            rejected.append({"model_name": name, "error": str(error)})
    return snapshots, rejected


def job_key(snapshot, settings):
    value = {"model_name": snapshot["model_name"], "model_sha256": snapshot["model_sha256"], "settings": settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_queue(root, settings, prepare, evaluate, *, names=None, retry_failed=False,
              watch=False, poll_seconds=15):
    """Caller holds evaluation_lock. evaluate(name, sha, prepared) returns a run receipt."""
    directory = Path(root) / "queue"
    inbox = directory / "inbox"
    for folder in (directory, inbox):
        if _is_link(folder): raise RuntimeError("Queue directories cannot be symlinks/junctions")
        folder.mkdir(parents=True, exist_ok=True)
    state = load_state(directory)
    prepared = None
    attempted = set()  # A failed watch job must not be retried forever in one invocation.
    issues = False
    try:
        while True:
            snapshots, rejected = scan_models(inbox, names)
            state["rejected"] = rejected
            for item in rejected: print("Queue skipped:", item["model_name"], item["error"], flush=True)
            issues |= bool(rejected)
            current = []
            for snapshot in snapshots:
                key = job_key(snapshot, settings)
                job = next((item for item in state["jobs"] if item["id"] == key), None)
                if job is None:
                    job = {**snapshot, "id": key, "settings": settings, "status": "pending",
                           "created_at_utc": now(), "attempts": 0}
                    state["jobs"].append(job)
                if job["status"] == "completed":
                    # A successful evaluation with failed upload is still completed.
                    # Upload-only retries do not run the model a second time.
                    if Path(job.get("archive_path", "")).is_file() and Path(job.get("result_dir", "")).is_dir():
                        continue
                    job.update(status="pending", error="Local result artifact is missing; evaluating again")
                if job["status"] == "failed" and not retry_failed:
                    issues = True; continue
                if key not in attempted: current.append(job)
            save_state(directory, state)
            if current and prepared is None:
                print("Queue: verifying/preparing shared COCO data once.", flush=True)
                prepared = prepare()
            for index, job in enumerate(current, 1):
                attempted.add(job["id"])
                job.update(status="running", started_at_utc=now(), attempts=job["attempts"] + 1, error=None)
                save_state(directory, state)
                print(f"Queue {index}/{len(current)}: {job['model_name']}", flush=True)
                try:
                    outcome = evaluate(job["model_name"], job["model_sha256"], prepared)
                    job.update(outcome)
                    job["status"] = "completed" if outcome["returncode"] in (0, 2) else "failed"
                    if outcome["returncode"]: issues = True
                except KeyboardInterrupt as error:
                    job.update(status="interrupted", error="Stopped; this job restarts on next queue invocation")
                    for key in ("result_dir", "archive_path"):
                        if hasattr(error,key): job[key]=getattr(error,key)
                    raise
                except Exception as error:
                    job.update(status="failed", error=str(error), returncode=1)
                    issues = True
                finally:
                    job["finished_at_utc"] = now()
                    save_state(directory, state)
            if not watch: break
            print(f"Queue waiting: {inbox} (poll {poll_seconds}s; Ctrl+C to stop)", flush=True)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        save_state(directory, state)
        print("Queue stopped; saved state:", directory / "state.json", flush=True)
        return 130
    except Exception:
        save_state(directory, state)
        raise
    print("Queue finished. Status:", directory / "state.json", "Summary:", directory / "summary.csv", flush=True)
    return 1 if issues else 0
