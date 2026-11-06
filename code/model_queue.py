"""Persistent sequential model queue. Standard library only; no model copies."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import datetime as dt
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import uuid

from github_upload import infer_user, _is_link, _sha256


class ResilientConsole:
    """Tee parent console output to disk; a disconnected terminal is optional.

    Only errors from the console are suppressed. Log/data filesystem errors
    still propagate, rather than disguising a failing SD card as a disconnect.
    """
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile
        self.lock = threading.RLock()
        self.detached = False

    def write(self, text):
        with self.lock:
            self.logfile.write(text)
            self.logfile.flush()
            if not self.detached:
                try: self.console.write(text); self.console.flush()
                except OSError as error:
                    if error.errno not in (errno.EIO, errno.EPIPE, errno.EBADF, errno.ENXIO): raise
                    self.detached = True
                    self.logfile.write(f"\n[console disconnected: errno={error.errno}; evaluation continues]\n")
                    self.logfile.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.logfile.flush()
            if not self.detached:
                try: self.console.flush()
                except OSError as error:
                    if error.errno not in (errno.EIO, errno.EPIPE, errno.EBADF, errno.ENXIO): raise
                    self.detached = True

    @property
    def encoding(self): return "utf-8"

    def isatty(self): return False


@contextmanager
def console_logging(root):
    directory = Path(root) / "queue"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "service.log"
    if _is_link(directory) or _is_link(path): raise RuntimeError("Service log cannot be a symlink")
    old_out, old_err = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as log:
        try:
            sys.stdout, sys.stderr = ResilientConsole(old_out, log), ResilientConsole(old_err, log)
            print("\nService started:", now(), "PID:", os.getpid(), flush=True)
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


class QueueInterrupted(KeyboardInterrupt):
    def __init__(self, number):
        self.signal_number = number
        self.signal_name = signal.Signals(number).name
        super().__init__("Received " + self.signal_name)


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
    if _is_link(path): raise RuntimeError("Queue state must not be a symlink")
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"format":1,"jobs":[]}
    if state.get("format") != 1 or not isinstance(state.get("jobs"), list):
        raise RuntimeError("Unrecognized queue state; existing state was preserved")
    # Recover uploads durably recorded before cleanup, even after a power loss.
    for job in state['jobs']:
        result=job.get('result_dir')
        if not result:continue
        record_path=Path(directory)/'uploaded'/(Path(result).name+'.json')
        if record_path.is_file() and not _is_link(record_path):
            record=json.loads(record_path.read_text(encoding='utf-8'))
            if (record.get('result_dir')==result and record.get('model_name')==job.get('model_name')
                    and record.get('upload_receipt',{}).get('model_sha256')==job.get('model_sha256')):
                job.update(status='completed',upload_status='uploaded',cleanup_status=record['cleanup_status'],
                           upload_receipt=record['upload_receipt'])
    history=Path(directory)/'uploaded'
    if history.exists() and not _is_link(history):
        for record_path in history.glob('*.json'):
            if _is_link(record_path):continue
            record=json.loads(record_path.read_text(encoding='utf-8'))
            receipt=record.get('upload_receipt',{})
            if receipt.get('status')!='uploaded' or any(j.get('result_dir')==record.get('result_dir') for j in state['jobs']):continue
            state['jobs'].append({'id':'uploaded:'+record_path.stem,'model_name':record['model_name'],
                'model_sha256':receipt['model_sha256'],'result_dir':record['result_dir'],
                'archive_path':record['archive_path'],'status':'completed','upload_status':'uploaded',
                'cleanup_status':record['cleanup_status'],'upload_receipt':receipt,'attempts':1})
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
def interrupt_signals(keep_running_on_hangup=False):
    """Let ordinary service/SSH shutdown use the same cleanup as Ctrl+C."""
    previous = {}
    def stop(signum, frame): raise QueueInterrupted(signum)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            handler = signal.SIG_IGN if name == "SIGHUP" and keep_running_on_hangup else stop
            previous[number] = signal.signal(number, handler)
    try: yield
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)


def file_signature(info):
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev)


def scan_models(inbox, names=None, cache=None):
    inbox = Path(inbox)
    selected = list(names) if names is not None else sorted(
        p.name for p in inbox.iterdir() if p.suffix.lower() in (".onnx", ".onix"))
    snapshots, rejected = [], []
    for name in dict.fromkeys(selected):
        path = inbox / name
        try:
            if Path(name).name != name: raise ValueError("Expected a filename only")
            infer_user(path)
            if _is_link(path) or not path.is_file(): raise ValueError("Expected a regular model file")
            before = path.stat()
            cached = (cache or {}).get(name)
            digest = cached[1] if cached and cached[0] == file_signature(before) else _sha256(path)
            after = path.stat()
            if file_signature(before) != file_signature(after):
                raise ValueError("Model is still changing; upload as .part and rename when finished")
            if cache is not None: cache[name] = (file_signature(after), digest)
            snapshots.append({"model_name": name, "model_sha256": digest, "model_file_bytes": after.st_size,
                              "file_signature": list(file_signature(after))})
        except (ValueError, RuntimeError, OSError) as error:
            rejected.append({"model_name": name, "error": str(error)})
    return snapshots, rejected


def job_key(snapshot, settings):
    value = {"model_name": snapshot["model_name"], "model_sha256": snapshot["model_sha256"], "settings": settings}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class QueueStatus:
    """Low-overhead status heartbeat: directory metadata only, never model hashing.

    The monitor remains active during an inference subprocess, so newly arrived
    models appear in status.txt without loading another model or interrupting it.
    """
    STAGES = {"starting": "대기열 확인 중", "preparing": "공통 이미지 준비·검증 중",
              "data_preparation": "데이터 확인", "audit": "ONNX 모델 검사",
              "infer": "CPU 추론", "score": "mAP 계산", "packaging": "결과 ZIP 생성",
              "upload": "GitHub 업로드", "ready": "새 모델 기다리는 중",
              "stopped": "프로그램 종료", "error": "오류 확인 필요"}

    def __init__(self, root, *, watch, poll_seconds, names=None, interval=5):
        self.directory = Path(root) / "queue"
        self.inbox = self.directory / "inbox"
        self.watch, self.poll_seconds, self.names = watch, poll_seconds, set(names) if names else None
        self.interval = min(interval, poll_seconds)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.known = {}
        self.model = None
        self.phase = "starting"
        self.image_current = self.image_total = None
        self.completed = self.failed = 0
        self.completed_models = []
        self.reason = ""
        self.error = None

    def sync_jobs(self, snapshots, jobs, settings):
        by_id = {job["id"]: job for job in jobs}
        with self.lock:
            self.known = {item["model_name"]: {
                "signature": tuple(item["file_signature"]),
                "status": by_id[job_key(item, settings)]["status"]} for item in snapshots}
            self.completed = sum(job["status"] == "completed" for job in jobs)
            self.failed = sum(job["status"] == "failed" for job in jobs)
            self.completed_models = list(dict.fromkeys(job['model_name'] for job in jobs if job['status']=='completed'))

    def activity(self, phase, model=None, *, refresh=True, image_current=None, image_total=None):
        with self.lock:
            if model != self.model:
                self.image_current = self.image_total = None
            self.phase, self.model = phase, model
            if image_current is not None: self.image_current = image_current
            if image_total is not None: self.image_total = image_total
        if refresh: self.refresh()

    def refresh(self):
        with self.lock:
            waiting, invalid, transferring = [], [], 0
            inventory_error = None
            try:
                for path in sorted(self.inbox.iterdir(), key=lambda value: value.name):
                    if path.name.endswith(".part"):
                        transferring += 1; continue
                    if path.suffix.lower() not in (".onnx", ".onix"): continue
                    if self.names is not None and path.name not in self.names: continue
                    try:
                        infer_user(path)
                        if _is_link(path) or not path.is_file(): raise ValueError("not a regular file")
                        signature = file_signature(path.stat())
                        known = self.known.get(path.name)
                        if known and known["signature"] == signature:
                            if known["status"] in ("completed", "failed"): continue
                            if path.name == self.model: continue
                        waiting.append(path.name)
                    except (OSError, ValueError, RuntimeError): invalid.append(path.name)
            except OSError as error:
                inventory_error = str(error)
            stopped = self.phase in ("stopped", "error")
            if stopped:
                label = "종료됨 — 자동 실행하지 않습니다" if self.phase == "stopped" else "오류로 종료됨"
            elif inventory_error:
                label = "대기열 폴더 확인 오류 — 사용 가능 여부 확인 필요"
            elif self.model:
                label = "작업 실행 중"
            elif self.phase == "preparing":
                label = "공통 환경 준비 중"
            elif self.phase == "starting":
                label = "대기열 확인 중"
            elif waiting:
                label = "대기 작업 있음 — 순서대로 자동 실행 예정"
            else:
                label = "바로 사용 가능 — 새 모델을 기다리는 중"
            kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
            lines = ["RT-DETR 모델 평가 서버 상태", f"갱신 시각: {kst}", f"상태: {label}",
                     f"실행 중인 모델: {self.model or '없음'}",
                     f"현재 단계: {self.STAGES.get(self.phase, self.phase)}",
                     f"대기 중인 모델: {len(waiting)}건"]
            lines += [f"  {index}. {name}" for index, name in enumerate(waiting, 1)]
            if self.image_current is not None:
                lines.append(f"추론 진행: {self.image_current}/{self.image_total}장")
            lines += [f"전송 중(.part): {transferring}건", f"누적 완료: {self.completed}건",
                      f"누적 실패(자동 반복 안 함): {self.failed}건", f"파일명/형식 확인 필요: {len(invalid)}건"]
            lines += ['완료된 모델:']+[f'  {name}' for name in self.completed_models]
            lines += [f"  확인 필요: {name}" for name in invalid]
            lines += [f"감시 모드: {'계속 대기' if self.watch else '현재 목록만 처리'}",
                      f"새 모델 탐색 간격: {self.poll_seconds}초", f"상태 갱신 간격: 최대 {self.interval}초",
                      f"프로세스 PID: {os.getpid()}", f"결과 저장 위치: {self.directory.parent / 'results'}",
                      "갱신 시각이 오래 멈췄다면 프로그램이 실행 중인지 확인하세요."]
            if self.reason: lines.append("안내: " + self.reason)
            if inventory_error: lines.append("폴더 오류: " + inventory_error)
            destination = self.directory / "status.txt"
            if _is_link(destination): raise RuntimeError("Queue status must not be a symlink")
            temporary = self.directory / (".status_" + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
                # Windows readers can briefly deny replacement while opening the
                # TXT. Keep atomic writes and retry that transient sharing lock.
                for attempt in range(10):
                    try:
                        os.replace(temporary, destination)
                        break
                    except PermissionError:
                        if os.name != "nt" or attempt == 9: raise
                        time.sleep(.025)
            finally:
                temporary.unlink(missing_ok=True)

    def start(self):
        self.refresh()
        def loop():
            while not self.stop_event.wait(self.interval):
                try: self.refresh()
                except Exception as error:
                    self.error = str(error)
                    print("Status update failed:", self.error, flush=True)
                    return
        self.thread = threading.Thread(target=loop, name="queue-status", daemon=True)
        self.thread.start()

    def close(self, phase="stopped", reason=""):
        self.stop_event.set()
        if self.thread is not None: self.thread.join(timeout=10)
        with self.lock:
            self.phase, self.model, self.reason = phase, None, reason
        self.refresh()


def run_queue(root, settings, prepare, evaluate, *, names=None, retry_failed=False,
              watch=False, poll_seconds=15):
    """Caller holds evaluation_lock. evaluate(name, sha, prepared, progress) returns a receipt."""
    directory = Path(root) / "queue"
    inbox = directory / "inbox"
    for folder in (directory, inbox):
        if _is_link(folder): raise RuntimeError("Queue directories cannot be symlinks/junctions")
        folder.mkdir(parents=True, exist_ok=True)
    state = load_state(directory)
    state["service"] = {"pid": os.getpid(), "status": "running", "started_at_utc": now(), "watch": watch}
    monitor = QueueStatus(root, watch=watch, poll_seconds=poll_seconds, names=names)
    monitor.start()
    prepared = None
    cache = {}
    attempted = set()  # A failed watch job must not be retried forever in one invocation.
    issues = False
    final_phase, final_reason = "stopped", "지정한 모델 처리가 끝났습니다."
    selected_names = names
    try:
        while True:
            if monitor.error: raise RuntimeError("Status file could not be updated: " + monitor.error)
            snapshots, rejected = scan_models(inbox, selected_names, cache)
            if not watch and selected_names is None:
                selected_names = [item["model_name"] for item in snapshots]
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
                    if job.get('upload_status')=='uploaded' and job.get('cleanup_status') in ('complete','partial','pending'):
                        continue  # Intentional cleanup is not a missing-artifact failure.
                    # A successful evaluation with failed upload is still completed.
                    # Upload-only retries do not run the model a second time.
                    if Path(job.get("archive_path", "")).is_file() and Path(job.get("result_dir", "")).is_dir():
                        continue
                    job.update(status="pending", error="Local result artifact is missing; evaluating again")
                if job["status"] == "failed" and not retry_failed:
                    issues = True; continue
                if key not in attempted:
                    job["status"] = "pending"
                    current.append(job)
            monitor.sync_jobs(snapshots, state["jobs"], settings)
            save_state(directory, state)
            if current and prepared is None:
                monitor.activity("preparing")
                print("Queue: verifying/preparing shared COCO data once.", flush=True)
                prepared = prepare()
            # Rescan after each job, including new arrivals during the last inference.
            for job in current[:1]:
                attempted.add(job["id"])
                job.update(status="running", started_at_utc=now(), attempts=job["attempts"] + 1, error=None)
                save_state(directory, state)
                monitor.sync_jobs(snapshots, state["jobs"], settings)
                monitor.activity("audit", job["model_name"])
                print(f"Queue: {job['model_name']} (waiting {len(current)-1})", flush=True)
                def progress(event):
                    phase = event.get("stage", job.get("stage", "audit"))
                    if "result_dir" in event: job["result_dir"] = event["result_dir"]
                    stage_changed = phase != job.get("stage")
                    job["stage"] = phase
                    monitor.activity(phase, job["model_name"], refresh=stage_changed,
                        image_current=event.get("image_current"), image_total=event.get("image_total"))
                    if stage_changed: save_state(directory, state)
                try:
                    outcome = evaluate(job["model_name"], job["model_sha256"], prepared, progress)
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
                    monitor.sync_jobs(snapshots, state["jobs"], settings)
            monitor.activity("ready")
            if selected_names is not None:
                removed={item['model_name'] for item in state['jobs'] if item.get('cleanup_status')=='complete'}
                selected_names=[name for name in selected_names if name not in removed or (inbox/name).exists()]
            if current: continue
            if not watch: break
            print(f"Queue waiting: {inbox} (poll {poll_seconds}s; Ctrl+C to stop)", flush=True)
            time.sleep(poll_seconds)
    except KeyboardInterrupt as error:
        final_reason = "중단 신호를 받았습니다: " + (str(error) or "KeyboardInterrupt / Ctrl+C")
        state["service"]["interruption"] = final_reason
        save_state(directory, state)
        print("Queue stopped; saved state:", directory / "state.json", flush=True)
        return 130
    except Exception as error:
        final_phase, final_reason = "error", str(error)
        failure = {"time_utc": now(), "type": type(error).__name__, "errno": getattr(error,"errno",None),
                   "error": str(error), "traceback": traceback.format_exc()}
        state["service"]["failure"] = failure
        path = directory / "service_failure.log"
        if _is_link(path): raise RuntimeError("Service failure log cannot be a symlink") from error
        with path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(failure,ensure_ascii=False,indent=2)+"\n")
        save_state(directory, state)
        raise
    finally:
        state["service"].update(status=final_phase, finished_at_utc=now())
        save_state(directory, state)
        monitor.close(final_phase, final_reason)
    print("Queue finished. Status:", directory / "state.json", "Summary:", directory / "summary.csv", flush=True)
    return 1 if issues else 0
