#!/usr/bin/env python3
"""RT-DETR CPU benchmark engine, launched by the standalone test.py.

Data, model audit, inference and COCO scoring run in separate stages. Successful
results and the selected model are published to their user1..user4 Git branch.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

# -------- 사용자가 변경할 설정 --------
MODEL_NAME = os.environ.get("RTDETR_MODEL_NAME", "user1_v1.onnx")
GITHUB_REPOSITORY = os.environ.get("RTDETR_REPO", "foxyaef/rtdetr-raspberrypi-eval")
UPLOAD_TO_GITHUB = os.environ.get("RTDETR_UPLOAD", "1") == "1"
CPU_THREADS = int(os.environ.get("RTDETR_THREADS", "4"))
LATENCY_REPEATS = int(os.environ.get("RTDETR_REPEATS", "1"))
# ------------------------------------
ROOT = Path(os.environ.get("RTDETR_WORKSPACE_ROOT", Path(__file__).resolve().parent.parent)).resolve()
PROTOCOL_PATH = Path(os.environ.get("RTDETR_PROTOCOL_PATH", ROOT/"protocol.json")).resolve()
POWER = {"status": "not_measured", "power_mean_watts": None,
         "power_peak_watts": None, "energy_joules": None,
         "reason": "No external electrical power meter configured; software estimates are not substituted."}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def distribution(values):
    if not values: return {"mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    def percentile(p):
        pos = (len(ordered)-1)*p; low = int(pos); high = min(low+1, len(ordered)-1)
        return ordered[low] + (ordered[high]-ordered[low])*(pos-low)
    return {"mean": statistics.mean(values), "p50": percentile(.5), "p95": percentile(.95)}


def optional_text(path):
    try: return Path(path).read_text(encoding="utf-8").strip().strip("\x00")
    except (OSError, UnicodeError): return None


def temperature():
    value = optional_text("/sys/class/thermal/thermal_zone0/temp")
    try: return float(value)/1000 if value else None
    except ValueError: return None


def throttled():
    if not shutil.which("vcgencmd"): return None
    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError): return None


def system_info():
    import psutil
    versions = {}
    for package in ["numpy", "Pillow", "onnx", "onnxruntime", "pycocotools", "psutil"]:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    try: swap_total = psutil.swap_memory().total
    except (OSError, RuntimeError, psutil.Error): swap_total = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
            "raspberry_pi_model": optional_text("/proc/device-tree/model"),
            "cpu_logical_count": psutil.cpu_count(), "RAM_total_bytes": psutil.virtual_memory().total,
            "RAM_available_bytes": psutil.virtual_memory().available,
            "swap_total_bytes": swap_total, "versions": versions,
            "temperature_celsius": temperature(), "throttled": throttled(),
            "provider": "CPUExecutionProvider", "power": POWER}


class ResourceSampler:
    """20ms peak sampling; compact 1Hz CSV; model process only, no upload/data."""
    def __init__(self, output, interval=.02):
        import psutil
        self.psutil = psutil; self.process = psutil.Process(); self.interval = interval
        self.cores = psutil.cpu_count() or 1; self.stop = threading.Event(); self.thread = None
        self.stage = "session_creation"; self.phases = {}; self.rss_base = self.process.memory_info().rss
        self.rss_peak = self.rss_base; self.system_memory_peak = 0
        self.cpu_peak = 0.; self.system_cpu_peak = 0.; self.temp_peak = None
        self.output = Path(output); self.started = None; self.error = None

    def memory(self):
        rss = self.process.memory_info().rss
        used = self.psutil.virtual_memory().total-self.psutil.virtual_memory().available
        self.rss_peak = max(self.rss_peak, rss); self.system_memory_peak = max(self.system_memory_peak, used)
        phase = self.phases.setdefault(self.stage, {"rss_peak_bytes": 0, "cpu_peak_percent_one_core_100": 0.})
        phase["rss_peak_bytes"] = max(phase["rss_peak_bytes"], rss)
        return rss, used, phase

    def _poll(self):
        try:
            self.process.cpu_percent(None); self.psutil.cpu_percent(None)
            last_log = -1.
            with self.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["elapsed_seconds", "phase", "process_CPU_percent_one_core_100", "system_CPU_percent",
                                 "RSS_bytes", "system_memory_used_bytes", "RSS_peak_so_far_bytes", "temperature_celsius"])
                while not self.stop.wait(self.interval):
                    cpu = self.process.cpu_percent(None); system_cpu = self.psutil.cpu_percent(None)
                    rss, used, phase = self.memory()
                    self.cpu_peak = max(self.cpu_peak, cpu); self.system_cpu_peak = max(self.system_cpu_peak, system_cpu)
                    phase["cpu_peak_percent_one_core_100"] = max(phase["cpu_peak_percent_one_core_100"], cpu)
                    elapsed = time.perf_counter()-self.started
                    if elapsed-last_log >= 1:
                        temp = temperature()
                        if temp is not None: self.temp_peak = temp if self.temp_peak is None else max(self.temp_peak,temp)
                        writer.writerow([elapsed,self.stage,cpu,system_cpu,rss,used,self.rss_peak,temp])
                        last_log = elapsed
        except Exception as error: self.error = str(error)

    def start(self):
        self.started = time.perf_counter(); self.cpu_start = self.process.cpu_times()
        self.memory(); self.thread = threading.Thread(target=self._poll, daemon=True); self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread is not None: self.thread.join()
        self.memory()
        duration = time.perf_counter()-self.started; cpu_end = self.process.cpu_times()
        cpu_s = cpu_end.user+cpu_end.system-self.cpu_start.user-self.cpu_start.system
        hwm = None
        if sys.platform.startswith("linux"):
            import resource
            hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        return {"sample_interval_seconds": self.interval, "CSV_log_interval_seconds": 1,
                "measurement_seconds": duration, "sampler_error": self.error,
                "cpu_logical_count": self.cores,
                "cpu_process_peak_percent_one_core_100": self.cpu_peak,
                "cpu_process_peak_percent_machine_normalized": min(100.,self.cpu_peak/self.cores),
                "cpu_process_mean_percent_one_core_100": 100*cpu_s/max(duration,1e-9),
                "cpu_system_peak_percent": self.system_cpu_peak,
                "rss_baseline_bytes": self.rss_base, "rss_sampled_peak_bytes": self.rss_peak,
                "rss_os_high_watermark_bytes": hwm,
                "rss_peak_bytes": max(self.rss_peak,hwm or 0),
                "system_memory_used_peak_bytes": self.system_memory_peak,
                "temperature_peak_celsius": self.temp_peak,"phases": self.phases,
                "scope": "Inference child: session creation, warmup, measured inference, prediction streaming and local artifact writes. Data preparation, raw ONNX audit, COCO scoring and Git upload excluded. OS RSS high-watermark includes child imports.",
                "CPU_note": "Process CPU uses 100% per logical CPU, so 4 fully used CPUs may report 400%; machine-normalized field divides by logical CPU count. Peaks are sampled observations."}


def audit_model(model):
    import onnx
    from collections import Counter
    graph = onnx.load(str(model), load_external_data=False)
    def tensors(g):
        yield from g.initializer
        for sparse in g.sparse_initializer: yield sparse.values; yield sparse.indices
        for node in g.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH: yield from tensors(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs: yield from tensors(child)
                elif attr.type == onnx.AttributeProto.TENSOR: yield attr.t
                elif attr.type == onnx.AttributeProto.TENSORS: yield from attr.tensors
                elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                    yield attr.sparse_tensor.values; yield attr.sparse_tensor.indices
                elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for sparse in attr.sparse_tensors: yield sparse.values; yield sparse.indices
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in tensors(graph.graph)):
        raise ValueError("Use one self-contained ONNX file; external weight files are not supported.")
    dtypes = Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in graph.graph.initializer)
    ops = Counter(n.op_type for n in graph.graph.node)
    onnx.checker.check_model(str(model))
    return {"status":"passed","self_contained":True,"model_sha256":sha256(model),
            "initializer_dtypes":dict(dtypes),"op_counts":dict(ops),
            "has_INT8_storage":bool(dtypes["INT8"] or dtypes["UINT8"]),
            "QDQ_node_count":ops["QuantizeLinear"]+ops["DequantizeLinear"],
            "note":"Storage/graph evidence only; not proof of native INT8 arithmetic or speedup."}


def check_interface(session):
    import numpy as np
    inputs = {v.name:v for v in session.get_inputs()}; outputs = {v.name for v in session.get_outputs()}
    if set(inputs)!={"images","orig_target_sizes"} or not {"labels","boxes","scores"}<=outputs:
        raise ValueError("Expected official RT-DETR deploy inputs images/orig_target_sizes and outputs labels/boxes/scores")
    if inputs["images"].type not in ("tensor(float)","tensor(float16)"):
        raise ValueError("images must be float32/float16")
    if inputs["orig_target_sizes"].type not in ("tensor(int64)","tensor(int32)"):
        raise ValueError("orig_target_sizes must be int64/int32")
    for name,expected in [("images",[1,3,640,640]),("orig_target_sizes",[1,2])]:
        shape = inputs[name].shape
        if len(shape)!=len(expected) or any(isinstance(a,int) and a!=b for a,b in zip(shape,expected)):
            raise ValueError(f"Unexpected input shape {name}: {shape}")
    return (np.float16 if inputs["images"].type=="tensor(float16)" else np.float32,
            np.int32 if inputs["orig_target_sizes"].type=="tensor(int32)" else np.int64)


def load_image(item,data_dir,image_dtype,size_dtype):
    import numpy as np
    from PIL import Image
    with Image.open(data_dir/"val2017"/item["file_name"]) as source:
        image=source.convert("RGB"); width,height=image.size
        if (width,height)!=(item["width"],item["height"]): raise ValueError("COCO image dimensions changed")
        array=np.asarray(image.resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)/255.
    return np.ascontiguousarray(array.transpose(2,0,1)[None].astype(image_dtype)),np.array([[width,height]],dtype=size_dtype)


def decode(outputs,item,category_ids):
    import numpy as np
    labels,boxes,scores=[np.asarray(v) for v in outputs]
    if labels.ndim!=2 or labels.shape[0]!=1 or scores.shape!=labels.shape or boxes.shape!=(*labels.shape,4):
        raise ValueError("Unexpected detection output dimensions")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all(): raise ValueError("Nonfinite output")
    if np.any(scores<0) or np.any(scores>1): raise ValueError("Scores must be probabilities")
    if not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=len(category_ids)):
        raise ValueError("Expected contiguous COCO labels 0..79")
    records=[]
    for label,box,confidence in zip(labels[0],boxes[0],scores[0]):
        x1,y1,x2,y2=map(float,box)
        if confidence>0 and x2>x1 and y2>y1:
            records.append({"image_id":item["id"],"category_id":category_ids[int(label)],
                            "bbox":[x1,y1,x2-x1,y2-y1],"score":float(confidence)})
    return records


def infer_worker(config):
    import numpy as np
    import onnxruntime as ort
    output=Path(config["result_dir"]); data=Path(config["data_dir"])
    annotation=read_json(data/"annotations/evaluation.json")
    images=sorted(annotation["images"],key=lambda x:x["file_name"])[:config["limit"]]
    categories=[c["id"] for c in sorted(annotation["categories"],key=lambda c:c["id"])]
    options=ort.SessionOptions(); options.intra_op_num_threads=config["threads"]; options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("optimization.disable_specified_optimizers","GatherSliceToSplitFusion")
    sampler=ResourceSampler(output/"resource_samples.csv"); sampler.start()
    times={k:[] for k in ["preprocess_ms","inference_ms","output_conversion_ms","end_to_end_ms"]}
    prediction_count=0; first=True
    try:
        load_start=time.perf_counter()
        session=ort.InferenceSession(config["model_path"],options,providers=["CPUExecutionProvider"])
        session.disable_fallback()
        if session.get_providers()!=["CPUExecutionProvider"]: raise RuntimeError("CPU provider was not selected exclusively")
        load_seconds=time.perf_counter()-load_start
        image_dtype,size_dtype=check_interface(session)
        sampler.stage="warmup"
        for item in images[:min(10,len(images))]:
            array,size=load_image(item,data,image_dtype,size_dtype)
            session.run(["labels","boxes","scores"],{"images":array,"orig_target_sizes":size})
        sampler.stage="inference"
        with (output/"predictions.json").open("w",encoding="utf-8") as pred_file, (output/"latency.csv").open("w",newline="",encoding="utf-8") as time_file:
            pred_file.write("["); writer=csv.writer(time_file)
            writer.writerow(["image_id","preprocess_ms","inference_first_ms","output_conversion_ms","end_to_end_ms","inference_all_repeats_ms"])
            for index,item in enumerate(images,1):
                begin=time.perf_counter(); array,size=load_image(item,data,image_dtype,size_dtype); loaded=time.perf_counter()
                feed={"images":array,"orig_target_sizes":size}; repetitions=[]
                for repeat in range(config["repeats"]):
                    tic=time.perf_counter(); outputs=session.run(["labels","boxes","scores"],feed); done=time.perf_counter()
                    repetitions.append((done-tic)*1000)
                    if repeat==0:
                        rows=decode(outputs,item,categories); converted=time.perf_counter()
                        pre=(loaded-begin)*1000; post=(converted-done)*1000; e2e=(converted-begin)*1000
                        times["preprocess_ms"].append(pre); times["output_conversion_ms"].append(post); times["end_to_end_ms"].append(e2e)
                        for row in rows:
                            if not first: pred_file.write(",")
                            pred_file.write(json.dumps(row,allow_nan=False,separators=(",",":"))); first=False
                        prediction_count+=len(rows)
                times["inference_ms"].extend(repetitions)
                writer.writerow([item["id"],pre,repetitions[0],post,e2e,json.dumps(repetitions)])
                if index%25==0 or index==len(images): print(f"Inference {index}/{len(images)}",flush=True)
            pred_file.write("]\n")
    finally:
        resource=sampler.finish()
    if resource["sampler_error"]: raise RuntimeError("Resource sampler failed: "+resource["sampler_error"])
    resource.update(model_load_seconds=load_seconds, latency={k:distribution(v) for k,v in times.items()},
                    inference_runs_timed=len(times["inference_ms"]),inference_images_timed=len(images),
                    warmup_images=min(10,len(images)),latency_repeats=config["repeats"],power=POWER)
    write_json(output/"inference_metrics.json",resource)
    write_json(output/"inference_provenance.json",{"onnxruntime_version":ort.__version__,"numpy_version":np.__version__,
                "providers":session.get_providers(),"provider_options":session.get_provider_options(),
                "evaluated_image_ids":[i["id"] for i in images],"prediction_count":prediction_count,
                "timing_policy":"Separate warmup, then all selected images. Preprocess+decode/E2E measured on first repeat only; session.run measured on all repeats. Result serialization and file writes excluded from latency; streaming reduces RSS vs Colab accumulated predictions."})


def score_worker(config):
    from coco_metrics import evaluate_predictions
    output=Path(config["result_dir"]); provenance=read_json(output/"inference_provenance.json")
    scores=evaluate_predictions(Path(config["data_dir"])/"annotations/evaluation.json",read_json(output/"predictions.json"),
                                threshold=.5,image_ids=provenance["evaluated_image_ids"])
    rows=scores.pop("class_wise")
    with (output/"class_metrics.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/"accuracy.json",scores)


def run_stage(stage, config_path, output, progress=None):
    command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker",stage,"--config",str(config_path)]
    with (output/(stage+".log")).open("w",encoding="utf-8") as log:
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace") as child:
            try:
                for line in child.stdout:
                    log.write(line); print(line.rstrip(),flush=True)
                    if progress is not None and stage=="infer":
                        match=re.fullmatch(r"Inference (\d+)/(\d+)",line.strip())
                        if match: progress({"stage":stage,"image_current":int(match[1]),"image_total":int(match[2])})
                returncode=child.wait()
            except KeyboardInterrupt:
                child.terminate(); child.wait(); raise
    if returncode:
        suffix=" (possible OS out-of-memory termination)" if returncode in (-9,137) else ""
        raise RuntimeError(f"{stage} failed with exit {returncode}{suffix}; see {stage}.log")


def package_results(output):
    # Stable archive excludes its later upload receipt; retries remain idempotent.
    archive=Path(str(output)+".zip")
    temporary=archive.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_symlink() or bool(getattr(path,"is_junction",lambda:False)()):
                    raise ValueError("Result archive cannot include symlinks/junctions")
                if path.is_file() and path!=output/"upload_status.json":
                    bundle.write(path,path.relative_to(output).as_posix())
        os.replace(temporary,archive)
    finally:
        temporary.unlink(missing_ok=True)
    print("Local results:",output,"\nZIP:",archive,flush=True)
    return str(archive)


def upload(model,output,repo,archive=None):
    from github_upload import upload_run
    try:
        result=upload_run(model,output,repo=repo,archive_path=archive)
        write_json(output/"upload_status.json",result); print("GitHub upload:",json.dumps(result,ensure_ascii=False),flush=True)
        return True
    except Exception as error:
        # Uploader redacts credential-bearing Git errors before exposing them.
        write_json(output/"upload_status.json",{"status":"failed","error":str(error)})
        print("Upload failed; local results preserved. Retry with --upload-only",output,flush=True)
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",help="Single filename within models/; bypasses queue auto-detection")
    parser.add_argument("--queue",action="store_true",help="Evaluate queue/inbox/ in filename order")
    parser.add_argument("--models",nargs="+",help="Queue filenames in the desired order")
    parser.add_argument("--watch",action="store_true",help="Keep checking queue/inbox/ for new models")
    parser.add_argument("--once",action="store_true",help="Process the current queue only, then exit")
    parser.add_argument("--poll-seconds",type=float,default=15)
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--queue-status",action="store_true")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/coco")
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--limit",type=int,default=1000,help="3 for smoke; 1000 for full evaluation")
    parser.add_argument("--repeats",type=int,default=LATENCY_REPEATS)
    parser.add_argument("--threads",type=int,default=CPU_THREADS)
    parser.add_argument("--repo",default=GITHUB_REPOSITORY)
    parser.add_argument("--no-upload",action="store_true")
    parser.add_argument("--no-download",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--upload-only",type=Path,help="Retry uploading a saved run without repeating inference")
    parser.add_argument("--worker",choices=["audit","infer","score"],help=argparse.SUPPRESS)
    parser.add_argument("--config",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        config=read_json(args.config)
        if args.worker=="audit":write_json(Path(config["result_dir"])/"quantization_audit.json",audit_model(Path(config["model_path"])))
        elif args.worker=="infer":infer_worker(config)
        else:score_worker(config)
        return 0
    from model_queue import evaluation_lock, interrupt_signals, run_queue
    if args.queue_status:
        state=ROOT/"queue/status.txt"
        print(state.read_text(encoding="utf-8") if state.exists() else "Queue has not been started.")
        return 0
    for folder in [ROOT/"code", ROOT/"models", args.results_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    if not args.no_download: args.data_dir.mkdir(parents=True, exist_ok=True)
    from github_upload import infer_user
    from coco_data import prepare_data
    if args.prepare_only:
        prepare_data(args.data_dir.resolve(),download=not args.no_download); return 0
    if not 1<=args.limit<=1000 or not 1<=args.repeats<=5 or not 1<=args.threads<=64:
        parser.error("limit=1..1000, repeats=1..5, threads=1..64")
    if not 2<=args.poll_seconds<=60: parser.error("poll-seconds=2..60")
    if sys.maxsize<=2**32:parser.error("Use a 64-bit OS and 64-bit Python for ONNX Runtime")
    if args.once and args.watch: parser.error("--once cannot be combined with --watch")
    queue_requested=args.queue or args.watch or args.once or bool(args.models) or args.retry_failed
    if args.model and queue_requested: parser.error("Choose --model or a queue, not both")
    inbox=ROOT/"queue/inbox"
    automatic=not args.model and not args.upload_only
    if args.upload_only and queue_requested: parser.error("--upload-only cannot be combined with a queue")
    if queue_requested or automatic:
        settings={"limit":args.limit,"repeats":args.repeats,"threads":args.threads,
                  "data_dir":str(args.data_dir.resolve()),"results_dir":str(args.results_dir.resolve()),
                  "protocol_sha256":sha256(PROTOCOL_PATH),
                  "runtime_bundle_sha256":os.environ.get("RTDETR_BUNDLE_SHA256",sha256(__file__))}
        def evaluate(name,digest,prepared,progress):
            return evaluate_model(args,name=name,model_directory=inbox,expected_sha=digest,prepared=prepared,progress=progress)
        with evaluation_lock(ROOT), interrupt_signals():
            return run_queue(ROOT,settings,
                lambda:prepare_data(args.data_dir.resolve(),download=not args.no_download),evaluate,
                names=args.models,retry_failed=args.retry_failed,watch=not args.once,poll_seconds=args.poll_seconds)
    if args.upload_only:
        output=args.upload_only.resolve(); provenance=read_json(output/"provenance.json")
        model=ROOT/provenance.get("model_relative_path","models/"+provenance["model_name"])
        infer_user(model)
        if (model.is_symlink() or model.parent.resolve() not in [(ROOT/"models").resolve(),inbox.resolve()]
                or model.name!=provenance["model_name"] or sha256(model)!=provenance["model_sha256"]):
            raise ValueError("Saved model path/hash does not match this result")
        with evaluation_lock(ROOT), interrupt_signals():
            archive=package_results(output)
            okay=upload(model,output,args.repo,archive); return 0 if okay else 2
    with evaluation_lock(ROOT), interrupt_signals():
        return evaluate_model(args,name=args.model or MODEL_NAME)["returncode"]


def evaluate_model(args,*,name,model_directory=None,expected_sha=None,prepared=None,progress=None):
    from github_upload import infer_user, _is_link
    from coco_data import prepare_data
    model_directory=Path(model_directory or ROOT/"models")
    model=model_directory/name
    user=infer_user(model)
    if Path(name).name!=name or _is_link(model):
        raise ValueError("Put the selected single-file model in its input folder and pass its filename")
    if not model.exists():
        if expected_sha: raise ValueError("Queued model was removed before evaluation")
        print("Model not found; preparing the account-local environment/data first.",flush=True)
        prepare_data(args.data_dir.resolve(),download=not args.no_download)
        print("Preparation complete. Put your model at:",model,flush=True)
        print("Then run the same test.py command again. No benchmark or upload was performed.",flush=True)
        return {"returncode":0,"status":"not_started"}
    if not model.is_file(): raise ValueError("The selected model must be a regular file")
    model=model.resolve(); source_hash=sha256(model)
    if expected_sha is not None and source_hash!=expected_sha:
        raise ValueError("Queued model changed after registration; submit it under a new version name")
    if args.limit!=1000:print(f"SMOKE/PARTIAL: {args.limit} images. Accuracy is not the final 1000-image result.",flush=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output=args.results_dir.resolve()/(model.stem+"_"+stamp+"_"+uuid.uuid4().hex[:8]); output.mkdir(parents=True)
    config={"model_path":str(model),"result_dir":str(output),"data_dir":str(args.data_dir.resolve()),
            "limit":args.limit,"repeats":args.repeats,"threads":args.threads}
    config_path=output/"run_config.json"; write_json(config_path,config)
    if progress: progress({"stage":"data_preparation","result_dir":str(output)})
    write_json(output/"system_info.json",system_info())
    write_json(output/"provenance.json",{"model_name":model.name,"model_sha256":source_hash,"user_branch":user,
               "model_relative_path":model.relative_to(ROOT).as_posix(),
               "test_script_sha256":sha256(__file__),"launcher_sha256":os.environ.get("RTDETR_LAUNCHER_SHA256"),
               "runtime_bundle_sha256":os.environ.get("RTDETR_BUNDLE_SHA256"),
               "created_at_utc":stamp,"full_evaluation":args.limit==1000})
    stage="data_preparation"
    try:
        if prepared is None: prepared=prepare_data(args.data_dir.resolve(),download=not args.no_download)
        ann_path=Path(prepared["annotation_path"])
        provenance=read_json(output/"provenance.json")
        provenance.update(evaluation_annotations_sha256=sha256(ann_path),data_manifest_sha256=sha256(prepared["manifest_path"]),
                          data_verification=prepared["verification"],cpu_threads=args.threads,latency_repeats=args.repeats,
                          provider="CPUExecutionProvider",protocol_sha256=sha256(PROTOCOL_PATH))
        write_json(output/"provenance.json",provenance)
        shutil.copy2(PROTOCOL_PATH,output/"protocol.json")
        for stage in ["audit","infer","score"]:
            write_json(output/"evaluation_state.json",{"status":"running","stage":stage})
            if progress:
                event={"stage":stage}
                if stage=="infer": event.update(image_current=0,image_total=args.limit)
                progress(event)
            print("Stage:",stage,flush=True); run_stage(stage,config_path,output,progress=progress)
        if sha256(model)!=source_hash:raise RuntimeError("Model file changed while evaluation was running")
        resource=read_json(output/"inference_metrics.json"); accuracy=read_json(output/"accuracy.json")
        metrics={**accuracy,"resource":resource,"model_file_bytes":model.stat().st_size,
                 "model_file_MiB":model.stat().st_size/1048576,"image_count":args.limit,"full_evaluation":args.limit==1000}
        write_json(output/"metrics.json",metrics)
        provenance.update(read_json(output/"inference_provenance.json")); write_json(output/"provenance.json",provenance)
        write_json(output/"system_info_after.json",system_info())
        write_json(output/"errors.json",[])
        write_json(output/"evaluation_state.json",{"status":"complete","stage":"complete","full_evaluation":args.limit==1000})
        print(json.dumps({"mAP50_95":accuracy["mAP50_95"],"file_MiB":metrics["model_file_MiB"],
                          "E2E_p50_ms":resource["latency"]["end_to_end_ms"]["p50"],
                          "CPU_peak_percent_machine":resource["cpu_process_peak_percent_machine_normalized"],
                          "RSS_peak_MiB":resource["rss_peak_bytes"]/1048576,"power":POWER},ensure_ascii=False,indent=2),flush=True)
    except (Exception,KeyboardInterrupt) as error:
        write_json(output/"errors.json",[{"stage":stage,"type":type(error).__name__,"error":str(error)}])
        write_json(output/"evaluation_state.json",{"status":"interrupted" if isinstance(error,KeyboardInterrupt) else "failed","stage":stage})
        (output/"failure.log").write_text(traceback.format_exc(),encoding="utf-8")
        archive=package_results(output); print(str(error),file=sys.stderr)
        if isinstance(error,KeyboardInterrupt):
            error.result_dir=str(output); error.archive_path=archive; raise
        return {"returncode":1,"result_dir":str(output),"archive_path":archive,"error":str(error),"upload_status":"not_attempted"}
    should_upload=UPLOAD_TO_GITHUB and not args.no_upload
    if progress: progress({"stage":"packaging"})
    archive=package_results(output)
    if progress and should_upload: progress({"stage":"upload"})
    okay=upload(model,output,args.repo,archive) if should_upload else True
    if not should_upload:write_json(output/"upload_status.json",{"status":"disabled"})
    return {"returncode":0 if okay else 2,"result_dir":str(output),"archive_path":archive,
            "upload_status":read_json(output/"upload_status.json")["status"]}


if __name__=="__main__":
    raise SystemExit(main())
