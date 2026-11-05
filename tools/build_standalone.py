#!/usr/bin/env python3
"""Rebuild code/test.py from its reviewed sources; no runtime network fetches."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent.parent
FILES = ["code/benchmark.py", "code/coco_data.py", "code/coco_metrics.py",
         "code/github_upload.py", "code/model_queue.py", "protocol.json", "requirements.txt"]


def build():
    entries = {}
    for name in FILES:
        content = (ROOT / name).read_text(encoding="utf-8")
        entries[name] = {"content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    raw = json.dumps({"format": 1, "files": entries}, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    lines = "\n".join("    " + repr(encoded[start:start+100]) for start in range(0,len(encoded),100))
    template = (ROOT / "tools/standalone_launcher.py.in").read_text(encoding="utf-8")
    source = template.replace("@@BUNDLE_SHA256@@", digest).replace("@@PAYLOAD_LINES@@", lines)
    compile(source, "test.py", "exec")
    destination = ROOT / "code/test.py"
    destination.write_text(source, encoding="utf-8", newline="\n")
    print(json.dumps({"path": str(destination), "bundle_sha256": digest,
                      "bundle_bytes": len(raw), "standalone_bytes": len(source.encode("utf-8"))}))
    return destination


if __name__ == "__main__": build()
