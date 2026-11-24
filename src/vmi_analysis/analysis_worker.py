"""Execute a reviewed local analysis snapshot using the application's Python."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


def run(manifest):
    manifest = Path(manifest).resolve(strict=True)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != "vmi.analysis-run/v1" or record.get("id") != manifest.parent.name:
        raise ValueError("Choose a VMI analysis run record.")
    script = manifest.with_name("script.py")
    raw = script.read_bytes()
    if sha256(raw).hexdigest() != record["script_sha256"]:
        raise ValueError("The approved Python snapshot changed. Review and submit it again.")
    for name, entry in record["inputs"].items():
        info = Path(entry["path"]).stat()
        if info.st_size != entry["size"] or info.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Input {name} changed after it was selected. Review and submit it again.")
    os.chdir(manifest.with_name("outputs"))
    sys.argv[:] = [str(script), str(manifest)]
    print(f"Running {record['title']} · {record['id']}", flush=True)
    exec(compile(raw, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    run(args.manifest)


if __name__ == "__main__":
    main()
