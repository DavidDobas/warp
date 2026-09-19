"""Fetch Git LFS asset files (USD, NanoVDB, PNG) from the upstream Warp repository.

The repository stores these assets as LFS pointer files; without ``git lfs`` installed a checkout
holds only the pointers and the mesh, volume and example tests skip or fail. This script resolves
every pointer under ``warp/examples/assets`` and ``warp/tests/assets`` through GitHub's LFS batch API.
"""

import json
import os
import re
import sys
import urllib.request

LFS_BATCH_URL = "https://github.com/NVIDIA/warp.git/info/lfs/objects/batch"


def find_pointers(root):
    pointers = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                with open(path, "rb") as f:
                    head = f.read(256)
            except OSError:
                continue
            if head.startswith(b"version https://git-lfs"):
                text = head.decode()
                oid = re.search(r"oid sha256:([0-9a-f]+)", text).group(1)
                size = int(re.search(r"size (\d+)", text).group(1))
                pointers[path] = (oid, size)
    return pointers


def fetch(pointers):
    body = json.dumps(
        {"operation": "download", "transfer": ["basic"], "objects": [{"oid": o, "size": s} for o, s in pointers.values()]}
    ).encode()
    request = urllib.request.Request(
        LFS_BATCH_URL,
        data=body,
        headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"},
    )
    response = json.load(urllib.request.urlopen(request, timeout=60))
    by_oid = {o["oid"]: o for o in response["objects"]}
    for path, (oid, size) in pointers.items():
        obj = by_oid[oid]
        if "actions" not in obj:
            print(f"no download for {path}: {obj.get('error')}")
            continue
        action = obj["actions"]["download"]
        data = urllib.request.urlopen(urllib.request.Request(action["href"], headers=action.get("header", {})), timeout=300).read()
        if len(data) != size:
            raise RuntimeError(f"{path}: expected {size} bytes, got {len(data)}")
        with open(path, "wb") as f:
            f.write(data)
        print(f"fetched {os.path.relpath(path)} ({size} bytes)")


if __name__ == "__main__":
    home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pointers = {}
    for sub in ("warp/examples/assets", "warp/tests/assets"):
        pointers.update(find_pointers(os.path.join(home, sub)))
    if not pointers:
        print("no LFS pointer files found; nothing to do")
        sys.exit(0)
    fetch(pointers)
