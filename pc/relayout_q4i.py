"""Rewrite a Q4 model file in the interleaved "q4i" layout of matvec_q4i (rows in tiles of 64, see
shaders/common.h): same tensors, offsets and sizes; every "q4" tensor becomes "q4i". The rest is copied.

Run: python relayout_q4i.py [name]   (models/<name>.bin + .json -> models/<name>i.bin + .json, e.g. qwen3.5-4b-q4i)
"""

import json
import sys
from pathlib import Path

import numpy as np

from q4 import interleave_q4

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5-4b-q4"
    meta = json.loads((MODELS / f"{name}.json").read_text())
    src = open(MODELS / f"{name}.bin", "rb")
    dst = open(MODELS / f"{name}i.bin", "wb")
    pos = 0
    for key, t in sorted(meta["tensors"].items(), key=lambda kv: kv[1]["offset"]):
        if t["offset"] > pos:
            dst.write(bytes(t["offset"] - pos))
        src.seek(t["offset"])
        data = src.read(t["size"])
        if t.get("fmt") == "q4":
            rows, cols = t["rows"], t["cols"]
            packed = np.frombuffer(data[:t["scales"]], np.uint8).reshape(rows, cols // 2)
            d16 = np.frombuffer(data[t["scales"]:], np.float16).reshape(rows, cols // 32)
            nib, sc = interleave_q4(packed, d16)
            data = nib + sc
            t["fmt"] = "q4i"
        dst.write(data)
        pos = t["offset"] + t["size"]
        print(f"{key:<16} {t.get('fmt', 'f32')}", end="\r")
    dst.close()
    (MODELS / f"{name}i.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {MODELS / (name + 'i.bin')} ({pos / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
