"""Split a converted DarkGarlic model (model.idx + its .bin) into one shard per console for the cluster.

  python pc/shard_model.py cluster/profiles/qwen3.5-9b-2x-test.json --idx chat/assets_9b/model.idx \
         --bin models/qwen3.5-9b-q4s.bin --out models/cluster-9b-2x
  -> models/cluster-9b-2x/node0/{model.idx,node0.bin}, node1/..., and a SHA-256 list

  python pc/shard_model.py --self-test      # builds a tiny fake model, shards it, checks every byte

Every shard keeps ALL the tensor records of model.idx in the same order (engine.c finds a layer's tensors by
position), but a tensor that is not on this node has offset 0 and size 0; its rows / cols / fmt stay, so the engine
still knows the shapes. A shard's file holds, in this order:

  [GPU part: final_norm, the node's layers, lm_head on the last node][embedding rows on node 0][ids on every node]

and its model.idx gets the "PSHD" trailer after the records (engine.c load_index reads it):

  "PSHD", u32 version (1), node, nodes, layer_first, layer_end (exclusive), flags (1 = has embedding rows,
  2 = has lm_head), char name[32] (the model's display / API name)

A model.idx without the trailer is a whole model: the app behaves exactly as before.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

HDR = struct.Struct("<4s10I2f128sQI")   # PMDL / PMQ5 header (192 bytes)
Q5X = struct.Struct("<8I")              # PMQ5 extension (32 bytes)
REC = struct.Struct("<QQIIQI")          # one tensor (36 bytes)
SHD = struct.Struct("<4s6I32s")         # the shard trailer (60 bytes)
FMT_IDS = 7


def read_idx(path):
    d = Path(path).read_bytes()
    h = list(HDR.unpack_from(d, 0))
    magic = h[0]
    if magic not in (b"PMDL", b"PMQ5"):
        sys.exit(f"{path}: not a model.idx")
    ext = list(Q5X.unpack_from(d, HDR.size)) if magic == b"PMQ5" else None
    rec0 = HDR.size + (Q5X.size if ext else 0)
    n = h[-1]
    recs = [list(REC.unpack_from(d, rec0 + i * REC.size)) for i in range(n)]
    tail = d[rec0 + n * REC.size:]
    if tail[:4] == b"PSHD":
        sys.exit(f"{path} is already a shard")
    return dict(header=h, ext=ext, recs=recs)


def layer_full(ext, full_mask, i):
    """Same rule as chat/layer_mask.h. A Python int shift by >= 32 is 0, which would mark Qwen3
    layers 32.. and Qwen3.5's later full-attention layers as DeltaNet."""
    if ext is None:
        return True
    if i < 32:
        return ((full_mask >> i) & 1) != 0
    interval = 0
    for b in range(32):
        if (full_mask >> b) & 1:
            interval = b + 1
            break
    if not interval:
        return False
    for b in range(32):
        expect = ((b + 1) % interval) == 0
        got = ((full_mask >> b) & 1) != 0
        if expect != got:
            return False
    return ((i + 1) % interval) == 0


def layout(idx):
    """Index of every tensor: embed, final_norm, per-layer lists, head, ids."""
    h, ext, recs = idx["header"], idx["ext"], idx["recs"]
    n_layers = h[1]
    full_mask = ext[5] if ext else 0xFFFFFFFF
    flags = ext[7] if ext else 0
    layers, b = [], 2
    for i in range(n_layers):
        k = 8 if layer_full(ext, full_mask, i) else 10
        layers.append(list(range(b, b + k)))
        b += k
    vocab_map = (flags >> 2) & 1
    has_head = flags & 1
    ids = len(recs) - 1 if vocab_map else None
    head = (len(recs) - 1 - vocab_map) if has_head else None
    if b + (1 if has_head else 0) + vocab_map != len(recs):
        sys.exit(f"model.idx: {len(recs)} records, expected {b + has_head + vocab_map} for {n_layers} layers")
    return dict(embed=0, final_norm=1, layers=layers, head=head, ids=ids)


def shard(idx, src, node, nodes, first, end, name, out_dir, chunk=32 << 20):
    lay = layout(idx)
    recs = [list(r) for r in idx["recs"]]
    last = node == nodes - 1
    gpu = [lay["final_norm"]] + [t for i in range(first, end) for t in lay["layers"][i]]
    if last and lay["head"] is not None:
        gpu.append(lay["head"])
    disk = ([lay["embed"]] if node == 0 else []) + ([lay["ids"]] if lay["ids"] is not None else [])
    keep = set(gpu) | set(disk)
    for i, r in enumerate(recs):
        if i not in keep:
            r[0], r[1] = 0, 0           # offset, size: not on this node (shape and format stay)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_name = f"node{node}.bin"
    pos = 0
    with open(src, "rb") as f, open(out_dir / bin_name, "wb") as g:
        def copy(i):
            nonlocal pos
            off, size = idx["recs"][i][0], idx["recs"][i][1]
            pad = (-pos) % 256                       # (engine gpu_alloc and the kernels: 256-byte aligned rows)
            g.write(b"\0" * pad)
            pos += pad
            recs[i][0] = pos
            f.seek(off)
            left = size
            while left:
                b = f.read(min(chunk, left))
                if not b:
                    sys.exit(f"{src}: short read at tensor {i}")
                g.write(b)
                left -= len(b)
            pos += size
        for i in gpu:
            copy(i)
        gpu_size = pos
        for i in disk:
            copy(i)
    h = list(idx["header"])
    path = f"/data/darkgarlic/{bin_name}".encode()
    h[13] = path.ljust(128, b"\0")[:128]           # weights path
    h[14] = gpu_size                                # weights_size = the GPU part
    body = HDR.pack(*h) + (Q5X.pack(*idx["ext"]) if idx["ext"] else b"")
    body += b"".join(REC.pack(*r) for r in recs)
    flags = (1 if node == 0 else 0) | (2 if last else 0)
    body += SHD.pack(b"PSHD", 1, node, nodes, first, end, flags, name.encode()[:31].ljust(32, b"\0"))
    (out_dir / "model.idx").write_bytes(body)
    return out_dir / bin_name, gpu_size, recs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(profile, idx_path, bin_path, out):
    prof = json.loads(Path(profile).read_text(encoding="utf-8"))
    idx = read_idx(idx_path)
    n_layers = idx["header"][1]
    nodes = prof["nodes"]
    ranges = [tuple(n["layers"]) for n in nodes]
    if ranges[0][0] != 0 or ranges[-1][1] != n_layers or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        sys.exit(f"profile layers {ranges} do not cover 0..{n_layers} in order")
    lines = []
    for n in nodes:
        k = n["node"]
        b, gs, _ = shard(idx, bin_path, k, len(nodes), n["layers"][0], n["layers"][1], prof.get("name", "DarkGarlic"),
                         Path(out) / f"node{k}")
        lines.append(f"{sha256(b)}  node{k}/{b.name}  ({b.stat().st_size} bytes, GPU part {gs >> 20} MiB)")
        print(lines[-1])
    (Path(out) / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def self_test():
    """A fake PMQ5 model of 6 layers (every 3rd full attention), sharded 3 ways: each tensor's bytes must come back."""
    import random
    rnd = random.Random(1)
    n_layers, mask = 6, 0b100100
    names = ["embed", "final_norm"]
    for i in range(n_layers):
        k = 8 if mask >> i & 1 else 10
        names += [f"{i}.{j}" for j in range(k)]
    names += ["lm_head", "ids"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        blobs, recs, pos = {}, [], 0
        with open(td / "m.bin", "wb") as f:
            for nm in names:
                size = rnd.randint(1, 3000)
                b = bytes(rnd.getrandbits(8) for _ in range(size))
                blobs[nm] = b
                recs.append([pos, size, 7, 9, 0, FMT_IDS if nm == "ids" else 6])
                f.write(b)
                pos += size
        h = [b"PMQ5", n_layers, 256, 512, 4, 1, 256, 100, 8192, 1, 2, 1e-6, 1e7, b"/data/x.bin", pos, len(recs)]
        ext = [2, 4, 128, 128, 64, mask, 3, 1 | 2 | 4]
        (td / "model.idx").write_bytes(HDR.pack(*h) + Q5X.pack(*ext) + b"".join(REC.pack(*r) for r in recs))
        prof = dict(name="Test", nodes=[dict(node=0, layers=[0, 2]), dict(node=1, layers=[2, 5]),
                                        dict(node=2, layers=[5, 6])])
        (td / "p.json").write_text(json.dumps(prof))
        run(td / "p.json", td / "model.idx", td / "m.bin", td / "out")
        seen = {nm: 0 for nm in names}
        for k in range(3):
            d = (td / "out" / f"node{k}" / "model.idx").read_bytes()
            sh = SHD.unpack_from(d, len(d) - SHD.size)
            assert sh[0] == b"PSHD" and sh[2] == k and sh[3] == 3, sh
            rec0 = HDR.size + Q5X.size
            data = (td / "out" / f"node{k}" / f"node{k}.bin").read_bytes()
            gpu_size = HDR.unpack_from(d, 0)[14]
            for i, nm in enumerate(names):
                off, size = REC.unpack_from(d, rec0 + i * REC.size)[:2]
                if size:
                    assert data[off:off + size] == blobs[nm], (k, nm)
                    assert off % 256 == 0, (k, nm, off)
                    seen[nm] += 1
                    if nm in ("embed", "ids"):
                        assert off >= gpu_size, (k, nm)   # disk-only data after the GPU part
                    else:
                        assert off + size <= gpu_size, (k, nm)
        for nm, c in seen.items():
            want = 3 if nm in ("final_norm", "ids") else 1
            assert c == want, (nm, c)
    # Layers past the 32-bit mask. A Python shift by >= 32 is 0, which used to call those layers DeltaNet.
    pmdl = layout({"header": [b"PMDL", 40], "ext": None, "recs": [None] * (2 + 40 * 8)})
    if any(len(pmdl["layers"][i]) != 8 for i in range(40)):
        sys.exit("PMDL 40: a layer was not full attention")
    n_full = sum(1 for i in range(64) if (i + 1) % 4 == 0)
    nrec = 2 + n_full * 8 + (64 - n_full) * 10 + 1
    ext = (0, 0, 0, 0, 0, 0x88888888, 0, 1)
    q = layout({"header": [b"PMQ5", 64], "ext": ext, "recs": [None] * nrec})
    if len(q["layers"][32]) != 10 or len(q["layers"][35]) != 8 or len(q["layers"][63]) != 8:
        sys.exit("PMQ5 64: layer kinds past 31 are wrong")
    if q["head"] != nrec - 1:
        sys.exit("PMQ5 64: lm_head index")
    print("shard_model self-test: OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--idx")
    ap.add_argument("--bin")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.profile and a.idx and a.bin and a.out):
        ap.error("profile, --idx, --bin and --out")
    run(a.profile, a.idx, a.bin, a.out)


if __name__ == "__main__":
    main()
