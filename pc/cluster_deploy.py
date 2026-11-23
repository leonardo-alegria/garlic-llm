"""Copy a sharded model to the consoles of a cluster over GoldHEN's FTP server, and write the head's cluster.txt.

  python pc/cluster_deploy.py cluster/profiles/qwen3.5-9b-2x-test.json models/cluster-9b-2x \
         --hosts 192.168.1.20 192.168.1.21

--hosts gives the consoles in node order (node 0 = the head, the one you hold the controller for); without it the
"host" fields of the profile are used. Each console gets /data/darkgarlic/model.idx and /data/darkgarlic/node<k>.bin
(uploaded as .part and renamed when the size checks out); the head also gets /data/darkgarlic/cluster.txt.
--remove deletes /data/darkgarlic/model.idx from every console instead: the app goes back to the packaged model.
"""
import argparse
import ftplib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build as b  # noqa: E402

DIR = "/data/darkgarlic"


def connect(host):
    ftp = ftplib.FTP()
    ftp.connect(host, b.FTP_PORT, timeout=60)
    ftp.login()
    try:
        ftp.mkd(DIR)
    except ftplib.error_perm:
        pass   # (already there)
    return ftp


def put(ftp, data_or_path, name):
    part = f"{DIR}/{name}.part"
    t0 = time.time()
    if isinstance(data_or_path, (bytes, bytearray)):
        ftp.storbinary(f"STOR {part}", io.BytesIO(data_or_path))
        want = len(data_or_path)
    else:
        with open(data_or_path, "rb") as f:
            ftp.storbinary(f"STOR {part}", f, blocksize=4 << 20)
        want = Path(data_or_path).stat().st_size
    lines = []
    ftp.retrlines(f"LIST {part}", lines.append)   # (GoldHEN's SIZE truncates to 32 bits; LIST does not)
    size = int(lines[0].split()[4]) if lines else -1
    if size != want:
        sys.exit(f"{name}: uploaded {size} bytes of {want}")
    try:
        ftp.delete(f"{DIR}/{name}")
    except ftplib.error_perm:
        pass
    ftp.rename(part, f"{DIR}/{name}")
    dt = max(time.time() - t0, 1e-3)
    print(f"    {name}: {size / 2**20:.0f} MiB in {dt:.0f} s ({size / 2**20 / dt:.1f} MiB/s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profile")
    ap.add_argument("shards", nargs="?", help="the --out folder of pc/shard_model.py")
    ap.add_argument("--hosts", nargs="+")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    prof = json.loads(Path(a.profile).read_text(encoding="utf-8"))
    nodes = prof["nodes"]
    hosts = a.hosts or [n["host"] for n in nodes]
    if len(hosts) != len(nodes) or any(h.startswith("PS4_") for h in hosts):
        sys.exit(f"give the {len(nodes)} console addresses in node order with --hosts")
    for n, host in zip(nodes, hosts):
        k = n["node"]
        print(f"node {k} ({'head' if k == 0 else 'worker'}) at {host}: layers {n['layers'][0]}..{n['layers'][1] - 1}")
        ftp = connect(host)
        if a.remove:
            try:
                ftp.delete(f"{DIR}/model.idx")
                print("    removed model.idx (back to the packaged model)")
            except ftplib.error_perm:
                print("    no model.idx there")
            ftp.quit()
            continue
        if not a.shards:
            sys.exit("give the shards folder")
        src = Path(a.shards) / f"node{k}"
        put(ftp, src / f"node{k}.bin", f"node{k}.bin")
        put(ftp, src / "model.idx", "model.idx")   # last: the app only switches once the weights are complete
        if k == 0:
            lines = ["# DarkGarlic cluster: written by pc/cluster_deploy.py", f"name = {prof.get('name', 'DarkGarlic')}"]
            lines += [f"worker = {h}   # node {i + 1}" for i, h in enumerate(hosts[1:])]
            put(ftp, ("\n".join(lines) + "\n").encode(), "cluster.txt")
        ftp.quit()
    if not a.remove:
        print("Start DarkGarlic on the workers first, then on the head.")


if __name__ == "__main__":
    main()
