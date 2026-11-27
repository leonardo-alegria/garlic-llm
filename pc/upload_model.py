"""Upload model files to /data/garlic/models on the PS4 over FTP.

Once the models are in /data/garlic/models on the console, build the app without --full:
  python build_chat.py --upload
The package is then ~39 MiB (built in ~2 seconds) instead of 5.7 GiB.

Usage:
  python pc/upload_model.py                        # uploads all models/*.bin (skips any already on the console)
  python pc/upload_model.py models/qwen3.5-9b-q4s.bin ...
"""
import ftplib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build as b  # noqa: E402

REMOTE_DIR = "/data/garlic/models"


def remote_size(ftp, path):
    lines = []
    try:
        ftp.retrlines(f"LIST {path}", lines.append)
    except ftplib.all_errors:
        return -1
    if not lines:
        return -1
    parts = lines[0].split()
    return int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else -1


def ensure_dirs(ftp):
    for d in ("/data/garlic", REMOTE_DIR):
        try:
            ftp.mkd(d)
        except ftplib.error_perm:
            pass


args = []
force = False
it = iter(sys.argv[1:])
for a in it:
    if a in ("--ip", "-ip"):
        next(it, None)
    elif a in ("--force", "-f"):
        force = True
    elif not a.startswith("--ip="):
        args.append(a)

if args:
    force = True   # explicit file arguments always upload (e.g. after patching a model)
files = [Path(a) for a in args] if args else sorted((b.ROOT / "models").glob("*.bin"))
if not files:
    sys.exit("No model .bin files found in models/.")

ftp = b.ps4_ftp(timeout=60)
ensure_dirs(ftp)
ftp.quit()

for p in files:
    want = p.stat().st_size
    target = f"{REMOTE_DIR}/{p.name}"
    ftp = b.ps4_ftp(timeout=60)
    if not force and remote_size(ftp, target) == want:
        print(f"{p.name}: already in {REMOTE_DIR} ({want / 2**20:.0f} MiB), skipping", flush=True)
        ftp.quit()
        continue
    if not force and remote_size(ftp, f"/data/{p.name}") == want:
        ftp.rename(f"/data/{p.name}", target)
        print(f"{p.name}: moved /data/{p.name} -> {target}", flush=True)
        ftp.quit()
        continue
    t0 = time.time()
    with open(p, "rb") as f:
        ftp.storbinary(f"STOR {target}.part", f, blocksize=4 << 20)
    size = remote_size(ftp, f"{target}.part")
    if size != want:
        sys.exit(f"{p.name}: uploaded {size} bytes of {want}")
    try:
        ftp.delete(target)
    except ftplib.error_perm:
        pass
    ftp.rename(f"{target}.part", target)
    ftp.quit()
    dt = max(time.time() - t0, 1e-3)
    print(f"{p.name}: {size / 2**20:.0f} MiB -> {target} in {dt:.0f} s ({size / 2**20 / dt:.1f} MiB/s)", flush=True)
