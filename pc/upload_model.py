"""Upload big model files to /data on the PS4 (as NAME.part, renamed when complete; LIST checks the size, since
GoldHEN's SIZE truncates to 32 bits).  python upload_model.py models/a.bin [models/b.bin ...]"""
import ftplib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build as b  # noqa: E402

for p in map(Path, sys.argv[1:]):
    t0 = time.time()
    ftp = b.ps4_ftp(timeout=60)
    with open(p, "rb") as f:
        ftp.storbinary(f"STOR /data/{p.name}.part", f, blocksize=4 << 20)
    lines = []
    ftp.retrlines(f"LIST /data/{p.name}.part", lines.append)
    size = int(lines[0].split()[4]) if lines else -1
    if size != p.stat().st_size:
        sys.exit(f"{p.name}: uploaded {size} bytes of {p.stat().st_size}")
    try:
        ftp.delete(f"/data/{p.name}")   # (the older copy of the same file, replaced by the new one)
    except ftplib.error_perm:
        pass
    ftp.rename(f"/data/{p.name}.part", f"/data/{p.name}")
    ftp.quit()
    print(f"{p.name}: {size / 2**20:.0f} MiB in {time.time() - t0:.0f} s ({size / 2**20 / (time.time() - t0):.1f} MiB/s)",
          flush=True)
