"""One-command setup for Maple Preview (deepgrove, 20B MoE / 1B active) on Garlic LLM PS4.

Downloads the official GGUF + tokenizer from Hugging Face, converts weights with --t1i
(no re-quantization), compiles shaders/maple.cl into kernels.bin, and optionally uploads
the finished folder to /data/garlic/models/maple/ on the PS4 over FTP.

Usage:
  python pc/maple/prepare_maple.py
  python pc/maple/prepare_maple.py --upload --ip 192.168.1.29
"""
import ftplib
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import build as b  # noqa: E402

GGUF_URL = "https://huggingface.co/deepgrove/maple-preview-GGUF/resolve/main/maple-preview-TQ1_0-head-Q4_K.gguf"
TOK_URL = "https://huggingface.co/deepgrove/maple-preview/resolve/main/tokenizer.json"

SRC_DIR = ROOT / "models" / "maple"
GGUF_PATH = SRC_DIR / "maple-preview-TQ1_0-head-Q4_K.gguf"
TOK_PATH = SRC_DIR / "tokenizer.json"
OUT_DIR = ROOT / "models" / "maple-ps4"
REMOTE_DIR = "/data/garlic/models/maple"
PS4_FILES = ["model.idx", "maple-ternary-t1i.bin", "maple-embed.bin", "tokenizer.bin", "kernels.bin"]


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "garlic-llm"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        want = int(resp.headers.get("Content-Length", "0"))
    if dest.exists() and want and dest.stat().st_size == want:
        print(f"[skip] {dest.name} already downloaded ({want / 2**20:.1f} MiB)", flush=True)
        return
    print(f"[download] {dest.name} ({want / 2**20:.1f} MiB) <- {url}", flush=True)
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl:
        subprocess.run([curl, "-L", "-C", "-", "--fail", "-o", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)
    if want and dest.stat().st_size != want:
        sys.exit(f"Download size mismatch for {dest.name}: got {dest.stat().st_size}, expected {want}")


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


def upload_folder(out_dir: Path):
    ftp = b.ps4_ftp(timeout=60)
    for d in ("/data/garlic", "/data/garlic/models", REMOTE_DIR):
        try:
            ftp.mkd(d)
        except ftplib.error_perm:
            pass
    ftp.quit()
    for name in PS4_FILES:
        p = out_dir / name
        if not p.exists():
            sys.exit(f"Missing converted file: {p}")
        want = p.stat().st_size
        target = f"{REMOTE_DIR}/{name}"
        ftp = b.ps4_ftp(timeout=60)
        if remote_size(ftp, target) == want:
            print(f"[skip] {name}: already on PS4 ({want / 2**20:.1f} MiB)", flush=True)
            ftp.quit()
            continue
        t0 = time.time()
        with open(p, "rb") as f:
            ftp.storbinary(f"STOR {target}.part", f, blocksize=4 << 20)
        got = remote_size(ftp, f"{target}.part")
        if got != want:
            sys.exit(f"{name}: uploaded {got} of {want} bytes")
        try:
            ftp.delete(target)
        except ftplib.error_perm:
            pass
        ftp.rename(f"{target}.part", target)
        ftp.quit()
        dt = max(time.time() - t0, 1e-3)
        print(f"[upload] {name}: {want / 2**20:.1f} MiB -> {target} in {dt:.1f}s ({want / 2**20 / dt:.1f} MiB/s)", flush=True)


def main():
    download(TOK_URL, TOK_PATH)
    download(GGUF_URL, GGUF_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    need_convert = not all((OUT_DIR / f).exists() for f in ["model.idx", "maple-ternary-t1i.bin", "maple-embed.bin", "tokenizer.bin"])
    if need_convert or "--force" in sys.argv:
        print("[convert] Running pc/maple/convert_maple.py --t1i ...", flush=True)
        subprocess.run([sys.executable, str(HERE / "convert_maple.py"), "--t1i", str(GGUF_PATH), str(OUT_DIR)], check=True)
    else:
        print(f"[skip] Converted files already present in {OUT_DIR}", flush=True)

    kpath = OUT_DIR / "kernels.bin"
    if not kpath.exists() or "--force" in sys.argv:
        print("[kernels] Compiling shaders/maple.cl -> kernels.bin ...", flush=True)
        subprocess.run([sys.executable, str(HERE / "build_kernels.py"), str(kpath)], check=True)
    else:
        print(f"[skip] {kpath} already exists", flush=True)

    if "--upload" in sys.argv:
        upload_folder(OUT_DIR)
    else:
        print(f"\nReady in {OUT_DIR}. Upload to PS4 with:\n  python pc/maple/prepare_maple.py --upload --ip <PS4_IP>")


if __name__ == "__main__":
    main()
