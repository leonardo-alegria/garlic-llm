"""Build Garlic LLM (chat/*.c + MicroPython + chat/assets) as a fake PKG with the OpenOrbis toolchain (see build.py).

Run: python build_chat.py           -> build/chat/IV0000-PSAI00001_00-PS4LOCALAICHAT01.pkg ("Garlic LLM", Qwen3.5-9B)
     python build_chat.py --4b      -> "Garlic LLM 4B" (PSAI00002), a second app that installs next to the first
     --upload                       -> also copies the package to the console (/data/pkg/), see PS4_IP in build.py
     --full                         -> the all-in-one package, with the weights inside: models/<file> goes to
                                       /app0/models, where the app looks first (chat/engine.c weights_locate)
The package gets chat/assets plus the files of its model: chat/assets_9b or chat/assets_4b.
Icon and home screen background: chat/sce_sys (pc/make_brand.py).
Without --full the model weights are not in the package: they stay in /data (see model.idx).
"""

import os
import shutil
import subprocess
import sys

import build as b

ROOT = b.ROOT
OUT = ROOT / "build" / "chat"
PKGDIR = OUT / "pkg"
if "--4b" in sys.argv:
    TITLE, VERSION, TITLE_ID, SUFFIX = "Garlic LLM 4B", "1.17", "PSAI00002", "02"
    REMOTE = "/data/pkg/Garlic-LLM-4B.pkg"
else:
    TITLE, VERSION, TITLE_ID, SUFFIX = "Garlic LLM", "1.19", "PSAI00001", "01"
    REMOTE = "/data/pkg/Garlic-LLM.pkg"
FULL = "--full" in sys.argv
# the weights of the all-in-one package (the names the .idx files give, see weights_locate)
FULL_MODELS = (["qwen3.5-4b-q4i.bin"] if "--4b" in sys.argv else ["qwen3.5-9b-q4s.bin", "qwen3.5-9b-mtp.bin"]) +     ["parakeet-v3-q4i.bin", "chatterbox-t3.bin", "chatterbox-s3gen-mf.bin"]
CONTENT_ID = f"IV0000-{TITLE_ID}_00-PS4LOCALAICHAT{SUFFIX}"
SOURCES = sorted((ROOT / "chat").glob("*.c"))
MP = ROOT / "chat" / "py" / "mp"
MP_SOURCES = sorted([*(MP / "py").glob("*.c"), *(MP / "extmod").glob("*.c"), *(MP / "shared" / "runtime").glob("*.c"),
                     ROOT / "chat" / "py" / "garlic_str.c"])
MP_FLAGS = [f"-I{ROOT / 'chat' / 'py'}", f"-I{MP}", "-DNDEBUG", "-std=gnu99", "-fno-strict-aliasing",
            "-Wno-unused-function", "-Wno-unused-variable", "-Wno-unused-but-set-variable"]
LIBS = ["-lc", "-lm", "-lkernel", "-lSceGnmDriver", "-lSceVideoOut", "-lScePad", "-lSceUserService", "-lSceSysmodule",
        "-lSceImeDialog", "-lSceCommonDialog", "-lSceAudioIn", "-lSceAudioOut", "-lSceSystemService", "-lSceNet"]


def compile_all():
    """Every C file, in parallel; MicroPython's objects are kept between builds while unchanged."""
    objdir = OUT / "obj"
    objdir.mkdir(parents=True, exist_ok=True)
    config = max(p.stat().st_mtime for p in (ROOT / "chat" / "py").glob("*.h"))
    jobs = []
    for src in SOURCES:
        model = ["-DGARLIC_4B"] if "--4b" in sys.argv else []
        jobs.append((src, objdir / (src.stem + ".o"), [*b.CFLAGS, *MP_FLAGS[:2], *model], True))
    for src in MP_SOURCES:
        obj = objdir / ("mp_" + src.parent.name + "_" + src.stem + ".o")
        fresh = obj.exists() and obj.stat().st_mtime > max(src.stat().st_mtime, config)
        jobs.append((src, obj, [*b.CFLAGS, *MP_FLAGS], not fresh))
    running = []
    for src, obj, flags, needed in jobs:
        if not needed:
            continue
        running.append((src, subprocess.Popen([str(b.ZIG), "cc", *map(str, flags), str(src), "-o", str(obj)], env=b.ENV)))
        if len(running) >= 8:
            for s, p in running:
                if p.wait():
                    sys.exit(f"compile failed: {s}")
            running = []
    for s, p in running:
        if p.wait():
            sys.exit(f"compile failed: {s}")
    return [obj for _, obj, _, _ in jobs]


def fix_gp4_dirs():
    """create-gp4 writes a fixed folder list; the package needs the real one (assets/py/tkinter...)."""
    gp4 = PKGDIR / "pkg.gp4"
    text = gp4.read_text(encoding="utf-8")
    start, end = text.index("<rootdir>"), text.index("</rootdir>") + len("</rootdir>")

    def tree(d, depth):
        out = ""
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            inner = tree(sub, depth + 1)
            pad = "\t" * depth
            out += f'{pad}<dir targ_name="{sub.name}">\n{inner}{pad}</dir>\n' if inner else f'{pad}<dir targ_name="{sub.name}" />\n'
        return out
    text = text[:start] + "<rootdir>\n" + tree(PKGDIR, 2) + "\t</rootdir>" + text[end:]
    gp4.write_text(text, encoding="utf-8")


def main():
    b.PKGDIR = PKGDIR            # build.pkgtool() runs in this folder
    shutil.rmtree(PKGDIR, ignore_errors=True)
    (PKGDIR / "sce_sys" / "about").mkdir(parents=True)
    (PKGDIR / "sce_module").mkdir()
    shutil.copytree(ROOT / "chat" / "assets", PKGDIR / "assets", ignore=shutil.ignore_patterns("__pycache__"))
    # the 9B: kernels.bin (chat_lab.cl), model.idx, mtp.idx; the 4B: model.idx (its kernels.bin is chat/assets')
    for f in (ROOT / "chat" / ("assets_4b" if "--4b" in sys.argv else "assets_9b")).iterdir():
        shutil.copy(f, PKGDIR / "assets" / f.name)
    if FULL:
        (PKGDIR / "models").mkdir()
        for name in FULL_MODELS:   # hard links: no second copy on the disk
            os.link(ROOT / "models" / name, PKGDIR / "models" / name)
    objs = compile_all()
    elf = OUT / "ps4ai.elf"
    b.run([b.ZIG, "ld.lld", "-m", "elf_x86_64", "-pie", "--script", b.OO / "link.x", "--eh-frame-hdr", *objs,
           "-L", b.OO / "lib", *LIBS, b.OO / "lib" / "crt1.o", "-o", elf])
    b.run([b.BIN / f"create-fself{b.EXE}", f"-in={elf}", f"-out={OUT / 'ps4ai.oelf'}", f"--eboot={PKGDIR / 'eboot.bin'}",
           "--paid", "0x3800000000000011"])
    sample = b.OO / "samples" / "hello_world"
    shutil.copy(sample / "sce_sys" / "about" / "right.sprx", PKGDIR / "sce_sys" / "about")
    shutil.copy(ROOT / "chat" / "sce_sys" / "icon0.png", PKGDIR / "sce_sys")
    shutil.copy(ROOT / "chat" / "sce_sys" / "pic1.png", PKGDIR / "sce_sys")
    for prx in (sample / "sce_module").glob("*.prx"):
        shutil.copy(prx, PKGDIR / "sce_module")
    sfo = "sce_sys/param.sfo"
    b.pkgtool("sfo_new", sfo)
    for key, typ, size, value in [("APP_TYPE", "Integer", 4, "1"), ("APP_VER", "Utf8", 8, VERSION),
                                  ("ATTRIBUTE", "Integer", 4, "0"), ("CATEGORY", "Utf8", 4, "gd"),
                                  ("CONTENT_ID", "Utf8", 48, CONTENT_ID), ("DOWNLOAD_DATA_SIZE", "Integer", 4, "0"),
                                  ("SYSTEM_VER", "Integer", 4, "0"), ("TITLE", "Utf8", 128, TITLE),
                                  ("TITLE_ID", "Utf8", 12, TITLE_ID), ("VERSION", "Utf8", 8, VERSION)]:
        b.pkgtool("sfo_setentry", sfo, key, "--type", typ, "--maxsize", str(size), "--value", value)
    files = ["eboot.bin", sfo, "sce_sys/about/right.sprx", "sce_sys/icon0.png", "sce_sys/pic1.png"]
    files += [f"sce_module/{p.name}" for p in (PKGDIR / "sce_module").glob("*.prx")]
    files += [p.relative_to(PKGDIR).as_posix() for p in sorted((PKGDIR / "assets").rglob("*")) if p.is_file()]
    if FULL:
        files += [f"models/{name}" for name in FULL_MODELS]
    b.run([b.BIN / f"create-gp4{b.EXE}", "-out", "pkg.gp4", f"--content-id={CONTENT_ID}", "--files", " ".join(files)],
          cwd=PKGDIR)
    fix_gp4_dirs()
    b.pkgtool("pkg_build", "pkg.gp4", str(OUT))
    pkg = OUT / f"{CONTENT_ID}.pkg"
    print("built", pkg, pkg.stat().st_size, "bytes")
    if "--upload" in sys.argv:
        ftp = b.ps4_ftp()
        with open(pkg, "rb") as f:
            ftp.storbinary(f"STOR {REMOTE}", f, blocksize=1 << 20)
        print(f"uploaded to {REMOTE}")
        ftp.quit()


if __name__ == "__main__":
    main()
