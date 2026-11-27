"""python pc/maple/build_kernels.py [out]: shaders/maple.cl -> kernels.bin (Garlic's PKRN format) for the Maple model folder. Fails on register spills or
unsupported kernel setups, as Garlic's pc/gpu.py does."""
import os, struct, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # the repository
_tc = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain") / "zig"
ZIG = os.environ.get("ZIG") or next((str(p) for p in (_tc / "zig", _tc / "zig.exe") if p.exists()), "zig")
CLANG_CL = [ZIG, "clang", "-x", "cl", "-cl-std=CL1.2", "-cl-no-stdinc", "-target", "amdgcn-amd-amdhsa",
            "-mcpu=bonaire", "-nogpulib", "-O3", "-ffast-math",
            "-Xclang", "-target-feature", "-Xclang", "-flat-for-global",    # FLAT instructions hang the PS4's GPU
            "-Xclang", "-target-feature", "-Xclang", "-enable-ds128"]       # ds128 is broken on the PS4's GPU


def elf_symbols(obj):
    shoff, = struct.unpack_from("<Q", obj, 0x28)
    entsize, num, strndx = struct.unpack_from("<HHH", obj, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", obj, shoff + i * entsize) for i in range(num)]
    names = secs[strndx][4]
    secname = [obj[names + s[0]: obj.index(b"\0", names + s[0])].decode() for s in secs]
    syms = {}
    for s in secs:
        if s[1] != 2:
            continue
        strtab = secs[s[6]][4]
        for off in range(s[4], s[4] + s[5], 24):
            name_off, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", obj, off)
            if name_off and shndx < len(secs):
                name = obj[strtab + name_off: obj.index(b"\0", strtab + name_off)].decode()
                syms[name] = (secname[shndx], value, size)
    return syms, {secname[i]: obj[s[4]: s[4] + s[5]] for i, s in enumerate(secs)}


def kernel_record(name, entry, kd):
    lds, scratch, _ = struct.unpack_from("<III", kd, 0)
    rsrc1, rsrc2, props = struct.unpack_from("<IIH", kd, 48)
    if scratch:
        sys.exit(f"{name} needs scratch memory (register spills): not supported")
    rsrc2 |= ((lds + 511) // 512) << 15
    kernarg, n = None, 0
    for bit, size in [(0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1)]:
        if props & (1 << bit):
            if bit == 3:
                kernarg = n
            elif bit not in (0, 5):
                sys.exit(f"{name} wants user SGPR feature {bit}: not supported")
            n += size
    return struct.pack("<32sIIIII", name.encode(), entry, rsrc1, rsrc2, n, kernarg or 0)


out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ROOT / "models" / "maple-ps4" / "kernels.bin"
obj = Path(tempfile.gettempdir()) / "maple_cl.o"
subprocess.run(CLANG_CL + ["-c", str(ROOT / "shaders" / "maple.cl"), "-o", str(obj)], check=True,
               stderr=subprocess.DEVNULL)
syms, sections = elf_symbols(obj.read_bytes())
text, recs, names = sections[".text"], [], []
for sym, (sec, value, _) in sorted(syms.items()):
    if sym.endswith(".kd"):
        name = sym[:-3]
        entry = syms[name][1]
        assert entry % 256 == 0, name
        recs.append(kernel_record(name, entry, sections[sec][value: value + 64]))
        names.append(name)
out.write_bytes(struct.pack("<4sII", b"PKRN", len(recs), len(text)) + b"".join(recs) + text)
print(f"{out}: {len(recs)} kernels, {len(text)} bytes of GPU code")
print(" ".join(names))
