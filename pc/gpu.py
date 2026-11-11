"""Run compute programs on the PS4 GPU: assemble, upload, build the command buffer, submit, time.

The Gpu class works through a host program on the console that is not part of this repository: `host` below is the
connection to it (garlic / garlic_size = its block of GPU memory, write / read / read32 of the console's memory, call
of a console function by name, wait, tsc_freq).

    from gpu import Gpu
    gpu = Gpu(host)
    prog = gpu.program("name")                   # shaders/name.s -> GCN machine code in GPU memory
    buf = gpu.alloc(4096)                        # GPU-visible memory (Garlic)
    secs = gpu.run([gpu.dispatch(prog, [*gpu.vsharp(buf, 4096)], groups=2)])

Layout of the Garlic block: command buffer at +0 (4 MiB), small scratch at +0x400000, programs from +0x410000,
data from +0x600000 (bump allocator).
"""

import os
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIG = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain") / "zig" / "zig.exe"
SHADERS = ROOT / "shaders"

# compute registers, as offsets from the SH register base (0x2C00)
START_X, NUM_THREAD_X, PGM_LO = 0x204, 0x207, 0x20C
PGM_RSRC1, RESOURCE_LIMITS, STATIC_THREAD_MGMT_SE0, TMPRING_SIZE, USER_DATA_0 = 0x212, 0x215, 0x216, 0x218, 0x240
EV_CS_PARTIAL_FLUSH = 7 | (4 << 8)
EV_BOTTOM_OF_PIPE_TS = 0x28 | (5 << 8)
EV_CACHE_FLUSH_AND_INV_TS = 0x14 | (5 << 8)
LABEL_DONE = 0x600DF00D
# ACQUIRE_MEM CP_COHER_CNTL: invalidate the per-CU vector L1 (TCL1) and the scalar cache (K$). L1 is not coherent
# between dispatches, so without this a kernel can read stale copies of what the previous kernel wrote.
COHER_TCL1_KCACHE = (1 << 22) | (1 << 27)
DCB_SIZE = 4 << 20


def pkt3(opcode, data, compute=False):
    return [(3 << 30) | ((len(data) - 1) << 16) | (opcode << 8) | (2 if compute else 0)] + list(data)


def set_sh(reg, *values):
    return pkt3(0x76, [reg, *values])


def eop(event, addr, data_sel, value=0):
    # EVENT_WRITE_EOP: data_sel 1 = write 32-bit value, 3 = write 64-bit GPU timestamp
    return pkt3(0x47, [event, addr & 0xFFFFFFFF, ((addr >> 32) & 0xFFFF) | (data_sel << 29), value, 0])


def text_section(obj):
    """Raw bytes of .text from an ELF64 object."""
    shoff, = struct.unpack_from("<Q", obj, 0x28)
    entsize, num, strndx = struct.unpack_from("<HHH", obj, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", obj, shoff + i * entsize) for i in range(num)]
    names = secs[strndx][4]
    for s in secs:
        if obj[names + s[0]: obj.index(b"\0", names + s[0])] == b".text":
            return obj[s[4]: s[4] + s[5]]
    raise ValueError("no .text")


class Program:
    def __init__(self, name, addr, code, vgprs, sgprs, user_sgprs, tgid_x=True, lds_bytes=0):
        self.name, self.addr, self.code = name, addr, code
        self.rsrc1 = ((vgprs - 1) // 4) | (((sgprs - 1) // 8) << 6) | (0xC0 << 12) | (1 << 21)
        self.rsrc2 = (user_sgprs << 1) | ((1 << 7) if tgid_x else 0) | (((lds_bytes + 511) // 512) << 15)


class Gpu:
    def __init__(self, host):
        self.host = host
        g = host.garlic
        self.dcb, self.scratch = g, g + DCB_SIZE
        self.args, self.label, self.stamps = self.scratch, self.scratch + 0x100, self.scratch + 0x200
        self.next_prog, self.next_data, self.end = g + DCB_SIZE + 0x10000, g + DCB_SIZE + 0x200000, g + host.garlic_size
        host.write(self.dcb, bytes(1024))
        host.call("sceGnmDrawInitDefaultHardwareState350", self.dcb, 256)
        self.init_state = host.read32(self.dcb, 256)
        self.gpu_hz = None

    def alloc(self, size, align=256):
        addr = (self.next_data + align - 1) // align * align
        if addr + size > self.end:
            raise MemoryError(f"Garlic block full ({size} bytes asked)")
        self.next_data = addr + size
        return addr

    def program(self, name, vgprs=16, sgprs=32, user_sgprs=4, tgid_x=True, lds_bytes=0):
        """Assemble shaders/<name>.s for gfx701 and copy its code into GPU memory."""
        src, obj = SHADERS / f"{name}.s", SHADERS / f"{name}.o"
        subprocess.run([str(ZIG), "cc", "-target", "amdgcn-amdhsa-none", "-mcpu=bonaire", "-c", str(src),
                        "-o", str(obj)], check=True)
        code = text_section(obj.read_bytes())
        addr = self.next_prog
        self.next_prog = (addr + len(code) + 255) // 256 * 256
        self.host.write(addr, code)
        return Program(name, addr, code, vgprs, sgprs, user_sgprs, tgid_x, lds_bytes)

    @staticmethod
    def vsharp(addr, size):
        """Buffer descriptor: base, stride 0, size in bytes, xyzw / 32-bit float format."""
        return [addr & 0xFFFFFFFF, (addr >> 32) & 0xFFFF, size, 0x27FAC]

    @staticmethod
    def dispatch(prog, user_data, groups, threads=64):
        words = set_sh(PGM_LO, (prog.addr >> 8) & 0xFFFFFFFF, (prog.addr >> 40) & 0xFF)
        words += set_sh(PGM_RSRC1, prog.rsrc1, prog.rsrc2)
        words += set_sh(NUM_THREAD_X, threads, 1, 1)
        words += set_sh(USER_DATA_0, *user_data)
        words += pkt3(0x15, [groups, 1, 1, 1], compute=True)   # DISPATCH_DIRECT
        return words

    def run(self, dispatch_list, timeout_ms=10000, profile=False):
        """Submit dispatches back to back; returns GPU seconds between start and end (needs calibrate()).
        profile=True also writes a GPU timestamp after each dispatch and returns (seconds, per-dispatch seconds)."""
        prof = self.scratch + 0x1000
        if profile and len(dispatch_list) > (0x10000 - 0x1000) // 8:
            raise ValueError("too many dispatches to profile")
        words = list(self.init_state)
        words += set_sh(RESOURCE_LIMITS, 0) + set_sh(STATIC_THREAD_MGMT_SE0, 0xFFFFFFFF, 0xFFFFFFFF)
        words += set_sh(TMPRING_SIZE, 0) + set_sh(START_X, 0, 0, 0)
        words += eop(EV_BOTTOM_OF_PIPE_TS, self.stamps, 3)
        for k, d in enumerate(dispatch_list):
            words += d
            words += pkt3(0x46, [EV_CS_PARTIAL_FLUSH])          # wait for the dispatch to finish...
            words += pkt3(0x58, [COHER_TCL1_KCACHE, 0xFFFFFFFF, 0, 0, 0, 0x0A])   # ...and drop stale L1 lines
            if profile:
                words += eop(EV_BOTTOM_OF_PIPE_TS, prof + 8 * k, 3)
        words += eop(EV_CACHE_FLUSH_AND_INV_TS, self.stamps + 8, 3)
        words += eop(EV_CACHE_FLUSH_AND_INV_TS, self.label, 1, LABEL_DONE)
        while len(words) % 8:
            words.append(0x80000000)
        blob = struct.pack(f"<{len(words)}I", *words)
        if len(blob) > DCB_SIZE:
            raise ValueError("command buffer too large")
        self.host.write(self.label, bytes(4) + bytes(16))
        self.host.write(self.stamps, bytes(16))
        self.host.write(self.dcb, blob)
        self.host.write(self.args, struct.pack("<QIIQI", self.dcb, len(blob), 0, 0, 0))
        r, _ = self.host.call("sceGnmSubmitCommandBuffers", 1, self.args, self.args + 8, self.args + 16, self.args + 24)
        if r & 0xFFFFFFFF:
            raise RuntimeError(f"sceGnmSubmitCommandBuffers -> {r & 0xFFFFFFFF:#x}")
        self.host.call("sceGnmSubmitDone")
        v, _, ok = self.host.wait(self.label, LABEL_DONE, timeout_ms=timeout_ms)
        if not ok:
            raise TimeoutError(f"GPU did not signal completion (label {v:#x})")
        t0, t1 = struct.unpack("<QQ", self.host.read(self.stamps, 16))
        secs = (t1 - t0) / self.gpu_hz if self.gpu_hz else t1 - t0
        if not profile:
            return secs
        n = len(dispatch_list)
        ts = struct.unpack(f"<{n}Q", self.host.read(prof, 8 * n))
        hz = self.gpu_hz or 1
        return secs, [(b - a) / hz for a, b in zip((t0,) + ts[:-1], ts)]

    def calibrate(self, pause=1.0):
        """GPU timestamp frequency, from two empty submissions timed with the PS4's CPU clock."""
        stamps = []
        for _ in range(2):
            cpu, _ = self.host.call("sceKernelReadTsc")
            self.run([])
            stamps.append((cpu, struct.unpack("<Q", self.host.read(self.stamps, 8))[0]))
            time.sleep(pause)
        (c0, g0), (c1, g1) = stamps
        self.gpu_hz = (g1 - g0) / ((c1 - c0) / self.host.tsc_freq)
        return self.gpu_hz


# ---- OpenCL C kernels compiled by LLVM (amdgcn-amd-amdhsa) -------------------------------------------------
# The compiler emits a kernel descriptor (.kd) per kernel with the exact COMPUTE_PGM_RSRC1/2 values and the list
# of user SGPRs the code expects. We feed those with our own PM4: s[0:3] private segment buffer (unused, zeros),
# then the kernel-argument pointer, then the workgroup id that the hardware appends.

CLANG = [str(ZIG), "clang", "-x", "cl", "-cl-std=CL1.2", "-cl-no-stdinc", "-target", "amdgcn-amd-amdhsa",
         "-mcpu=bonaire", "-nogpulib", "-O3", "-ffast-math",
         # global memory through buffer instructions (addr64): FLAT instructions hang the PS4's GPU
         "-Xclang", "-target-feature", "-Xclang", "-flat-for-global",
         # no ds_read_b128 / ds_write_b128: on the PS4's GPU they read and write the wrong LDS data
         # (float4 LDS arrays gave garbage; 64-bit and narrower DS ops are fine)
         "-Xclang", "-target-feature", "-Xclang", "-enable-ds128"]


def elf_symbols(obj):
    shoff, = struct.unpack_from("<Q", obj, 0x28)
    entsize, num, strndx = struct.unpack_from("<HHH", obj, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", obj, shoff + i * entsize) for i in range(num)]
    names = secs[strndx][4]
    secname = [obj[names + s[0]: obj.index(b"\0", names + s[0])].decode() for s in secs]
    syms = {}
    for s in secs:
        if s[1] != 2:                                   # SHT_SYMTAB
            continue
        strtab = secs[s[6]][4]
        for off in range(s[4], s[4] + s[5], 24):
            name_off, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", obj, off)
            if name_off and shndx < len(secs):
                name = obj[strtab + name_off: obj.index(b"\0", strtab + name_off)].decode()
                syms[name] = (secname[shndx], value, size)
    sections = {secname[i]: obj[s[4]: s[4] + s[5]] for i, s in enumerate(secs)}
    return syms, sections


class Kernel:
    def __init__(self, name, entry, kd):
        self.name, self.addr = name, entry
        self.lds, self.scratch, self.kernarg_size = struct.unpack_from("<III", kd, 0)
        rsrc1, rsrc2, props = struct.unpack_from("<IIH", kd, 48)
        if self.scratch:
            raise ValueError(f"{name} needs {self.scratch} bytes of scratch (spills); not supported")
        self.rsrc1 = rsrc1
        self.rsrc2 = rsrc2 | (((self.lds + 511) // 512) << 15)
        sizes = [(0, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 1)]     # private buf, dispatch, queue, kernarg...
        self.kernarg_sgpr, n = None, 0
        for bit, size in sizes:
            if props & (1 << bit):
                if bit == 3:
                    self.kernarg_sgpr = n
                elif bit not in (0, 5):
                    raise ValueError(f"{name} wants user SGPR feature {bit}; not supported")
                n += size
        self.user_sgprs = n


def load_cl(gpu, name):
    """Compile shaders/<name>.cl and upload its code; returns {kernel name: Kernel}."""
    src, obj = SHADERS / f"{name}.cl", SHADERS / f"{name}.o"
    subprocess.run(CLANG + ["-c", str(src), "-o", str(obj)], check=True)
    syms, sections = elf_symbols(obj.read_bytes())
    text = sections[".text"]
    base = (gpu.next_prog + 255) // 256 * 256
    gpu.host.write(base, text)
    gpu.next_prog = base + (len(text) + 255) // 256 * 256
    kernels = {}
    for sym, (sec, value, size) in syms.items():
        if sym.endswith(".kd"):
            kname = sym[:-3]
            entry = base + syms[kname][1]
            if entry % 256:
                raise ValueError(f"{kname} entry not 256-byte aligned")
            kernels[kname] = Kernel(kname, entry, sections[sec][value: value + 64])
    return kernels


class Args:
    """Kernel-argument buffers for one command buffer (each dispatch needs its own, they run later)."""

    def __init__(self, gpu, size=1 << 20):
        self.gpu, self.base = gpu, gpu.alloc(size)
        self.size, self.used, self.pending = size, 0, bytearray()

    def reset(self):
        self.used, self.pending = 0, bytearray()

    def add(self, packed):
        addr = self.base + self.used
        blob = bytes(packed) + bytes((-len(packed)) % 64)
        self.used += len(blob)
        if self.used > self.size:
            raise MemoryError("kernel-argument buffer full")
        self.pending += blob
        return addr

    def flush(self):
        if self.pending:
            self.gpu.host.write(self.base, bytes(self.pending))


def cl_dispatch(kernel, args_addr, groups, threads=64):
    user = [0, 0, 0, 0] + [args_addr & 0xFFFFFFFF, args_addr >> 32]
    words = set_sh(PGM_LO, (kernel.addr >> 8) & 0xFFFFFFFF, (kernel.addr >> 40) & 0xFF)
    words += set_sh(PGM_RSRC1, kernel.rsrc1, kernel.rsrc2)
    words += set_sh(NUM_THREAD_X, threads, 1, 1)
    words += set_sh(USER_DATA_0, *user[:kernel.user_sgprs])
    words += pkt3(0x15, [groups, 1, 1, 1], compute=True)
    return words
