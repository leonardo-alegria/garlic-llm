"""Shared build settings: where the tools are, the compiler flags and the connection to the console.

The app itself is built by build_chat.py. The tools are expected in toolchain/ (or in the folder named by the
GARLIC_TOOLCHAIN environment variable):
  OpenOrbis/PS4Toolchain   the OpenOrbis PS4 Toolchain (headers, libraries, link script, package tools)
  zig/zig.exe              Zig: its clang and lld compile the app and the GPU kernels
  dotnet/dotnet.exe        the .NET runtime that OpenOrbis' PkgTool needs
"""

import ftplib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLCHAIN = Path(os.environ.get("GARLIC_TOOLCHAIN") or ROOT / "toolchain")
OO = TOOLCHAIN / "OpenOrbis" / "PS4Toolchain"
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""                       # the tools' names on Windows and on Linux
ZIG = TOOLCHAIN / "zig" / f"zig{EXE}"
DOTNET = TOOLCHAIN / "dotnet" / "dotnet.exe"      # Windows only: PkgTool.Core.dll runs on it there
BIN = OO / ("bin/windows" if WIN else "bin/linux")
BUILD = ROOT / "build"
PKGDIR = BUILD / "pkg"

# The console's address for the uploads over FTP (GoldHEN's FTP server): set PS4_IP, and PS4_FTP_PORT if it is not 2121.
PS4_IP = os.environ.get("PS4_IP", "")
FTP_PORT = int(os.environ.get("PS4_FTP_PORT") or 2121)

CFLAGS = ["-target", "x86_64-freebsd-none", "-march=btver2", "-O2", "-fPIC", "-funwind-tables",
          "-fno-stack-protector", "-fshort-wchar", "-nostdinc", "-isystem", str(OO / "include"), "-Wall", "-c"]


ENV = dict(os.environ, DOTNET_ROLL_FORWARD="Major", OO_PS4_TOOLCHAIN=str(OO),
           DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1")   # PkgTool.Core needs no ICU library (Linux)


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=ENV, **kw)


def pkgtool(*args):
    if WIN:
        run([DOTNET, BIN / "PkgTool.Core.dll", *args], cwd=PKGDIR)
    else:
        run([BIN / "PkgTool.Core", *args], cwd=PKGDIR)


def ps4_ftp(timeout=30):
    """A logged-in FTP connection to the console."""
    if not PS4_IP:
        sys.exit("Set the PS4_IP environment variable to your console's address (and start GoldHEN's FTP server).")
    ftp = ftplib.FTP()
    ftp.connect(PS4_IP, FTP_PORT, timeout=timeout)
    ftp.login()
    return ftp
