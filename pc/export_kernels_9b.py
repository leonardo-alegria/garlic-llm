"""The 9B's GPU kernels: shaders/chat_lab.cl (chat.cl plus spec.h) -> chat/assets_9b/kernels.bin.
The 4B's kernels.bin (shaders/chat.cl) is written by export_chat.py into chat/assets.

Run: python export_kernels_9b.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_chat  # noqa: E402

if __name__ == "__main__":
    export_chat.OUT = HERE.parent / "chat" / "assets_9b"
    export_chat.export_kernels("chat_lab")
