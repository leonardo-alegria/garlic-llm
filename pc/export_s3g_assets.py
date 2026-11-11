"""App assets for S3Gen + HiFT (chat/engine_s3g.inc), from models/chatterbox-s3gen-mf.json and chatterbox-voice:
  chat/assets/s3g.idx   "PS3G", u32 count, char[128] weights path on the PS4, u64 file size, then per tensor
                        <32sQQQIIIII: name, offset, size, scales offset, rows, cols, fmt (0 f32, 1 q4i, 2 f16i,
                        3 f32t), conv input channels (0 = not a conv), true outputs (shape[0])
  chat/assets/s3g_voice.bin  "PS3V", u32 prompt tokens P, u32 prompt mel frames, then P x i32 tokens,
                        frames x 80 f32 prompt mel, 80 f32 speaker vector
Run: python export_s3g_assets.py   (and upload models/chatterbox-s3gen-mf.bin to /data on the PS4)
"""
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "chat" / "assets"
FMT = {"f32": 0, "q4i": 1, "f16i": 2, "f32t": 3}


def main():
    meta = json.loads((ROOT / "models" / "chatterbox-s3gen-mf.json").read_text())
    recs = b""
    for k, e in meta["tensors"].items():
        assert len(k) < 32, k
        recs += struct.pack("<32sQQQIIIII", k.encode(), e["offset"], e["size"], e.get("scales", 0), e.get("rows", 0),
                            e.get("cols", 0), FMT[e["fmt"]], e.get("conv", {}).get("cin", 0), e["shape"][0])
    head = struct.pack("<4sI128sQ", b"PS3G", len(meta["tensors"]), b"/data/chatterbox-s3gen-mf.bin", meta["size"])
    (OUT / "s3g.idx").write_bytes(head + recs)
    vm = json.loads((ROOT / "models" / "chatterbox-voice.json").read_text())
    blob = np.fromfile(ROOT / "models" / "chatterbox-voice.bin", np.uint8)
    get = lambda k, dt: blob[vm["tensors"][k]["offset"]: vm["tensors"][k]["offset"] + vm["tensors"][k]["size"]].view(dt)
    tok, mel, spk = get("prompt_tok", np.int32), get("prompt_mel", np.float32), get("spk", np.float32)
    (OUT / "s3g_voice.bin").write_bytes(struct.pack("<4sII", b"PS3V", len(tok), len(mel) // 80) + tok.tobytes()
                                        + mel.tobytes() + spk.tobytes())
    print(f"s3g.idx: {len(meta['tensors'])} tensors; s3g_voice.bin: {len(tok)} prompt tokens, {len(mel) // 80} frames")


if __name__ == "__main__":
    main()
