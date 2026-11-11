"""Russian voice of the app: cloned from a Mozilla Common Voice 17 (CC0) recording, cleaned from the original MP3s
with Resemble Enhance. Writes:
  chat/assets/t3_prefix.bin       the "ru" block <- all 34 rows of this voice's prefix, the other blocks unchanged
  chat/assets/s3g_voice_ru.bin    S3Gen's voice for Russian, same "PS3V" layout as s3g_voice.bin (pc/export_s3g_assets.py):
                                  prompt tokens, prompt mel [2P][80], spk = affine(normalize(x-vector)) of the
                                  meanflow decoder (as models/chatterbox-voice.bin), then the gain for its audio (f32)
Check: the built-in voice through the same code must give s3g_voice.bin byte for byte.

Run: python pc/export_ru_voice.py   (needs torch, to read the voices' conds.pt files)
"""
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))
CB = ROOT / "models" / "chatterbox"
VC = CB / "accent_ru" / "vc"
ASSETS = ROOT / "chat" / "assets"
NAME = "ru_voice"
# The Russian voice comes out 4.2 dB quieter than the built-in one: the app multiplies its audio by this gain
# (chat/audio_out.c, with a limiter at 0.99). The built-in voice file has no gain (1).
GAIN_DB = 4.2


def voice_bin(conds_path, P):
    import s3gen_ref as R
    v = R.load_voice_conds(conds_path, P=P)
    tok, mel, spk = v["prompt_tok"].astype(np.int32), v["prompt_mel"].astype(np.float32), v["spk"].astype(np.float32)
    # the official embed_ref keeps tokens = frames // 2, so an odd clip has one extra mel frame (this voice: 216 tokens,
    # 433 frames); the PS4 layout is exactly 2 frames per prompt token: the last 20 ms frame goes
    assert 2 * len(tok) <= mel.shape[0] <= 2 * len(tok) + 1, mel.shape
    mel = np.ascontiguousarray(mel[: 2 * len(tok)])
    return struct.pack("<4sII", b"PS3V", len(tok), mel.shape[0]) + tok.tobytes() + mel.tobytes() + spk.tobytes()


def main():
    import s3gen_ref as R
    from safetensors.numpy import load_file
    P = R.prepare(R.load_raw())   # as pc/convert_chatterbox_s3gen.py (load_converted differs by ~4e-8 in spk)
    same = voice_bin(R.CONDS, P) == (ASSETS / "s3g_voice.bin").read_bytes()
    print("built-in voice recomputed == s3g_voice.bin:", same)
    assert same
    vb = voice_bin(VC / "conds" / f"{NAME}.pt", P) + struct.pack("<f", 10 ** (GAIN_DB / 20))
    (ASSETS / "s3g_voice_ru.bin").write_bytes(vb)
    np_, nmel = struct.unpack("<II", vb[4:12])
    print(f"s3g_voice_ru.bin: {np_} prompt tokens ({np_ / 25:.2f} s), {nmel} mel frames, gain {GAIN_DB:+.1f} dB, "
          f"{len(vb)} bytes")

    pre = ASSETS / "t3_prefix.bin"
    backup = VC / "t3_prefix_before_ru.bin"
    if not backup.exists():
        shutil.copy2(pre, backup)
    d = bytearray(backup.read_bytes())
    assert d[:4] == b"PT3P"
    n = struct.unpack("<I", d[4:8])[0]
    langs = [bytes(d[8 + 4 * i: 12 + 4 * i]).rstrip(b"\0").decode() for i in range(n)]
    i = langs.index("ru")
    block = load_file(str(VC / "prefixes.safetensors"))[f"{NAME}_B"].astype(np.float16)
    assert block.shape == (34, 1024)
    off = 8 + 4 * n + i * 34 * 2048
    d[off: off + 34 * 2048] = block.tobytes()
    pre.write_bytes(bytes(d))
    print(f"t3_prefix.bin: blocks {langs}, '{langs[i]}' (block {i}) <- {NAME}_B; backup {backup.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
