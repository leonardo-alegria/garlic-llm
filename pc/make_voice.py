"""Convert any 5-15s reference .wav audio into a Garlic LLM PS3V voice file and optionally upload it to the PS4.

Custom voices live in /data/garlic/voices/<name>.bin on the console and are listed at GET /v1/audio/voices.
Use them in /v1/audio/speech or /v1/chat/completions with "voice": "<name>" (or "<lang>:<name>", e.g. "en:aria").

Requires torch + chatterbox-tts (or run via Modal: `python -m modal run pc/make_voice.py --wav aria.wav --name aria`).

Usage:
  python pc/make_voice.py aria.wav aria --ip 192.168.1.29
  python pc/make_voice.py aria.wav aria --out voices/aria.bin
"""

import ftplib
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import build as b  # noqa: E402


def build_ps3v(tok, mel, spk, gain_db=0.0):
    tok = np.asarray(tok, dtype=np.int32).reshape(-1)
    mel = np.asarray(mel, dtype=np.float32).reshape(-1, 80)
    spk = np.asarray(spk, dtype=np.float32).reshape(80)
    mel = np.ascontiguousarray(mel[: 2 * len(tok)])
    gain = float(10 ** (gain_db / 20.0))
    return struct.pack("<4sII", b"PS3V", len(tok), mel.shape[0]) + tok.tobytes() + mel.tobytes() + spk.tobytes() + struct.pack("<f", gain)


def convert_local(wav_path):
    try:
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        sys.exit(f"Missing dependency ({e}). Install torch, torchaudio, and chatterbox-tts to convert voices locally.")

    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        ref = model.s3gen.embed_ref(wav.squeeze(0).numpy(), 16000)
        spk = model.s3gen.flow.spk_embed_affine_layer(
            torch.nn.functional.normalize(ref["embedding"], dim=-1)
        )
    return build_ps3v(ref["prompt_token"].squeeze(0).cpu().numpy(),
                      ref["prompt_feat"].squeeze(0).cpu().numpy(),
                      spk.squeeze(0).cpu().numpy())


def upload_voice(data, name):
    ftp = b.ps4_ftp(timeout=30)
    for d in ("/data/garlic", "/data/garlic/voices"):
        try:
            ftp.mkd(d)
        except ftplib.error_perm:
            pass
    import io
    ftp.storbinary(f"STOR /data/garlic/voices/{name}.bin", io.BytesIO(data))
    ftp.quit()
    print(f"Uploaded /data/garlic/voices/{name}.bin ({len(data)} bytes)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 1:
        sys.exit("Usage: python pc/make_voice.py <ref.wav> [voice_name] [--ip <ps4_ip>] [--out <file.bin>]")
    wav_path = Path(args[0])
    name = (args[1] if len(args) > 1 else wav_path.stem).lower().replace(" ", "_")[:28]
    out_path = None
    for i, a in enumerate(sys.argv[:-1]):
        if a == "--out":
            out_path = Path(sys.argv[i + 1])
    data = convert_local(wav_path)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"Wrote {out_path} ({len(data)} bytes)")
    if b.PS4_IP:
        upload_voice(data, name)
    elif not out_path:
        out_path = ROOT / "models" / "voices" / f"{name}.bin"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"Wrote {out_path} ({len(data)} bytes); upload with --ip <PS4_IP>")


if __name__ == "__main__":
    main()
