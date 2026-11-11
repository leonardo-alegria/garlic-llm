"""Parakeet TDT 0.6B v3 (nvidia/parakeet-tdt-0.6b-v3, CC-BY 4.0) in plain NumPy, written step by step as the PS4
will run it, and checked against transformers' ParakeetForTDT.

  features   pre-emphasis 0.97, |STFT|^2 (n_fft 512, Hann 400, hop 160, centred, zero padding), 128 Slaney mels,
             log(x + 2^-24), per-mel mean / std over the valid frames (L // 160 of them; the extra centred frame
             is dropped)
  encoder    subsampling: conv 3x3 /2 (1 -> 256) + ReLU, 2 x [depthwise 3x3 /2 + pointwise 1x1 + ReLU],
             flatten (channel-major) 4096 -> linear 1024; 24 Conformer blocks:
               x += 0.5 FF(LN x);  x += RelPosMHSA(LN x);  x += Conv(LN x);  x += 0.5 FF(LN x);  x = LN x
             FF = linear 4096, SiLU, linear 1024; attention 8 heads of 128, Transformer-XL relative positions
             (bias_u, bias_v, linear_pos); Conv = pointwise 2048, GLU, depthwise 9, BatchNorm, SiLU, pointwise
  decoder    encoder projector 1024 -> 640; prediction net: embedding 640, 2-layer LSTM 640, projector 640;
             joint relu(enc + dec) -> 8193 tokens (8192 = blank) + 5 durations; TDT greedy decoding

Run: python parakeet_ref.py [wav]   (compares with transformers on the CPU)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "parakeet-tdt-0.6b-v3"
BLANK, VOCAB, N_DUR = 8192, 8193, 5


def load_weights(model_dir=MODEL):
    from safetensors.numpy import load_file
    return load_file(str(model_dir / "model.safetensors"))


def mel_filters():
    import librosa
    return librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney").astype(np.float32)


def features(audio, fb=None):
    """audio float32 [L] at 16 kHz -> normalised log-mel [T = L // 160, 128]."""
    fb = mel_filters() if fb is None else fb
    x = np.concatenate([audio[:1], audio[1:] - 0.97 * audio[:-1]]).astype(np.float32)
    n_fft, hop, win = 512, 160, 400
    w = np.hanning(win + 1)[:-1] if False else np.hanning(win).astype(np.float64)   # torch.hann_window(periodic=False)
    window = np.zeros(n_fft)
    window[(n_fft - win) // 2:(n_fft - win) // 2 + win] = w
    xp = np.pad(x.astype(np.float64), (n_fft // 2, n_fft // 2))
    T = len(audio) // hop
    frames = np.stack([xp[t * hop:t * hop + n_fft] * window for t in range(T)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2                       # [T, 257]
    mel = np.log(spec @ fb.T.astype(np.float64) + 2.0 ** -24)              # [T, 128]
    mean = mel.mean(0)
    std = np.sqrt(((mel - mean) ** 2).sum(0) / (T - 1))
    return ((mel - mean) / (std + 1e-5)).astype(np.float32)


def conv2d(x, w, b, stride, pad, groups=1):
    """x [C, H, W], w [O, C / groups, k, k] -> [O, H', W'] (plain NumPy, small sizes)."""
    C, H, W = x.shape
    O, Cg, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    out = np.zeros((O, Ho, Wo), np.float32)
    og = O // groups
    for g in range(groups):
        xs = xp[g * Cg:(g + 1) * Cg]
        cols = np.stack([xs[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride] for i in range(k) for j in range(k)],
                        axis=1)                                            # [Cg, k*k, Ho, Wo]
        cols = cols.reshape(Cg * k * k, Ho * Wo)
        wg = w[g * og:(g + 1) * og].reshape(og, Cg * k * k)
        out[g * og:(g + 1) * og] = (wg @ cols).reshape(og, Ho, Wo)
    return out + (b[:, None, None] if b is not None else 0)


def layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def silu(x):
    return x / (1 + np.exp(-x))


def rel_pos_emb(T, d=1024):
    pos = np.arange(T - 1, -T, -1, dtype=np.float32)
    inv = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    f = pos[:, None] * inv[None, :]
    return np.stack([np.sin(f), np.cos(f)], axis=-1).reshape(2 * T - 1, d)


def rel_shift(bd):
    """[H, T, 2T-1] -> [H, T, T]: entry (i, j) = bd[i, T - 1 - i + j] (relative distance i - j)."""
    H, T, P = bd.shape
    idx = (T - 1 - np.arange(T))[:, None] + np.arange(T)[None, :]
    return np.take_along_axis(bd, np.broadcast_to(idx, (H, T, T)), axis=2)


def encoder(feat, w):
    E = "encoder."
    x = feat[None]                                                          # [1, T, 128]
    x = np.maximum(conv2d(x, w[E + "subsampling.layers.0.weight"], w[E + "subsampling.layers.0.bias"], 2, 1), 0)
    for a, b in ((2, 3), (5, 6)):
        x = conv2d(x, w[E + f"subsampling.layers.{a}.weight"], w[E + f"subsampling.layers.{a}.bias"], 2, 1,
                   groups=x.shape[0])
        x = np.maximum(conv2d(x, w[E + f"subsampling.layers.{b}.weight"], w[E + f"subsampling.layers.{b}.bias"], 1, 0), 0)
    C, T, F = x.shape
    x = x.transpose(1, 0, 2).reshape(T, C * F)
    x = x @ w[E + "subsampling.linear.weight"].T + w[E + "subsampling.linear.bias"]
    pe = rel_pos_emb(T)
    H, D = 8, 128
    for i in range(24):
        p = f"{E}layers.{i}."
        L = lambda n: (w[p + n + ".weight"], w[p + n + ".bias"])
        h = layer_norm(x, *L("norm_feed_forward1"))
        h = silu(h @ w[p + "feed_forward1.linear1.weight"].T) @ w[p + "feed_forward1.linear2.weight"].T
        x = x + 0.5 * h
        h = layer_norm(x, *L("norm_self_att"))
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, H, D).transpose(1, 0, 2)
        r = (pe @ w[p + "self_attn.relative_k_proj.weight"].T).reshape(2 * T - 1, H, D).transpose(1, 0, 2)
        qu = q + w[p + "self_attn.bias_u"][:, None, :]
        qv = q + w[p + "self_attn.bias_v"][:, None, :]
        s = (qu @ k.transpose(0, 2, 1) + rel_shift(qv @ r.transpose(0, 2, 1))) / np.sqrt(D)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(T, H * D)
        x = x + o @ w[p + "self_attn.o_proj.weight"].T
        h = layer_norm(x, *L("norm_conv"))
        h = h @ w[p + "conv.pointwise_conv1.weight"][:, :, 0].T             # [T, 2048]
        h = h[:, :1024] / (1 + np.exp(-h[:, 1024:]))                       # GLU
        dw = w[p + "conv.depthwise_conv.weight"][:, 0, :]                  # [1024, 9]
        hp = np.pad(h, ((4, 4), (0, 0)))
        h = sum(hp[j:j + T] * dw[:, j] for j in range(9))
        h = (h - w[p + "conv.norm.running_mean"]) / np.sqrt(w[p + "conv.norm.running_var"] + 1e-5) \
            * w[p + "conv.norm.weight"] + w[p + "conv.norm.bias"]
        h = silu(h) @ w[p + "conv.pointwise_conv2.weight"][:, :, 0].T
        x = x + h
        h = layer_norm(x, *L("norm_feed_forward2"))
        h = silu(h @ w[p + "feed_forward2.linear1.weight"].T) @ w[p + "feed_forward2.linear2.weight"].T
        x = x + 0.5 * h
        x = layer_norm(x, *L("norm_out"))
    return x.astype(np.float32)


class Decoder:
    """Prediction network + joint, one step at a time (as the PS4 runs it)."""

    def __init__(self, w):
        self.w = w
        self.h = [np.zeros(640, np.float32) for _ in range(2)]
        self.c = [np.zeros(640, np.float32) for _ in range(2)]
        self.out = None

    def step(self, token):
        w = self.w
        x = w["decoder.embedding.weight"][token]
        for l in range(2):
            g = w[f"decoder.lstm.weight_ih_l{l}"] @ x + w[f"decoder.lstm.bias_ih_l{l}"] + \
                w[f"decoder.lstm.weight_hh_l{l}"] @ self.h[l] + w[f"decoder.lstm.bias_hh_l{l}"]
            i, f, gg, o = np.split(g, 4)
            sig = lambda z: 1 / (1 + np.exp(-z))
            self.c[l] = sig(f) * self.c[l] + sig(i) * np.tanh(gg)
            self.h[l] = sig(o) * np.tanh(self.c[l])
            x = self.h[l]
        self.out = w["decoder.decoder_projector.weight"] @ x + w["decoder.decoder_projector.bias"]

    def joint(self, enc_proj_t):
        w = self.w
        return w["joint.head.weight"] @ np.maximum(enc_proj_t + self.out, 0) + w["joint.head.bias"]


def tdt_greedy(enc, w, max_symbols=10):
    proj = enc @ w["encoder_projector.weight"].T + w["encoder_projector.bias"]
    dec = Decoder(w)
    dec.step(BLANK)
    out, t, at_frame = [], 0, 0
    while t < len(proj):
        logits = dec.joint(proj[t])
        tok = int(np.argmax(logits[:VOCAB]))
        dur = int(np.argmax(logits[VOCAB:]))
        if tok != BLANK:
            out.append(tok)
            dec.step(tok)
            at_frame += 1
        if tok == BLANK and dur == 0:
            dur = 1
        if dur == 0 and at_frame >= max_symbols:
            dur = 1
        if dur > 0:
            at_frame = 0
        t += dur
    return out


def read_wav(path):
    import wave
    try:
        f = wave.open(str(path))
    except wave.Error:          # float WAV (e.g. FLEURS): soundfile, resampled if needed
        import librosa
        a, _ = librosa.load(str(path), sr=16000, mono=True)
        return a.astype(np.float32)
    with f:
        assert f.getframerate() == 16000 and f.getsampwidth() == 2, (f.getframerate(), f.getsampwidth())
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
        return a.reshape(-1, f.getnchannels()).mean(1)


def main():
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoProcessor, ParakeetForTDT
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "build" / "test_en.wav")
    audio = read_wav(path)
    w = load_weights()
    feat = features(audio)
    enc = encoder(feat, w)
    toks = tdt_greedy(enc, w)
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    print("NumPy:       ", repr(tok.decode(toks)))

    proc = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = ParakeetForTDT.from_pretrained(str(MODEL) if (MODEL / "config.json").exists() else
                                          "nvidia/parakeet-tdt-0.6b-v3").eval()
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    T = len(audio) // 160
    f_ref = inputs.input_features[0, :T].numpy()
    print(f"features: {feat.shape} vs {tuple(inputs.input_features.shape)}, max abs diff {np.abs(feat - f_ref).max():.2e}")
    with torch.no_grad():
        e_ref = model.get_audio_features(**inputs).last_hidden_state[0].numpy()
        n = min(len(enc), len(e_ref))
        print(f"encoder: {enc.shape} vs {e_ref.shape}, rel diff {np.abs(enc[:n] - e_ref[:n]).max() / np.abs(e_ref).max():.2e}")
        g = model.generate(**inputs)
        seq = g.sequences[0].tolist() if hasattr(g, "sequences") else g[0].tolist()
    ref_text = proc.batch_decode([seq], skip_special_tokens=True)[0]
    print("transformers:", repr(ref_text))
    np.savez(ROOT / "build" / f"parakeet_ref_{Path(path).stem}.npz", feat=feat, enc=enc, tokens=np.array(toks))


if __name__ == "__main__":
    main()
