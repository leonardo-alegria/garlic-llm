"""Chatterbox S3Gen speech decoder (the distilled "meanflow" variant of ResembleAI/chatterbox-turbo, MIT) and its
HiFT vocoder in plain NumPy, written step by step as the PS4 will run it, and checked against the official torch
code (the `chatterbox` package).

Layout. Every activation is frames-major [T][C] (one row per frame, as shaders/parakeet.cl). Every product is
X @ W.T + b with W [out][in]. A Conv1d over frames is the same product: its weight is stored [O][k*C] tap-major
(W[o][j*C + c] = torch_W[o][c][j]) and the input "row" of output frame t is the k*C floats X[t*stride + j*dil][c]
(j = 0..k-1) of a zero-padded buffer. With dil = 1 that row is contiguous in memory (k consecutive frames), so a
matrix kernel reads it in place with ldx = stride * C: no im2col buffer on the GPU. ConvTranspose1d (HiFT) and the
encoder's nearest x2 upsampling + conv are rewritten as ordinary stride-1 convs with (phases x Cout) outputs whose
result [T][u*Cout] is, as memory, the frames-major output [u*T][Cout] (polyphase form, built by prepare()).

Pipeline (P prompt tokens, N new tokens, T25 = P + N tokens at 25 Hz, T50 = 2 * T25 mel frames, M = 2N new ones):
  encoder  tokens -> embedding 6561 x 512 -> linear + LayerNorm (x sqrt 512, folded into the LN) -> look-ahead
           layer (conv k4 over t..t+3, leaky ReLU 0.01, conv k3 over t-2..t, + residual) -> 6 Conformer layers
           [x += RelPosMHSA(LN x); x += FF(LN x)]: 8 heads x 64, ESPnet relative positions (bias_u, bias_v,
           linear_pos), FF 2048 SiLU, LN eps 1e-12, no conv module -> x2 upsample (3-tap conv to 2 x 512) ->
           linear + LN (x sqrt 512) -> 4 Conformer layers -> LayerNorm -> encoder_proj 512 -> 80 = mu [T50][80]
  CFM      meanflow, 2 Euler steps (t, r) = (0, .5), (.5, 1), no CFG: x += (r - t) * v(x | mu | spk | cond, t, r).
           x starts as Gaussian noise over ALL T50 frames (prompt frames included); cond = prompt mel on the first
           2P frames, zeros after; spk = 80-vector, the same on every frame. All T50 frames are re-decoded (full
           attention, so the prompt shapes the new frames) and the last M are kept.
           Estimator (1-D U-Net, causal convs): in [T50][320] -> resnet 320->256 + 4 transformers -> causal conv
           k3 (skip saved before it) -> 12 x (resnet + 4 transformers) -> concat(x, skip) 512 -> resnet + 4
           transformers -> causal conv k3 -> final block -> 1x1 conv 256 -> 80.
           resnet: h = block1(x) + tb; h = block2(h); out = h + res_conv(x); block = causal conv k3, LN, Mish.
           tb = Linear(Mish(temb)), temb = mixer([MLP(sin(t)) | MLP(sin(r))]): it only depends on (t, r), so
           prepare() computes the 14 x 2 vectors once (the GPU never runs the time MLPs).
           transformer: x += Attn(LN x) (8 heads x 64 over dim 256, full, scale 1/8); x += FF(LN x) (GELU, 1024).
  HiFT     mel [M][80] -> F0 (5 x [conv k3 + ELU], linear 512 -> 1, abs) -> NSF source at 24 kHz: 9 harmonics
           0.1 sin(2 pi cumsum(h f0 / 24000) + phi_h) on voiced samples (f0 > 10 Hz) + noise, linear 9 -> 1,
           tanh -> STFT (n_fft 16, hop 4, Hann, centred, reflect) [120M + 1][18] = re 0..8 | im 0..8.
           x = conv_pre k7 80 -> 512; 3 x [leaky 0.1, ConvTranspose x8 / x5 / x3 (reflection pad 1 frame after
           the last), + source branch (strided conv of the STFT, ResBlock), mean of 3 ResBlocks (k 3/7/11,
           dilations 1/3/5, Snake)]; leaky 0.01; conv_post k7 64 -> 18; magnitude = min(exp, 100), phase = sin;
           iSTFT -> [480 M] samples; clamp +-0.99.
  output   first 20 ms zeroed, next 20 ms faded in, the last token's 40 ms dropped (the watermark is skipped).

Run (needs torch and, for the comparison, the official `chatterbox` package):
  python pc/s3gen_ref.py                  validate vs torch (es, ru)
  python pc/s3gen_ref.py --quant [es,ru]  fp32 vs models/chatterbox-s3gen-mf
"""

import json
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # 4 threads unless set otherwise
    os.environ.setdefault(_v, "4")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CB = ROOT / "models" / "chatterbox"
WEIGHTS = CB / "hf_turbo" / "s3gen_meanflow.safetensors"
CONDS = CB / "results" / "voice_builtin_reclone" / "conds.pt"
MODEL = ROOT / "models" / "chatterbox-s3gen-mf"
VOICE = ROOT / "models" / "chatterbox-voice"
OUT = CB / "s3gen_port"
SR, HOP = 24000, 480                 # 24 kHz, 50 mel frames per second, 2 frames per speech token
F32 = np.float32
STEPS = ((0.0, 0.5), (0.5, 1.0))     # meanflow schedule: linspace(0, 1, 3), no cosine warp
N_ENC, N_ENC25 = 10, 6               # Conformer layers (the first 6 at 25 Hz)
N_RES, N_TF = 14, 56                 # CFM resnets / transformer blocks
UPS = ((8, 16, 256), (5, 11, 128), (3, 7, 64))          # HiFT ConvTranspose: stride, kernel, out channels
SRC_DOWN = ((30, 15, 7), (6, 3, 1), (1, 1, 0))          # source convs on the STFT: kernel, stride, padding
SRC_RES_K = (7, 7, 11)
RES_K = (3, 7, 11)
DILS = (1, 3, 5)


# ---- elementwise functions (as the kernels will compute them) --------------------------------------------
def silu(x):
    return x / (1 + np.exp(-x))


def mish(x):
    """x tanh(softplus x) = x n (n + 2) / (n (n + 2) + 2), n = e^x (x > 20: x)."""
    n = np.exp(np.minimum(x, F32(20)))
    p = n * (n + 2)
    return np.where(x > 20, x, x * p / (p + 2)).astype(F32)


def erfc_pos(z):
    """erfc(z) for z >= 0, relative error < 1.2e-7 (Numerical Recipes erfcc, Chebyshev fit)."""
    t = 1 / (1 + F32(0.5) * z)
    c = (-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587,
         -0.82215223, 0.17087277)
    p = F32(c[9])
    for k in range(8, 0, -1):
        p = F32(c[k]) + t * p
    return t * np.exp(-z * z + F32(c[0]) + t * p)


def gelu(x):
    """Exact (erf) GELU: x Phi(x), via erfc so both tails keep their relative precision."""
    c = F32(0.5) * erfc_pos(np.abs(x) * F32(0.7071067811865476))
    return np.where(x >= 0, x * (1 - c), x * c).astype(F32)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(F32)


def leaky(x, slope):
    return np.where(x >= 0, x, x * F32(slope)).astype(F32)


def snake(x, alpha, inv_alpha):
    """x + 1 / (alpha + 1e-9) sin^2(alpha x), per channel (the converter stores inv_alpha = 1 / (alpha + 1e-9))."""
    return x + inv_alpha * np.sin(x * alpha) ** 2


def layer_norm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + F32(eps)) * w + b


def softmax(s):
    s = np.exp(s - s.max(-1, keepdims=True))
    return s / s.sum(-1, keepdims=True)


# ---- convolutions on frames-major data ---------------------------------------------------------------------
def conv(x, w, b, k, pad=(0, 0), stride=1, dil=1, chunk=8192):
    """x [T][C] -> [T'][O]. w [O][>= k*C] tap-major (extra columns, if any, are the converter's zero padding).
    Output frame t reads frames t*stride + j*dil - pad[0], j < k, of x zero-padded by pad[0] / pad[1] frames."""
    T, C = x.shape
    xp = np.pad(x, ((pad[0], pad[1]), (0, 0)))
    To = (T + pad[0] + pad[1] - dil * (k - 1) - 1) // stride + 1
    wt = np.ascontiguousarray(w[:, :k * C].T)
    out = []
    for t0 in range(0, To, chunk):                         # chunks only bound the PC's im2col memory
        n = min(chunk, To - t0)
        cols = np.concatenate([xp[t0 * stride + j * dil: t0 * stride + j * dil + (n - 1) * stride + 1: stride]
                               for j in range(k)], axis=1)
        out.append(cols @ wt)
    y = np.concatenate(out)
    return y + b if b is not None else y


# ---- encoder ------------------------------------------------------------------------------------------------
def rel_pos_emb(T, d=512):
    """ESPnet relative positional encoding [2T - 1][d]: row i = distance T - 1 - i, interleaved sin / cos (fp32,
    computed like torch: pos * div_term rounded to fp32, then sin / cos)."""
    pos = np.arange(T - 1, -T, -1, dtype=F32)
    div = np.exp(np.arange(0, d, 2, dtype=F32) * F32(-(math.log(10000.0) / d)))
    f = pos[:, None] * div[None, :]
    pe = np.empty((2 * T - 1, d), F32)
    pe[:, 0::2] = np.sin(f)
    pe[:, 1::2] = np.cos(f)
    return pe


def rel_shift(bd):
    """[H, T, 2T-1] -> [H, T, T]: entry (i, j) = bd[i, T - 1 - i + j] (relative distance i - j)."""
    H, T, _ = bd.shape
    idx = (T - 1 - np.arange(T))[:, None] + np.arange(T)[None, :]
    return np.take_along_axis(bd, np.broadcast_to(idx, (H, T, T)), axis=2)


def conformer(x, P, i, pe, H=8, D=64):
    """x += RelPosMHSA(LN x); x += FF(LN x). QKV [T][1536] = q | k | v; R = pe @ Wpos.T [2T-1][512]."""
    p, T = f"enc.{i}.", len(x)
    h = layer_norm(x, P[p + "ln1_w"], P[p + "ln1_b"], 1e-12)
    qkv = h @ P[p + "wqkv"].T + P[p + "bqkv"]
    q, k, v = (qkv[:, n * 512:(n + 1) * 512].reshape(T, H, D).transpose(1, 0, 2) for n in range(3))
    r = (pe @ P[p + "wpos"].T).reshape(2 * T - 1, H, D).transpose(1, 0, 2)
    qu = q + P[p + "bias_u"].reshape(H, 1, D)
    qv = q + P[p + "bias_v"].reshape(H, 1, D)
    s = softmax((qu @ k.transpose(0, 2, 1) + rel_shift(qv @ r.transpose(0, 2, 1))) / F32(8))
    o = (s @ v).transpose(1, 0, 2).reshape(T, H * D)
    x = x + o @ P[p + "wo"].T + P[p + "bo"]
    h = layer_norm(x, P[p + "ln2_w"], P[p + "ln2_b"], 1e-12)
    return x + silu(h @ P[p + "ff1_w"].T + P[p + "ff1_b"]) @ P[p + "ff2_w"].T + P[p + "ff2_b"]


def encoder(tokens, P, trace=None):
    """tokens [T25] (prompt | new) -> mu [T50][80]. trace: optional dict that receives the intermediate stages."""
    tr = trace if trace is not None else {}
    T = len(tokens)
    x = P["enc.emb"][np.asarray(tokens)]
    x = layer_norm(x @ P["enc.in_w"].T + P["enc.in_b"], P["enc.in_ln_w"], P["enc.in_ln_b"], 1e-5)
    tr["embed"] = x
    h = leaky(conv(x, P["enc.la1_w"], P["enc.la1_b"], 4, pad=(0, 3)), 0.01)      # look-ahead: t .. t+3
    x = x + conv(h, P["enc.la2_w"], P["enc.la2_b"], 3, pad=(2, 0))                # causal: t-2 .. t
    tr["lookahead"] = x
    pe = rel_pos_emb(T)
    for i in range(N_ENC25):
        x = conformer(x, P, i, pe)
    tr["layers25"] = x
    x = conv(x, P["enc.up_w"], P["enc.up_b"], 3, pad=(2, 0)).reshape(2 * T, 512)  # [T][2*512] = [2T][512]
    tr["upsample"] = x
    x = layer_norm(x @ P["enc.up_in_w"].T + P["enc.up_in_b"], P["enc.up_ln_w"], P["enc.up_ln_b"], 1e-5)
    tr["up_embed"] = x
    pe = rel_pos_emb(2 * T)
    for i in range(N_ENC25, N_ENC):
        x = conformer(x, P, i, pe)
    x = layer_norm(x, P["enc.out_ln_w"], P["enc.out_ln_b"], 1e-5)
    tr["layers50"] = x
    mu = x @ P["enc.proj_w"].T + P["enc.proj_b"]
    tr["mu"] = mu
    return mu


# ---- CFM estimator ------------------------------------------------------------------------------------------
def causal_block(x, P, p):
    """causal conv k3 (frames t-2..t) -> LayerNorm over channels -> Mish."""
    return mish(layer_norm(conv(x, P[p + "_w"], P[p + "_b"], 3, pad=(2, 0)), P[p + "_ln_w"], P[p + "_ln_b"], 1e-5))


def resnet(x, P, n, step):
    p = f"cfm.res{n}."
    h = causal_block(x, P, p + "c1") + P[f"cfm.tb{step}.{n}"]
    h = causal_block(h, P, p + "c2")
    return h + x @ P[p + "res_w"].T + P[p + "res_b"]


def transformer(x, P, m, H=8, D=64):
    p, T = f"cfm.tf{m}.", len(x)
    h = layer_norm(x, P[p + "ln1_w"], P[p + "ln1_b"], 1e-5)
    qkv = h @ P[p + "wqkv"].T                                                   # [T][1536], no bias
    q, k, v = (qkv[:, n * 512:(n + 1) * 512].reshape(T, H, D).transpose(1, 0, 2) for n in range(3))
    o = (softmax(q @ k.transpose(0, 2, 1) * F32(0.125)) @ v).transpose(1, 0, 2).reshape(T, H * D)
    x = x + o @ P[p + "wo"].T + P[p + "bo"]
    h = layer_norm(x, P[p + "ln3_w"], P[p + "ln3_b"], 1e-5)
    return x + gelu(h @ P[p + "ff1_w"].T + P[p + "ff1_b"]) @ P[p + "ff2_w"].T + P[p + "ff2_b"]


def estimator(xin, P, step):
    """xin [T50][320] = x | mu | spk | cond -> velocity [T50][80]."""
    x = resnet(xin, P, 0, step)
    for m in range(4):
        x = transformer(x, P, m)
    skip = x
    x = conv(x, P["cfm.down_w"], P["cfm.down_b"], 3, pad=(2, 0))
    for n in range(1, 13):
        x = resnet(x, P, n, step)
        for m in range(4 * n, 4 * n + 4):
            x = transformer(x, P, m)
    x = resnet(np.concatenate([x, skip], axis=1), P, 13, step)
    for m in range(52, 56):
        x = transformer(x, P, m)
    x = conv(x, P["cfm.up_w"], P["cfm.up_b"], 3, pad=(2, 0))
    x = causal_block(x, P, "cfm.fin")
    return x @ P["cfm.proj_w"].T + P["cfm.proj_b"]


def cfm(mu, spk, prompt_mel, noise, P, trace=None):
    """mu [T50][80], spk [80], prompt_mel [2P][80], noise [T50][80] (N(0, 1)) -> mel of the new frames [M][80]."""
    T, Tp = len(mu), len(prompt_mel)
    cond = np.zeros((T, 80), F32)
    cond[:Tp] = prompt_mel
    spk_rows = np.broadcast_to(spk.astype(F32), (T, 80))
    x = noise.astype(F32)
    for step, (t, r) in enumerate(STEPS):
        v = estimator(np.concatenate([x, mu, spk_rows, cond], axis=1), P, step)
        if trace is not None:
            trace[f"v{step}"] = v
        x = x + F32(r - t) * v
    return x[Tp:]


# ---- HiFT vocoder -------------------------------------------------------------------------------------------
def f0_predictor(mel, P):
    """mel [M][80] -> f0 [M] in Hz."""
    x = mel
    for i in range(5):
        x = elu(conv(x, P[f"hift.f0c{i}_w"], P[f"hift.f0c{i}_b"], 3, pad=(1, 1)))
    return np.abs(x @ P["hift.f0_cls_w"] + P["hift.f0_cls_b"][0])


def harmonic_phase(f0, mode="gpu"):
    """Phase in cycles (0..1) of the 9 harmonics on every 24 kHz sample: frac(cumsum(f0 h / 24000)), [9][480M].
    mode "torch": as the official code (fp32 increments, cumsum accumulated in double by torch on the CPU, rounded
      to fp32, then % 1), so its rounding grows with the running sum (~2e-3 cycles after 10 s).
    mode "gpu": f0 is constant over each 480-sample frame, so phase(480 m + k) = frac(C_mh + (k + 1) inc_mh) with
      C_mh = frac(sum_{m' < m} 480 inc_m'h) (a scan over frames, in double, by one thread) and inc in fp32."""
    h = np.arange(1, 10, dtype=F32)
    if mode == "torch":
        inc = np.repeat(f0.astype(F32), HOP)[None, :] * h[:, None] / F32(SR)
        return np.remainder(np.cumsum(inc.astype(np.float64), axis=1).astype(F32), F32(1))
    inc = f0.astype(F32)[None, :] * h[:, None] / F32(SR)                   # [9][M]
    tot = np.cumsum(inc.astype(np.float64) * HOP, axis=1)
    start = np.remainder(np.concatenate([np.zeros((9, 1)), tot[:, :-1]], axis=1), 1.0).astype(F32)
    k1 = np.arange(1, HOP + 1, dtype=F32)
    ph = start[:, :, None] + k1[None, None, :] * inc[:, :, None]           # [9][M][480]
    return np.remainder(ph, F32(1)).reshape(9, -1)


def source(f0, P, phase_vec, noise, mode="gpu"):
    """NSF excitation: f0 [M] -> s [480M]. phase_vec [9] (uniform in -pi..pi, [0] = 0), noise [9][480M] N(0, 1)."""
    theta = F32(2 * np.pi) * harmonic_phase(f0, mode)
    sine = F32(0.1) * np.sin(theta + phase_vec.astype(F32)[:, None])
    uv = (np.repeat(f0.astype(F32), HOP) > 10).astype(F32)[None, :]
    amp = uv * F32(0.003) + (1 - uv) * F32(0.1) / F32(3)
    sine = sine * uv + amp * noise
    return np.tanh(P["hift.src_w"] @ sine + P["hift.src_b"][0])


def stft_basis():
    """[18][16]: rows 0..8 = window x cos(2 pi k n / 16), rows 9..17 = -window x sin (torch.stft, onesided)."""
    n = np.arange(16)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / 16)                             # periodic Hann
    k = np.arange(9)[:, None]
    return np.concatenate([w * np.cos(2 * np.pi * k * n / 16), -w * np.sin(2 * np.pi * k * n / 16)]).astype(F32)


def istft_basis():
    """[16][18]: frame sample n = window[n] x irfft(re | im)[n] (the imaginary DC / Nyquist terms vanish)."""
    n = np.arange(16)[:, None]
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / 16)
    k = np.arange(9)[None, :]
    a = np.where((k == 0) | (k == 8), 1.0, 2.0) / 16
    return np.concatenate([w * a * np.cos(2 * np.pi * k * n / 16), -w * a * np.sin(2 * np.pi * k * n / 16)],
                          axis=1).astype(F32)


def stft16(s):
    """s [L] -> [L/4 + 1][18] (n_fft 16, hop 4, centred with reflect padding of 8)."""
    xp = np.pad(s, 8, mode="reflect")
    F = len(s) // 4 + 1
    frames = np.lib.stride_tricks.sliding_window_view(xp, 16)[::4][:F]
    return frames @ stft_basis().T


def istft16(spec):
    """spec [F][18] (re | im) -> [4 (F - 1)]: inverse DFT x window, overlap-add (hop 4), / overlap-added
    window^2, centred (the first and last 8 samples dropped)."""
    F = len(spec)
    fr = spec @ istft_basis().T                                            # [F][16]
    y = np.zeros(4 * (F - 1) + 16, F32)
    env = np.zeros_like(y)
    w2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(16) / 16)).astype(F32) ** 2
    for j in range(4):                         # frames f = j mod 4 never overlap each other
        blk = fr[j::4]
        n = len(blk)
        y[4 * j: 4 * j + 16 * n] += blk.reshape(-1)
        env[4 * j: 4 * j + 16 * n] += np.tile(w2, n)
    return y[8: 8 + 4 * (F - 1)] / env[8: 8 + 4 * (F - 1)]


def resblock(x, P, p, k):
    """3 x [x += conv_k(Snake(conv_k,dil(Snake(x))))], dilations 1, 3, 5 then 1."""
    for l, d in enumerate(DILS):
        q = f"{p}.{l}."
        t = snake(x, P[q + "a1"], P[q + "ia1"])
        t = conv(t, P[q + "c1_w"], P[q + "c1_b"], k, pad=(d * (k - 1) // 2,) * 2, dil=d)
        t = snake(t, P[q + "a2"], P[q + "ia2"])
        x = x + conv(t, P[q + "c2_w"], P[q + "c2_b"], k, pad=((k - 1) // 2,) * 2)
    return x


def hift_decode(mel, s, P, trace=None):
    """mel [M][80], excitation s [480M] -> waveform [480M] (before the post-processing)."""
    st = stft16(s)                                                         # [120M + 1][18]
    x = conv(mel, P["hift.pre_w"], P["hift.pre_b"], 7, pad=(3, 3))         # [M][512]
    for i, (u, k_up, cout) in enumerate(UPS):
        x = leaky(x, 0.1)
        x = conv_transpose(x, P, f"hift.up{i}", u, k_up, cout)                     # [u T][cout]
        if i == 2:
            x = np.concatenate([x[1:2], x])                                # ReflectionPad1d((1, 0))
        k, stv, pd = SRC_DOWN[i]
        si = conv(st, P[f"hift.sd{i}_w"], P[f"hift.sd{i}_b"], k, pad=(pd, pd), stride=stv)
        x = x + resblock(si, P, f"hift.sr{i}", SRC_RES_K[i])
        xs = resblock(x, P, f"hift.rb{i}.0", RES_K[0])
        xs = xs + resblock(x, P, f"hift.rb{i}.1", RES_K[1])
        xs = xs + resblock(x, P, f"hift.rb{i}.2", RES_K[2])
        x = xs / F32(3)
        if trace is not None:
            trace[f"hift_stage{i}"] = x
    x = conv(leaky(x, 0.01), P["hift.post_w"], P["hift.post_b"], 7, pad=(3, 3))   # [120M + 1][18]
    mag = np.minimum(np.exp(x[:, :9]), F32(100))
    ph = np.sin(x[:, 9:])
    wav = istft16(np.concatenate([mag * np.cos(ph), mag * np.sin(ph)], axis=1))
    return np.clip(wav, F32(-0.99), F32(0.99))


def postprocess(wav, n_tokens):
    """Official: the first 20 ms zeroed and the next 20 ms faded in; the last token's 40 ms dropped."""
    n = SR // 50
    fade = np.zeros(2 * n, F32)
    fade[n:] = (np.cos(np.linspace(np.pi, 0, n, dtype=F32)) + 1) / 2
    wav = wav.copy()
    wav[:2 * n] *= fade
    return wav[:max(1, n_tokens - 1) * 2 * HOP]


def synthesize(tokens, P, voice, rng, mode="gpu", trace=None):
    """New speech tokens [N] -> 24 kHz waveform (post-processed). The random inputs come from rng."""
    pt = voice["prompt_tok"]
    T50 = 2 * (len(pt) + len(tokens))
    noise = rng.standard_normal((T50, 80)).astype(F32)
    mu = encoder(np.concatenate([pt, tokens]), P)
    mel = cfm(mu, voice["spk"], voice["prompt_mel"], noise, P)
    f0 = f0_predictor(mel, P)
    phase = rng.uniform(-np.pi, np.pi, 9).astype(F32)
    phase[0] = 0
    s = source(f0, P, phase, rng.standard_normal((9, len(mel) * HOP)).astype(F32), mode)
    if trace is not None:
        trace.update(mu=mu, mel=mel, f0=f0, s=s)
    return postprocess(hift_decode(mel, s, P), len(tokens))


# ---- weights: official safetensors -> GPU layout ----------------------------------------------------------
def load_raw(path=WEIGHTS):
    """The inference tensors of the official file (S3 tokenizer and CAMPPlus, for voice cloning, are skipped)."""
    from safetensors import safe_open
    with safe_open(str(path), "np") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith(("flow.", "mel2wav."))}


def taps(w):
    """torch Conv1d weight [O][C][k] -> [O][k*C] tap-major."""
    O, C, k = w.shape
    return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(O, k * C), dtype=F32)


def fold_wn(w, p):
    """weight_norm (dim 0): W = g v / ||v|| over all axes but the first."""
    g, v = w[p + ".parametrizations.weight.original0"], w[p + ".parametrizations.weight.original1"]
    n = np.sqrt((v.astype(np.float64) ** 2).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
    return (v * (g / n)).astype(F32)


def poly_groups(u, k, pad):
    """Polyphase geometry of a ConvTranspose1d (stride u, kernel k, padding pad): output frame u q + r gets
    W[:, :, j]^T x[q + (r + pad - j) / u] for every tap j with r + pad - j divisible by u. Consecutive phases r
    reading the same input window [lo, hi] (relative to q) form one group -> list of (r0, r1, lo, hi).
    x8 k16: (0-3: q-1..q) (4-7: q..q+1); x5 k11: (0-1: -1..0) (2: -1..1) (3-4: 0..1); x3 k7: (0) (1) (2) alike."""
    groups = []
    for r in range(u):
        offs = [(r + pad - j) // u for j in range(k) if (r + pad - j) % u == 0]
        lo, hi = min(offs), max(offs)
        if groups and groups[-1][2:] == (lo, hi):
            groups[-1] = (groups[-1][0], r + 1, lo, hi)
        else:
            groups.append((r, r + 1, lo, hi))
    return groups


def poly_transpose(w, u, pad):
    """ConvTranspose1d weight [Cin][Cout][k] -> one conv per phase group (see poly_groups): [(r1-r0)*Cout][n*Cin]
    over the n = hi - lo + 1 input frames q+lo .. q+hi (tap-major), no zero taps."""
    cin, cout, k = w.shape
    out = []
    for r0, r1, lo, hi in poly_groups(u, k, pad):
        m = np.zeros((r1 - r0, cout, hi - lo + 1, cin), np.float64)
        for r in range(r0, r1):
            for j in range(k):
                if (r + pad - j) % u == 0:
                    m[r - r0, :, (r + pad - j) // u - lo, :] += w[:, :, j].T
        out.append(m.reshape((r1 - r0) * cout, (hi - lo + 1) * cin).astype(F32))
    return out


def conv_transpose(x, P, p, u, k, cout):
    """Polyphase ConvTranspose1d: each phase group is a stride-1 conv; group (r0, r1) fills columns r0*cout ..
    r1*cout of Y [T][u*cout] (on the GPU: Y + r0*cout with ldy = u*cout), which is the output [u*T][cout]."""
    y = np.empty((len(x), u * cout), F32)
    for g, (r0, r1, lo, hi) in enumerate(poly_groups(u, k, (k - u) // 2)):
        y[:, r0 * cout:r1 * cout] = conv(x, P[f"{p}.{g}_w"], P[f"{p}.{g}_b"], hi - lo + 1, pad=(-lo, hi))
    return y.reshape(-1, cout)


def poly_upsample2(w):
    """Encoder Upsample1D: nearest x2, 4 zeros on the left, conv k5 [O][C][5] -> 3-tap conv over x frames q-2..q
    with 2*O outputs (output frame 2q + r = row q, columns r*O ...). Taps landing on the same frame are summed."""
    O, C, k = w.shape
    m = np.zeros((2, O, 3, C), np.float64)
    for r in range(2):
        for j in range(k):
            m[r, :, (r + j - 4) // 2 + 2, :] += w[:, :, j]
    return m.reshape(2 * O, 3 * C).astype(F32)


def time_biases(w, t, r):
    """The 14 resnet time biases for one Euler step: Linear_n(Mish(mixer([MLP(emb t) | MLP(emb r)])))."""
    E = "flow.decoder.estimator."

    def emb(v):                                    # SinusoidalPosEmb(320), scale 1000
        e = np.exp(np.arange(160, dtype=F32) * F32(-(math.log(10000) / 159)))
        e = F32(1000) * F32(v) * e
        h = np.concatenate([np.sin(e), np.cos(e)]) @ w[E + "time_mlp.linear_1.weight"].T + w[E + "time_mlp.linear_1.bias"]
        return silu(h) @ w[E + "time_mlp.linear_2.weight"].T + w[E + "time_mlp.linear_2.bias"]

    temb = mish(np.concatenate([emb(t), emb(r)]) @ w[E + "time_embed_mixer.weight"].T)
    return [temb @ w[res_key(n) + "mlp.1.weight"].T + w[res_key(n) + "mlp.1.bias"] for n in range(N_RES)]


def res_key(n):
    E = "flow.decoder.estimator."
    return E + ("down_blocks.0.0." if n == 0 else f"mid_blocks.{n - 1}.0." if n < 13 else "up_blocks.0.0.")


def tf_key(m):
    E = "flow.decoder.estimator."
    if m < 4:
        return E + f"down_blocks.0.1.{m}."
    if m < 52:
        return E + f"mid_blocks.{(m - 4) // 4}.1.{(m - 4) % 4}."
    return E + f"up_blocks.0.1.{m - 52}."


def prepare(w):
    """Official tensors -> dict of fp32 arrays in the GPU layout (what the converter writes, before quantizing).
    Weight norm folded, convs tap-major, LN x sqrt(512) folded, q | k | v concatenated, time MLPs precomputed."""
    P = {}
    xs = F32(math.sqrt(512))
    E = "flow.encoder."
    P["enc.emb"] = w["flow.input_embedding.weight"]
    for lin, ln, src in (("in", "in_ln", "embed"), ("up_in", "up_ln", "up_embed")):   # linear, LN, x sqrt(512)
        P[f"enc.{lin}_w"], P[f"enc.{lin}_b"] = w[E + src + ".out.0.weight"], w[E + src + ".out.0.bias"]
        P[f"enc.{ln}_w"], P[f"enc.{ln}_b"] = w[E + src + ".out.1.weight"] * xs, w[E + src + ".out.1.bias"] * xs
    P["enc.la1_w"] = taps(w[E + "pre_lookahead_layer.conv1.weight"])
    P["enc.la1_b"] = w[E + "pre_lookahead_layer.conv1.bias"]
    P["enc.la2_w"] = taps(w[E + "pre_lookahead_layer.conv2.weight"])
    P["enc.la2_b"] = w[E + "pre_lookahead_layer.conv2.bias"]
    P["enc.up_w"] = poly_upsample2(w[E + "up_layer.conv.weight"])
    P["enc.up_b"] = np.tile(w[E + "up_layer.conv.bias"], 2)
    for i in range(N_ENC):
        s = E + (f"encoders.{i}." if i < N_ENC25 else f"up_encoders.{i - N_ENC25}.")
        a, p = s + "self_attn.", f"enc.{i}."
        P[p + "ln1_w"], P[p + "ln1_b"] = w[s + "norm_mha.weight"], w[s + "norm_mha.bias"]
        P[p + "ln2_w"], P[p + "ln2_b"] = w[s + "norm_ff.weight"], w[s + "norm_ff.bias"]
        P[p + "wqkv"] = np.concatenate([w[a + f"linear_{n}.weight"] for n in "qkv"])
        P[p + "bqkv"] = np.concatenate([w[a + f"linear_{n}.bias"] for n in "qkv"])
        P[p + "wpos"] = w[a + "linear_pos.weight"]
        P[p + "bias_u"] = w[a + "pos_bias_u"].reshape(-1)
        P[p + "bias_v"] = w[a + "pos_bias_v"].reshape(-1)
        P[p + "wo"], P[p + "bo"] = w[a + "linear_out.weight"], w[a + "linear_out.bias"]
        P[p + "ff1_w"], P[p + "ff1_b"] = w[s + "feed_forward.w_1.weight"], w[s + "feed_forward.w_1.bias"]
        P[p + "ff2_w"], P[p + "ff2_b"] = w[s + "feed_forward.w_2.weight"], w[s + "feed_forward.w_2.bias"]
    P["enc.out_ln_w"], P["enc.out_ln_b"] = w[E + "after_norm.weight"], w[E + "after_norm.bias"]
    P["enc.proj_w"], P["enc.proj_b"] = w["flow.encoder_proj.weight"], w["flow.encoder_proj.bias"]
    P["spk_w"], P["spk_b"] = w["flow.spk_embed_affine_layer.weight"], w["flow.spk_embed_affine_layer.bias"]

    E = "flow.decoder.estimator."
    for step, (t, r) in enumerate(STEPS):
        for n, tb in enumerate(time_biases(w, t, r)):
            P[f"cfm.tb{step}.{n}"] = tb.astype(F32)

    def block(dst, src):                          # causal conv k3 + LayerNorm (+ Mish in the kernel)
        P[dst + "_w"], P[dst + "_b"] = taps(w[src + ".block.0.weight"]), w[src + ".block.0.bias"]
        P[dst + "_ln_w"], P[dst + "_ln_b"] = w[src + ".block.2.weight"], w[src + ".block.2.bias"]

    for n in range(N_RES):
        s, p = res_key(n), f"cfm.res{n}."
        block(p + "c1", s + "block1")
        block(p + "c2", s + "block2")
        P[p + "res_w"], P[p + "res_b"] = w[s + "res_conv.weight"][:, :, 0], w[s + "res_conv.bias"]
    for m in range(N_TF):
        s, p = tf_key(m), f"cfm.tf{m}."
        P[p + "ln1_w"], P[p + "ln1_b"] = w[s + "norm1.weight"], w[s + "norm1.bias"]
        P[p + "ln3_w"], P[p + "ln3_b"] = w[s + "norm3.weight"], w[s + "norm3.bias"]
        P[p + "wqkv"] = np.concatenate([w[s + f"attn1.to_{n}.weight"] for n in "qkv"])
        P[p + "wo"], P[p + "bo"] = w[s + "attn1.to_out.0.weight"], w[s + "attn1.to_out.0.bias"]
        P[p + "ff1_w"], P[p + "ff1_b"] = w[s + "ff.net.0.proj.weight"], w[s + "ff.net.0.proj.bias"]
        P[p + "ff2_w"], P[p + "ff2_b"] = w[s + "ff.net.2.weight"], w[s + "ff.net.2.bias"]
    P["cfm.down_w"], P["cfm.down_b"] = taps(w[E + "down_blocks.0.2.weight"]), w[E + "down_blocks.0.2.bias"]
    P["cfm.up_w"], P["cfm.up_b"] = taps(w[E + "up_blocks.0.2.weight"]), w[E + "up_blocks.0.2.bias"]
    block("cfm.fin", E + "final_block")
    P["cfm.proj_w"], P["cfm.proj_b"] = w[E + "final_proj.weight"][:, :, 0], w[E + "final_proj.bias"]

    H = "mel2wav."
    for i in range(5):
        P[f"hift.f0c{i}_w"] = taps(fold_wn(w, H + f"f0_predictor.condnet.{2 * i}"))
        P[f"hift.f0c{i}_b"] = w[H + f"f0_predictor.condnet.{2 * i}.bias"]
    P["hift.f0_cls_w"] = w[H + "f0_predictor.classifier.weight"][0]
    P["hift.f0_cls_b"] = w[H + "f0_predictor.classifier.bias"]
    P["hift.src_w"] = w[H + "m_source.l_linear.weight"][0]
    P["hift.src_b"] = w[H + "m_source.l_linear.bias"]
    P["hift.pre_w"], P["hift.pre_b"] = taps(fold_wn(w, H + "conv_pre")), w[H + "conv_pre.bias"]
    for i, (u, k, _) in enumerate(UPS):
        for g, (m, (r0, r1, _, _)) in enumerate(zip(poly_transpose(fold_wn(w, H + f"ups.{i}"), u, (k - u) // 2),
                                                   poly_groups(u, k, (k - u) // 2))):
            P[f"hift.up{i}.{g}_w"] = m
            P[f"hift.up{i}.{g}_b"] = np.tile(w[H + f"ups.{i}.bias"], r1 - r0)
        P[f"hift.sd{i}_w"] = taps(w[H + f"source_downs.{i}.weight"])
        P[f"hift.sd{i}_b"] = w[H + f"source_downs.{i}.bias"]

    def res(dst, src):
        for l in range(3):
            q = f"{dst}.{l}."
            for a in (1, 2):
                al = w[src + f".activations{a}.{l}.alpha"]
                P[q + f"a{a}"] = al
                P[q + f"ia{a}"] = (F32(1) / (al + F32(1e-9))).astype(F32)
                P[q + f"c{a}_w"] = taps(fold_wn(w, src + f".convs{a}.{l}"))
                P[q + f"c{a}_b"] = w[src + f".convs{a}.{l}.bias"]

    for i in range(3):
        res(f"hift.sr{i}", H + f"source_resblocks.{i}")
        for j in range(3):
            res(f"hift.rb{i}.{j}", H + f"resblocks.{3 * i + j}")
    P["hift.post_w"], P["hift.post_b"] = taps(fold_wn(w, H + "conv_post")), w[H + "conv_post.bias"]
    return {k: np.ascontiguousarray(v, dtype=F32) for k, v in P.items()}


def load_voice_conds(path=CONDS, P=None):
    """Voice from the official Conditionals (conds.pt, needs torch): prompt tokens [P], prompt mel [2P][80],
    x-vector [192] and spk = spk_embed_affine(normalize(x-vector)) [80] with THIS decoder's affine layer."""
    import torch
    g = torch.load(str(path), map_location="cpu", weights_only=True)["gen"]
    xv = g["embedding"][0].detach().numpy().astype(F32)
    v = dict(prompt_tok=g["prompt_token"][0].numpy().astype(np.int32),
             prompt_mel=g["prompt_feat"][0].detach().numpy().astype(F32), xvec=xv)
    if P is not None:
        v["spk"] = (P["spk_w"] @ (xv / max(np.linalg.norm(xv), 1e-12)) + P["spk_b"]).astype(F32)
    return v


# ---- weights: the converted file (models/chatterbox-s3gen-mf.bin + .json) ---------------------------------
def load_converted(name=MODEL):
    """Read the converted file and dequantize exactly as the GPU does: q4i -> (q - 8) * d in fp32 (exact: q d
    and 8 d are exact in fp32 because d is an fp16 value), f16i -> fp32, f32t transposed back. Returns the same
    dict as prepare()."""
    from q4 import dequantize_q4
    meta = json.loads(Path(str(name) + ".json").read_text())
    blob = np.fromfile(str(name) + ".bin", np.uint8)
    P = {}
    for key, e in meta["tensors"].items():
        raw = blob[e["offset"]: e["offset"] + e["size"]]
        shape = e["shape"]
        if e["fmt"] == "q4i":
            rows, cols = e["rows"], e["cols"]
            nb = cols // 32
            nib = raw[:e["scales"]].reshape(rows // 64, nb, 64, 16).transpose(0, 2, 1, 3).reshape(rows, nb * 16)
            sc = raw[e["scales"]:].view(np.float16).reshape(rows // 64, nb, 64).transpose(0, 2, 1).reshape(rows, nb)
            m = dequantize_q4(np.ascontiguousarray(nib), np.ascontiguousarray(sc))
            P[key] = m[:shape[0], :shape[1]] if len(shape) == 2 else m[:shape[0]]
        elif e["fmt"] == "f16i":
            rows, cols = e["rows"], e["cols"]
            m = raw.view(np.float16).reshape(rows // 64, cols, 64).transpose(0, 2, 1).reshape(rows, cols)
            P[key] = m[:shape[0], :shape[1]].astype(F32)
        elif e["fmt"] == "f32t":
            P[key] = raw.view(F32).reshape(e["cols"], e["rows"]).T[:shape[0]]
        elif e["fmt"] == "f32":
            P[key] = raw.view(F32).reshape(shape)
        else:
            raise ValueError(e["fmt"])
    return P


def load_voice(name=VOICE):
    meta = json.loads(Path(str(name) + ".json").read_text())
    blob = np.fromfile(str(name) + ".bin", np.uint8)
    v = {}
    for key, e in meta["tensors"].items():
        raw = blob[e["offset"]: e["offset"] + e["size"]]
        v[key] = raw.view(np.int32 if e["fmt"] == "i32" else F32).reshape(e["shape"])
    return v


# ---- validation against the official torch code -----------------------------------------------------------
def err(name, ref, x):
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    assert ref.shape == x.shape, (name, ref.shape, x.shape)
    d = np.abs(ref - x).max()
    snr = 10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-30))
    print(f"  {name:34s} {str(ref.shape):14s} max|ref| {np.abs(ref).max():9.3e}  max err {d:9.2e}  "
          f"rel {d / max(np.abs(ref).max(), 1e-30):8.1e}  SNR {snr:6.1f} dB", flush=True)
    return dict(max_abs=float(d), rel=float(d / max(np.abs(ref).max(), 1e-30)), snr_db=float(snr))


class Official:
    """The official S3Gen(meanflow=True) with hooks: captures the encoder stages, the estimator calls, and lets
    the caller supply the random numbers (CFM noise, SineGen phases / noise)."""

    def __init__(self):
        import torch
        from safetensors.torch import load_file
        from chatterbox.models.s3gen import S3Gen
        torch.set_num_threads(4)
        self.torch = torch
        self.m = S3Gen(meanflow=True)
        self.m.load_state_dict(load_file(str(WEIGHTS)), strict=False)
        self.m.eval()
        self.cap = {}
        enc = self.m.flow.encoder
        hook = lambda name, f: (lambda mod, i, o: self.cap.__setitem__(name, f(o)))
        first = lambda o: (o[0] if isinstance(o, tuple) else o)[0].detach().numpy().copy()
        enc.embed.register_forward_hook(hook("embed", first))
        enc.pre_lookahead_layer.register_forward_hook(hook("lookahead", first))
        enc.encoders[-1].register_forward_hook(hook("layers25", first))
        enc.up_layer.register_forward_hook(hook("upsample", lambda o: o[0][0].T.detach().numpy().copy()))
        enc.up_embed.register_forward_hook(hook("up_embed", first))
        enc.after_norm.register_forward_hook(hook("layers50", first))
        self.m.flow.encoder_proj.register_forward_hook(hook("mu", first))
        est = self.m.flow.decoder.estimator
        self.calls = []
        fwd = est.forward                          # basic_euler calls .forward directly (no module hooks)

        def est_wrap(x, **kw):
            out = fwd(x, **kw)
            self.calls.append({k: v.detach().clone() for k, v in kw.items() if hasattr(v, "detach")}
                              | {"x": x.detach().clone(), "out": out.detach().clone()})
            return out
        est.forward = est_wrap

    def flow(self, tokens, voice, noise):
        """Official flow.inference (finalize, 2 meanflow steps); noise [T50][80] replaces torch.randn_like."""
        torch = self.torch
        z = torch.from_numpy(np.ascontiguousarray(noise.T))[None]
        Tp = len(voice["prompt_mel"])
        orig = torch.randn_like
        torch.randn_like = lambda t, *a, **k: z.clone()
        try:
            with torch.inference_mode():
                mel, _ = self.m.flow.inference(
                    token=torch.tensor(tokens, dtype=torch.long)[None], token_len=torch.tensor([len(tokens)]),
                    prompt_token=torch.from_numpy(voice["prompt_tok"].astype(np.int64))[None],
                    prompt_token_len=torch.tensor([len(voice["prompt_tok"])]),
                    prompt_feat=torch.from_numpy(voice["prompt_mel"])[None], prompt_feat_len=None,
                    embedding=torch.from_numpy(voice["xvec"])[None], finalize=True, n_timesteps=2,
                    noised_mels=z[:, :, Tp:].clone(), meanflow=True)
        finally:
            torch.randn_like = orig
        return mel[0].T.numpy().copy()

    def hift(self, mel, phase, noise):
        """Official HiFT on mel [M][80] with the given SineGen phases [9] and noise [9][480M]: (f0, s, wav)."""
        torch = self.torch
        from chatterbox.models.s3gen import hifigan
        g = self.m.mel2wav
        nz = torch.from_numpy(noise)[None]

        class FixedUniform:
            def __init__(self, low, high):
                pass

            def sample(self, sample_shape):
                return torch.from_numpy(phase.copy()).reshape(1, 9, 1)

        orig_u, orig_r = hifigan.Uniform, torch.randn_like
        hifigan.Uniform = FixedUniform
        torch.randn_like = lambda t, *a, **k: nz.clone() if tuple(t.shape) == tuple(nz.shape) else torch.zeros_like(t)
        try:
            with torch.inference_mode():
                x = torch.from_numpy(np.ascontiguousarray(mel.T))[None]
                f0 = g.f0_predictor(x)
                wav, s = g.inference(speech_feat=x, cache_source=torch.zeros(1, 1, 0))
        finally:
            hifigan.Uniform, torch.randn_like = orig_u, orig_r
        return f0[0].numpy().copy(), s[0, 0].numpy().copy(), wav[0].numpy().copy()

    def source(self, f0, phase, noise):
        """Official f0 upsampling + SourceModuleHnNSF for an arbitrary f0 [M] (exercises the voiced branch)."""
        torch = self.torch
        from chatterbox.models.s3gen import hifigan
        g = self.m.mel2wav
        nz = torch.from_numpy(noise)[None]

        class FixedUniform:
            def __init__(self, low, high):
                pass

            def sample(self, sample_shape):
                return torch.from_numpy(phase.copy()).reshape(1, 9, 1)

        orig_u, orig_r = hifigan.Uniform, torch.randn_like
        hifigan.Uniform = FixedUniform
        torch.randn_like = lambda t, *a, **k: nz.clone() if tuple(t.shape) == tuple(nz.shape) else torch.zeros_like(t)
        try:
            with torch.inference_mode():
                up = g.f0_upsamp(torch.from_numpy(f0.astype(F32))[None, None]).transpose(1, 2)
                s, _, _ = g.m_source(up)
        finally:
            hifigan.Uniform, torch.randn_like = orig_u, orig_r
        return s[0, :, 0].numpy().copy()

    def decode(self, mel, s):
        torch = self.torch
        with torch.inference_mode():
            return self.m.mel2wav.decode(x=torch.from_numpy(np.ascontiguousarray(mel.T))[None],
                                         s=torch.from_numpy(s)[None, None])[0].numpy().copy()


def validate(langs=("es", "ru"), seed=0):
    import soundfile as sf
    OUT.mkdir(parents=True, exist_ok=True)
    toks = json.loads((CB / "results" / "speech_tokens_v3.json").read_text())
    print("loading weights ...", flush=True)
    raw = load_raw()
    P = prepare(raw)
    voice = load_voice_conds(P=P)
    ref = Official()
    with ref.torch.inference_mode():
        spk_t = ref.m.flow.spk_embed_affine_layer(ref.torch.nn.functional.normalize(
            ref.torch.from_numpy(voice["xvec"])[None], dim=1))[0].numpy()
    report = {}
    print("speaker vector:")
    report["spk"] = err("spk80 (affine of the x-vector)", spk_t, voice["spk"])
    for lang in langs:
        rng = np.random.default_rng(seed)
        tokens = np.array(toks[lang], np.int32)
        T50 = 2 * (len(voice["prompt_tok"]) + len(tokens))
        noise = rng.standard_normal((T50, 80)).astype(F32)
        M = 2 * len(tokens)
        phase = rng.uniform(-np.pi, np.pi, 9).astype(F32)
        phase[0] = 0
        snoise = rng.standard_normal((9, M * HOP)).astype(F32)
        print(f"\n[{lang}] {len(tokens)} tokens + {len(voice['prompt_tok'])} prompt -> T50 = {T50}, M = {M}", flush=True)
        R = report[lang] = {}
        ref.calls.clear()
        mel_t = ref.flow(tokens, voice, noise)
        tr = {}
        print(" encoder (same tokens):")
        mu = encoder(np.concatenate([voice["prompt_tok"], tokens]), P, tr)
        for k in ("embed", "lookahead", "layers25", "upsample", "up_embed", "layers50", "mu"):
            R["enc_" + k] = err(k, ref.cap[k], tr[k])
        print(" CFM estimator (same inputs as each official call):")
        for step, c in enumerate(ref.calls):
            xin = np.concatenate([c["x"][0].numpy().T, c["mu"][0].numpy().T,
                                  np.broadcast_to(c["spks"][0].numpy(), (T50, 80)), c["cond"][0].numpy().T], axis=1)
            assert abs(float(c["t"][0]) - STEPS[step][0]) < 1e-7 and abs(float(c["r"][0]) - STEPS[step][1]) < 1e-7
            R[f"cfm_v{step}"] = err(f"velocity step {step} (t={STEPS[step][0]})", c["out"][0].numpy().T,
                                    estimator(xin.astype(F32), P, step))
        print(" CFM, 2 steps from the same noise (own mu):")
        mel = cfm(mu, voice["spk"], voice["prompt_mel"], noise, P)
        R["cfm_mel"] = err("mel (new frames)", mel_t, mel)
        print(" HiFT (official mel as input):")
        f0_t, s_t, wav_t = ref.hift(mel_t, phase, snoise)
        R["f0"] = err("f0", f0_t, f0_predictor(mel_t, P))
        R["source_torch"] = err("source s (official f0, torch phase)", s_t, source(f0_t, P, phase, snoise, "torch"))
        R["source_gpu"] = err("source s (official f0, GPU phase)", s_t, source(f0_t, P, phase, snoise, "gpu"))
        # the predicted f0 stays below the 10 Hz voicing threshold (this checkpoint's F0 head outputs ~0..1), so
        # the sine branch is masked out above; check it with a synthetic contour: 0 (unvoiced) then 90..260 Hz
        fs = np.where(np.arange(M) < M // 5, 0, 175 + 85 * np.sin(np.arange(M) / 9.0)).astype(F32)
        s_syn = ref.source(fs, phase, snoise)
        R["source_voiced_torch"] = err("source s (synthetic f0, torch phase)", s_syn, source(fs, P, phase, snoise, "torch"))
        R["source_voiced_gpu"] = err("source s (synthetic f0, GPU phase)", s_syn, source(fs, P, phase, snoise, "gpu"))
        wd = hift_decode(mel_t, s_t, P, tr)
        R["decode"] = err("decode (official mel and s)", wav_t, wd)
        print(" end to end (tokens -> waveform, all NumPy, GPU phase):")
        f0 = f0_predictor(mel, P)
        wav = hift_decode(mel, source(f0, P, phase, snoise, "gpu"), P)
        R["wav_e2e"] = err("waveform", wav_t, wav)
        pp_t, pp = postprocess(wav_t, len(tokens)), postprocess(wav, len(tokens))
        R["wav_e2e_post"] = err("waveform, post-processed", pp_t, pp)
        wav_tp = hift_decode(mel, source(f0, P, phase, snoise, "torch"), P)
        R["wav_e2e_torchphase"] = err("waveform (torch phase mode)", wav_t, wav_tp)
        sf.write(OUT / f"{lang}_official.wav", pp_t, SR, subtype="PCM_16")
        sf.write(OUT / f"{lang}_numpy_fp32.wav", pp, SR, subtype="PCM_16")
    (OUT / "validation.json").write_text(json.dumps(report, indent=1))
    return report


def logmel_dist(a, b):
    """Mean |log10 mel| difference (dB/10) between two waveforms: 80 mels, n_fft 1024, hop 256 (perceptual-ish)."""
    import librosa
    m = [np.log10(librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=1024, hop_length=256,
                                                 n_mels=80) + 1e-5) for x in (a, b)]
    n = min(m[0].shape[1], m[1].shape[1])
    return float(np.abs(m[0][:, :n] - m[1][:, :n]).mean())


def quant_report(langs=("es", "en", "pt", "ru"), seed=0):
    """fp32 (prepare) vs the converted file, same random inputs: mel / waveform errors, written as WAVs."""
    import soundfile as sf
    toks = json.loads((CB / "results" / "speech_tokens_v3.json").read_text())
    P32 = prepare(load_raw())
    PQ = load_converted()
    voice = load_voice()
    meta = json.loads(Path(str(MODEL) + ".json").read_text())
    path = OUT / "quant_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}     # languages accumulate across runs
    report["policy"] = meta.get("policy")
    for lang in langs:
        tokens = np.array(toks[lang], np.int32)
        tr32 = {}
        w32 = synthesize(tokens, P32, voice, np.random.default_rng(seed), trace=tr32)
        sf.write(OUT / f"{lang}_numpy_fp32.wav", w32, SR, subtype="PCM_16")
        for tag, P in (("q", PQ),):
            tr = {}
            w = synthesize(tokens, P, voice, np.random.default_rng(seed), trace=tr)
            sf.write(OUT / f"{lang}_numpy_{tag}.wav", w, SR, subtype="PCM_16")
            print(f"[{lang}] {tag}:", flush=True)
            r = report[f"{lang}/{tag}"] = {
                "mu": err("mu", tr32["mu"], tr["mu"]), "mel": err("mel", tr32["mel"], tr["mel"]),
                "f0": err("f0", tr32["f0"], tr["f0"]), "wav": err("waveform", w32, w)}
            r["logmel_dist"] = logmel_dist(w32, w)
            print(f"  log-mel distance {r['logmel_dist']:.4f}")
        rng = np.random.default_rng(seed + 1)          # vocoder alone: fp32 mel and excitation, both HiFTs
        f0 = f0_predictor(tr32["mel"], P32)
        ph = rng.uniform(-np.pi, np.pi, 9).astype(F32)
        ph[0] = 0
        nz = rng.standard_normal((9, len(tr32["mel"]) * HOP)).astype(F32)
        s = source(f0, P32, ph, nz)
        a, b = hift_decode(tr32["mel"], s, P32), hift_decode(tr32["mel"], s, PQ)
        print(f"[{lang}] quantized HiFT alone (fp32 mel and source):")
        report[f"{lang}/hift_only"] = err("waveform", a, b) | {"logmel_dist": logmel_dist(a, b)}
        print(f"  log-mel distance {report[f'{lang}/hift_only']['logmel_dist']:.4f}")
        path.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--quant" in sys.argv:
        i = sys.argv.index("--quant")
        quant_report(*([sys.argv[i + 1].split(",")] if len(sys.argv) > i + 1 else []))
    else:
        sys.path.insert(0, str(CB / "src" / "src"))
        validate()
