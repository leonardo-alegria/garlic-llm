"""Convert Chatterbox-Turbo's meanflow S3Gen (s3gen_meanflow.safetensors of ResembleAI/chatterbox-turbo, MIT) to the
PS4 format: one file, tensors on 256-byte boundaries, index in models/<name>.json (as the Parakeet / Qwen files).
The tensors are pc/s3gen_ref.py's prepare() dict (the reference runs from either), so the GPU layout is:

  matrices    W [O][I], Y = X @ W.T + b over frames-major X [T][I]. q | k | v concatenated (rows 3 x 512).
  convs       W [O][k*C] tap-major: W[o][j*C + c] = torch_W[o][c][j]. The input row of output frame t is the k*C
              floats starting at frame t*stride - pad_left of the activation buffer (zero-padded frames around it):
              with dilation 1 they are contiguous, so a matrix kernel reads them in place with ldx = stride*C.
              When k*C is not a multiple of 32, cols are padded with zero weights to the next multiple: the kernel
              then also reads the first floats of the following frame (x 0), so buffers need that much finite
              slack at the end. Dilated convs (HiFT ResBlocks) read tap j at frame t + j*dil (32-float blocks never
              straddle taps since C % 32 == 0).
  polyphase   ConvTranspose1d x u (HiFT) is split into phase groups (s3gen_ref.poly_groups), each a stride-1 conv
              over 2 or 3 input frames with (r1 - r0)*Cout outputs written at Y + r0*Cout with ldy = u*Cout; the
              encoder's nearest-x2 + conv k5 is one 3-tap conv with 2*512 outputs. Either way the result
              [T][u*Cout] IS the frames-major output [u*T][Cout] (see the "conv" metadata: phases, u, cout).
  folded      weight norm (g v / |v|), LayerNorm x sqrt(512) after the encoder input linears, Snake 1 / (alpha +
              1e-9) stored as ia*, and the CFM time path: for the fixed 2-step schedule the 14 resnet time biases
              cfm.tb{step}.{n} [256] are precomputed (no time MLP / mixer weights on the PS4).

Formats: q4i = pc/q4.py (blocks of 32 along a row, rows padded to a multiple of 64, interleaved for mm_q4i),
quantized with the MSE-searched scale (same format); f16i = fp16, rows in tiles of 64, per tile [cols][64]
(lane = output row, coalesced); f32t = fp32 transposed Wt[c][o], o padded to 64 (mm_f32t: the 80 / 18-output
projections, the two small source convs, the speaker affine layer); f32 = plain little-endian (norms, biases,
alphas, time biases, F0 head, source mixer, STFT bases).

Policy (default --hift f16): every big matrix of the encoder and the CFM is q4i (convs included: their Q4 error
stays well inside the model's own seed-to-seed spread); the HiFT convs are f16i (Q4 there changes the audio far more
than fp16), except the F0 predictor convs, kept q4i: the F0 head only decides voiced / unvoiced at 10 Hz and this
checkpoint never gets near it (it outputs ~0..1), so the waveform does not depend on their precision. --hift q4
puts the whole HiFT in q4i (71 MiB total).

The voice (prompt tokens, prompt mel 2P x 80, the 80-d speaker vector for THIS decoder, the 192-d x-vector it comes
from) goes to models/chatterbox-voice.bin + .json in the same style. Note: the base S3Gen's speaker affine layer
differs from the meanflow one, so the 80-d speaker vector is recomputed here from the x-vector.

Run: python pc/convert_chatterbox_s3gen.py [--hift f16|q4]   (needs torch, to read the voice's conds.pt)
     -> models/chatterbox-s3gen-mf.bin + .json, models/chatterbox-voice.bin + .json
"""

import json
import sys
from pathlib import Path

import numpy as np

import s3gen_ref as R
from q4 import interleave_q4, quantize_q4_mse

ROOT = Path(__file__).resolve().parent.parent
NAME = "chatterbox-s3gen-mf"

# Conv metadata for the engine: key -> (k, pad_left, pad_right, stride, dilation, C_in, outputs per input frame)
CONVS = {"enc.la1_w": (4, 0, 3, 1, 1, 512, 1), "enc.la2_w": (3, 2, 0, 1, 1, 512, 1),
         "enc.up_w": (3, 2, 0, 1, 1, 512, 2),
         "cfm.down_w": (3, 2, 0, 1, 1, 256, 1), "cfm.up_w": (3, 2, 0, 1, 1, 256, 1),
         "cfm.fin_w": (3, 2, 0, 1, 1, 256, 1),
         "hift.pre_w": (7, 3, 3, 1, 1, 80, 1), "hift.post_w": (7, 3, 3, 1, 1, 64, 1)}
for _n in range(R.N_RES):
    CONVS[f"cfm.res{_n}.c1_w"] = (3, 2, 0, 1, 1, 320 if _n == 0 else 512 if _n == 13 else 256, 1)
    CONVS[f"cfm.res{_n}.c2_w"] = (3, 2, 0, 1, 1, 256, 1)
for _i in range(5):
    CONVS[f"hift.f0c{_i}_w"] = (3, 1, 1, 1, 1, 80 if _i == 0 else 512, 1)
for _i, (_u, _k, _c) in enumerate(R.UPS):
    for _g, (_r0, _r1, _lo, _hi) in enumerate(R.poly_groups(_u, _k, (_k - _u) // 2)):
        CONVS[f"hift.up{_i}.{_g}_w"] = (_hi - _lo + 1, -_lo, _hi, 1, 1, 2 * _c, _r1 - _r0)
    _ks, _st, _pd = R.SRC_DOWN[_i]
    CONVS[f"hift.sd{_i}_w"] = (_ks, _pd, _pd, _st, 1, 18, 1)
    for _blk, _kk in [(f"sr{_i}", R.SRC_RES_K[_i])] + [(f"rb{_i}.{_j}", R.RES_K[_j]) for _j in range(3)]:
        for _l, _d in enumerate(R.DILS):
            CONVS[f"hift.{_blk}.{_l}.c1_w"] = (_kk, _d * (_kk - 1) // 2, _d * (_kk - 1) // 2, 1, _d, _c, 1)
            CONVS[f"hift.{_blk}.{_l}.c2_w"] = (_kk, (_kk - 1) // 2, (_kk - 1) // 2, 1, 1, _c, 1)

# fp32 matrices, stored transposed for mm_f32t (f32t): the 80 / 18-output projections (mu, velocity, log-magnitude
# and phase: precision), the small source convs and the speaker affine layer
F32_KEYS = {"enc.proj_w", "cfm.proj_w", "hift.post_w", "spk_w", "hift.sd1_w", "hift.sd2_w"}


def part(key):
    return "encoder" if key.startswith("enc.") else "cfm" if key.startswith("cfm.") else \
        "hift" if key.startswith("hift.") else "other"


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.offset, self.index = 0, {}

    def put(self, key, data, **meta):
        pad = (-self.offset) % 256
        self.f.write(bytes(pad))
        self.offset += pad
        self.f.write(data)
        self.index[key] = dict(offset=self.offset, size=len(data), **meta)
        self.offset += len(data)

    def f32(self, key, a, **meta):
        a = np.ascontiguousarray(a, dtype=np.float32)
        self.put(key, a.tobytes(), fmt="f32", shape=list(a.shape), **meta)

    def i32(self, key, a):
        a = np.ascontiguousarray(a, dtype=np.int32)
        self.put(key, a.tobytes(), fmt="i32", shape=list(a.shape))

    def padded(self, m):
        rows, cols = m.shape
        return np.pad(m, ((0, (-rows) % 64), (0, (-cols) % 32))).astype(np.float32)

    def f32t(self, key, m, **meta):
        """Small fp32 matrix for mm_f32t: transposed, Wt[c][o], outputs padded to a multiple of 64."""
        rows, cols = m.shape
        t = np.ascontiguousarray(np.pad(m, ((0, (-rows) % 64), (0, 0))).T, dtype=np.float32)
        self.put(key, t.tobytes(), fmt="f32t", shape=list(m.shape), rows=t.shape[1], cols=cols, **meta)

    def q4i(self, key, m, **meta):
        mp = self.padded(m)
        packed, d16 = quantize_q4_mse(mp)
        nib, sc = interleave_q4(packed, d16)
        self.put(key, nib + sc, fmt="q4i", shape=list(m.shape), rows=mp.shape[0], cols=mp.shape[1],
                 scales=len(nib), **meta)

    def f16i(self, key, m, **meta):
        mp = self.padded(m)
        rows, cols = mp.shape
        t = mp.astype(np.float16).reshape(rows // 64, 64, cols).transpose(0, 2, 1)
        self.put(key, np.ascontiguousarray(t).tobytes(), fmt="f16i", shape=list(m.shape), rows=rows, cols=cols,
                 **meta)

    def close(self):
        self.f.close()


def main(hift_fmt="q4"):
    print("reading + preparing ...", flush=True)
    P = R.prepare(R.load_raw())
    P["hift.stft_basis"] = R.stft_basis()
    P["hift.istft_basis"] = R.istft_basis()
    w = Writer(ROOT / "models" / f"{NAME}.bin")
    sizes, fp32_sizes = {}, {}
    for key, a in P.items():
        meta = {}
        if key in CONVS:
            k, pl, pr, st, dl, cin, up = CONVS[key]
            assert a.shape[1] == k * cin, (key, a.shape, k, cin)
            meta["conv"] = dict(k=k, pad=[pl, pr], stride=st, dil=dl, cin=cin, outputs_per_frame=up)
            if key.startswith("hift.up"):              # polyphase group: Y + r0 * cout, ldy = u * cout
                i, g = int(key[7]), int(key[9])
                u, kk, cout = R.UPS[i]
                r0, r1, _, _ = R.poly_groups(u, kk, (kk - u) // 2)[g]
                meta["conv"].update(phases=[r0, r1], u=u, cout=cout)
        big = a.ndim == 2 and key not in F32_KEYS and a.shape[0] >= 64
        if big and key.startswith("hift.") and not key.startswith("hift.f0c") and hift_fmt == "f16":
            w.f16i(key, a, **meta)
        elif big:
            w.q4i(key, a, **meta)
        elif key in F32_KEYS:
            w.f32t(key, a, **meta)
        else:
            w.f32(key, a, **meta)
        e = w.index[key]
        sizes[part(key)] = sizes.get(part(key), 0) + e["size"]
        fp32_sizes[part(key)] = fp32_sizes.get(part(key), 0) + a.size * 4
    w.close()
    by_fmt = {}
    for e in w.index.values():
        by_fmt[e["fmt"]] = by_fmt.get(e["fmt"], 0) + e["size"]
    meta = dict(
        size=w.offset, tensors=w.index, policy=dict(hift=hift_fmt, q4="q4i, MSE-searched block scale"),
        model=dict(name="chatterbox-turbo s3gen meanflow", sample_rate=R.SR, hop=R.HOP, mel=80, d_enc=512,
                   heads=8, d_head=64, enc_layers=R.N_ENC, enc_layers_25hz=R.N_ENC25, d_cfm=256,
                   cfm_resnets=R.N_RES, cfm_transformers=R.N_TF, steps=[list(s) for s in R.STEPS], vocab=6561,
                   hift_ups=[list(u) for u in R.UPS], hift_src_down=[list(s) for s in R.SRC_DOWN],
                   hift_res_k=list(R.RES_K), hift_src_res_k=list(R.SRC_RES_K), hift_dil=list(R.DILS),
                   ln_eps=dict(enc_layers=1e-12, enc_in=1e-5, enc_out=1e-5, cfm=1e-5)),
        sizes_bytes=sizes, sizes_fp32_bytes=fp32_sizes, sizes_by_format=by_fmt)
    (ROOT / "models" / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    MiB = 2 ** 20
    print(f"wrote models/{NAME}.bin: {w.offset / MiB:.2f} MiB, {len(w.index)} tensors (HiFT {hift_fmt})")
    for k in sizes:
        print(f"  {k:8s} {sizes[k] / MiB:7.2f} MiB   (fp32 {fp32_sizes[k] / MiB:7.2f} MiB)")
    for k, v in by_fmt.items():
        print(f"  {k:8s} {v / MiB:7.2f} MiB")

    # voice: official Conditionals of the built-in voice (re-cloned), spk projected with this decoder's layer
    v = R.load_voice_conds(P=P)
    w = Writer(ROOT / "models" / "chatterbox-voice.bin")
    w.i32("prompt_tok", v["prompt_tok"])
    w.f32("prompt_mel", v["prompt_mel"])
    w.f32("spk", v["spk"])
    w.f32("xvec", v["xvec"])
    w.close()
    (ROOT / "models" / "chatterbox-voice.json").write_text(json.dumps(dict(
        size=w.offset, tensors=w.index, source=str(R.CONDS.relative_to(ROOT)).replace("\\", "/"),
        note="spk = spk_embed_affine_layer(normalize(xvec)) of the meanflow S3Gen; prompt_mel fp32 [2P][80]"),
        indent=1))
    print(f"wrote models/chatterbox-voice.bin: {w.offset / 1024:.1f} KiB ({len(v['prompt_tok'])} prompt tokens)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main("q4" if "--hift" in sys.argv and sys.argv[sys.argv.index("--hift") + 1] == "q4" else "f16")
