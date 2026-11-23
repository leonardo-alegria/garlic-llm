"""Write the data files the standalone chat app reads from /app0/assets (see chat/*.c for the readers):

  kernels.bin    compiled GPU code of shaders/chat.cl (qwen3.cl for a Qwen3 model) + one record per kernel (entry,
                 RSRC1/2, user SGPRs)
  model.idx      model shape + where every tensor sits in the weights file (which stays in /data on the PS4)
  tokenizer.bin  byte-level BPE: raw bytes of every token id, merge table in rank order, byte -> id map
  font.bin       text font: Selawik and Noto Sans (+ Arimo, DejaVu Sans and the Noto symbol fonts) + Twemoji
                 (CC-BY 4.0) incl. flags and sequences
  font_bold.bin / font_italic.bin  bold / italic (markdown), Latin + Cyrillic
  font_small.bin text at 24 px (hints, chips);  font_big.bin  titles (semibold at 40 px + keyboard emoji)
  mono.bin       code (Inconsolata, Cascadia Mono for Cyrillic)

The font files are read from pc/fonts_libres and the emoji from third_party/twemoji: both are downloaded, not part of
the repository (CREDITS.md lists them).

All little endian. Run: python export_chat.py [model name] [hf model dir]
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

from gpu import CLANG, SHADERS, Kernel, elf_symbols

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "chat" / "assets"
MAX_SEQ = 2048
EXTRA_CHARS = "‘’‚“”„–—…•◦▪€™←↑→↓↕·"
TEXT_FACE = "segoeui.ttf"
TEXT_RANGES = [(32, 127), (160, 0x250), (0x400, 0x460), (0x2010, 0x2028)]   # Latin, Cyrillic, punctuation


def export_kernels(shader="qwen3"):
    src, obj = SHADERS / f"{shader}.cl", SHADERS / f"{shader}.o"
    subprocess.run(CLANG + ["-c", str(src), "-o", str(obj)], check=True)
    syms, sections = elf_symbols(obj.read_bytes())
    text = sections[".text"]
    recs = []
    for sym, (sec, value, _size) in sorted(syms.items()):
        if sym.endswith(".kd"):
            name = sym[:-3]
            entry = syms[name][1]
            assert entry % 256 == 0, name
            k = Kernel(name, entry, sections[sec][value: value + 64])
            recs.append(struct.pack("<32sIIIII", name.encode(), entry, k.rsrc1, k.rsrc2, k.user_sgprs, k.kernarg_sgpr))
    data = struct.pack("<4sII", b"PKRN", len(recs), len(text)) + b"".join(recs) + text
    (OUT / "kernels.bin").write_bytes(data)
    print(f"kernels.bin: {len(recs)} kernels from {shader}.cl, {len(text)} bytes of code")


def export_model_index(name):
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    c, t = meta["config"], meta["tensors"]
    if c.get("model_type") == "qwen3_5_text":
        return export_model_index_qwen35(name, meta)
    order = ["embed", "final_norm"]
    for i in range(c["num_hidden_layers"]):
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown")]
    path = f"/data/{name}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMDL", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"], MAX_SEQ,
                       151645, 151643, c["rms_norm_eps"], c["rope_theta"], path, meta["size"], len(order))
    recs = []
    for k in order:
        e = t[k]
        fmt = {"q4": 2, "q8": 1}.get(e.get("fmt"), 0)
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), fmt))
    (OUT / "model.idx").write_bytes(head + b"".join(recs))
    print(f"model.idx: {len(order)} tensors, weights at {path.decode()} ({meta['size'] / 2**20:.0f} MiB)")


QWEN35_MAX_SEQ = 8192   # MAX_SEQ in shaders/qwen35.cl


def export_model_index_qwen35(name, meta):
    """PMQ5: the PMDL header (eos = <|im_end|>, eos2 = <|endoftext|>) + "<8I" (linear key heads, value heads,
    key / value head dims, RoPE dims, mask of full-attention layers, <|im_start|>, 0); then per tensor "<QQIIQI" in
    this order: embed, final_norm, then per layer attn_norm, mlp_norm, and either (full attention) q_norm, k_norm,
    wqkv, wo or (Gated DeltaNet) w_in, conv, A, dt_bias, gnorm, wout; then wgu, wdown."""
    c, t = meta["config"], meta["tensors"]
    order, mask = ["embed", "final_norm"], 0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        mask |= full << i
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    flags = 0
    if "lm_head" in t:                       # untied output head (9B): last record
        order.append("lm_head")
        flags |= 1
    if t["embed"].get("fmt") == "q4row":     # input embedding rows stay in the file, past gpu_size
        flags |= 2
    rp = c["rope_parameters"]
    rope_dim = int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0))
    path = f"/data/{name}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMQ5", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"],
                       QWEN35_MAX_SEQ, 248046, 248044, c["rms_norm_eps"], rp["rope_theta"], path,
                       meta.get("gpu_size", meta["size"]), len(order))
    head += struct.pack("<8I", c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"], rope_dim, mask, 248045, flags)
    recs = []
    for k in order:
        e = t[k]
        fmt = {"q4": 2, "q8": 1, "q4i": 3, "q4row": 4, "q3i": 5}.get(e.get("fmt"), 0)
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), fmt))
    (OUT / "model.idx").write_bytes(head + b"".join(recs))
    print(f"model.idx (Qwen3.5, flags {flags}): {len(order)} tensors, full attention mask {mask:#010x}, weights at "
          f"{path.decode()} ({meta['size'] / 2**20:.0f} MiB)")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokenizer(model_dir):
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    dec = bytes_to_unicode()
    n = max(max(vocab.values()), max(a["id"] for a in tj["added_tokens"])) + 1
    raw = [b""] * n
    for tok, i in vocab.items():
        raw[i] = bytes(dec[ch] for ch in tok)
    special = [0] * n
    for a in tj["added_tokens"]:
        raw[a["id"]] = a["content"].encode()
        special[a["id"]] = 1
    byte_id = [vocab[next(ch for ch, b in dec.items() if b == x)] for x in range(256)]
    mrec = []
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        mrec.append(struct.pack("<III", vocab[a], vocab[b], vocab[a + b]))
    out = struct.pack("<4sII", b"PTOK", n, len(mrec)) + struct.pack("<256I", *byte_id)
    out += b"".join(struct.pack("<BH", special[i], len(r)) + r for i, r in enumerate(raw))
    out += b"".join(mrec)
    (OUT / "tokenizer.bin").write_bytes(out)
    print(f"tokenizer.bin: {n} tokens, {len(mrec)} merges")


# Glyph sources, in order of preference: the text faces, the symbol faces for arrows / math / shapes, Twemoji
# (third_party/twemoji, CC-BY 4.0) for emoji, including flags, keycaps and joined sequences. A symbol Twemoji lacks
# comes from the symbol faces, in one colour.
FONTS = "C:/Windows/Fonts/"
TWEMOJI = ROOT / "third_party" / "twemoji" / "assets" / "72x72"
SKIP = set(range(0x200B, 0x2010)) | set(range(0xFE00, 0xFE10)) | set(range(0x1F3FB, 0x1F400)) | {0x20E3}
SEQ_BASE = 0x110000                    # records for multi-codepoint emoji use codepoints past Unicode
MONO_RANGES = [(32, 127), (160, 384), (0x400, 0x460), (0x2010, 0x2028), (0x2190, 0x2200), (0x2500, 0x25A0)]

# Two sets of faces, by role; within a role the first face that has a character draws it. A face is
# (file, weight of a variable font or None, scale[, highest codepoint it may draw]). "windows" are the Windows faces
# the app first had (not ours to ship); "free" (pc/fonts_libres, SIL OFL 1.1 except DejaVu's own free licence) look
# as close to them as we found: Selawik is Microsoft's open stand-in for Segoe UI with the same widths, but Latin
# only, so Noto Sans (scaled to Segoe UI's x-height) draws Cyrillic, Greek and the rest, and Noto Sans Italic the
# italics (Selawik has none). With "free" every font keeps the line height and baseline of the face it replaces
# (LINE below), so the layout does not move.
FREE = ROOT / "pc" / "fonts_libres"
NOTO_X = 0.5 / 0.536                   # Segoe UI's x-height / Noto Sans's
FACE_SETS = {
    "windows": {
        "text": [(FONTS + "segoeui.ttf", None, 1)], "bold": [(FONTS + "segoeuib.ttf", None, 1)],
        "italic": [(FONTS + "segoeuii.ttf", None, 1)], "semibold": [(FONTS + "seguisb.ttf", None, 1)],
        "arial": [(FONTS + "arial.ttf", None, 1)], "symbols": [(FONTS + "seguisym.ttf", None, 1)],
        "emoji": [(FONTS + "seguiemj.ttf", None, 1)], "mono": [(FONTS + "consola.ttf", None, 1)],
        "zh": [(FONTS + "msyh.ttc", None, 1)], "ko": [(FONTS + "malgun.ttf", None, 1)],
    },
    "free": {
        "text": [(f"{FREE}/Selawik/selawk.ttf", None, 1), (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 400, NOTO_X)],
        "bold": [(f"{FREE}/Selawik/selawkb.ttf", None, 1), (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 700, NOTO_X)],
        "italic": [(f"{FREE}/NotoSans/NotoSans-Italic[wdth,wght].ttf", 400, NOTO_X)],
        "semibold": [(f"{FREE}/Selawik/selawksb.ttf", None, 1),
                     (f"{FREE}/NotoSans/NotoSans[wdth,wght].ttf", 600, NOTO_X)],
        "arial": [(f"{FREE}/Arimo/Arimo[wght].ttf", 400, 1, 0x1FFF)],   # letters only: its arrows / math are narrow
        # DejaVu Sans (scaled to Segoe UI's x-height) has the arrows, math and shapes closest to Segoe UI Symbol's
        "symbols": [(f"{FREE}/DejaVu/DejaVuSans.ttf", None, 0.914),
                    (f"{FREE}/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf", None, 1),
                    (f"{FREE}/NotoSansSymbols/NotoSansSymbols[wght].ttf", 400, 1),
                    (f"{FREE}/NotoSansMath/NotoSansMath-Regular.ttf", None, 1),
                    (f"{FREE}/NotoMusic/NotoMusic-Regular.ttf", None, 1)],
        "emoji": [],                   # Twemoji only; the symbols Twemoji lacks come from "symbols", in one color
        # Inconsolata was drawn after Consolas; scaled to Consolas's width. Cascadia Mono (light, same width) adds
        # the Cyrillic Inconsolata lacks.
        "mono": [(f"{FREE}/Inconsolata/Inconsolata[wdth,wght].ttf", 400, 28 / 26),   # 14 px a letter at 26, as
                 (f"{FREE}/CascadiaMono/CascadiaMono[wght].ttf", 350, 24 / 26)],      # Consolas

        "zh": [(f"{FREE}/NotoSansSC/NotoSansSC[wght].ttf", 400, 1)],
        "ko": [(f"{FREE}/NotoSansKR/NotoSansKR[wght].ttf", 400, 1)],
    },
}
LINE = {"mono": (1521, 527, 2048)}     # (ascender, descender, units per em) of Consolas; the rest: Segoe UI's
SEGOE_LINE = (2210, 514, 2048)
face_set = "free"                      # --fonts=free | windows (the Windows ones: only to compare)
font_out = OUT                         # --out=DIR: where the font files go


def load_face(spec, size):
    """A PIL font for (file, weight, scale) at size px (scaled); variable fonts set to the weight."""
    from PIL import ImageFont
    path, weight, scale = spec[:3]
    f = ImageFont.truetype(path, size if scale == 1 else max(1, int(round(size * scale))), index=0)
    if weight is not None:
        names = [a.get("name", b"") for a in f.get_variation_axes()]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        f.set_variation_by_axes([weight if n.lower() in ("weight", "wght") else a["default"]
                                 for n, a in zip(names, f.get_variation_axes())])
    return f


def font_cmaps(specs):
    from fontTools.ttLib import TTFont
    return {s: {cp for cp in TTFont(s[0], lazy=True, fontNumber=0).getBestCmap() if len(s) < 4 or cp <= s[3]}
            for s in specs}


def cjk_codepoints():
    """Common East Asian characters: the 6,763 hanzi of GB2312, kana, CJK punctuation and full-width forms
    (Microsoft YaHei), and the 2,350 hangul syllables of KS X 1001 (Malgun Gothic)."""
    def charset(encoding, rows, cols):
        out = set()
        for b1 in rows:
            for b2 in cols:
                try:
                    out.add(ord(bytes([b1, b2]).decode(encoding)))
                except UnicodeDecodeError:
                    pass
        return out
    yahei = charset("gb2312", range(0xB0, 0xF8), range(0xA1, 0xFF))
    yahei |= set(range(0x3000, 0x3040)) | set(range(0x3040, 0x3100)) | set(range(0xFF01, 0xFF5F))
    hangul = charset("euc-kr", range(0xB0, 0xC9), range(0xA1, 0xFF))
    return {cp: "zh" for cp in yahei} | {cp: "ko" for cp in hangul if 0xAC00 <= cp <= 0xD7A3}


def twemoji_files():
    """{codepoint tuple without FE0F: png path}"""
    out = {}
    for p in sorted(TWEMOJI.glob("*.png")):
        cps = tuple(int(x, 16) for x in p.stem.split("-") if x != "fe0f")
        out.setdefault(cps, p)
    return out


def export_font(fname, size, full=True, role="text", emoji=(), emoji_size=None):
    """PFN2: header <4sHHII (magic, line height, ascent, glyph count, bitmap bytes); per glyph <IHHhhHBBI
    (codepoint, width, height, left, top, advance, kind 0 = alpha / 1 = RGBA, pad, bitmap offset), sorted by
    codepoint; then the bitmaps (1 or 4 bytes per pixel); then, if any, the emoji sequences: <4sI ("SEQS", count)
    and per sequence <BBHI (length, 0, 0, glyph index) followed by its codepoints (u32, FE0F removed).
    role: the faces of FACE_SETS[face_set] that draw it ("text", "bold", "italic", "semibold", "mono")."""
    from PIL import Image, ImageDraw
    fs = FACE_SETS[face_set]
    roles = ("emoji", "text", "arial", "symbols", "zh", "ko") if full else (role,)
    specs = list(dict.fromkeys(s for r in roles + (role, "arial") for s in fs[r]))
    cmaps = font_cmaps(specs)
    faces = {s: load_face(s, size) for s in specs}
    base = fs[role][0]
    if face_set == "windows":
        ascent, descent = faces[base].getmetrics()
    else:                                              # the replaced face's line, so nothing on screen moves
        a, d, upm = LINE.get(role, SEGOE_LINE)
        ascent, descent = -(-size * a // upm), -(-size * d // upm)
    emoji_px = emoji_size or int(round(size * 0.95))
    tw = twemoji_files() if full else {}
    plan, seqs = {}, []

    def first(chain, cp):
        return next((s for s in chain if cp in cmaps[s]), None)
    if full:
        for cps, path in tw.items():
            if len(cps) == 1 and cps[0] >= 0x2300 and cps[0] not in SKIP:
                plan[cps[0]] = path
        for s in fs["emoji"]:
            for cp in sorted(cmaps[s]):
                if cp >= 0x2600 and cp not in SKIP and cp not in plan:
                    plan[cp] = s
        for s in fs["text"] + fs["arial"] + fs["symbols"]:   # first face that has the character wins
            for cp in cmaps[s]:
                if cp >= 32 and cp not in SKIP and cp not in plan and not (0xD800 <= cp < 0xE000):
                    plan[cp] = s
        for cp, r in cjk_codepoints().items():         # East Asian text the fonts above lack
            s = first(fs[r], cp)
            if cp not in plan and s:
                plan[cp] = s
        for k, (cps, path) in enumerate(sorted((c, p) for c, p in tw.items() if len(c) > 1)):
            plan[SEQ_BASE + k] = path
            seqs.append(cps)
    else:
        cps_wanted = [cp for lo, hi in (MONO_RANGES if role == "mono" else TEXT_RANGES)
                      for cp in range(lo, hi)] + [ord(c) for c in EXTRA_CHARS] + [0xFFFD]
        for cp in cps_wanted:
            s = first(fs[role], cp)
            if s or cp == 0xFFFD:
                plan[cp] = s or base
        if emoji:                                       # extra Twemoji (the keyboard's emoji, drawn big)
            tw = twemoji_files()
            for e in emoji:
                cps = tuple(ord(c) for c in e if ord(c) != 0xFE0F)
                if len(cps) == 1:
                    plan[cps[0]] = tw[cps]
                else:
                    plan[SEQ_BASE + len(seqs)] = tw[cps]
                    seqs.append(cps)
    plan.setdefault(0xFFFD, fs["arial"][0] if full else base)
    recs, bitmaps = [], bytearray()
    index = {}
    for cp in sorted(plan):
        src = plan[cp]
        index[cp] = len(recs)
        if not isinstance(src, tuple):                 # a Twemoji PNG
            img = Image.open(src).convert("RGBA").resize((emoji_px, emoji_px), Image.LANCZOS)
            top = ascent - emoji_px + int(round(size * 0.12)) if not emoji_size else (ascent + descent - emoji_px) // 2
            recs.append(struct.pack("<IHHhhHBBI", cp, emoji_px, emoji_px, 1, top, emoji_px + 2, 1, 0, len(bitmaps)))
            bitmaps += img.tobytes()
            continue
        ch, f = chr(cp), faces[src]
        color = src in fs["emoji"]
        dy = ascent - f.getmetrics()[0]                # line the font up on the base font's baseline
        left, top, right, bottom = f.getbbox(ch)
        w, h = max(0, right - left), max(0, bottom - top)
        adv = int(round(f.getlength(ch)))
        data = b""
        if w and h:
            img = Image.new("RGBA" if color else "L", (w, h), (0, 0, 0, 0) if color else 0)
            ImageDraw.Draw(img).text((-left, -top), ch, font=f, fill=(255, 255, 255, 255) if color else 255,
                                     embedded_color=color)
            data = img.tobytes()
            if not any(data):
                data, w, h = b"", 0, 0
        recs.append(struct.pack("<IHHhhHBBI", cp, w, h, left, top + dy, adv, 1 if color and data else 0, 0, len(bitmaps)))
        bitmaps += data
    head = struct.pack("<4sHHII", b"PFN2", ascent + descent, ascent, len(recs), len(bitmaps))
    tail = b""
    if seqs:
        tail = struct.pack("<4sI", b"SEQS", len(seqs))
        for k, cps in enumerate(seqs):
            tail += struct.pack("<BBHI", len(cps), 0, 0, index[SEQ_BASE + k]) + struct.pack(f"<{len(cps)}I", *cps)
    (font_out / fname).write_bytes(head + b"".join(recs) + bytes(bitmaps) + tail)
    n_tw = sum(1 for v in plan.values() if not isinstance(v, tuple))
    print(f"{fname}: {len(recs)} glyphs at {size}px ({n_tw} Twemoji incl. {len(seqs)} sequences), "
          f"{(len(head) + 20 * len(recs) + len(bitmaps) + len(tail)) / 2**20:.1f} MiB")


def main():
    global face_set, font_out
    argv = [a for a in sys.argv if not a.startswith("--")]
    name = argv[1] if len(argv) > 1 else "qwen3.5-4b-q4"
    model_dir = ROOT / "models" / (argv[2] if len(argv) > 2 else "Qwen3.5-4B")
    for a in sys.argv:
        if a.startswith("--fonts="):
            face_set = a.split("=", 1)[1]
        elif a.startswith("--out="):
            font_out = Path(a.split("=", 1)[1])
    OUT.mkdir(parents=True, exist_ok=True)
    font_out.mkdir(parents=True, exist_ok=True)
    if "--fonts-only" not in sys.argv:
        export_kernels("chat" if name.startswith("qwen3.5") else "qwen3")   # chat.cl = qwen35 + parakeet
        export_model_index(name)
        export_tokenizer(model_dir)
    if "--no-fonts" in sys.argv:
        return
    print(f"fonts: {face_set} -> {font_out}")
    export_font("font.bin", 30, full=True)
    export_font("font_bold.bin", 30, full=False, role="bold")
    export_font("font_italic.bin", 30, full=False, role="italic")
    export_font("font_small.bin", 24, full=False, role="text")
    from gen_emoji_pages import keyboard_emoji
    export_font("font_big.bin", 40, full=False, role="semibold", emoji=keyboard_emoji(), emoji_size=25)   # keys ~25 px
    export_font("mono.bin", 26, full=False, role="mono")


if __name__ == "__main__":
    main()
