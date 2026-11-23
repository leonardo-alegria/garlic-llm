"""DarkGarlic LLM brand images, drawn here (no external artwork):

  chat/assets/logo.bin     PIMG: "PIMG", u32 count, per image <HHI (width, height, offset of RGBA bytes); the
                           garlic mark at 48 and 168 px (straight alpha)
  chat/sce_sys/icon0.png   512 x 512 home screen icon
  chat/sce_sys/pic1.png    1920 x 1080 home screen background

The mark is a white garlic bulb in a black Dracula cape, with two fangs. Red, black and white only.

Run: python make_brand.py   (writes build/brand_preview.png too)
"""

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "pc" / "fonts_libres" / "Selawik" / "selawksb.ttf"   # Selawik Semibold (SIL OFL 1.1)
if not FONT.exists():
    FONT = Path(r"C:\Windows\Fonts\segoeuib.ttf")   # same widths as Selawik; used when the OFL face is not checked out
SS = 4   # supersampling


def _path(start, *segments):
    """Points along cubic Bezier segments ((control 1, control 2, end), ...), in the mark's 100-unit grid."""
    pts = [start]
    for c1, c2, end in segments:
        p0 = pts[-1]
        for i in range(1, 25):
            t = i / 24
            a, b, c, e = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            pts.append((a * p0[0] + b * c1[0] + c * c2[0] + e * end[0], a * p0[1] + b * c1[1] + c * c2[1] + e * end[1]))
    return pts


# The mark on a 100 x 100 grid (the same paths as the website's SVG): one closed bulb with a short tip, the middle
# clove in a lighter tone and a seam on each side.
BULB = _path((50, 15), ((51, 23), (56, 29), (66, 35)), ((80, 43), (88, 55), (88, 67)), ((88, 81), (74, 90), (50, 90)),
             ((26, 90), (12, 81), (12, 67)), ((12, 55), (20, 43), (34, 35)), ((44, 29), (49, 23), (50, 15)))
CLOVE = _path((50, 21), ((40, 38), (35, 70), (43, 89.6)), ((47.67, 90.13), (52.33, 90.13), (57, 89.6)),
              ((65, 70), (60, 38), (50, 21)))
SEAMS = [_path((44, 31), ((27, 44), (22, 68), (31, 87))), _path((56, 31), ((73, 44), (78, 68), (69, 87)))]


# Cape on the same 100-unit grid: high collar, wings past the bulb, hem at the bottom.
CAPE = [(6, 96), (2, 50), (16, 32), (30, 18), (40, 30), (50, 10), (60, 30), (70, 18), (84, 32), (98, 50), (94, 96)]
LINING = [(18, 94), (14, 56), (28, 40), (40, 34), (50, 26), (60, 34), (72, 40), (86, 56), (82, 94)]
FANGS = [[(40, 76), (45, 104), (51, 80)], [(49, 80), (55, 104), (60, 76)]]


def _poly(draw, pts, k, fill):
    draw.polygon([(x * k, y * k) for x, y in pts], fill=fill)


def garlic(size):
    """White garlic bulb, black Dracula cape with a red lining, two fangs."""
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = S / 100
    _poly(d, CAPE, k, (8, 8, 8, 255))
    _poly(d, LINING, k, (176, 16, 32, 255))
    _poly(d, BULB, k, (244, 244, 244, 255))
    _poly(d, CLOVE, k, (255, 255, 255, 255))
    for seam in SEAMS:
        d.line([(x * k, y * k) for x, y in seam], fill=(150, 150, 150, 255), width=max(1, int(0.014 * S)), joint="curve")
    # a short red mouth, then the fangs in front of it
    d.polygon([(38 * k, 74 * k), (62 * k, 74 * k), (58 * k, 80 * k), (42 * k, 80 * k)], fill=(122, 12, 24, 255))
    for fang in FANGS:
        _poly(d, fang, k, (255, 255, 255, 255))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0.43 * S, 0.38 * S, 0.50 * S, 0.58 * S], fill=140)
    mask = mask.filter(ImageFilter.GaussianBlur(0.02 * S))
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.putalpha(Image.composite(mask, Image.new("L", img.size, 0), img.getchannel("A")))
    img.alpha_composite(white)
    return img.resize((size, size), Image.LANCZOS)


def night_background(w, h, glow_at=(0.5, 0.75), glow_r=0.7):
    """Black field with a red pool of light."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    gx, gy = glow_at[0] * w, glow_at[1] * h
    R = glow_r * max(w, h)
    for y in range(h):
        t = y / (h - 1)
        top = (32, 6, 8)
        bot = (0, 0, 0)
        base = [top[i] + (bot[i] - top[i]) * t for i in range(3)]
        for x in range(w):
            dist = math.hypot(x - gx, y - gy) / R
            g = max(0.0, 1 - dist) ** 2
            px[x, y] = tuple(min(255, int(base[i] + g * (150, 8, 16)[i] * 0.65)) for i in range(3))
    return img


def icon():
    W = 512
    img = night_background(W, W, (0.5, 0.40), 0.8).convert("RGBA")
    g = garlic(340)
    glow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([90, 30, 422, 360], fill=(176, 16, 32, 80))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(36)))
    img.alpha_composite(g, (86, 8))
    f = ImageFont.truetype(str(FONT), 54)
    d = ImageDraw.Draw(img)
    text = "DarkGarlic"
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, 372), text, font=f, fill=(255, 255, 255))
    sub = ImageFont.truetype(str(FONT), 28)
    sw = d.textlength("LLM", font=sub)
    d.text(((W - sw) / 2, 432), "LLM", font=sub, fill=(255, 42, 48))
    d.rectangle([W / 2 - 36, 470, W / 2 + 36, 474], fill=(176, 16, 32))
    return img.convert("RGB")


def background():
    W, H = 1920, 1080
    small = night_background(W // 4, H // 4, (0.72, 0.55), 0.65).resize((W, H), Image.BICUBIC).convert("RGBA")
    g = garlic(640)
    g.putalpha(g.getchannel("A").point(lambda a: int(a * 0.34)))
    small.alpha_composite(g, (1140, 220))
    return small.convert("RGB")


def hero():
    """Title card for the README: the mark and the wordmark on a clean black and red field."""
    img = night_background(1920, 1080, (0.5, 0.38), 0.85).convert("RGBA")
    g = garlic(480)
    img.alpha_composite(g, ((1920 - 480) // 2, 90))
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(str(FONT), 88)
    small = ImageFont.truetype(str(FONT), 40)
    name, sub = "DarkGarlic", "LLM"
    nw, sw = d.textlength(name, font=big), d.textlength(sub, font=small)
    gap = 24
    x = (1920 - (nw + gap + sw)) / 2
    y = 680
    d.text((x, y), name, font=big, fill=(255, 255, 255))
    d.text((x + nw + gap, y + 42), sub, font=small, fill=(255, 42, 48))
    d.rectangle([(1920 - 96) / 2, 820, (1920 + 96) / 2, 826], fill=(176, 16, 32))
    return img.convert("RGB")


def main():
    images = [garlic(48), garlic(168)]
    head = b"PIMG" + struct.pack("<I", len(images))
    offset = len(head) + 8 * len(images)
    recs, data = b"", b""
    for im in images:
        recs += struct.pack("<HHI", im.width, im.height, offset + len(data))
        data += im.tobytes()
    (ROOT / "chat" / "assets" / "logo.bin").write_bytes(head + recs + data)
    sce = ROOT / "chat" / "sce_sys"
    sce.mkdir(exist_ok=True)
    ic = icon()
    ic.save(sce / "icon0.png")
    bg = background()
    bg.save(sce / "pic1.png")
    hero().save(ROOT / "docs" / "chat.png")
    prev = Image.new("RGB", (512 + 20 + 960, 540), (20, 20, 20))
    prev.paste(ic, (0, 14))
    prev.paste(bg.resize((960, 540)), (532, 0))
    (ROOT / "build").mkdir(exist_ok=True)
    prev.save(ROOT / "build" / "brand_preview.png")
    print("logo.bin", len(head + recs + data), "bytes; icon0.png, pic1.png written")


if __name__ == "__main__":
    main()
