"""Garlic LLM brand images, drawn here (no external artwork):

  chat/assets/logo.bin     PIMG: "PIMG", u32 count, per image <HHI (width, height, offset of RGBA bytes); the
                           garlic mark at 48 and 168 px (straight alpha)
  chat/sce_sys/icon0.png   512 x 512 home screen icon
  chat/sce_sys/pic1.png    1920 x 1080 home screen background

Run: python make_brand.py   (writes build/brand_preview.png too)
"""

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "pc" / "fonts_libres" / "Selawik" / "selawksb.ttf"   # Selawik Semibold (SIL OFL 1.1)
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


def garlic(size):
    """The mark: a closed garlic bulb with a short tip, in warm off-white."""
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = S / 100
    d.polygon([(x * k, y * k) for x, y in BULB], fill=(230, 218, 196, 255))
    d.polygon([(x * k, y * k) for x, y in CLOVE], fill=(246, 240, 228, 255))
    for seam in SEAMS:
        d.line([(x * k, y * k) for x, y in seam], fill=(201, 184, 156, 255), width=max(1, int(0.014 * S)), joint="curve")
    # soft highlight on the middle clove
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0.43 * S, 0.40 * S, 0.49 * S, 0.66 * S], fill=150)
    mask = mask.filter(ImageFilter.GaussianBlur(0.025 * S))
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.putalpha(Image.composite(mask, Image.new("L", img.size, 0), img.getchannel("A")))
    img.alpha_composite(white)
    shade = Image.new("L", img.size, 0)   # a little shadow under the bulb
    ImageDraw.Draw(shade).ellipse([0.14 * S, 0.72 * S, 0.86 * S, 0.96 * S], fill=70)
    shade = shade.filter(ImageFilter.GaussianBlur(0.05 * S))
    dark = Image.new("RGBA", img.size, (120, 98, 70, 255))
    dark.putalpha(Image.composite(shade, Image.new("L", img.size, 0), img.getchannel("A")))
    img.alpha_composite(dark)
    return img.resize((size, size), Image.LANCZOS)


def blue_background(w, h, glow_at=(0.5, 0.75), glow_r=0.7):
    img = Image.new("RGB", (w, h))
    px = img.load()
    gx, gy = glow_at[0] * w, glow_at[1] * h
    R = glow_r * max(w, h)
    for y in range(h):
        t = y / (h - 1)
        top = (11, 30, 64)
        bot = (4, 10, 26)
        base = [top[i] + (bot[i] - top[i]) * t for i in range(3)]
        for x in range(w):
            dist = math.hypot(x - gx, y - gy) / R
            g = max(0.0, 1 - dist) ** 2
            px[x, y] = tuple(int(base[i] + g * (0, 70, 160)[i] * 0.55) for i in range(3))
    return img


def icon():
    W = 512
    img = blue_background(W, W, (0.5, 0.42), 0.75).convert("RGBA")
    g = garlic(300)
    glow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([116, 56, 396, 336], fill=(63, 169, 245, 90))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(40)))
    img.alpha_composite(g, (106, 38))
    f = ImageFont.truetype(str(FONT), 66)
    d = ImageDraw.Draw(img)
    text = "Garlic LLM"
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, 372), text, font=f, fill=(255, 255, 255))
    d.rectangle([W / 2 - 40, 462, W / 2 + 40, 466], fill=(63, 169, 245))
    return img.convert("RGB")


def background():
    W, H = 1920, 1080
    small = blue_background(W // 4, H // 4, (0.72, 0.55), 0.6).resize((W, H), Image.BICUBIC).convert("RGBA")
    g = garlic(620)
    g.putalpha(g.getchannel("A").point(lambda a: a * 0.16))
    small.alpha_composite(g, (1150, 260))
    return small.convert("RGB")


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
    prev = Image.new("RGB", (512 + 20 + 960, 540), (20, 20, 20))
    prev.paste(ic, (0, 14))
    prev.paste(bg.resize((960, 540)), (532, 0))
    (ROOT / "build").mkdir(exist_ok=True)
    prev.save(ROOT / "build" / "brand_preview.png")
    print("logo.bin", len(head + recs + data), "bytes; icon0.png, pic1.png written")


if __name__ == "__main__":
    main()
