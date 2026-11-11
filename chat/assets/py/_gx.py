# Shared helpers of Garlic's turtle / tkinter / pygame: colours and which key each controller button sends.
import _garlic

COLORS = {
    "black": 0x000000, "white": 0xFFFFFF, "red": 0xFF0000, "green": 0x008000, "blue": 0x0000FF, "yellow": 0xFFFF00,
    "orange": 0xFFA500, "purple": 0x800080, "pink": 0xFFC0CB, "brown": 0xA52A2A, "gray": 0x808080, "grey": 0x808080,
    "cyan": 0x00FFFF, "magenta": 0xFF00FF, "lime": 0x00FF00, "navy": 0x000080, "teal": 0x008080, "olive": 0x808000,
    "maroon": 0x800000, "silver": 0xC0C0C0, "gold": 0xFFD700, "violet": 0xEE82EE, "indigo": 0x4B0082,
    "turquoise": 0x40E0D0, "coral": 0xFF7F50, "salmon": 0xFA8072, "khaki": 0xF0E68C, "beige": 0xF5F5DC,
    "tan": 0xD2B48C, "crimson": 0xDC143C, "orchid": 0xDA70D6, "plum": 0xDDA0DD, "lavender": 0xE6E6FA,
    "skyblue": 0x87CEEB, "lightblue": 0xADD8E6, "lightgreen": 0x90EE90, "lightgray": 0xD3D3D3, "lightgrey": 0xD3D3D3,
    "lightyellow": 0xFFFFE0, "lightpink": 0xFFB6C1, "lightcyan": 0xE0FFFF, "darkblue": 0x00008B, "darkgreen": 0x006400,
    "darkred": 0x8B0000, "darkgray": 0xA9A9A9, "darkgrey": 0xA9A9A9, "darkorange": 0xFF8C00, "darkviolet": 0x9400D3,
    "darkcyan": 0x008B8B, "forestgreen": 0x228B22, "seagreen": 0x2E8B57, "springgreen": 0x00FF7F,
    "royalblue": 0x4169E1, "steelblue": 0x4682B4, "dodgerblue": 0x1E90FF, "deepskyblue": 0x00BFFF,
    "midnightblue": 0x191970, "slategray": 0x708090, "dimgray": 0x696969, "dimgrey": 0x696969, "whitesmoke": 0xF5F5F5,
    "gainsboro": 0xDCDCDC, "snow": 0xFFFAFA, "ivory": 0xFFFFF0, "hotpink": 0xFF69B4, "deeppink": 0xFF1493,
    "tomato": 0xFF6347, "orangered": 0xFF4500, "chocolate": 0xD2691E, "sienna": 0xA0522D, "firebrick": 0xB22222,
    "goldenrod": 0xDAA520, "greenyellow": 0xADFF2F, "chartreuse": 0x7FFF00, "lawngreen": 0x7CFC00,
    "yellowgreen": 0x9ACD32, "aquamarine": 0x7FFFD4, "aqua": 0x00FFFF, "fuchsia": 0xFF00FF, "wheat": 0xF5DEB3,
    "systembuttonface": 0xF0F0F0, "systemwindow": 0xFFFFFF, "systembuttontext": 0x000000, "lightsteelblue": 0xB0C4DE,
    "lightslategray": 0x778899, "mediumseagreen": 0x3CB371, "mediumpurple": 0x9370DB, "palegreen": 0x98FB98,
    "lightcoral": 0xF08080, "darkslategray": 0x2F4F4F, "darkslategrey": 0x2F4F4F, "cornflowerblue": 0x6495ED,
    "lightgoldenrodyellow": 0xFAFAD2, "mintcream": 0xF5FFFA, "honeydew": 0xF0FFF0, "aliceblue": 0xF0F8FF,
}


def color(c, default=0, unit=255):
    """Name, "#rgb", "#rrggbb", (r, g, b) or (r, g, b, a) -> 0xRRGGBB (alpha < 255 -> 0xTTRRGGBB, TT = 255 - a)."""
    if c is None:
        return default
    if isinstance(c, int):
        return c
    if isinstance(c, str):
        s = c.strip().lower().replace(" ", "")
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = h[0] * 2 + h[1] * 2 + h[2] * 2
            try:
                return int(h[:6], 16)
            except ValueError:
                return default
        if s in COLORS:
            return COLORS[s]
        if s.startswith(("rgb(", "rgba(", "hsl(", "hsla(")) and s.endswith(")"):   # CSS colours (the 9B writes them)
            try:
                v = s[s.index("(") + 1:-1].split(",")
                if s.startswith("rgb"):
                    r, g, b = [int(float(x[:-1]) * 2.55 + 0.5) if x.endswith("%") else int(float(x)) for x in v[:3]]
                else:
                    h = float(v[0].replace("deg", "")) / 360 % 1
                    sat, lig = float(v[1].rstrip("%")) / 100, float(v[2].rstrip("%")) / 100
                    q = lig * (1 + sat) if lig < 0.5 else lig + sat - lig * sat
                    p = 2 * lig - q

                    def ch(t):
                        t %= 1
                        if t < 1 / 6:
                            return p + (q - p) * 6 * t
                        if t < 0.5:
                            return q
                        return p + (q - p) * (2 / 3 - t) * 6 if t < 2 / 3 else p
                    r, g, b = int(ch(h + 1 / 3) * 255 + 0.5), int(ch(h) * 255 + 0.5), int(ch(h - 1 / 3) * 255 + 0.5)
                return max(0, min(255, r)) << 16 | max(0, min(255, g)) << 8 | max(0, min(255, b))
            except (ValueError, IndexError):
                return default
        for suffix in "1234":   # tk variants such as "gray20", "blue2"
            if s.endswith(suffix) and s[:-1] in COLORS:
                return COLORS[s[:-1]]
        if s.startswith(("gray", "grey")) and s[4:].isdigit():
            v = int(int(s[4:]) * 2.55)
            return v << 16 | v << 8 | v
        return default
    try:
        v = list(c)
    except TypeError:
        return default
    if len(v) >= 3:
        if unit == 1 or (all(isinstance(x, float) for x in v[:3]) and max(v[:3]) <= 1.0):
            v = [int(x * 255) for x in v[:3]] + v[3:]
        r, g, b = [max(0, min(255, int(x))) for x in v[:3]]
        rgb = r << 16 | g << 8 | b
        if len(v) > 3 and int(v[3]) < 255:
            rgb |= (255 - max(0, int(v[3]))) << 24
        return rgb
    return default


LANG = _garlic.lang() if hasattr(_garlic, "lang") else "es"
_NAMES = {"es": {"space": "Espacio", "return": "Enter", "escape": "Esc", "arrows": "Mover"},
          "en": {"space": "Space", "return": "Enter", "escape": "Esc", "arrows": "Move"},
          "pt": {"space": "Espaço", "return": "Enter", "escape": "Esc", "arrows": "Mover"},
          "ru": {"space": "Пробел", "return": "Enter", "escape": "Esc", "arrows": "Двигать"}}


def key_label(k):
    n = _NAMES.get(LANG, _NAMES["en"])
    return n.get(k.lower(), k.upper() if len(k) == 1 else k)


def keymap(used):
    """used: key names the program reacts to, in order of appearance, lower case ("up", "w", "space", "return",
    "p"...). Returns {button: key} for the D-pad and the face buttons, and sets the footer hints."""
    m = {}
    arrows = [k for k in ("up", "down", "left", "right") if k in used]
    wasd = [k for k in ("w", "s", "a", "d") if k in used]
    if arrows or not wasd:
        m.update(up="up", down="down", left="left", right="right")
    else:
        m.update(up="w", down="s", left="a", right="d")
    taken = set(m.values()) | {"up", "down", "left", "right", "w", "a", "s", "d"}
    rest = [k for k in used if k not in taken]
    prefer = [k for k in ("space", "return") if k in rest]
    order = prefer + [k for k in rest if k not in prefer and k != "q"] + (["q"] if "q" in rest else [])   # quit last
    for button, k in zip(("cross", "square", "triangle", "l1", "r1"), order):
        m[button] = k
    hints = [("up", key_label("arrows"))]
    for button in ("cross", "square", "triangle", "l1", "r1"):
        if button in m:
            hints.append((button, key_label(m[button])))
    _garlic.hints(hints)
    return m
