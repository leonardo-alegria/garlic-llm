# colorsys for Garlic (as CPython's: values from 0 to 1).
def rgb_to_hsv(r, g, b):
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0, 0.0, mx
    d = mx - mn
    rc, gc, bc = (mx - r) / d, (mx - g) / d, (mx - b) / d
    h = bc - gc if r == mx else 2.0 + rc - bc if g == mx else 4.0 + gc - rc
    return (h / 6.0) % 1.0, d / mx, mx


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i % 6]


def rgb_to_hls(r, g, b):
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mn + mx) / 2.0
    if mx == mn:
        return 0.0, l, 0.0
    d = mx - mn
    s = d / (mx + mn) if l <= 0.5 else d / (2.0 - mx - mn)
    rc, gc, bc = (mx - r) / d, (mx - g) / d, (mx - b) / d
    h = bc - gc if r == mx else 2.0 + rc - bc if g == mx else 4.0 + gc - rc
    return (h / 6.0) % 1.0, l, s


def _v(m1, m2, hue):
    hue %= 1.0
    if hue < 1 / 6:
        return m1 + (m2 - m1) * hue * 6.0
    if hue < 0.5:
        return m2
    if hue < 2 / 3:
        return m1 + (m2 - m1) * (2 / 3 - hue) * 6.0
    return m1


def hls_to_rgb(h, l, s):
    if s == 0.0:
        return l, l, l
    m2 = l * (1.0 + s) if l <= 0.5 else l + s - l * s
    m1 = 2.0 * l - m2
    return _v(m1, m2, h + 1 / 3), _v(m1, m2, h), _v(m1, m2, h - 1 / 3)
