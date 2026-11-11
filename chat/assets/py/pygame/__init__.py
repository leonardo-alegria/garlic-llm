# pygame for Garlic LLM on the PS4: display, Surface, Rect, draw, font, event, key, time, mouse (and silent mixer),
# drawn through _garlic and played with the controller. The D-pad and the buttons send the keys the program
# checks for (read from its source); the footer shows which button is which.
import _garlic
import _gx
import math as _pymath
import sys

# ---- constants ----
QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, USEREVENT = 256, 768, 769, 1025, 1026, 1024, 32850
ACTIVEEVENT, VIDEORESIZE, JOYBUTTONDOWN, JOYAXISMOTION, WINDOWCLOSE = 32768, 32769, 1539, 1536, 32787
SRCALPHA, HWSURFACE, DOUBLEBUF, FULLSCREEN, RESIZABLE, NOFRAME, SCALED = 65536, 1, 1073741824, -2147483648, 16, 32, 512
BLEND_RGBA_MULT, BLEND_ADD = 8, 1
K_BACKSPACE, K_TAB, K_RETURN, K_ESCAPE, K_SPACE, K_DELETE = 8, 9, 13, 27, 32, 127
K_UP, K_DOWN, K_RIGHT, K_LEFT = 1073741906, 1073741905, 1073741903, 1073741904
K_LSHIFT, K_RSHIFT, K_LCTRL, K_RCTRL, K_KP_ENTER = 1073742049, 1073742053, 1073742048, 1073742052, 1073741912
KMOD_NONE = 0
_KEYNAMES = {K_UP: "up", K_DOWN: "down", K_LEFT: "left", K_RIGHT: "right", K_SPACE: "space", K_RETURN: "return",
             K_ESCAPE: "escape", K_BACKSPACE: "backspace", K_TAB: "tab", K_LSHIFT: "left shift", K_KP_ENTER: "enter"}
for _c in "abcdefghijklmnopqrstuvwxyz":
    globals()["K_" + _c] = ord(_c)
    _KEYNAMES[ord(_c)] = _c
for _c in "0123456789":
    globals()["K_" + _c] = ord(_c)
    _KEYNAMES[ord(_c)] = _c
_NAME2KEY = {v: k for k, v in _KEYNAMES.items()}

error = RuntimeError


def init():
    return (6, 0)


def quit():
    pass


def get_init():
    return True


def get_sdl_version():
    return (2, 28, 0)


# ---- Rect ----
class Rect:
    def __init__(self, *a):
        if len(a) == 1:
            a = a[0]
            if isinstance(a, Rect):
                a = (a.x, a.y, a.w, a.h)
            elif len(a) == 2:
                a = (a[0][0], a[0][1], a[1][0], a[1][1])
        elif len(a) == 2:
            a = (a[0][0], a[0][1], a[1][0], a[1][1])
        self.x, self.y, self.w, self.h = [int(round(v)) if isinstance(v, float) else int(v) for v in a[:4]]

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __getitem__(self, i):
        return (self.x, self.y, self.w, self.h)[i]

    def __len__(self):
        return 4

    def __repr__(self):
        return "<rect(%d, %d, %d, %d)>" % (self.x, self.y, self.w, self.h)

    def __eq__(self, o):
        try:
            return tuple(self) == tuple(Rect(o))
        except Exception:
            return False

    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)

    # derived positions
    def _get(self, name):
        x, y, w, h = self.x, self.y, self.w, self.h
        return {"left": x, "right": x + w, "top": y, "bottom": y + h, "width": w, "height": h, "size": (w, h),
                "centerx": x + w // 2, "centery": y + h // 2, "center": (x + w // 2, y + h // 2),
                "topleft": (x, y), "topright": (x + w, y), "bottomleft": (x, y + h), "bottomright": (x + w, y + h),
                "midtop": (x + w // 2, y), "midbottom": (x + w // 2, y + h), "midleft": (x, y + h // 2),
                "midright": (x + w, y + h // 2)}[name]

    def __getattr__(self, name):
        try:
            return self._get(name)
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, v):
        if name in ("x", "y", "w", "h"):
            object.__setattr__(self, name, int(v))
            return
        x, y, w, h = self.x, self.y, self.w, self.h
        if name == "left":
            self.x = v
        elif name == "right":
            self.x = v - w
        elif name == "top":
            self.y = v
        elif name == "bottom":
            self.y = v - h
        elif name == "width":
            self.w = v
        elif name == "height":
            self.h = v
        elif name == "size":
            self.w, self.h = v
        elif name == "centerx":
            self.x = v - w // 2
        elif name == "centery":
            self.y = v - h // 2
        elif name == "center":
            self.x, self.y = v[0] - w // 2, v[1] - h // 2
        elif name == "topleft":
            self.x, self.y = v
        elif name == "topright":
            self.x, self.y = v[0] - w, v[1]
        elif name == "bottomleft":
            self.x, self.y = v[0], v[1] - h
        elif name == "bottomright":
            self.x, self.y = v[0] - w, v[1] - h
        elif name == "midtop":
            self.x, self.y = v[0] - w // 2, v[1]
        elif name == "midbottom":
            self.x, self.y = v[0] - w // 2, v[1] - h
        elif name == "midleft":
            self.x, self.y = v[0], v[1] - h // 2
        elif name == "midright":
            self.x, self.y = v[0] - w, v[1] - h // 2
        else:
            object.__setattr__(self, name, v)

    def move(self, dx, dy=None):
        if dy is None:
            dx, dy = dx
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def move_ip(self, dx, dy=None):
        if dy is None:
            dx, dy = dx
        self.x += int(dx)
        self.y += int(dy)

    def inflate(self, dx, dy=None):
        if dy is None:
            dx, dy = dx
        return Rect(self.x - dx // 2, self.y - dy // 2, self.w + dx, self.h + dy)

    def inflate_ip(self, dx, dy=None):
        r = self.inflate(dx, dy)
        self.x, self.y, self.w, self.h = r.x, r.y, r.w, r.h

    def clamp_ip(self, other):
        o = Rect(other)
        self.x = max(o.x, min(self.x, o.x + o.w - self.w))
        self.y = max(o.y, min(self.y, o.y + o.h - self.h))

    def clamp(self, other):
        r = self.copy()
        r.clamp_ip(other)
        return r

    def colliderect(self, o):
        o = o if isinstance(o, Rect) else Rect(o)
        return self.x < o.x + o.w and o.x < self.x + self.w and self.y < o.y + o.h and o.y < self.y + self.h

    def collidepoint(self, x, y=None):
        if y is None:
            x, y = x
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def collidelist(self, rects):
        for i, r in enumerate(rects):
            if self.colliderect(r):
                return i
        return -1

    def collidelistall(self, rects):
        return [i for i, r in enumerate(rects) if self.colliderect(r)]

    def contains(self, o):
        o = Rect(o)
        return self.x <= o.x and self.y <= o.y and o.x + o.w <= self.x + self.w and o.y + o.h <= self.y + self.h

    def union(self, o):
        o = Rect(o)
        x, y = min(self.x, o.x), min(self.y, o.y)
        return Rect(x, y, max(self.x + self.w, o.x + o.w) - x, max(self.y + self.h, o.y + o.h) - y)

    def clip(self, o):
        o = Rect(o)
        x, y = max(self.x, o.x), max(self.y, o.y)
        r, b = min(self.x + self.w, o.x + o.w), min(self.y + self.h, o.y + o.h)
        return Rect(x, y, max(0, r - x), max(0, b - y))

    def update(self, *a):   # (pygame 2) set all four: update(x, y, w, h), update((x, y), (w, h)), update(rect)
        r = Rect(*a)
        self.x, self.y, self.w, self.h = r.x, r.y, r.w, r.h

    def unionall(self, rects):
        r = self.copy()
        for o in rects:
            r = r.union(o)
        return r

    def normalize(self):
        if self.w < 0:
            self.x += self.w
            self.w = -self.w
        if self.h < 0:
            self.y += self.h
            self.h = -self.h


FRect = Rect


class Color:   # (not a tuple subclass: MicroPython has no tuple.__new__ for subclasses)
    def __init__(self, *a):
        if len(a) == 1 and not isinstance(a[0], (tuple, list, Color)):
            v = _gx.color(a[0])
            a = (v >> 16 & 255, v >> 8 & 255, v & 255)
        elif len(a) == 1:
            a = tuple(a[0])
        v = [int(x) for x in a] + [255] * (4 - len(a))
        self.r, self.g, self.b, self.a = v[:4]

    def _t(self):
        return (self.r, self.g, self.b, self.a)

    def __iter__(self):
        return iter(self._t())

    def __getitem__(self, i):
        return self._t()[i]

    def __len__(self):
        return 4

    def __eq__(self, o):
        try:
            return self._t() == tuple(Color(o))
        except Exception:
            return False

    def __repr__(self):
        return "Color(%d, %d, %d, %d)" % self._t()

    def lerp(self, o, t):
        o = Color(o)
        return Color(*[int(x + (y - x) * t) for x, y in zip(self._t(), o._t())])


def _rgb(c):
    return _gx.color(tuple(c) if isinstance(c, Color) else c)


# ---- Surfaces: the screen draws at once; other surfaces record what is drawn on them and replay it when blitted ----
class Surface:
    def __init__(self, size=(1, 1), flags=0, depth=32, *a, **k):
        self._size = (int(size[0]), int(size[1]))
        self._ops = []
        self._screen = False
        self._alpha = 255
        self._colorkey = None

    def get_size(self):
        return self._size

    def get_width(self):
        return self._size[0]

    def get_height(self):
        return self._size[1]

    def get_rect(self, **kw):
        r = Rect(0, 0, self._size[0], self._size[1])
        for k, v in kw.items():
            setattr(r, k, v)
        return r

    def get_flags(self):
        return 0

    def _op(self, *op):
        if self._screen:
            _run_op(op, 0, 0)
        else:
            if op[0] == "fill" and op[2] is None:
                self._ops = []
            self._ops.append(op)

    def fill(self, color, rect=None, special_flags=0):
        c = _rgb(color)
        if rect is None:
            if self._screen:
                _garlic.rect(0, 0, self._size[0], self._size[1], c, None)
            else:
                self._op("fill", c, None)
        else:
            r = Rect(rect)
            self._op("rect", c, (r.x, r.y, r.w, r.h), 0)
        return Rect(0, 0, self._size[0], self._size[1])

    def blit(self, src, dest=(0, 0), area=None, special_flags=0):
        if isinstance(dest, Rect):
            x, y = dest.x, dest.y
        else:
            x, y = dest[0], dest[1]
        self._op("blit", src, (x, y))
        return Rect(x, y, src._size[0], src._size[1])

    def blits(self, seq, doreturn=1):
        return [self.blit(*s) for s in seq]

    def set_alpha(self, a, flags=0):
        self._alpha = 255 if a is None else int(a)

    def get_alpha(self):
        return self._alpha

    def set_colorkey(self, *a):
        self._colorkey = a[0] if a else None

    def convert(self, *a):
        return self

    convert_alpha = convert

    def copy(self):
        s = Surface(self._size)
        s._ops = list(self._ops)
        s._alpha = self._alpha
        return s

    def subsurface(self, rect):
        return self

    def get_at(self, pos):
        return Color(0, 0, 0)

    def set_at(self, pos, color):
        self._op("rect", _rgb(color), (pos[0], pos[1], 1, 1), 0)

    def lock(self):
        pass

    unlock = lock

    def get_bounding_rect(self, *a):
        return self.get_rect()


def _run_op(op, ox, oy, alpha=255):
    k = op[0]
    if k == "fill":
        pass
    elif k == "rect":
        c, (x, y, w, h), width = op[1], op[2], op[3]
        if alpha < 255:
            c = (c & 0xFFFFFF) | ((255 - alpha * (255 - (c >> 24)) // 255) << 24)
        if width:
            _garlic.rect(ox + x, oy + y, w, h, None, c & 0xFFFFFF, width)
        else:
            _garlic.rect(ox + x, oy + y, w, h, c, None)
    elif k == "oval":
        c, (x, y, w, h), width = op[1], op[2], op[3]
        if width:
            _garlic.oval(ox + x, oy + y, w, h, None, c, width)
        else:
            _garlic.oval(ox + x, oy + y, w, h, c, None)
    elif k == "line":
        _garlic.line(ox + op[2], oy + op[3], ox + op[4], oy + op[5], op[1], op[6])
    elif k == "poly":
        pts = [v + (ox if i % 2 == 0 else oy) for i, v in enumerate(op[2])]
        if op[3]:
            _garlic.poly(pts, None, op[1], op[3])
        else:
            _garlic.poly(pts, op[1], None)
    elif k == "text":
        _garlic.text(ox + op[1], oy + op[2], op[3], op[4], op[5], op[6], 0)
    elif k == "blit":
        src, (x, y) = op[1], op[2]
        if isinstance(src, _TextSurface):
            if src._bg is not None:
                _garlic.rect(ox + x, oy + y, src._size[0], src._size[1], src._bg, None)
            _garlic.text(ox + x, oy + y, src._text, src._px, src._bold, src._color, 0)
        else:
            a = alpha * src._alpha // 255
            for o in src._ops:
                if o[0] == "fill":
                    c = o[1]
                    if a < 255 or (c >> 24):
                        cov = a * (255 - (c >> 24)) // 255
                        c = (c & 0xFFFFFF) | ((255 - cov) << 24)
                    if src._colorkey is None or _rgb(src._colorkey) != (c & 0xFFFFFF):
                        _garlic.rect(ox + x, oy + y, src._size[0], src._size[1], c, None)
                else:
                    _run_op(o, ox + x, oy + y, a)


class _TextSurface(Surface):
    def __init__(self, text, px, bold, color, bg):
        w, h = _garlic.text_size(text or " ", px, bold)
        Surface.__init__(self, (max(1, int(w)), int(h)))
        self._text, self._px, self._bold, self._color, self._bg = text, px, bold, color, bg


# ---- display ----
class _Display:
    def __init__(self):
        self._screen = None

    def set_mode(self, size=(0, 0), flags=0, depth=0, display=0, vsync=0):
        if not size or size[0] == 0:
            size = (800, 600)
        s = Surface(size)
        s._screen = True
        self._screen = s
        _garlic.begin(int(size[0]), int(size[1]), 0)
        _keys()
        return s

    def get_surface(self):
        return self._screen

    def set_caption(self, title, icontitle=None):
        _garlic.set_title(str(title))

    def get_caption(self):
        return ("pygame", "pygame")

    def set_icon(self, *a):
        pass

    def flip(self):
        _garlic.frame()

    def update(self, *a):
        _garlic.frame()

    def init(self):
        pass

    def quit(self):
        pass

    def get_init(self):
        return True

    def Info(self):
        class _I:
            current_w, current_h = 1920, 1080
        return _I()

    def toggle_fullscreen(self):
        pass

    def get_window_size(self):
        return self._screen.get_size() if self._screen else (800, 600)


display = _Display()


# ---- draw ----
class _Draw:
    def rect(self, surf, color, rect, width=0, border_radius=0, *a, **k):
        r = Rect(rect)
        surf._op("rect", _rgb(color), (r.x, r.y, r.w, r.h), width)
        return r

    def circle(self, surf, color, center, radius, width=0, *a):
        x, y = center
        surf._op("oval", _rgb(color), (x - radius, y - radius, 2 * radius, 2 * radius), width)
        return Rect(x - radius, y - radius, 2 * radius, 2 * radius)

    def ellipse(self, surf, color, rect, width=0):
        r = Rect(rect)
        surf._op("oval", _rgb(color), (r.x, r.y, r.w, r.h), width)
        return r

    def line(self, surf, color, start, end, width=1):
        surf._op("line", _rgb(color), start[0], start[1], end[0], end[1], width)
        return Rect(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]) + 1, abs(end[1] - start[1]) + 1)

    def aaline(self, surf, color, start, end, blend=1):
        return self.line(surf, color, start, end, 1)

    def lines(self, surf, color, closed, points, width=1):
        pts = list(points)
        for i in range(len(pts) - 1):
            self.line(surf, color, pts[i], pts[i + 1], width)
        if closed and len(pts) > 2:
            self.line(surf, color, pts[-1], pts[0], width)

    aalines = lines

    def polygon(self, surf, color, points, width=0):
        flat = []
        for p in points:
            flat += [p[0], p[1]]
        surf._op("poly", _rgb(color), flat, width)

    def arc(self, surf, color, rect, start, stop, width=1):
        r = Rect(rect)
        cx, cy, rx, ry = r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2
        n = 24
        prev = None
        for i in range(n + 1):
            a = start + (stop - start) * i / n
            p = (cx + rx * _pymath.cos(a), cy - ry * _pymath.sin(a))
            if prev:
                self.line(surf, color, prev, p, width)
            prev = p


draw = _Draw()


# ---- font ----
class Font:
    def __init__(self, name=None, size=24, bold=False, italic=False):
        self._px = size * (0.72 if name is None else 1.0)
        self._bold = bold

    def render(self, text, antialias=True, color=(255, 255, 255), background=None):
        return _TextSurface(str(text), self._px, self._bold, _rgb(color), _rgb(background) if background is not None else None)

    def size(self, text):
        w, h = _garlic.text_size(str(text) or " ", self._px, self._bold)
        return (int(w), int(h))

    def get_height(self):
        return int(_garlic.text_size("Ag", self._px, self._bold)[1])

    get_linesize = get_height

    def set_bold(self, b):
        self._bold = b

    def set_italic(self, i):
        pass

    def set_underline(self, u):
        pass


class _FontModule:
    Font = Font

    def SysFont(self, name, size, bold=False, italic=False):
        return Font("sys", size, bold)

    def init(self):
        pass

    def quit(self):
        pass

    def get_init(self):
        return True

    def get_fonts(self):
        return ["arial"]

    def match_font(self, *a, **k):
        return None

    def get_default_font(self):
        return "freesansbold.ttf"


font = _FontModule()


# ---- events and keys ----
_map = None
_held_keys = set()
_queue = []


def _keys():
    """Which key each controller button sends, from the K_ names the program uses."""
    global _map
    if _map is not None:
        return _map
    src = _garlic.source()
    used = []
    i = 0
    while True:
        i = src.find("K_", i)
        if i < 0:
            break
        j = i + 2
        while j < len(src) and (src[j].isalpha() or src[j].isdigit() or src[j] == "_"):
            j += 1
        name = src[i + 2:j].lower()
        name = {"return": "return", "kp_enter": "return", "space": "space"}.get(name, name)
        if name in ("up", "down", "left", "right", "space", "return", "escape") or len(name) == 1:
            if name not in used and name != "escape":
                used.append(name)
        i = j
    if "MOUSEBUTTONDOWN" in src or "mouse.get_pressed" in src:
        _garlic.mouse_mode(True)
        _garlic_mouse[0] = True
    _map = _gx.keymap(used)
    return _map


class Event:
    def __init__(self, type, d=None, **kw):
        self.type = type
        self.dict = dict(d or {}, **kw)
        for k, v in self.dict.items():
            setattr(self, k, v)

    def __repr__(self):
        return "<Event(%d %r)>" % (self.type, self.dict)


EventType = Event


def _poll():
    m = _keys()
    for kind, button, x, y in _garlic.events():
        if button in ("cross", "square") and _garlic_mouse[0]:
            et = MOUSEBUTTONDOWN if kind == "down" else MOUSEBUTTONUP
            _queue.append(Event(et, pos=(int(x), int(y)), button=1 if button == "cross" else 3))
            if button == "cross" and m.get("cross") is None:
                continue
        k = m.get(button)
        if not k:
            continue
        code = _NAME2KEY.get(k, ord(k[0]))
        if kind == "down":
            _held_keys.add(code)
            _queue.append(Event(KEYDOWN, key=code, unicode=k if len(k) == 1 else (" " if k == "space" else ""), mod=0, scancode=0))
        else:
            _held_keys.discard(code)
            _queue.append(Event(KEYUP, key=code, mod=0, scancode=0))


_garlic_mouse = [False]


class _EventModule:
    Event = Event

    def get(self, eventtype=None, pump=True, exclude=None):
        _poll()
        out = [e for e in _queue if eventtype is None or e.type == eventtype or (isinstance(eventtype, (list, tuple)) and e.type in eventtype)]
        _queue[:] = [e for e in _queue if e not in out]
        return out

    def poll(self):
        _poll()
        return _queue.pop(0) if _queue else Event(0)

    def wait(self, timeout=0):
        while True:
            _poll()
            if _queue:
                return _queue.pop(0)
            _garlic.frame()

    def pump(self):
        _poll()

    def clear(self, *a):
        _poll()
        _queue.clear()

    def post(self, e):
        _queue.append(e)

    def set_allowed(self, *a):
        pass

    set_blocked = set_allowed

    def custom_type(self):
        return USEREVENT + 1


event = _EventModule()


class _Pressed:
    def __getitem__(self, code):
        return code in _held_keys

    def __len__(self):
        return 512


class _KeyModule:
    def get_pressed(self):
        _poll()
        return _Pressed()

    def name(self, code):
        return _KEYNAMES.get(code, "")

    def key_code(self, name):
        return _NAME2KEY.get(name, 0)

    def set_repeat(self, *a):
        pass

    def get_mods(self):
        return 0

    def get_focused(self):
        return True


key = _KeyModule()


class _Mouse:
    def get_pos(self):
        x, y = _garlic.mouse()
        return (int(x), int(y))

    def get_pressed(self, num_buttons=3):
        h = _garlic.held()
        return ("cross" in h, False, "square" in h)

    def set_visible(self, v):
        pass

    def get_visible(self):
        return True

    def set_pos(self, *a):
        pass


mouse = _Mouse()


# ---- time ----
class Clock:
    def __init__(self):
        self._last = _garlic.ticks()
        self._fps = 0

    def tick(self, framerate=0):
        now = _garlic.ticks()
        if framerate:
            wait = 1.0 / framerate - (now - self._last)
            if wait > 0.002:
                _garlic.wait(wait)
            now = _garlic.ticks()
        dt = now - self._last
        self._fps = 1 / dt if dt > 0 else 0
        self._last = now
        return int(dt * 1000)

    tick_busy_loop = tick

    def get_fps(self):
        return self._fps

    def get_time(self):
        return 16

    def get_rawtime(self):
        return 16


class _TimeModule:
    Clock = Clock

    def get_ticks(self):
        return int(_garlic.ticks() * 1000)

    def wait(self, ms):
        _garlic.wait(ms / 1000)
        return ms

    delay = wait

    def set_timer(self, ev, ms, loops=0):
        pass


time = _TimeModule()


# ---- sound: silent ----
class _Sound:
    def __init__(self, *a, **k):
        pass

    def play(self, *a, **k):
        pass

    def stop(self):
        pass

    def set_volume(self, v):
        pass

    def get_length(self):
        return 0


class _Music:
    def load(self, *a):
        pass

    def play(self, *a, **k):
        pass

    def stop(self):
        pass

    def set_volume(self, v):
        pass

    def pause(self):
        pass

    def unpause(self):
        pass

    def get_busy(self):
        return False


class _Mixer:
    Sound = _Sound
    music = _Music()

    def init(self, *a, **k):
        pass

    def pre_init(self, *a, **k):
        pass

    def quit(self):
        pass

    def get_init(self):
        return True

    def Channel(self, *a):
        return _Sound()


mixer = _Mixer()


class _Image:
    def load(self, path, *a):
        raise FileNotFoundError("No file found: " + str(path))

    def save(self, *a):
        pass


image = _Image()


class _Transform:
    def scale(self, surf, size, *a):
        return surf

    smoothscale = scale

    def rotate(self, surf, angle):
        return surf

    def flip(self, surf, x, y):
        return surf

    def rotozoom(self, surf, angle, scale):
        return surf


transform = _Transform()


class _Sprite:
    class Sprite:
        def __init__(self, *groups):
            self._groups = []
            for g in groups:
                g.add(self)

        def kill(self):
            for g in list(self._groups):
                g.remove(self)

        def alive(self):
            return bool(self._groups)

        def update(self, *a, **k):
            pass

        def add(self, *groups):
            for g in groups:
                g.add(self)

        def remove(self, *groups):
            for g in groups:
                g.remove(self)

        def groups(self):
            return list(self._groups)

    class Group:
        def __init__(self, *sprites):
            self._s = []
            for s in sprites:
                self.add(s)

        def add(self, *sprites):
            for s in sprites:
                if isinstance(s, (list, tuple)):
                    self.add(*s)
                elif s not in self._s:
                    self._s.append(s)
                    s._groups.append(self)

        def remove(self, *sprites):
            for s in sprites:
                if s in self._s:
                    self._s.remove(s)
                    s._groups.remove(self)

        def has(self, s):
            return s in self._s

        def sprites(self):
            return list(self._s)

        def update(self, *a, **k):
            for s in list(self._s):
                s.update(*a, **k)

        def draw(self, surf):
            for s in self._s:
                if hasattr(s, "image") and hasattr(s, "rect"):
                    surf.blit(s.image, s.rect)

        def empty(self):
            for s in list(self._s):
                self.remove(s)

        def __iter__(self):
            return iter(list(self._s))

        def __len__(self):
            return len(self._s)

        def __contains__(self, s):
            return s in self._s

        def __bool__(self):
            return bool(self._s)

    RenderPlain = RenderUpdates = OrderedUpdates = LayeredUpdates = GroupSingle = Group

    def spritecollide(self, sprite, group, dokill, collided=None):
        hits = [s for s in group if sprite.rect.colliderect(s.rect)]
        if dokill:
            for s in hits:
                s.kill()
        return hits

    def spritecollideany(self, sprite, group, collided=None):
        for s in group:
            if sprite.rect.colliderect(s.rect):
                return s
        return None

    def groupcollide(self, a, b, dokilla, dokillb, collided=None):
        out = {}
        for s in list(a):
            hits = self.spritecollide(s, b, dokillb)
            if hits:
                out[s] = hits
                if dokilla:
                    s.kill()
        return out

    def collide_rect(self, a, b):
        return a.rect.colliderect(b.rect)


sprite = _Sprite()


class _Math:
    class Vector2:
        def __init__(self, x=0, y=None):
            if y is None:
                x, y = (x[0], x[1]) if isinstance(x, (tuple, list, _Math.Vector2)) else (x, x)
            self.x, self.y = float(x), float(y)

        def __getitem__(self, i):
            return (self.x, self.y)[i]

        def __iter__(self):
            return iter((self.x, self.y))

        def __len__(self):
            return 2

        def __add__(self, o):
            return _Math.Vector2(self.x + o[0], self.y + o[1])

        def __sub__(self, o):
            return _Math.Vector2(self.x - o[0], self.y - o[1])

        def __mul__(self, k):
            if isinstance(k, (int, float)):
                return _Math.Vector2(self.x * k, self.y * k)
            return self.x * k[0] + self.y * k[1]

        __rmul__ = __mul__

        def __truediv__(self, k):
            return _Math.Vector2(self.x / k, self.y / k)

        def __neg__(self):
            return _Math.Vector2(-self.x, -self.y)

        def __eq__(self, o):
            return self.x == o[0] and self.y == o[1]

        def __repr__(self):
            return "Vector2(%g, %g)" % (self.x, self.y)

        def length(self):
            return _pymath.sqrt(self.x * self.x + self.y * self.y)

        magnitude = length

        def length_squared(self):
            return self.x * self.x + self.y * self.y

        def normalize(self):
            l = self.length() or 1
            return _Math.Vector2(self.x / l, self.y / l)

        def normalize_ip(self):
            l = self.length() or 1
            self.x /= l
            self.y /= l

        def distance_to(self, o):
            return _pymath.sqrt((self.x - o[0]) ** 2 + (self.y - o[1]) ** 2)

        def dot(self, o):
            return self.x * o[0] + self.y * o[1]

        def rotate(self, deg):
            a = _pymath.radians(deg)
            return _Math.Vector2(self.x * _pymath.cos(a) - self.y * _pymath.sin(a), self.x * _pymath.sin(a) + self.y * _pymath.cos(a))

        def copy(self):
            return _Math.Vector2(self.x, self.y)

    Vector3 = Vector2


math = _Math()
Vector2 = _Math.Vector2


class _Joystick:
    def init(self):
        pass

    def get_count(self):
        return 0


joystick = _Joystick()
