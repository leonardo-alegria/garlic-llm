# turtle for Garlic LLM on the PS4: the parts of Python's turtle that programs use (Screen, Turtle, keys, timers,
# drawing, write), drawn through _garlic and played with the controller. Coordinates as in turtle: origin in the
# middle, y up.
import _garlic
import _gx
import math

_items = []      # what has been drawn: ("line", owner, x1, y1, x2, y2, color, width) / ("poly"...) / ("text"...) / ("dot"...)
_turtles = []
_screen = None
_mode255 = 1.0


def _col(c, default=0):
    return _gx.color(c, default, 255 if _mode255 == 255 else 1)


class _Screen:
    def __init__(self):
        self._w, self._h = 800, 600
        self._bg = 0xFFFFFF
        self._tracer = 1
        self._press, self._release = {}, {}
        self._clicks = {}
        self._timers = []
        self._title = "turtle"
        self._begun = False
        self._dirty = True
        self._map = None
        self._last = 0
        self._ops = 0

    # window
    def setup(self, width=0.5, height=0.75, startx=None, starty=None):
        self._w = int(width * 1920) if isinstance(width, float) and width <= 1 else int(width)
        self._h = int(height * 1080) if isinstance(height, float) and height <= 1 else int(height)
        self._begun = False
        self._dirty = True

    def screensize(self, canvwidth=None, canvheight=None, bg=None):
        if canvwidth:
            self.setup(canvwidth, canvheight or self._h)
        if bg is not None:
            self.bgcolor(bg)
        return self._w, self._h

    def window_width(self):
        return self._w

    def window_height(self):
        return self._h

    def title(self, t):
        self._title = str(t)
        _garlic.set_title(self._title)

    def bgcolor(self, *c):
        if not c:
            return self._bg
        self._bg = _col(c[0] if len(c) == 1 else c, self._bg)
        self._dirty = True

    def bgpic(self, *a):
        pass

    def colormode(self, m=None):
        global _mode255
        if m is None:
            return _mode255
        _mode255 = m

    def tracer(self, n=None, delay=None):
        if n is None:
            return self._tracer
        self._tracer = n

    def delay(self, d=None):
        return 0

    def mode(self, m=None):
        return "standard"

    def register_shape(self, *a, **k):
        pass

    addshape = register_shape

    def getcanvas(self):
        return self

    def turtles(self):
        return list(_turtles)

    # input
    def listen(self, *a):
        pass

    def onkeypress(self, fun, key=None):
        self._press[(key or "*").lower()] = fun
        self._map = None

    def onkeyrelease(self, fun, key=None):
        self._release[(key or "*").lower()] = fun
        self._map = None

    onkey = onkeyrelease

    def onclick(self, fun, btn=1, add=None):
        self._clicks[btn] = fun
        _garlic.mouse_mode(True)

    onscreenclick = onclick

    def ontimer(self, fun, t=0):
        self._timers.append((_garlic.ticks() + t / 1000.0, fun))

    # loop
    def _begin(self):
        if not self._begun:
            _garlic.begin(self._w, self._h, self._bg)
            _garlic.set_title(self._title)
            self._begun = True

    def _keys(self):
        if self._map is None:
            used = []
            for k in list(self._press) + list(self._release):
                k = {"return": "return", "enter": "return", "space": "space", " ": "space"}.get(k, k)
                if k != "*" and k not in used:
                    used.append(k)
            self._map = _gx.keymap(used)
        return self._map

    def _process(self):
        km = self._keys()
        for kind, button, x, y in _garlic.events():
            tx, ty = x - self._w / 2, self._h / 2 - y
            if self._clicks and button in ("cross", "square"):
                if kind == "down":
                    f = self._clicks.get(1 if button == "cross" else 3)
                    if f:
                        f(tx, ty)
                continue
            key = km.get(button)
            if not key:
                continue
            table = self._press if kind == "down" else self._release
            for name in (key, "*"):
                f = table.get(name.lower())
                if f:
                    f()
                    break
        now = _garlic.ticks()
        due = [t for t in self._timers if t[0] <= now]
        if due:
            self._timers = [t for t in self._timers if t[0] > now]
            for _, f in due:
                f()

    def _draw(self):
        self._begin()
        _garlic.clear(self._bg)
        W2, H2 = self._w / 2, self._h / 2
        for it in _items:
            k = it[0]
            if k == "line":
                _garlic.line(it[2] + W2, H2 - it[3], it[4] + W2, H2 - it[5], it[6], it[7])
            elif k == "poly":
                pts = []
                for i in range(0, len(it[2]), 2):
                    pts.append(it[2][i] + W2)
                    pts.append(H2 - it[2][i + 1])
                _garlic.poly(pts, it[3], it[4], it[5])
            elif k == "dot":
                r = it[4] / 2
                _garlic.oval(it[2] + W2 - r, H2 - it[3] - r, 2 * r, 2 * r, it[5], None)
            elif k == "text":
                _garlic.text(it[2] + W2, H2 - it[3], it[4], it[5], it[6], it[7], it[8])
        for t in _turtles:
            if t._vis:
                t._draw_shape(W2, H2)
        self._dirty = False

    def update(self):
        self._process()
        self._draw()
        _garlic.frame()
        self._last = _garlic.ticks()
        self._ops = 0

    def _changed(self):
        """After a drawing step: with the tracer on, show it now and then (an animation, like turtle)."""
        self._dirty = True
        if self._tracer:
            self._ops += 1
            now = _garlic.ticks()
            if now - self._last > 1 / 30 or self._ops > 400:
                self.update()

    def mainloop(self):
        while True:
            self._process()
            if self._dirty:
                self._draw()
            _garlic.frame()

    done = mainloop

    def exitonclick(self):
        self.mainloop()

    def bye(self):
        raise SystemExit

    def clear(self):
        _items.clear()
        for t in list(_turtles):
            t._vis = False
        _turtles.clear()
        self._dirty = True

    clearscreen = clear
    reset = clear

    def textinput(self, title, prompt):
        return input(prompt + " ")

    def numinput(self, title, prompt, default=None, minval=None, maxval=None):
        s = input(prompt + " ")
        try:
            return float(s)
        except ValueError:
            return default


def Screen():
    global _screen
    if _screen is None:
        _screen = _Screen()
    return _screen


_SHAPES = {
    "classic": [(0, 0), (-9, 5), (-7, 0), (-9, -5)],
    "arrow": [(10, 0), (0, 10), (0, -10)],
    "triangle": [(10, 0), (-5, 8.66), (-5, -8.66)],
    "turtle": [(16, 0), (14, 2), (10, 1), (7, 4), (9, 7), (8, 9), (5, 6), (1, 7), (-3, 5), (-6, 8), (-8, 6), (-5, 4),
               (-7, 0), (-5, -4), (-8, -6), (-6, -8), (-3, -5), (1, -7), (5, -6), (8, -9), (9, -7), (7, -4), (10, -1),
               (14, -2)],
}


class Turtle:
    def __init__(self, shape="classic", visible=True, **kw):
        Screen()
        self._x = self._y = 0.0
        self._h = 0.0
        self._pen = True
        self._pc = self._fc = 0x000000
        self._ps = 1
        self._shape = shape
        self._vis = visible
        self._sw = self._sl = 1.0
        self._fill = None
        _turtles.append(self)
        _screen._changed()

    # drawing state
    def _move(self, x, y):
        if self._pen:
            _items.append(("line", self, self._x, self._y, x, y, self._pc, self._ps))
        if self._fill is not None:
            self._fill += [x, y]
        self._x, self._y = float(x), float(y)
        _screen._changed()

    def forward(self, d):
        a = math.radians(self._h)
        self._move(self._x + d * math.cos(a), self._y + d * math.sin(a))

    fd = forward

    def backward(self, d):
        self.forward(-d)

    bk = back = backward

    def right(self, a):
        self._h = (self._h - a) % 360
        _screen._changed()

    rt = right

    def left(self, a):
        self.right(-a)

    lt = left

    def goto(self, x, y=None):
        if y is None:
            x, y = x
        self._move(x, y)

    setpos = setposition = goto

    def teleport(self, x=None, y=None, **kw):
        p = self._pen
        self._pen = False
        self._move(self._x if x is None else x, self._y if y is None else y)
        self._pen = p

    def setx(self, x):
        self._move(x, self._y)

    def sety(self, y):
        self._move(self._x, y)

    def setheading(self, a):
        self._h = a % 360
        _screen._changed()

    seth = setheading

    def home(self):
        self._move(0, 0)
        self._h = 0

    def circle(self, radius, extent=None, steps=None):
        extent = 360 if extent is None else extent
        n = steps or max(8, min(90, int(abs(radius) * abs(extent) / 360 / 2) + 8))
        step = extent / n
        length = 2 * radius * math.sin(math.radians(step) / 2)
        tr = _screen._tracer
        _screen._tracer = 0
        self.left(step / 2)
        for _ in range(n):
            self.forward(length)
            self.left(step)
        self.left(-step / 2)
        _screen._tracer = tr
        _screen._changed()

    def dot(self, size=None, *color):
        c = _col(color[0] if len(color) == 1 else color, self._pc) if color else self._pc
        _items.append(("dot", self, self._x, self._y, size or max(self._ps + 4, 2 * self._ps), c))
        _screen._changed()

    def stamp(self):
        pts = self._shape_points()
        flat = []
        for x, y in pts:
            flat += [x, y]
        _items.append(("poly", self, flat, self._fc, self._pc, 1))
        _screen._changed()
        return len(_items)

    def clearstamps(self, n=None):
        _items[:] = [it for it in _items if not (it[1] is self and it[0] == "poly")]
        _screen._changed()

    def clearstamp(self, sid):
        self.clearstamps()

    def write(self, arg, move=False, align="left", font=("Arial", 8, "normal")):
        size = font[1] if len(font) > 1 else 8
        bold = len(font) > 2 and "bold" in str(font[2])
        anchor = {"left": 6, "center": 7, "right": 8}.get(align, 6)
        _items.append(("text", self, self._x, self._y, str(arg), size * 1.33, bold, self._pc, anchor))
        _screen._changed()

    # queries
    def pos(self):
        return (self._x, self._y)

    position = pos

    def xcor(self):
        return self._x

    def ycor(self):
        return self._y

    def heading(self):
        return self._h

    def towards(self, x, y=None):
        if y is None:
            x, y = x.pos() if isinstance(x, Turtle) else x
        return math.degrees(math.atan2(y - self._y, x - self._x)) % 360

    def distance(self, x, y=None):
        if y is None:
            x, y = x.pos() if isinstance(x, Turtle) else x
        return math.sqrt((x - self._x) ** 2 + (y - self._y) ** 2)

    # pen
    def penup(self):
        self._pen = False

    pu = up = penup

    def pendown(self):
        self._pen = True

    pd = down = pendown

    def isdown(self):
        return self._pen

    def pensize(self, w=None):
        if w is None:
            return self._ps
        self._ps = w

    width = pensize

    def speed(self, s=None):
        return 0

    def color(self, *a):
        if not a:
            return (self._pc, self._fc)
        if len(a) == 2 or (len(a) == 1 and isinstance(a[0], (tuple, list)) and len(a[0]) == 2 and not isinstance(a[0][0], (int, float))):
            p, f = a if len(a) == 2 else a[0]
            self._pc, self._fc = _col(p, self._pc), _col(f, self._fc)
        else:
            c = _col(a[0] if len(a) == 1 else a, self._pc)
            self._pc = self._fc = c
        _screen._changed()

    def pencolor(self, *a):
        if not a:
            return self._pc
        self._pc = _col(a[0] if len(a) == 1 else a, self._pc)
        _screen._changed()

    def fillcolor(self, *a):
        if not a:
            return self._fc
        self._fc = _col(a[0] if len(a) == 1 else a, self._fc)
        _screen._changed()

    def begin_fill(self):
        self._fill = [self._x, self._y]
        self._fill_at = len(_items)

    def end_fill(self):
        if self._fill and len(self._fill) >= 6:   # under the lines drawn while filling
            _items.insert(min(self._fill_at, len(_items)), ("poly", self, self._fill, self._fc, None, 0))
        self._fill = None
        _screen._changed()

    def filling(self):
        return self._fill is not None

    def clear(self):
        _items[:] = [it for it in _items if it[1] is not self]
        _screen._changed()

    def reset(self):
        self.clear()
        self._x = self._y = self._h = 0.0
        self._pen = True
        self._pc = self._fc = 0

    # looks
    def hideturtle(self):
        self._vis = False
        _screen._changed()

    ht = hideturtle

    def showturtle(self):
        self._vis = True
        _screen._changed()

    st = showturtle

    def isvisible(self):
        return self._vis

    def shape(self, name=None):
        if name is None:
            return self._shape
        self._shape = name
        _screen._changed()

    def shapesize(self, stretch_wid=None, stretch_len=None, outline=None):
        if stretch_wid is None and stretch_len is None:
            return (self._sw, self._sl, 1)
        self._sw = stretch_wid if stretch_wid is not None else self._sw
        self._sl = stretch_len if stretch_len is not None else self._sw
        _screen._changed()

    turtlesize = shapesize

    def getscreen(self):
        return _screen

    def undo(self):
        pass

    def setundobuffer(self, n):
        pass

    def tilt(self, a):
        pass

    def onclick(self, fun, btn=1, add=None):
        _screen.onclick(lambda x, y: fun(x, y) if self.distance(x, y) < 20 * max(self._sw, self._sl) else None, btn)

    def ondrag(self, *a, **k):
        pass

    def onrelease(self, *a, **k):
        pass

    def _shape_points(self):
        if self._shape == "square":
            w, l = 10 * self._sl, 10 * self._sw
            pts = [(-w, -l), (w, -l), (w, l), (-w, l)]
        elif self._shape == "circle":
            pts = [(10 * self._sl * math.cos(i * math.pi / 12), 10 * self._sw * math.sin(i * math.pi / 12)) for i in range(24)]
        else:
            pts = [(x * self._sl, y * self._sw) for x, y in _SHAPES.get(self._shape, _SHAPES["classic"])]
        a = math.radians(self._h)
        ca, sa = math.cos(a), math.sin(a)
        if self._shape in ("square", "circle"):
            return [(self._x + x, self._y + y) for x, y in pts]
        return [(self._x + x * ca - y * sa, self._y + x * sa + y * ca) for x, y in pts]

    def _draw_shape(self, W2, H2):
        if self._shape == "circle" and self._sw == self._sl:
            r = 10 * self._sw
            _garlic.oval(self._x + W2 - r, H2 - self._y - r, 2 * r, 2 * r, self._fc, None)
            return
        if self._shape == "square":
            w, l = 10 * self._sl, 10 * self._sw
            _garlic.rect(self._x + W2 - w, H2 - self._y - l, 2 * w, 2 * l, self._fc, None)
            return
        flat = []
        for x, y in self._shape_points():
            flat += [x + W2, H2 - y]
        _garlic.poly(flat, self._fc, self._pc, 1)


Pen = RawTurtle = RawPen = Turtle
TurtleScreen = _Screen
_default = None


def _pen():
    global _default
    if _default is None:
        _default = Turtle()
    return _default


def _wrap_turtle(name):
    return lambda *a, **k: getattr(_pen(), name)(*a, **k)


def _wrap_screen(name):
    return lambda *a, **k: getattr(Screen(), name)(*a, **k)


for _n in ("forward fd backward bk back right rt left lt goto setpos setposition teleport setx sety setheading seth home "
           "circle dot stamp clearstamps write pos position xcor ycor heading towards distance penup pu up pendown pd down "
           "isdown pensize width speed color pencolor fillcolor begin_fill end_fill filling clear reset hideturtle ht "
           "showturtle st isvisible shape shapesize turtlesize undo").split():
    globals()[_n] = _wrap_turtle(_n)
for _n in ("setup screensize window_width window_height title bgcolor bgpic colormode tracer delay mode register_shape "
           "addshape listen onkeypress onkeyrelease onkey onclick onscreenclick ontimer update mainloop done exitonclick "
           "bye clearscreen textinput numinput turtles").split():
    globals()[_n] = _wrap_screen(_n)
del _n


class Terminator(Exception):
    pass


class Vec2D(tuple):
    pass
