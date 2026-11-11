# tkinter for Garlic LLM on the PS4: windows made of Frame / Label / Button / Entry / Canvas / Checkbutton...,
# laid out with pack / grid / place, drawn through _garlic and used with the controller: the D-pad moves a
# highlight between buttons (or sends arrow keys when the program listens to them), X presses, square is the right
# click, an Entry opens the app's keyboard. messagebox shows a dialog inside the window.
import _garlic
import _gx

END = "end"
INSERT = "insert"
ACTIVE = "active"
ANCHOR = "anchor"
ALL = "all"
LEFT, RIGHT, TOP, BOTTOM = "left", "right", "top", "bottom"
BOTH, X, Y, NONE = "both", "x", "y", "none"
N, S, E, W, NE, NW, SE, SW, NS, EW, NSEW, CENTER = "n", "s", "e", "w", "ne", "nw", "se", "sw", "ns", "ew", "nsew", "center"
NORMAL, DISABLED, HIDDEN, READONLY = "normal", "disabled", "hidden", "readonly"
HORIZONTAL, VERTICAL = "horizontal", "vertical"
RAISED, SUNKEN, FLAT, GROOVE, RIDGE, SOLID = "raised", "sunken", "flat", "groove", "ridge", "solid"
WORD, CHAR = "word", "char"
YES, NO, TRUE, FALSE = True, False, True, False
SINGLE, BROWSE, MULTIPLE, EXTENDED = "single", "browse", "multiple", "extended"
ARC, PIESLICE, CHORD = "arc", "pieslice", "chord"
TclError = RuntimeError

SYS_BG = 0xF0F0F0
_root = None
_widgets_dirty = True
_timers = []
_timer_id = [0]
_key_binds = {}      # keysym (lower) or "*" -> [handlers] (bind_all / root / canvas key bindings)
_running = [False]


def _dirty():
    global _widgets_dirty
    _widgets_dirty = True


def _px(font):
    """Font size in window pixels and bold, from a tuple ("Arial", 12, "bold"), "Arial 12 bold" or a Font."""
    if font is None:
        return 12, False
    if hasattr(font, "_size"):
        return font._px(), font._bold
    if isinstance(font, str):
        parts = font.replace("{", "").replace("}", "").split()
    else:
        parts = [str(p) for p in font]
    size, bold = 9, False
    for p in parts[1:]:
        try:
            size = int(float(p))
        except ValueError:
            bold = bold or "bold" in p
    return (-size if size < 0 else size * 1.333), bold


class Event:
    def __init__(self, **kw):
        self.x = self.y = self.x_root = self.y_root = 0
        self.keysym = self.char = ""
        self.keycode = self.num = self.delta = 0
        self.widget = None
        self.type = ""
        self.state = 0
        self.width = self.height = 0
        for k, v in kw.items():
            setattr(self, k, v)


def _norm_seq(seq):
    """'<KeyPress-Up>' / '<Up>' / '<Key-a>' / 'a' / '<Button-1>' -> ('key', 'up') / ('key', '*') / ('click', 1)."""
    s = seq.strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    parts = s.split("-")
    if parts[0] in ("Button", "ButtonPress", "ButtonRelease") or parts[0].isdigit():
        n = parts[-1] if parts[-1].isdigit() else "1"
        return ("click", int(n))
    if parts[0] in ("Key", "KeyPress", "KeyRelease"):
        k = parts[1] if len(parts) > 1 else "*"
    elif parts[0] in ("Motion", "B1-Motion", "Enter", "Leave", "Configure", "FocusIn", "FocusOut", "Destroy",
                      "MouseWheel", "Double", "Control", "Shift", "Alt", "Return-release"):
        return ("other", s)
    else:
        k = parts[-1]
    return ("key", k.lower())


class Variable:
    def __init__(self, master=None, value=None, name=None):
        self._v = self._default() if value is None else value
        self._traces = []

    def _default(self):
        return ""

    def get(self):
        return self._v

    def set(self, v):
        self._v = v
        _dirty()
        for f in self._traces:
            f("", "", "write")

    def trace_add(self, mode, cb):
        self._traces.append(cb)

    def trace(self, mode, cb):
        self._traces.append(cb)

    trace_variable = trace


class StringVar(Variable):
    def get(self):
        return str(self._v)


class IntVar(Variable):
    def _default(self):
        return 0

    def get(self):
        try:
            return int(self._v)
        except (TypeError, ValueError):
            return 0


class DoubleVar(Variable):
    def _default(self):
        return 0.0

    def get(self):
        return float(self._v)


class BooleanVar(Variable):
    def _default(self):
        return False

    def get(self):
        return bool(self._v)


class Misc:
    _defaults = {}
    _focusable = False

    def _init(self, master, kw):
        global _root
        if master is None and not isinstance(self, Tk):
            master = _root if _root else Tk()
        self.master = master
        self.children = {}
        self._kids = []
        self._opts = dict(self._defaults)
        self._opts.update(kw)
        self._mgr = None
        self._g = {}
        self._binds = {}
        self._x = self._y = 0
        self._w = self._h = 1
        self._rows, self._cols = {}, {}
        if master is not None:
            master._kids.append(self)
            master.children[str(len(master._kids))] = self
        _dirty()

    # options
    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        if not kw:
            return dict(self._opts)
        self._opts.update(kw)
        _dirty()

    config = configure

    def cget(self, key):
        v = self._opts.get(key, "")
        if key == "text" and self._opts.get("textvariable") is not None:
            return self._opts["textvariable"].get()
        return v

    def __setitem__(self, k, v):
        self.configure(**{k: v})

    def __getitem__(self, k):
        return self.cget(k)

    def keys(self):
        return list(self._opts)

    # geometry
    def pack(self, cnf=None, **kw):
        self._mgr, self._g = "pack", kw
        _dirty()
        return self

    def grid(self, cnf=None, **kw):
        self._mgr, self._g = "grid", kw
        _dirty()
        return self

    def place(self, cnf=None, **kw):
        self._mgr, self._g = "place", kw
        _dirty()
        return self

    pack_configure = pack
    grid_configure = grid
    place_configure = place

    def pack_forget(self):
        self._mgr = None
        _dirty()

    grid_forget = grid_remove = place_forget = pack_forget

    def grid_info(self):
        return dict(self._g)

    def grid_rowconfigure(self, i, weight=0, minsize=0, **kw):
        for r in (i if isinstance(i, (list, tuple)) else [i]):
            self._rows[r] = (weight, minsize)
        _dirty()

    def grid_columnconfigure(self, i, weight=0, minsize=0, **kw):
        for c in (i if isinstance(i, (list, tuple)) else [i]):
            self._cols[c] = (weight, minsize)
        _dirty()

    rowconfigure = grid_rowconfigure
    columnconfigure = grid_columnconfigure

    def grid_propagate(self, flag=None):
        self._opts["_noprop"] = flag is False

    pack_propagate = grid_propagate

    def grid_slaves(self, row=None, column=None):
        return [k for k in self._kids if k._mgr == "grid" and (row is None or k._g.get("row", 0) == row)
                and (column is None or k._g.get("column", 0) == column)]

    def pack_slaves(self):
        return [k for k in self._kids if k._mgr == "pack"]

    def winfo_children(self):
        return [k for k in self._kids if not getattr(k, "_destroyed", False)]

    def winfo_width(self):
        return int(self._w)

    def winfo_height(self):
        return int(self._h)

    def winfo_reqwidth(self):
        return int(self._req()[0])

    def winfo_reqheight(self):
        return int(self._req()[1])

    def winfo_x(self):
        return int(self._x)

    def winfo_y(self):
        return int(self._y)

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def winfo_exists(self):
        return not getattr(self, "_destroyed", False)

    def winfo_toplevel(self):
        return _root

    def destroy(self):
        self._destroyed = True
        if self.master is not None and self in self.master._kids:
            self.master._kids.remove(self)
        _dirty()

    # events and time
    def bind(self, seq, func=None, add=None):
        kind, what = _norm_seq(seq)
        if kind == "key":
            if self is _root or isinstance(self, Canvas) or isinstance(self, Toplevel):
                _key_binds.setdefault(what, []).append((func, self))
            else:
                self._binds.setdefault(("key", what), []).append(func)
        else:
            self._binds.setdefault((kind, what), []).append(func)
        _app_changed()

    def bind_all(self, seq, func=None, add=None):
        kind, what = _norm_seq(seq)
        if kind == "key":
            _key_binds.setdefault(what, []).append((func, self))
        else:
            self._binds.setdefault((kind, what), []).append(func)
        _app_changed()

    bind_class = bind_all

    def unbind(self, seq, funcid=None):
        kind, what = _norm_seq(seq)
        self._binds.pop((kind, what), None)
        _key_binds.pop(what, None)
        _app_changed()

    unbind_all = unbind

    def after(self, ms, func=None, *args):
        if func is None:
            _garlic.wait(ms / 1000)
            return None
        _timer_id[0] += 1
        _timers.append((_garlic.ticks() + ms / 1000.0, _timer_id[0], func, args))
        return "after#%d" % _timer_id[0]

    def after_idle(self, func, *args):
        return self.after(0, func, *args)

    def after_cancel(self, tid):
        if tid:
            n = int(str(tid).split("#")[-1])
            _timers[:] = [t for t in _timers if t[1] != n]

    def focus_set(self):
        _app.focus = self if self._focusable else _app.focus
        _dirty()

    focus = focus_force = focus_set

    def focus_get(self):
        return _app.focus

    def update(self):
        _app.step(frame=True)

    def update_idletasks(self):
        pass

    def mainloop(self, n=0):
        _app.loop()

    def quit(self):
        _running[0] = False

    def event_generate(self, seq, **kw):
        kind, what = _norm_seq(seq)
        for f in self._binds.get((kind, what), []):
            f(Event(widget=self, **kw))

    def option_add(self, *a, **k):
        pass

    def tk_setPalette(self, *a, **k):
        pass

    def lift(self, *a):
        pass

    lower = tkraise = lift

    def wait_window(self, w=None):
        pass

    def grab_set(self):
        pass

    def grab_release(self):
        pass

    def register(self, f, *a):
        return f

    # drawing helpers
    def _bg(self):
        bg = self._opts.get("bg", self._opts.get("background"))
        if bg is None:
            return self.master._bg() if self.master is not None else SYS_BG
        return _gx.color(bg, SYS_BG)

    def _font(self):
        return _px(self._opts.get("font"))

    def _text(self):
        tv = self._opts.get("textvariable")
        if tv is not None:
            return str(tv.get())
        return str(self._opts.get("text", ""))

    def _text_size(self, text, size, bold):
        lines = text.split("\n") if text else [""]
        w = 0
        h = 0
        for ln in lines:
            tw, th = _garlic.text_size(ln or " ", size, bold)
            w = max(w, tw if ln else 0)
            h += th * 0.92
        return w, h

    def _pad(self, key, default=0):
        v = self._opts.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _req(self):
        return self._container_req()

    def _container_req(self):
        kids = [k for k in self._kids if k._mgr in ("pack", "grid") and not getattr(k, "_destroyed", False)]
        w, h = _pack_req([k for k in kids if k._mgr == "pack"])
        gw, gh = _grid_req(self, [k for k in kids if k._mgr == "grid"])
        bw = self._pad("bd", self._pad("borderwidth", 0)) * 2 + self._pad("padx") * 2
        bh = self._pad("bd", self._pad("borderwidth", 0)) * 2 + self._pad("pady") * 2
        rw, rh = max(w, gw) + bw, max(h, gh) + bh
        if "width" in self._opts and (self._opts.get("_noprop") or not kids):
            rw = float(self._opts["width"])
        if "height" in self._opts and (self._opts.get("_noprop") or not kids):
            rh = float(self._opts["height"])
        return max(rw, 1), max(rh, 1)

    def _layout(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h
        kids = [k for k in self._kids if not getattr(k, "_destroyed", False)]
        b = self._pad("bd", self._pad("borderwidth", 0))
        ix, iy = x + b + self._pad("padx"), y + b + self._pad("pady")
        iw, ih = w - 2 * (b + self._pad("padx")), h - 2 * (b + self._pad("pady"))
        if isinstance(self, LabelFrame):
            iy += 14
            ih -= 14
        _pack_layout([k for k in kids if k._mgr == "pack"], ix, iy, iw, ih)
        _grid_layout(self, [k for k in kids if k._mgr == "grid"], ix, iy, iw, ih)
        for k in kids:
            if k._mgr == "place":
                g = k._g
                rw, rh = k._req()
                kw_, kh_ = float(g.get("width", rw)), float(g.get("height", rh))
                if "relwidth" in g:
                    kw_ = iw * float(g["relwidth"])
                if "relheight" in g:
                    kh_ = ih * float(g["relheight"])
                kx = ix + float(g.get("x", 0)) + iw * float(g.get("relx", 0))
                ky = iy + float(g.get("y", 0)) + ih * float(g.get("rely", 0))
                a = g.get("anchor", "nw")
                if a in ("e", "ne", "se"):
                    kx -= kw_
                elif a in ("n", "s", "center"):
                    kx -= kw_ / 2
                if a in ("s", "se", "sw"):
                    ky -= kh_
                elif a in ("e", "w", "center"):
                    ky -= kh_ / 2
                k._layout(kx, ky, kw_, kh_)

    def _draw(self):
        bg = self._opts.get("bg", self._opts.get("background"))
        if bg is not None or isinstance(self, (Tk, Toplevel)):
            _garlic.rect(self._x, self._y, self._w, self._h, self._bg(), None)
        if isinstance(self, LabelFrame):
            _garlic.rect(self._x + 2, self._y + 8, self._w - 4, self._h - 10, None, 0xA0A0A0, 1)
            t = self._text()
            if t:
                size, bold = self._font()
                tw, th = _garlic.text_size(t, size, bold)
                _garlic.rect(self._x + 8, self._y, tw + 6, th, self._bg(), None)
                _garlic.text(self._x + 11, self._y, t, size, bold, _gx.color(self._opts.get("fg"), 0), 0)
        self._draw_kids()

    def _draw_kids(self):
        for k in self._kids:
            if k._mgr and not getattr(k, "_destroyed", False):
                k._draw()


def _pack_req(kids):
    w = h = accw = acch = 0
    for k in kids:
        rw, rh = k._req()
        g = k._g
        rw += 2 * _padv(g.get("padx", 0)) + 2 * float(g.get("ipadx", 0))
        rh += 2 * _padv(g.get("pady", 0)) + 2 * float(g.get("ipady", 0))
        side = g.get("side", "top")
        if side in ("top", "bottom"):
            w = max(w, accw + rw)
            acch += rh
        else:
            h = max(h, acch + rh)
            accw += rw
    return max(w, accw), max(h, acch)


def _padv(p):
    if isinstance(p, (tuple, list)):
        return (float(p[0]) + float(p[-1])) / 2
    return float(p)


def _pack_layout(kids, x, y, w, h):
    if not kids:
        return
    reqs = []
    for k in kids:
        rw, rh = k._req()
        g = k._g
        reqs.append((rw + 2 * float(g.get("ipadx", 0)), rh + 2 * float(g.get("ipady", 0))))
    # extra space for children with expand=True, along their side
    tot_v = sum(reqs[i][1] + 2 * _padv(k._g.get("pady", 0)) for i, k in enumerate(kids) if k._g.get("side", "top") in ("top", "bottom"))
    tot_h = sum(reqs[i][0] + 2 * _padv(k._g.get("padx", 0)) for i, k in enumerate(kids) if k._g.get("side", "top") in ("left", "right"))
    exp_v = [k for k in kids if k._g.get("expand") and k._g.get("side", "top") in ("top", "bottom")]
    exp_h = [k for k in kids if k._g.get("expand") and k._g.get("side", "top") in ("left", "right")]
    extra_v = max(0, h - tot_v) / len(exp_v) if exp_v else 0
    extra_h = max(0, w - tot_h) / len(exp_h) if exp_h else 0
    cx, cy, cw, ch = x, y, w, h
    for k, (rw, rh) in zip(kids, reqs):
        g = k._g
        side = g.get("side", "top")
        px, py = _padv(g.get("padx", 0)), _padv(g.get("pady", 0))
        fill = g.get("fill", "none")
        anchor = g.get("anchor", "center")
        if side in ("top", "bottom"):
            ph = rh + 2 * py + (extra_v if k in exp_v else 0)
            py0 = cy if side == "top" else cy + ch - ph
            kw_ = cw - 2 * px if fill in ("x", "both") else min(rw, cw - 2 * px)
            kh_ = ph - 2 * py if fill in ("y", "both") else rh
            kx = cx + px + (0 if anchor in ("w", "nw", "sw") else (cw - 2 * px - kw_) if anchor in ("e", "ne", "se") else (cw - 2 * px - kw_) / 2)
            ky = py0 + py + (ph - 2 * py - kh_) / 2
            k._layout(kx, ky, kw_, kh_)
            if side == "top":
                cy += ph
            ch -= ph
        else:
            pw = rw + 2 * px + (extra_h if k in exp_h else 0)
            px0 = cx if side == "left" else cx + cw - pw
            kw_ = pw - 2 * px if fill in ("x", "both") else rw
            kh_ = ch - 2 * py if fill in ("y", "both") else min(rh, ch - 2 * py)
            kx = px0 + px + (pw - 2 * px - kw_) / 2
            ky = cy + py + (0 if anchor in ("n", "ne", "nw") else (ch - 2 * py - kh_) if anchor in ("s", "se", "sw") else (ch - 2 * py - kh_) / 2)
            k._layout(kx, ky, kw_, kh_)
            if side == "left":
                cx += pw
            cw -= pw


def _grid_sizes(master, kids):
    cols, rows = {}, {}
    span_items = []
    for k in kids:
        g = k._g
        r, c = int(g.get("row", 0)), int(g.get("column", 0))
        rs, cs = int(g.get("rowspan", 1)), int(g.get("columnspan", 1))
        rw, rh = k._req()
        rw += 2 * _padv(g.get("padx", 0)) + 2 * float(g.get("ipadx", 0))
        rh += 2 * _padv(g.get("pady", 0)) + 2 * float(g.get("ipady", 0))
        if cs == 1:
            cols[c] = max(cols.get(c, 0), rw)
        else:
            span_items.append(("c", c, cs, rw))
        if rs == 1:
            rows[r] = max(rows.get(r, 0), rh)
        else:
            span_items.append(("r", r, rs, rh))
    for kind, start, n, need in span_items:
        d = cols if kind == "c" else rows
        have = sum(d.get(i, 0) for i in range(start, start + n))
        if have < need:
            for i in range(start, start + n):
                d[i] = d.get(i, 0) + (need - have) / n
    for i, (wt, mn) in master._cols.items():
        cols[i] = max(cols.get(i, 0), mn)
    for i, (wt, mn) in master._rows.items():
        rows[i] = max(rows.get(i, 0), mn)
    return cols, rows


def _grid_req(master, kids):
    if not kids:
        return 0, 0
    cols, rows = _grid_sizes(master, kids)
    return sum(cols.values()), sum(rows.values())


def _grid_layout(master, kids, x, y, w, h):
    if not kids:
        return
    cols, rows = _grid_sizes(master, kids)
    nc, nr = max(cols) + 1, max(rows) + 1
    cw = [cols.get(i, 0) for i in range(nc)]
    rh = [rows.get(i, 0) for i in range(nr)]
    for sizes, conf, total in ((cw, master._cols, w), (rh, master._rows, h)):
        weights = [conf.get(i, (0, 0))[0] for i in range(len(sizes))]
        extra = total - sum(sizes)
        if sum(weights) > 0 and extra > 0:
            for i in range(len(sizes)):
                sizes[i] += extra * weights[i] / sum(weights)
    ox = x + max(0, (w - sum(cw)) / 2) if sum(c[0] for c in master._cols.values()) == 0 else x
    oy = y + max(0, (h - sum(rh)) / 2) if sum(r[0] for r in master._rows.values()) == 0 else y
    if isinstance(master, (Tk, Toplevel)) or True:
        ox, oy = x, y   # tk places the grid at the top-left of the master
    xs = [ox]
    for c in cw:
        xs.append(xs[-1] + c)
    ys = [oy]
    for r in rh:
        ys.append(ys[-1] + r)
    for k in kids:
        g = k._g
        r, c = int(g.get("row", 0)), int(g.get("column", 0))
        rs, cs = int(g.get("rowspan", 1)), int(g.get("columnspan", 1))
        px, py = _padv(g.get("padx", 0)), _padv(g.get("pady", 0))
        cx0, cx1 = xs[c] + px, xs[min(c + cs, nc)] - px
        cy0, cy1 = ys[r] + py, ys[min(r + rs, nr)] - py
        rw, rh_ = k._req()
        rw += 2 * float(g.get("ipadx", 0))
        rh_ += 2 * float(g.get("ipady", 0))
        st = str(g.get("sticky", "")).lower()
        kw_ = cx1 - cx0 if ("e" in st and "w" in st) else min(rw, cx1 - cx0)
        kh_ = cy1 - cy0 if ("n" in st and "s" in st) else min(rh_, cy1 - cy0)
        kx = cx0 if "w" in st else cx1 - kw_ if "e" in st else (cx0 + cx1 - kw_) / 2
        ky = cy0 if "n" in st else cy1 - kh_ if "s" in st else (cy0 + cy1 - kh_) / 2
        k._layout(kx, ky, kw_, kh_)


class Tk(Misc):
    _defaults = {}

    def __init__(self, screenName=None, baseName=None, className="Tk", useTk=1, **kw):
        global _root
        self._init(None, kw)
        if _root is None:
            _root = self
            _app.root = self
        self._title = "tk"
        self._geom = None
        _garlic.set_title(self._title)

    def title(self, t=None):
        if t is None:
            return self._title
        self._title = str(t)
        _garlic.set_title(self._title)

    wm_title = title

    def geometry(self, g=None):
        if g is None:
            return "%dx%d" % (self._w, self._h)
        size = str(g).split("+")[0]
        if "x" in size:
            a, b = size.split("x")
            self._geom = (int(a), int(b))
            _dirty()

    wm_geometry = geometry

    def resizable(self, *a, **k):
        pass

    def protocol(self, *a, **k):
        pass

    def iconbitmap(self, *a, **k):
        pass

    def iconphoto(self, *a, **k):
        pass

    def attributes(self, *a, **k):
        pass

    wm_attributes = attributes

    def minsize(self, *a):
        pass

    def maxsize(self, *a):
        pass

    def state(self, *a):
        return "normal"

    def overrideredirect(self, *a):
        pass

    def destroy(self):
        _running[0] = False
        raise SystemExit

    def _req(self):
        w, h = self._container_req()
        if self._geom:   # never smaller than what the widgets need: on the TV nothing gets cut off
            return max(w, self._geom[0]), max(h, self._geom[1])
        return w, h


class Toplevel(Tk):
    def __init__(self, master=None, **kw):
        Misc._init(self, master or _root, kw)
        self._title = "tk"
        self._geom = None
        self._mgr = "place"
        self._g = {"relx": 0.5, "rely": 0.5, "anchor": "center"}

    def destroy(self):
        Misc.destroy(self)

    def _draw(self):
        _garlic.rect(self._x - 2, self._y - 2, self._w + 4, self._h + 4, None, 0x404040, 2)
        Misc._draw(self)


class Frame(Misc):
    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)

    def _draw(self):
        Misc._draw(self)
        relief = self._opts.get("relief")
        b = self._pad("bd", self._pad("borderwidth", 0))
        if b and relief not in (None, "flat"):
            _garlic.rect(self._x, self._y, self._w, self._h, None, 0x909090, b)


class LabelFrame(Frame):
    pass


class Label(Misc):
    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)

    def _req(self):
        size, bold = self._font()
        tw, th = self._text_size(self._text(), size, bold)
        cw = _garlic.text_size("0", size, bold)[0]
        if "width" in self._opts and self._opts["width"]:
            tw = float(self._opts["width"]) * cw
        if "height" in self._opts and self._opts["height"]:
            th = float(self._opts["height"]) * th / max(1, len(self._text().split("\n")))
        px = self._pad("padx", 1) + self._pad("bd", self._pad("borderwidth", 0))
        py = self._pad("pady", 1) + self._pad("bd", self._pad("borderwidth", 0))
        return tw + 2 * px + 2, th + 2 * py

    def _fg(self, default=0x000000):
        return _gx.color(self._opts.get("fg", self._opts.get("foreground")), default)

    def _draw_text(self, fg):
        size, bold = self._font()
        text = self._text()
        lines = text.split("\n")
        lh = _garlic.text_size("Ag", size, bold)[1] * 0.92
        total = lh * len(lines)
        anchor = self._opts.get("anchor", "center")
        justify = self._opts.get("justify", "center")
        px = self._pad("padx", 1) + 2
        y = self._y + (self._h - total) / 2
        if anchor in ("n", "ne", "nw"):
            y = self._y + self._pad("pady", 1)
        elif anchor in ("s", "se", "sw"):
            y = self._y + self._h - total - self._pad("pady", 1)
        for ln in lines:
            if anchor in ("w", "nw", "sw") or justify == "left" and anchor == "center" and len(lines) > 1:
                _garlic.text(self._x + px, y, ln, size, bold, fg, 0)
            elif anchor in ("e", "ne", "se"):
                _garlic.text(self._x + self._w - px, y, ln, size, bold, fg, 2)
            else:
                _garlic.text(self._x + self._w / 2, y, ln, size, bold, fg, 1)
            y += lh

    def _draw(self):
        bg = self._opts.get("bg", self._opts.get("background"))
        if bg is not None:
            _garlic.rect(self._x, self._y, self._w, self._h, self._bg(), None)
        relief = self._opts.get("relief")
        b = self._pad("bd", self._pad("borderwidth", 0))
        if b and relief not in (None, "flat"):
            _garlic.rect(self._x, self._y, self._w, self._h, None, 0x909090, b)
        self._draw_text(self._fg())


Message = Label


class Button(Label):
    _focusable = True

    def _req(self):
        size, bold = self._font()
        text = self._text()
        tw, th = self._text_size(text, size, bold)
        cw = _garlic.text_size("0", size, bold)[0]
        lh = _garlic.text_size("Ag", size, bold)[1] * 0.92
        if self._opts.get("width"):
            tw = float(self._opts["width"]) * cw
        if self._opts.get("height"):
            th = float(self._opts["height"]) * lh
        px = self._pad("padx", 1) + 3
        py = self._pad("pady", 1) + 3
        return tw + 2 * px + 2, th + 2 * py

    def invoke(self):
        if self._opts.get("state", "normal") != "disabled":
            f = self._opts.get("command")
            if f:
                return f()

    def flash(self):
        pass

    def _draw(self):
        disabled = self._opts.get("state", "normal") == "disabled"
        bg = _gx.color(self._opts.get("bg", self._opts.get("background")), 0xE1E1E1)
        relief = self._opts.get("relief", "raised")
        _garlic.rect(self._x, self._y, self._w, self._h, bg, None)
        if relief not in ("flat",):
            _garlic.rect(self._x, self._y, self._w, self._h, None, 0x7A7A7A if relief == "sunken" else 0xADADAD, 1)
        fg = self._fg(0)
        if disabled:
            fg = _gx.color(self._opts.get("disabledforeground"), 0x8C8C8C) if "fg" not in self._opts and "foreground" not in self._opts else fg
        self._draw_text(fg)


class Checkbutton(Button):
    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        if self._opts.get("variable") is None:
            self._opts["variable"] = IntVar()

    def _on(self):
        v = self._opts["variable"].get()
        return v == self._opts.get("onvalue", 1) or v is True

    def invoke(self):
        var = self._opts["variable"]
        var.set(self._opts.get("offvalue", 0) if self._on() else self._opts.get("onvalue", 1))
        f = self._opts.get("command")
        if f:
            f()

    def select(self):
        self._opts["variable"].set(self._opts.get("onvalue", 1))

    def deselect(self):
        self._opts["variable"].set(self._opts.get("offvalue", 0))

    def toggle(self):
        self.invoke()

    def _req(self):
        w, h = Label._req(self)
        return w + 24, max(h, 22)

    def _draw(self):
        bg = self._opts.get("bg", self._opts.get("background"))
        if bg is not None:
            _garlic.rect(self._x, self._y, self._w, self._h, self._bg(), None)
        bx, by = self._x + 4, self._y + self._h / 2 - 7
        _garlic.rect(bx, by, 14, 14, 0xFFFFFF, 0x333333, 1)
        if self._on():
            _garlic.line(bx + 3, by + 7, bx + 6, by + 11, 0x0070D1, 2)
            _garlic.line(bx + 6, by + 11, bx + 12, by + 3, 0x0070D1, 2)
        size, bold = self._font()
        _garlic.text(bx + 20, self._y + self._h / 2, self._text(), size, bold, self._fg(0), 3)


class Radiobutton(Checkbutton):
    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        if self._opts.get("variable") is None:
            self._opts["variable"] = StringVar()

    def _on(self):
        return self._opts["variable"].get() == self._opts.get("value")

    def invoke(self):
        self._opts["variable"].set(self._opts.get("value"))
        f = self._opts.get("command")
        if f:
            f()

    def _draw(self):
        bg = self._opts.get("bg", self._opts.get("background"))
        if bg is not None:
            _garlic.rect(self._x, self._y, self._w, self._h, self._bg(), None)
        bx, by = self._x + 4, self._y + self._h / 2 - 7
        _garlic.oval(bx, by, 14, 14, 0xFFFFFF, 0x333333, 1)
        if self._on():
            _garlic.oval(bx + 4, by + 4, 6, 6, 0x0070D1, None)
        size, bold = self._font()
        _garlic.text(bx + 20, self._y + self._h / 2, self._text(), size, bold, self._fg(0), 3)


class Entry(Misc):
    _focusable = True
    _defaults = {"width": 20}

    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        self._val = ""

    def _get(self):
        tv = self._opts.get("textvariable")
        return str(tv.get()) if tv is not None else self._val

    def _set(self, v):
        tv = self._opts.get("textvariable")
        if tv is not None:
            tv.set(v)
        else:
            self._val = v
        _dirty()

    def get(self):
        return self._get()

    def _index(self, i):
        if i in ("end", "insert"):
            return len(self._get())
        return int(i)

    def insert(self, i, s):
        v = self._get()
        i = self._index(i)
        self._set(v[:i] + str(s) + v[i:])

    def delete(self, a, b=None):
        v = self._get()
        a = self._index(a)
        b = a + 1 if b is None else self._index(b)
        self._set(v[:a] + v[b:])

    def icursor(self, *a):
        pass

    def select_range(self, *a):
        pass

    def selection_range(self, *a):
        pass

    def _req(self):
        size, bold = self._font()
        cw = _garlic.text_size("0", size, bold)[0]
        lh = _garlic.text_size("Ag", size, bold)[1]
        b = self._pad("bd", self._pad("borderwidth", 2))
        return float(self._opts.get("width", 20)) * cw + 2 * b + 4, lh + 2 * b + 2

    def _edit(self):
        if self._opts.get("state", "normal") in ("disabled", "readonly"):
            return
        s = _garlic.input()
        self._set(s)
        for f in self._binds.get(("key", "return"), []):
            f(Event(widget=self, keysym="Return"))

    def _draw(self):
        b = self._pad("bd", self._pad("borderwidth", 2))
        _garlic.rect(self._x, self._y, self._w, self._h, _gx.color(self._opts.get("bg"), 0xFFFFFF), 0x7A7A7A, 1)
        size, bold = self._font()
        v = self._get()
        if self._opts.get("show"):
            v = self._opts["show"] * len(v)
        fg = _gx.color(self._opts.get("fg"), 0)
        j = self._opts.get("justify", "left")
        if j == "right":
            _garlic.text(self._x + self._w - b - 3, self._y + self._h / 2, v, size, bold, fg, 5)
        elif j == "center":
            _garlic.text(self._x + self._w / 2, self._y + self._h / 2, v, size, bold, fg, 4)
        else:
            _garlic.text(self._x + b + 3, self._y + self._h / 2, v, size, bold, fg, 3)


class Text(Entry):
    _defaults = {"width": 80, "height": 24}

    def get(self, a="1.0", b="end"):
        return self._get() + "\n"

    def insert(self, i, s, *tags):
        self._set(self._get() + str(s))

    def delete(self, a, b=None):
        self._set("")

    def see(self, *a):
        pass

    def tag_configure(self, *a, **k):
        pass

    tag_config = tag_add = tag_remove = mark_set = tag_configure

    def _req(self):
        size, bold = self._font()
        cw = _garlic.text_size("0", size, bold)[0]
        lh = _garlic.text_size("Ag", size, bold)[1]
        return float(self._opts.get("width", 80)) * cw + 8, float(self._opts.get("height", 24)) * lh + 8

    def _draw(self):
        _garlic.rect(self._x, self._y, self._w, self._h, _gx.color(self._opts.get("bg"), 0xFFFFFF), 0x7A7A7A, 1)
        size, bold = self._font()
        lh = _garlic.text_size("Ag", size, bold)[1]
        lines = self._get().split("\n")
        rows = max(1, int((self._h - 8) / lh))
        y = self._y + 4
        for ln in lines[-rows:]:
            _garlic.text(self._x + 4, y, ln, size, bold, _gx.color(self._opts.get("fg"), 0), 0)
            y += lh


class Listbox(Misc):
    _focusable = True
    _defaults = {"width": 20, "height": 10}

    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        self._items = []
        self._sel = 0

    def insert(self, i, *items):
        for it in items:
            if i in ("end",):
                self._items.append(str(it))
            else:
                self._items.insert(int(i), str(it))
        _dirty()

    def delete(self, a, b=None):
        if a == 0 and b in ("end", None) and b is not None:
            self._items = []
        else:
            del self._items[int(a)]
        _dirty()

    def get(self, a, b=None):
        if b is None:
            return self._items[int(a)]
        return tuple(self._items)

    def size(self):
        return len(self._items)

    def curselection(self):
        return (self._sel,) if self._items else ()

    def selection_set(self, i, *a):
        self._sel = int(i)
        _dirty()

    select_set = selection_set

    def selection_clear(self, *a):
        pass

    def activate(self, i):
        self._sel = int(i)

    def see(self, i):
        pass

    def _req(self):
        size, bold = self._font()
        cw = _garlic.text_size("0", size, bold)[0]
        lh = _garlic.text_size("Ag", size, bold)[1]
        return float(self._opts.get("width", 20)) * cw + 8, float(self._opts.get("height", 10)) * lh + 8

    def _draw(self):
        _garlic.rect(self._x, self._y, self._w, self._h, 0xFFFFFF, 0x7A7A7A, 1)
        size, bold = self._font()
        lh = _garlic.text_size("Ag", size, bold)[1]
        y = self._y + 4
        for i, it in enumerate(self._items):
            if y + lh > self._y + self._h:
                break
            if i == self._sel:
                _garlic.rect(self._x + 2, y, self._w - 4, lh, 0x0078D7, None)
            _garlic.text(self._x + 6, y, it, size, bold, 0xFFFFFF if i == self._sel else 0, 0)
            y += lh


class Scale(Label):
    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        self._v = kw.get("from_", 0)

    def get(self):
        return self._v

    def set(self, v):
        self._v = v
        _dirty()


class Scrollbar(Frame):
    def set(self, *a):
        pass

    def _req(self):
        return 1, 1


class Spinbox(Entry):
    pass


class Canvas(Misc):
    _defaults = {"width": 378, "height": 265}

    def __init__(self, master=None, cnf=None, **kw):
        self._init(master, kw)
        self._items = {}
        self._order = []
        self._next = 1

    def _req(self):
        b = self._pad("bd", self._pad("borderwidth", 0)) + self._pad("highlightthickness", 0)
        return float(self._opts.get("width", 378)) + 2 * b, float(self._opts.get("height", 265)) + 2 * b

    def _add(self, kind, coords, kw):
        if len(coords) == 1 and isinstance(coords[0], (list, tuple)):
            coords = coords[0]
        flat = []
        for c in coords:
            if isinstance(c, (list, tuple)):
                flat += [float(v) for v in c]
            else:
                flat.append(float(c))
        i = self._next
        self._next += 1
        tags = kw.get("tags", kw.get("tag", ()))
        if isinstance(tags, str):
            tags = (tags,)
        self._items[i] = [kind, flat, dict(kw), tuple(tags)]
        self._order.append(i)
        _dirty()
        return i

    def create_rectangle(self, *c, **kw):
        return self._add("rect", c, kw)

    def create_oval(self, *c, **kw):
        return self._add("oval", c, kw)

    def create_line(self, *c, **kw):
        return self._add("line", c, kw)

    def create_polygon(self, *c, **kw):
        return self._add("poly", c, kw)

    def create_text(self, *c, **kw):
        return self._add("text", c, kw)

    def create_arc(self, *c, **kw):
        return self._add("arc", c, kw)

    def create_image(self, *c, **kw):
        return self._add("none", c, kw)

    def create_window(self, *c, **kw):
        return self._add("none", c, kw)

    def _ids(self, tag):
        if tag in ("all", ALL):
            return list(self._order)
        if isinstance(tag, int):
            return [tag] if tag in self._items else []
        if isinstance(tag, str) and tag.isdigit():
            return [int(tag)] if int(tag) in self._items else []
        return [i for i in self._order if tag in self._items[i][3]]

    def delete(self, *tags):
        for t in tags:
            for i in self._ids(t):
                self._items.pop(i, None)
                if i in self._order:
                    self._order.remove(i)
        _dirty()

    def itemconfig(self, tag, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        for i in self._ids(tag):
            self._items[i][2].update(kw)
            if "tags" in kw:
                t = kw["tags"]
                self._items[i][3] = (t,) if isinstance(t, str) else tuple(t)
        _dirty()

    itemconfigure = itemconfig

    def itemcget(self, tag, key):
        ids = self._ids(tag)
        return self._items[ids[0]][2].get(key, "") if ids else ""

    def coords(self, tag, *c):
        ids = self._ids(tag)
        if not ids:
            return []
        if not c:
            return list(self._items[ids[0]][1])
        flat = []
        for v in (c[0] if len(c) == 1 and isinstance(c[0], (list, tuple)) else c):
            flat += [float(x) for x in v] if isinstance(v, (list, tuple)) else [float(v)]
        self._items[ids[0]][1] = flat
        _dirty()

    def move(self, tag, dx, dy):
        for i in self._ids(tag):
            cs = self._items[i][1]
            for k in range(0, len(cs) - 1, 2):
                cs[k] += dx
                cs[k + 1] += dy
        _dirty()

    def moveto(self, tag, x=0, y=0):
        for i in self._ids(tag):
            cs = self._items[i][1]
            dx, dy = float(x) - cs[0], float(y) - cs[1]
            for k in range(0, len(cs) - 1, 2):
                cs[k] += dx
                cs[k + 1] += dy
        _dirty()

    def bbox(self, tag="all"):
        xs, ys = [], []
        for i in self._ids(tag):
            cs = self._items[i][1]
            xs += cs[0::2]
            ys += cs[1::2]
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def find_all(self):
        return tuple(self._order)

    def find_withtag(self, tag):
        return tuple(self._ids(tag))

    def find_overlapping(self, x1, y1, x2, y2):
        out = []
        for i in self._order:
            b = self.bbox(i)
            if b and b[0] <= x2 and b[2] >= x1 and b[1] <= y2 and b[3] >= y1:
                out.append(i)
        return tuple(out)

    def find_closest(self, x, y):
        best, bd = None, 1e18
        for i in self._order:
            b = self.bbox(i)
            if b:
                d = ((b[0] + b[2]) / 2 - x) ** 2 + ((b[1] + b[3]) / 2 - y) ** 2
                if d < bd:
                    best, bd = i, d
        return (best,) if best else ()

    def gettags(self, i):
        return self._items[i][3] if i in self._items else ()

    def type(self, i):
        return self._items[i][0] if i in self._items else None

    def tag_bind(self, tag, seq, func=None, add=None):
        kind, what = _norm_seq(seq)
        self._binds.setdefault(("tag", kind, what), []).append((tag, func))
        _app_changed()

    def tag_raise(self, tag, *a):
        for i in self._ids(tag):
            self._order.remove(i)
            self._order.append(i)
        _dirty()

    lift = tag_raise

    def tag_lower(self, tag, *a):
        for i in self._ids(tag):
            self._order.remove(i)
            self._order.insert(0, i)
        _dirty()

    def addtag_withtag(self, new, tag):
        for i in self._ids(tag):
            self._items[i][3] = self._items[i][3] + (new,)

    def dtag(self, *a):
        pass

    def canvasx(self, x):
        return x

    def canvasy(self, y):
        return y

    def config(self, cnf=None, **kw):
        return Misc.configure(self, cnf, **kw)

    configure = config

    def _draw(self):
        b = self._pad("bd", self._pad("borderwidth", 0)) + self._pad("highlightthickness", 0)
        ox, oy = self._x + b, self._y + b
        _garlic.rect(self._x, self._y, self._w, self._h, _gx.color(self._opts.get("bg", self._opts.get("background")), SYS_BG), None)
        for i in self._order:
            kind, cs, o, tags = self._items[i]
            if o.get("state") == "hidden":
                continue
            fill = o.get("fill")
            outline = o.get("outline", "black" if kind in ("rect", "oval", "arc") else None)
            width = float(o.get("width", 1))
            if kind in ("rect", "oval") and len(cs) >= 4:
                x0, y0, x1, y1 = cs[0], cs[1], cs[2], cs[3]
                fc = _gx.color(fill) if fill not in (None, "") else None
                oc = _gx.color(outline) if outline not in (None, "") and width > 0 else None
                fn = _garlic.rect if kind == "rect" else _garlic.oval
                fn(ox + min(x0, x1), oy + min(y0, y1), abs(x1 - x0), abs(y1 - y0), fc, oc, width)
            elif kind == "line" and len(cs) >= 4:
                col = _gx.color(fill if fill not in (None, "") else "black")
                for k in range(0, len(cs) - 3, 2):
                    _garlic.line(ox + cs[k], oy + cs[k + 1], ox + cs[k + 2], oy + cs[k + 3], col, width)
            elif kind == "poly" and len(cs) >= 6:
                pts = [ox + v if k % 2 == 0 else oy + v for k, v in enumerate(cs)]
                fc = _gx.color(fill if fill is not None else "black") if fill != "" else None
                oc = _gx.color(outline) if outline not in (None, "") else None
                _garlic.poly(pts, fc, oc, width)
            elif kind == "arc" and len(cs) >= 4:
                import math
                x0, y0, x1, y1 = cs[0], cs[1], cs[2], cs[3]
                st, ext = float(o.get("start", 0)), float(o.get("extent", 90))
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
                pts = [ox + cx, oy + cy] if o.get("style", "pieslice") == "pieslice" else []
                for k in range(25):
                    a = math.radians(st + ext * k / 24)
                    pts += [ox + cx + rx * math.cos(a), oy + cy - ry * math.sin(a)]
                fc = _gx.color(fill) if fill not in (None, "") and o.get("style") != "arc" else None
                _garlic.poly(pts, fc, _gx.color(outline) if outline else None, width)
            elif kind == "text" and len(cs) >= 2:
                size, bold = _px(o.get("font"))
                anchor = {"nw": 0, "n": 1, "ne": 2, "w": 3, "center": 4, "e": 5, "sw": 6, "s": 7, "se": 8}.get(o.get("anchor", "center"), 4)
                col = _gx.color(fill if fill not in (None, "") else "black")
                text = str(o.get("text", ""))
                lines = text.split("\n")
                lh = _garlic.text_size("Ag", size, bold)[1] * 0.92
                y = oy + cs[1] - (lh * (len(lines) - 1) / 2 if anchor in (3, 4, 5) else 0)
                for ln in lines:
                    _garlic.text(ox + cs[0], y, ln, size, bold, col, anchor)
                    y += lh


class _Font:
    def __init__(self, root=None, font=None, family="Arial", size=9, weight="normal", slant="roman", **kw):
        if font is not None:
            s, b = _px(font)
            self._size, self._bold = s / 1.333, b
        else:
            self._size, self._bold = size, weight == "bold"

    def _px(self):
        return -self._size if self._size < 0 else self._size * 1.333

    def configure(self, **kw):
        if "size" in kw:
            self._size = kw["size"]
        if "weight" in kw:
            self._bold = kw["weight"] == "bold"
        _dirty()

    config = configure

    def measure(self, text):
        return int(_garlic.text_size(text, self._px(), self._bold)[0])

    def metrics(self, *a):
        return int(self._px() * 1.2)

    def actual(self, *a):
        return {"size": self._size}


class _App:
    """The window's loop: layout, drawing, the focus highlight, controller input, timers and dialogs."""

    def __init__(self):
        self.root = None
        self.focus = None
        self.keymap = None
        self.size = None
        self.mouse = False

    def focusables(self):
        out = []

        def walk(w):
            for k in w._kids:
                if k._mgr and not getattr(k, "_destroyed", False):
                    if k._focusable:
                        out.append(k)
                    walk(k)
        if self.root:
            walk(self.root)
        return out

    def keys_wanted(self):
        used = []
        for k in _key_binds:
            k = {"enter": "return", " ": "space"}.get(k, k)
            if k != "*" and k not in used:
                used.append(k)
        return used

    def relayout(self):
        global _widgets_dirty
        r = self.root
        w, h = r._req()
        w, h = max(int(w), 120), max(int(h), 60)
        if self.size != (w, h):
            self.size = (w, h)
            _garlic.begin(w, h, r._bg())
        r._layout(0, 0, w, h)
        if self.keymap is None:
            used = self.keys_wanted()
            clicks = any(("click", 1) in wd._binds or any(k[0] == "tag" for k in wd._binds) for wd in self.all_widgets())
            self.mouse = clicks and not self.focusables()
            _garlic.mouse_mode(self.mouse)
            if used or "*" in _key_binds:
                self.keymap = _gx.keymap(used if used else ["space"])
            else:
                self.keymap = {}
                hints = [("up", _gx.key_label("arrows")), ("cross", "OK")]
                if any(("click", 3) in wd._binds for wd in self.all_widgets()):
                    hints.append(("square", "🚩"))
                _garlic.hints(hints if self.focusables() else [])

    def all_widgets(self):
        out = []

        def walk(w):
            out.append(w)
            for k in w._kids:
                walk(k)
        if self.root:
            walk(self.root)
        return out

    def draw(self):
        global _widgets_dirty
        self.relayout()
        _garlic.clear(self.root._bg())
        self.root._draw()
        f = self.focus
        if f is not None and f in self.focusables() and not self.keymap:
            _garlic.rect(f._x - 2, f._y - 2, f._w + 4, f._h + 4, None, 0x0070D1, 3)
        _widgets_dirty = False

    def move_focus(self, d):
        fs = self.focusables()
        if not fs:
            return
        f = self.focus if self.focus in fs else None
        if f is None:
            self.focus = fs[0]
            return
        cx, cy = f._x + f._w / 2, f._y + f._h / 2
        best, bd = None, 1e18
        for k in fs:
            if k is f:
                continue
            kx, ky = k._x + k._w / 2, k._y + k._h / 2
            dx, dy = kx - cx, ky - cy
            if d == "right" and dx <= 1 or d == "left" and dx >= -1 or d == "down" and dy <= 1 or d == "up" and dy >= -1:
                continue
            main, side = (abs(dx), abs(dy)) if d in ("left", "right") else (abs(dy), abs(dx))
            score = main + 3 * side
            if score < bd:
                best, bd = k, score
        if best is not None:
            self.focus = best

    def send_key(self, key, down=True):
        ev = Event(keysym=key.capitalize() if len(key) > 1 and key not in ("space",) else key, char=key if len(key) == 1 else (" " if key == "space" else ""),
                   type="KeyPress" if down else "KeyRelease")
        if key in ("up", "down", "left", "right", "return", "escape"):
            ev.keysym = {"up": "Up", "down": "Down", "left": "Left", "right": "Right", "return": "Return", "escape": "Escape"}[key]
        if not down:
            return
        for f, w in _key_binds.get(key, []) + _key_binds.get("*", []):
            ev.widget = w
            f(ev)

    def click_at(self, x, y, num):
        hit = None
        for w in self.all_widgets():
            if w._x <= x < w._x + w._w and w._y <= y < w._y + w._h:
                hit = w
        w = hit
        while w is not None:
            if isinstance(w, Canvas):
                cx, cy = x - w._x, y - w._y
                for key, lst in list(w._binds.items()):
                    if key[0] == "tag" and key[2] == num:
                        for tag, f in lst:
                            ids = w._ids(tag)
                            for i in w.find_overlapping(cx, cy, cx, cy):
                                if i in ids:
                                    f(Event(x=cx, y=cy, num=num, widget=w))
                                    break
            fs = w._binds.get(("click", num))
            if fs:
                for f in fs:
                    f(Event(x=x - w._x, y=y - w._y, num=num, widget=w))
                return
            if num == 1 and isinstance(w, Button):
                w.invoke()
                return
            w = w.master

    def handle(self):
        for kind, button, x, y in _garlic.events():
            if self.mouse and button in ("cross", "square"):
                if kind == "down":
                    self.click_at(x, y, 1 if button == "cross" else 3)
                continue
            if self.keymap:
                key = self.keymap.get(button)
                if key and kind == "down":
                    self.send_key(key)
                if kind == "down" and button in ("l1", "r1") and self.focusables():
                    self.move_focus("left" if button == "l1" else "right")
                if kind == "down" and button == "cross" and not key and self.focus:
                    self.activate(self.focus, 1)
                continue
            if kind != "down":
                continue
            if button in ("up", "down", "left", "right"):
                self.move_focus(button)
                _dirty()
            elif button == "cross":
                if self.focus is None:
                    self.move_focus("down")
                    _dirty()
                else:
                    self.activate(self.focus, 1)
            elif button == "square" and self.focus is not None:
                self.activate(self.focus, 3)
            elif button == "triangle" and isinstance(self.focus, Listbox):
                pass

    def activate(self, w, num):
        if num == 3:
            for f in w._binds.get(("click", 3), []) + w._binds.get(("click", 2), []):
                f(Event(x=w._w / 2, y=w._h / 2, num=3, widget=w))
            return
        if isinstance(w, Entry):
            w._edit()
            return
        if isinstance(w, Listbox):
            w._sel = (w._sel + 1) % max(1, len(w._items))
            for f in w._binds.get(("other", "<ListboxSelect>"), []):
                f(Event(widget=w))
            _dirty()
            return
        fs = w._binds.get(("click", 1))
        if fs:
            for f in fs:
                f(Event(x=w._w / 2, y=w._h / 2, num=1, widget=w))
        if hasattr(w, "invoke") and not (fs and not w._opts.get("command")):
            w.invoke()
        _dirty()

    def timers(self):
        now = _garlic.ticks()
        due = [t for t in _timers if t[0] <= now]
        if due:
            _timers[:] = [t for t in _timers if t[0] > now]
            due.sort(key=lambda t: t[0])
            for t in due:
                t[2](*t[3])

    def step(self, frame=True):
        if self.root is None:
            return
        if self.focus is None and not self.keymap:
            fs = self.focusables()
            if fs:
                self.focus = fs[0]
        self.handle()
        self.timers()
        if _widgets_dirty:
            self.draw()
        if frame:
            _garlic.frame()

    def loop(self):
        _running[0] = True
        if self.root is not None:
            self.draw()
        while _running[0]:
            self.step()


_app = _App()


def _app_changed():
    _app.keymap = None
    _dirty()


def mainloop(n=0):
    _app.loop()


def NoDefaultRoot():
    pass


class PhotoImage:
    def __init__(self, *a, **k):
        self._w = k.get("width", 1)
        self._h = k.get("height", 1)

    def width(self):
        return self._w

    def height(self):
        return self._h

    def subsample(self, *a):
        return self

    zoom = subsample


def _dialog(title, message, buttons):
    """A modal dialog drawn inside the window; returns the index of the chosen button."""
    if _app.root is None:
        Tk()
    _app.draw()
    w, h = _app.size
    sel = 0
    while True:
        _app.draw()
        bw = min(w - 40, 420)
        bh = 150
        size = 14
        tw, th = _garlic.text_size(str(message), size, False)
        lines = str(message).split("\n")
        bh = 100 + th * 1.0 * len(lines)
        x0, y0 = (w - bw) / 2, (h - bh) / 2
        _garlic.rect(0, 0, w, h, 0x7F000000 | 0x000000, None)   # dim (alpha ~50%)
        _garlic.rect(x0, y0, bw, bh, 0xFFFFFF, 0x0070D1, 2)
        _garlic.rect(x0, y0, bw, 30, 0x0070D1, None)
        _garlic.text(x0 + 10, y0 + 15, str(title), 13, True, 0xFFFFFF, 3)
        y = y0 + 44
        for ln in lines:
            _garlic.text(x0 + bw / 2, y, ln, size, False, 0x111111, 1)
            y += th
        n = len(buttons)
        for i, b in enumerate(buttons):
            bx = x0 + bw / 2 + (i - (n - 1) / 2) * 110 - 48
            by = y0 + bh - 42
            _garlic.rect(bx, by, 96, 30, 0x0070D1 if i == sel else 0xE1E1E1, 0x0070D1 if i == sel else 0xADADAD, 1)
            _garlic.text(bx + 48, by + 15, b, 13, i == sel, 0xFFFFFF if i == sel else 0, 4)
        _garlic.frame()
        for kind, button, x, y_ in _garlic.events():
            if kind != "down":
                continue
            if button in ("left", "up"):
                sel = (sel - 1) % n
            elif button in ("right", "down"):
                sel = (sel + 1) % n
            elif button == "cross":
                _dirty()
                return sel


class _MessageBox:
    def showinfo(self, title=None, message=None, **kw):
        _dialog(title or "", message or "", ["OK"])
        return "ok"

    showwarning = showerror = showinfo

    def askyesno(self, title=None, message=None, **kw):
        yes, no = {"es": ("Sí", "No"), "pt": ("Sim", "Não"), "ru": ("Да", "Нет")}.get(_gx.LANG, ("Yes", "No"))
        return _dialog(title or "", message or "", [yes, no]) == 0

    def askokcancel(self, title=None, message=None, **kw):
        return _dialog(title or "", message or "", ["OK", "Cancel"]) == 0

    askretrycancel = askokcancel

    def askquestion(self, title=None, message=None, **kw):
        return "yes" if self.askyesno(title, message) else "no"

    def askyesnocancel(self, title=None, message=None, **kw):
        return self.askyesno(title, message)


messagebox = _MessageBox()


class _SimpleDialog:
    def askstring(self, title, prompt, **kw):
        return input(prompt + " ")

    def askinteger(self, title, prompt, **kw):
        try:
            return int(input(prompt + " "))
        except ValueError:
            return None

    def askfloat(self, title, prompt, **kw):
        try:
            return float(input(prompt + " "))
        except ValueError:
            return None


simpledialog = _SimpleDialog()


class _FontModule:
    Font = _Font
    BOLD, NORMAL, ITALIC, ROMAN = "bold", "normal", "italic", "roman"

    def families(self, *a):
        return ("Arial",)

    def nametofont(self, name):
        return _Font()


font = _FontModule()


class _Ttk:
    Button, Label, Frame, Entry, Checkbutton, Radiobutton = Button, Label, Frame, Entry, Checkbutton, Radiobutton
    LabelFrame, Scale, Scrollbar, Spinbox = LabelFrame, Scale, Scrollbar, Spinbox

    class Style:
        def __init__(self, *a, **k):
            pass

        def configure(self, *a, **k):
            pass

        def theme_use(self, *a):
            pass

        def map(self, *a, **k):
            pass

    class Combobox(Entry):
        def current(self, i=None):
            vals = self._opts.get("values", [])
            if i is not None and vals:
                self._set(str(vals[i]))

    class Progressbar(Frame):
        def start(self, *a):
            pass

        def stop(self):
            pass

        def step(self, *a):
            pass

    class Separator(Frame):
        def _req(self):
            return 2, 2

    class Notebook(Frame):
        def add(self, child, **kw):
            pass

    class Treeview(Listbox):
        pass


ttk = _Ttk()
