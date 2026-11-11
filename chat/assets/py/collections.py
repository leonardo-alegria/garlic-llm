# collections for Garlic: MicroPython's built-in one (deque, namedtuple, OrderedDict) plus Counter and defaultdict.
import sys

_p = list(sys.path)
sys.path.clear()
try:
    from collections import *
    from collections import deque, namedtuple, OrderedDict
finally:
    sys.path.extend(_p)
del _p


class defaultdict(dict):
    def __init__(self, default_factory=None, *a, **kw):
        super().__init__()   # (MicroPython: the dict itself is made here, empty)
        self.default_factory = default_factory
        for k, v in (dict(*a, **kw)).items():
            self[k] = v

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if self.default_factory is None:
                raise
            v = self[key] = self.default_factory()
            return v

    def __missing__(self, key):
        return self[key]

    def copy(self):
        return defaultdict(self.default_factory, self.items())


class Counter(dict):
    def __init__(self, iterable=None, **kw):
        super().__init__()
        self.update(iterable, **kw)

    def __getitem__(self, key):
        return self.get(key, 0)

    def __missing__(self, key):
        return 0

    def __s(self, k, v):
        super().__setitem__(k, v)

    def update(self, iterable=None, **kw):
        if iterable is not None:
            if isinstance(iterable, dict):
                for k, v in iterable.items():
                    self.__s(k, self[k] + v)
            else:
                for k in iterable:
                    self.__s(k, self[k] + 1)
        for k, v in kw.items():
            self.__s(k, self[k] + v)

    def subtract(self, iterable=None, **kw):
        if iterable is not None:
            items = iterable.items() if isinstance(iterable, dict) else ((k, 1) for k in iterable)
            for k, v in items:
                self.__s(k, self[k] - v)

    def most_common(self, n=None):
        items = sorted(self.items(), key=lambda kv: kv[1], reverse=True)
        return items if n is None else items[:n]

    def elements(self):
        for k, v in self.items():
            for _ in range(v):
                yield k

    def total(self):
        return sum(self.values())

    def __add__(self, other):
        c = Counter(self)
        c.update(other)
        return Counter({k: v for k, v in c.items() if v > 0})

    def __sub__(self, other):
        c = Counter(self)
        c.subtract(other)
        return Counter({k: v for k, v in c.items() if v > 0})


class ChainMap(dict):
    def __init__(self, *maps):
        super().__init__()
        for m in reversed(maps):
            self.update(m)
        self.maps = list(maps)
