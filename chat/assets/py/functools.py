# functools for Garlic.
def reduce(function, iterable, *initial):
    it = iter(iterable)
    value = initial[0] if initial else next(it)
    for x in it:
        value = function(value, x)
    return value


def partial(func, *args, **kw):
    def p(*a, **k):
        kk = dict(kw)
        kk.update(k)
        return func(*(args + a), **kk)
    return p


def wraps(wrapped, *a, **k):
    return lambda f: f


def update_wrapper(wrapper, wrapped, *a, **k):
    return wrapper


def lru_cache(maxsize=128, typed=False):
    def deco(f):
        memo = {}

        def g(*args):
            if args not in memo:
                memo[args] = f(*args)
            return memo[args]
        return g
    if callable(maxsize):
        return deco(maxsize)
    return deco


cache = lru_cache


def cmp_to_key(cmp):
    class K:
        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, o):
            return cmp(self.obj, o.obj) < 0

        def __gt__(self, o):
            return cmp(self.obj, o.obj) > 0

        def __eq__(self, o):
            return cmp(self.obj, o.obj) == 0

        def __le__(self, o):
            return cmp(self.obj, o.obj) <= 0

        def __ge__(self, o):
            return cmp(self.obj, o.obj) >= 0
    return K


def total_ordering(cls):
    return cls
