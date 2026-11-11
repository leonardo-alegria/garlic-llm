# itertools for Garlic (the CPython functions programs use, as generators).
def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step


def cycle(iterable):
    saved = list(iterable)
    while saved:
        for x in saved:
            yield x


def repeat(obj, times=None):
    if times is None:
        while True:
            yield obj
    for _ in range(times):
        yield obj


def _chain(its):
    for it in its:
        for x in it:
            yield x


class chain:   # (a class: MicroPython functions cannot carry chain.from_iterable)
    def __init__(self, *iterables):
        self._g = _chain(iterables)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._g)

    @staticmethod
    def from_iterable(its):
        return _chain(its)


def islice(iterable, *args):
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    else:
        start, stop, step = args[0] or 0, args[1], (args[2] if len(args) > 2 else 1) or 1
    i, nxt = 0, start
    for x in iterable:
        if stop is not None and i >= stop:
            return
        if i == nxt:
            yield x
            nxt += step
        i += 1


def accumulate(iterable, func=None, initial=None):
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for x in it:
        total = func(total, x) if func else total + x
        yield total


def product(*iterables, repeat=1):
    pools = [tuple(p) for p in iterables] * repeat
    result = [[]]
    for pool in pools:
        result = [r + [y] for r in result for y in pool]
    for r in result:
        yield tuple(r)


def permutations(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    for idx in product(range(n), repeat=r):
        if len(set(idx)) == r:
            yield tuple(pool[i] for i in idx)


def combinations(iterable, r):
    pool = tuple(iterable)
    for idx in permutations(range(len(pool)), r):
        if sorted(idx) == list(idx):
            yield tuple(pool[i] for i in idx)


def combinations_with_replacement(iterable, r):
    pool = tuple(iterable)
    for idx in product(range(len(pool)), repeat=r):
        if sorted(idx) == list(idx):
            yield tuple(pool[i] for i in idx)


def zip_longest(*iterables, fillvalue=None):
    its = [iter(i) for i in iterables]
    while True:
        row, alive = [], 0
        for it in its:
            try:
                row.append(next(it))
                alive += 1
            except StopIteration:
                row.append(fillvalue)
        if not alive:
            return
        yield tuple(row)


def groupby(iterable, key=None):
    key = key or (lambda x: x)
    group, k, first = [], None, True
    for x in iterable:
        kx = key(x)
        if first or kx != k:
            if not first:
                yield k, iter(group)
            group, k, first = [x], kx, False
        else:
            group.append(x)
    if not first:
        yield k, iter(group)


def starmap(func, iterable):
    for args in iterable:
        yield func(*args)


def takewhile(pred, iterable):
    for x in iterable:
        if not pred(x):
            return
        yield x


def dropwhile(pred, iterable):
    it = iter(iterable)
    for x in it:
        if not pred(x):
            yield x
            break
    for x in it:
        yield x


def filterfalse(pred, iterable):
    for x in iterable:
        if not (pred or bool)(x):
            yield x


def compress(data, selectors):
    for d, s in zip(data, selectors):
        if s:
            yield d


def pairwise(iterable):
    it = iter(iterable)
    try:
        a = next(it)
    except StopIteration:
        return
    for b in it:
        yield a, b
        a = b


def tee(iterable, n=2):
    saved = list(iterable)
    return tuple(iter(saved) for _ in range(n))
