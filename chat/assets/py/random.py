# random for Garlic: MicroPython's built-in random plus what programs use from CPython's (shuffle, sample, choices,
# gauss...). The built-in is reached with an empty sys.path.
import sys

_p = list(sys.path)
sys.path.clear()
try:
    from random import *
    from random import getrandbits, seed, random, randrange, randint, choice, uniform
finally:
    sys.path.extend(_p)
del _p


def shuffle(x):
    for i in range(len(x) - 1, 0, -1):
        j = randrange(i + 1)
        x[i], x[j] = x[j], x[i]


def sample(population, k, counts=None):
    pool = list(population)
    if counts:
        pool = [v for v, c in zip(pool, counts) for _ in range(c)]
    if k > len(pool):
        raise ValueError("sample larger than population")
    for i in range(k):
        j = randrange(i, len(pool))
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def choices(population, weights=None, cum_weights=None, k=1):
    pool = list(population)
    if weights is None and cum_weights is None:
        return [pool[randrange(len(pool))] for _ in range(k)]
    if cum_weights is None:
        cum_weights, t = [], 0
        for w in weights:
            t += w
            cum_weights.append(t)
    total = cum_weights[-1]
    out = []
    for _ in range(k):
        r = random() * total
        i = 0
        while i < len(cum_weights) - 1 and cum_weights[i] <= r:
            i += 1
        out.append(pool[i])
    return out


def gauss(mu=0.0, sigma=1.0):
    import math
    u = 1.0 - random()
    return mu + sigma * math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * random())


normalvariate = gauss


def triangular(low=0.0, high=1.0, mode=None):
    u = random()
    c = 0.5 if mode is None else (mode - low) / (high - low)
    if u > c:
        u, c, low, high = 1.0 - u, 1.0 - c, high, low
    return low + (high - low) * (u * c) ** 0.5


def expovariate(lambd=1.0):
    import math
    return -math.log(1.0 - random()) / lambd


def betavariate(a, b):
    return random()


def getstate():
    return None


def setstate(s):
    pass


class Random:
    def __init__(self, x=None):
        if x is not None:
            seed(x)

    def __getattr__(self, name):
        return globals()[name]


SystemRandom = Random
