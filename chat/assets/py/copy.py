# copy for Garlic: copy and deepcopy of lists, dicts, sets, tuples and plain objects.
class Error(Exception):
    pass


def copy(x):
    if isinstance(x, list):
        return list(x)
    if isinstance(x, dict):
        return type(x)(x) if type(x) is not dict else dict(x)
    if isinstance(x, set):
        return set(x)
    if hasattr(x, "__copy__"):
        return x.__copy__()
    if hasattr(x, "__dict__") and not isinstance(x, type):
        y = object.__new__(type(x))
        y.__dict__.update(x.__dict__)
        return y
    return x


def deepcopy(x, memo=None):
    if memo is None:
        memo = {}
    i = id(x)
    if i in memo:
        return memo[i]
    if isinstance(x, list):
        y = []
        memo[i] = y
        y.extend(deepcopy(v, memo) for v in x)
    elif isinstance(x, tuple):
        y = tuple(deepcopy(v, memo) for v in x)
    elif isinstance(x, dict):
        y = {}
        memo[i] = y
        for k, v in x.items():
            y[deepcopy(k, memo)] = deepcopy(v, memo)
    elif isinstance(x, set):
        y = set(deepcopy(v, memo) for v in x)
    elif hasattr(x, "__deepcopy__"):
        y = x.__deepcopy__(memo)
    elif hasattr(x, "__dict__") and not isinstance(x, type) and not callable(x):
        y = object.__new__(type(x))
        memo[i] = y
        for k, v in x.__dict__.items():
            setattr(y, k, deepcopy(v, memo))
    else:
        y = x
    return y
