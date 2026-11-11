# typing for Garlic: the names exist and can be subscripted (List[int]); nothing is checked, as in CPython.
class _Any:
    def __getitem__(self, k):
        return self

    def __call__(self, *a, **k):
        return a[0] if a else None

    def __or__(self, o):
        return self

    def __ror__(self, o):
        return self


Any = Union = Optional = List = Dict = Tuple = Set = FrozenSet = Callable = Iterable = Iterator = Sequence = _Any()
Mapping = MutableMapping = Type = Generator = Literal = ClassVar = Final = NoReturn = Deque = DefaultDict = _Any()
Counter = NamedTuple = TypedDict = Protocol = Annotated = Self = _Any()
TYPE_CHECKING = False


def TypeVar(name, *a, **k):
    return _Any()


def NewType(name, tp):
    return lambda x: x


def cast(tp, v):
    return v


def overload(f):
    return f


def final(f):
    return f


class Generic:
    pass
