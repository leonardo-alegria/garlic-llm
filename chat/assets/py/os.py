# os for Garlic: what text programs call (clearing the screen, the terminal size); there are no files.
import _garlic

name = "posix"
sep = "/"
linesep = "\n"
curdir = "."
environ = {}


def system(cmd):
    if str(cmd).strip() in ("cls", "clear"):
        _garlic.console_clear()
    return 0


def getcwd():
    return "/"


def listdir(path="."):
    return []


def getenv(key, default=None):
    return environ.get(key, default)


def get_terminal_size(*a):
    return (80, 24)


def urandom(n):
    import random
    return bytes(random.getrandbits(8) for _ in range(n))


def getpid():
    return 1


def remove(p):
    raise OSError(2, "No such file: " + str(p))


unlink = remove


def makedirs(*a, **k):
    pass


mkdir = makedirs


class _Path:
    sep = "/"

    def exists(self, p):
        return False

    def isfile(self, p):
        return False

    def isdir(self, p):
        return False

    def join(self, *p):
        return "/".join(str(x).rstrip("/") for x in p)

    def basename(self, p):
        return str(p).rsplit("/", 1)[-1]

    def dirname(self, p):
        return str(p).rsplit("/", 1)[0] if "/" in str(p) else ""

    def abspath(self, p):
        return str(p)

    def splitext(self, p):
        p = str(p)
        i = p.rfind(".")
        return (p[:i], p[i:]) if i > 0 and i > p.rfind("/") else (p, "")

    def expanduser(self, p):
        return str(p)


path = _Path()
