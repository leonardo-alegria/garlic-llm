# textwrap for Garlic: wrap, fill, dedent, indent, shorten.
def wrap(text, width=70, **kw):
    lines, line = [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = word if not line else line + " " + word
    if line:
        lines.append(line)
    return lines


def fill(text, width=70, **kw):
    return "\n".join(wrap(text, width))


def shorten(text, width, placeholder=" [...]"):
    t = " ".join(str(text).split())
    return t if len(t) <= width else t[:max(0, width - len(placeholder))].rsplit(" ", 1)[0] + placeholder


def dedent(text):
    lines = text.split("\n")
    ind = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    n = min(ind) if ind else 0
    return "\n".join(l[n:] if l.strip() else "" for l in lines)


def indent(text, prefix, predicate=None):
    out = []
    for l in text.split("\n"):
        out.append(prefix + l if (predicate(l) if predicate else l.strip()) else l)
    return "\n".join(out)
