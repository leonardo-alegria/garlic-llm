# time for Garlic: MicroPython's built-in time plus CPython's perf_counter, monotonic, localtime, strftime...
import sys

_p = list(sys.path)
sys.path.clear()
try:
    from time import *
    from time import time, time_ns, sleep, sleep_ms, sleep_us, ticks_ms, ticks_us, ticks_diff, ticks_add
finally:
    sys.path.extend(_p)
del _p

_t0 = time()


def perf_counter():
    return time() - _t0


monotonic = process_time = perf_counter


def perf_counter_ns():
    return int(perf_counter() * 1e9)


monotonic_ns = process_time_ns = perf_counter_ns

timezone = 0
altzone = 0
daylight = 0
tzname = ("UTC", "UTC")


class struct_time(tuple):
    def __getattr__(self, name):
        i = ("tm_year", "tm_mon", "tm_mday", "tm_hour", "tm_min", "tm_sec", "tm_wday", "tm_yday", "tm_isdst").index(name)
        return self[i]


def _days_to_ymd(days):   # days since 1970-01-01 -> (y, m, d)
    days += 719468
    era = days // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + 1 if m <= 2 else y), m, d


def gmtime(secs=None):
    s = int(time() if secs is None else secs)
    days, rem = divmod(s, 86400)
    y, m, d = _days_to_ymd(days)
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    yday = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)[m - 1] + d + (1 if leap and m > 2 else 0)
    return struct_time((y, m, d, rem // 3600, rem // 60 % 60, rem % 60, (days + 3) % 7, yday, 0))


localtime = gmtime


def mktime(t):
    y, m, d, hh, mm, ss = t[:6]
    if m <= 2:
        y, m = y - 1, m + 12
    days = 365 * y + y // 4 - y // 100 + y // 400 + (153 * (m - 3) + 2) // 5 + d - 719469
    return float(days * 86400 + hh * 3600 + mm * 60 + ss)


_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def strftime(fmt, t=None):
    t = localtime() if t is None else t
    rep = {"Y": "%04d" % t[0], "y": "%02d" % (t[0] % 100), "m": "%02d" % t[1], "d": "%02d" % t[2],
           "H": "%02d" % t[3], "I": "%02d" % ((t[3] - 1) % 12 + 1), "M": "%02d" % t[4], "S": "%02d" % t[5],
           "p": "AM" if t[3] < 12 else "PM", "A": _DAYS[t[6]], "a": _DAYS[t[6]][:3], "B": _MONTHS[t[1] - 1],
           "b": _MONTHS[t[1] - 1][:3], "j": "%03d" % t[7], "%": "%", "Z": "UTC", "z": "+0000"}
    rep["c"] = "%s %s %2d %s:%s:%s %s" % (rep["a"], rep["b"], t[2], rep["H"], rep["M"], rep["S"], rep["Y"])
    rep["x"] = rep["m"] + "/" + rep["d"] + "/" + rep["y"]
    rep["X"] = rep["H"] + ":" + rep["M"] + ":" + rep["S"]
    out, i = [], 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            out.append(rep.get(fmt[i + 1], fmt[i:i + 2]))
            i += 2
        else:
            out.append(fmt[i])
            i += 1
    return "".join(out)


def asctime(t=None):
    return strftime("%c", t)


def ctime(secs=None):
    return asctime(localtime(secs))
