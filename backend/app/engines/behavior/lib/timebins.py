"""Local-time context: timezone, calendar, bins, slots, dayparts, cadence class.

STATUS: implemented. Constants, signatures and semantics are frozen
(docs/lib3/helpers_api.md).

Why centralised: every seasonal model (B03 buckets, B07 rhythm, B13 budget
phases), every Mondrian stratum (B24, B25) and every portrait window must
agree on what "Monday 09:00 local on a workday" means. A misconfigured tz or
calendar shifts every bin at once, so this is the single place it is computed.
zoneinfo based, DST-correct (local fields come from the aware datetime, the
slot index from the local wall clock), and holiday / 调休 (make-up workday)
aware: a date in calendar.holidays is a nonworkday even on a weekday, a date
in calendar.makeup_workdays is a workday even on a weekend.

tctx dict (feature.tctx):
    hour_local : float, local hour + minute/60 + second/3600 (10.0 at 10:00)
    dow        : int, 0 = Monday .. 6 = Sunday (local)
    day_type   : 'workday' | 'nonworkday'
    bin48      : int, floor(hour_local) + 24 * (day_type == 'nonworkday')
    bin168     : int, dow * 24 + floor(hour_local)
    slot       : int, floor(local_epoch_s / 900) where local_epoch_s =
                 ts + utcoffset(ts) (15-min slot index since the local epoch)
    daypart    : 'wd_day' | 'wd_night' | 'nwd_day' | 'nwd_night'
                 (day = day_hours[0] <= hour_local < day_hours[1])
    cc         : int, cadence class = nearest of CADENCE_CLASSES to dt in log space
                 (ties go to the larger class)
B01 test: tz Asia/Shanghai, ts = 2026-10-01T02:00Z, holiday -> hour_local 10,
day_type nonworkday, daypart nwd_day.

Implementation notes:
  * One local clock. tctx derives every field from the local epoch second
    L = ts + utcoffset(ts) (hour = (L mod 86400) / 3600, date = epoch + L // 86400),
    so hour_local, bin48, bin168 and slot can never disagree at a boundary
    (datetime.fromtimestamp rounds to the microsecond, L does not). The offset
    is looked up at floor(ts): tz transitions sit on whole seconds, so this is
    exact for any real ts.
  * DST. Across spring-forward a local hour does not exist and its slots are
    never produced by slot_of; slot_bounds returns them as empty intervals at
    the transition instant. Across fall-back a local hour happens twice and
    both passes share the same slot / bin; slot_bounds returns the first pass
    (the pre-transition, "earlier" offset). A B07 slot that happens twice is
    simply active if either pass was.
  * Cadence class compares dt^2 with the product of neighbouring classes
    (dt is nearer c_lo than c_hi in log space iff dt^2 < c_lo * c_hi), which
    makes the tie at the geometric mean (e.g. dt = 1800 between 900 and 3600)
    exact instead of subject to log rounding.
  * NaN policy. A non-finite ts / hour / dt is a caller bug, not "undefined"
    data, and raises ValueError instead of producing a NaN bin index that
    would silently land in the wrong stratum. decode_tctx maps non-finite
    fields back to None (hour_local stays as is) and rejects a day_type /
    daypart index outside its name tuple.
  * Calendar normalises its sets to datetime.date (strings are parsed), and
    day_type / tctx also accept the raw ctx.config['calendar'] mapping.
"""
from __future__ import annotations

import datetime as _dt
import functools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

DEFAULT_TZ = "Asia/Shanghai"
DEFAULT_DAY_HOURS: Tuple[int, int] = (8, 20)
SLOT_S = 900
CADENCE_CLASSES: Tuple[int, ...] = (60, 300, 900, 3600)
DAYPARTS: Tuple[str, ...] = ("wd_day", "wd_night", "nwd_day", "nwd_night")
DAY_TYPES: Tuple[str, ...] = ("workday", "nonworkday")
# float32 encoding order of feature.tctx (add_vec); strings are indices into
# DAY_TYPES / DAYPARTS. slot < 2^24 until ~2448 so float32 is exact.
TCTX_FIELDS: Tuple[str, ...] = ("hour_local", "dow", "day_type", "bin48", "bin168",
                                "slot", "daypart", "cc")

_DAY_S = 86400
_EPOCH_ORDINAL = _dt.date(1970, 1, 1).toordinal()     # 1970-01-01 was a Thursday
_EPOCH_DOW = 3
_EPOCH_NAIVE = _dt.datetime(1970, 1, 1)
_INT_FIELDS = ("dow", "bin48", "bin168", "slot", "cc")
_CAL_CACHE_MAX = 64

TzLike = Union[str, _dt.tzinfo]


@dataclass(frozen=True, slots=True)
class Calendar:
    holidays: FrozenSet[_dt.date] = field(default_factory=frozenset)
    makeup_workdays: FrozenSet[_dt.date] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Normalise to frozenset[date]: Calendar(holidays={'2026-10-01'}) would
        # otherwise never match a date and silently turn a holiday into a workday.
        for name in ("holidays", "makeup_workdays"):
            v = getattr(self, name)
            if not (isinstance(v, frozenset)
                    and all(type(x) is _dt.date for x in v)):
                object.__setattr__(self, name, frozenset(_as_date(x) for x in _items(v)))


_EMPTY_CALENDAR = Calendar()
_cal_cache: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], Calendar] = {}


# ---------------------------------------------------------------- calendar
def _as_date(x: Any) -> _dt.date:
    if isinstance(x, _dt.datetime):          # datetime is a date subclass: drop the time
        return x.date()
    if isinstance(x, _dt.date):
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            return _dt.date.fromisoformat(s)
        except ValueError:
            try:
                return _dt.datetime.fromisoformat(s).date()
            except ValueError:
                pass
    raise ValueError(f"calendar date must be 'YYYY-MM-DD' or datetime.date, got {x!r}")


def _items(v: Any) -> Tuple[Any, ...]:
    if v is None:
        return ()
    if isinstance(v, (str, _dt.date)):       # a lone date, not a list of characters
        return (v,)
    return tuple(v)


def parse_calendar(cfg: Optional[Mapping[str, Any]]) -> Calendar:
    """ctx.config['calendar'] {holidays: [...], makeup_workdays: [...]} -> Calendar.
    Items may be 'YYYY-MM-DD' strings or datetime.date; None -> empty calendar.
    Cached by id/contents (called every tick)."""
    if cfg is None:
        return _EMPTY_CALENDAR
    if isinstance(cfg, Calendar):
        return cfg
    if not isinstance(cfg, Mapping):
        raise ValueError(f"calendar config must be a mapping, got {type(cfg).__name__}")
    hol, mk = _items(cfg.get("holidays")), _items(cfg.get("makeup_workdays"))
    # Keyed by contents (str and date are hashable), so an in-place edit of the
    # config list is picked up on the next tick; a small bounded dict suffices
    # because a deployment has one or two calendars.
    try:
        key = (hol, mk)
        cal = _cal_cache.get(key)
    except TypeError:                        # unhashable item: parse without caching
        key, cal = None, None
    if cal is not None:
        return cal
    cal = Calendar(holidays=frozenset(_as_date(x) for x in hol),
                   makeup_workdays=frozenset(_as_date(x) for x in mk))
    if key is not None:
        if len(_cal_cache) >= _CAL_CACHE_MAX:
            _cal_cache.clear()
        _cal_cache[key] = cal
    return cal


def day_type(d: _dt.date, calendar: Optional[Calendar] = None) -> str:
    """'workday' | 'nonworkday': makeup_workdays -> workday; holidays -> nonworkday;
    else Mon-Fri workday, Sat-Sun nonworkday."""
    if isinstance(d, _dt.datetime):
        d = d.date()
    if calendar is not None:
        if not isinstance(calendar, Calendar):   # raw ctx.config['calendar'] mapping
            calendar = parse_calendar(calendar)
        if d in calendar.makeup_workdays:
            return "workday"
        if d in calendar.holidays:
            return "nonworkday"
    return "workday" if d.weekday() < 5 else "nonworkday"


# ---------------------------------------------------------------- cadence
def cadence_class(dt_s: float) -> int:
    """Nearest of CADENCE_CLASSES to dt_s by |log(dt/c)|, ties to the larger."""
    dt = float(dt_s)
    if math.isnan(dt):
        raise ValueError("cadence_class: dt is NaN")
    if dt <= 0.0:                            # log-space limit dt -> 0+
        return CADENCE_CLASSES[0]
    for lo, hi in zip(CADENCE_CLASSES[:-1], CADENCE_CLASSES[1:]):
        # nearer lo iff log(dt/lo) < log(hi/dt) iff dt^2 < lo*hi (exact tie -> hi)
        if dt * dt < lo * hi:
            return lo
    return CADENCE_CLASSES[-1]


# ---------------------------------------------------------------- clock
@functools.lru_cache(maxsize=64)
def _zone_by_name(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {tz!r}") from exc


def _zone(tz: Optional[TzLike]) -> _dt.tzinfo:
    if isinstance(tz, _dt.tzinfo):
        return tz
    return _zone_by_name(tz or DEFAULT_TZ)


def _check_ts(ts: float) -> float:
    t = float(ts)
    if not math.isfinite(t):
        raise ValueError(f"timestamp must be finite, got {ts!r}")
    return t


def _offset_s(t: float, zone: _dt.tzinfo) -> int:
    """utcoffset (seconds) in effect at UTC epoch t. Looked up at floor(t):
    transitions are on whole seconds, and an integer-valued timestamp is
    converted without the microsecond rounding of fromtimestamp."""
    d = _dt.datetime.fromtimestamp(math.floor(t), tz=zone)
    off = d.utcoffset()
    return 0 if off is None else int(off.total_seconds())


def _local_epoch_s(t: float, zone: _dt.tzinfo) -> float:
    return t + _offset_s(t, zone)


def local_datetime(ts: float, tz: str = DEFAULT_TZ) -> _dt.datetime:
    """Aware local datetime of a UTC epoch ts (zoneinfo.ZoneInfo(tz), cached)."""
    return _dt.datetime.fromtimestamp(_check_ts(ts), tz=_zone(tz))


def slot_of(ts: float, tz: str = DEFAULT_TZ) -> int:
    """15-min local slot index: floor((ts + utcoffset(ts)) / 900)."""
    t = _check_ts(ts)
    return int(_local_epoch_s(t, _zone(tz)) // SLOT_S)


def _first_offset_change(a: int, b: int, zone: _dt.tzinfo) -> int:
    """Smallest integer t in (a, b] whose offset differs from offset(a);
    precondition: offset(b) != offset(a). Bisection, ~log2(b - a) lookups."""
    off_a = _offset_s(a, zone)
    while b - a > 1:
        m = (a + b) // 2
        if _offset_s(m, zone) == off_a:
            a = m
        else:
            b = m
    return b


def slot_bounds(slot: int, tz: str = DEFAULT_TZ) -> Tuple[float, float]:
    """UTC epoch [start, end) of a local slot. Across a DST fold the earlier
    offset is used; a slot that does not exist (spring-forward gap) returns
    an empty interval (start == end)."""
    zone = _zone(tz)
    l0 = int(slot) * SLOT_S                  # local epoch seconds of the slot start
    l1 = l0 + SLOT_S
    wall = _EPOCH_NAIVE + _dt.timedelta(seconds=l0)
    # fold=0 is the first pass of a repeated wall time (pre-transition offset)
    t0 = int(round(wall.replace(tzinfo=zone, fold=0).timestamp()))
    off0 = _offset_s(t0, zone)
    if t0 + off0 == l0:
        start = t0
    else:
        # Wall time l0 is in a spring-forward gap: fold=1 maps it with the
        # post-transition offset to an instant before the jump. The slot then
        # starts at the transition T (if any of it survives the gap) or is empty.
        t1 = int(round(wall.replace(tzinfo=zone, fold=1).timestamp()))
        lo, hi = min(t0, t1) - 1, max(t0, t1)
        if _offset_s(lo, zone) == _offset_s(hi, zone):   # defensive: no jump found
            start = t0
        else:
            T = _first_offset_change(lo, hi, zone)
            off0 = _offset_s(T, zone)
            l_after = T + off0                           # wall clock right after the jump
            if l_after >= l1:
                return float(T), float(T)
            start, l0 = T, l_after
    # End of the first pass: normally start + remaining slot length, cut short
    # by a transition inside the slot (fall-back at / inside the slot, or a
    # gap that begins mid-slot).
    end = start + (l1 - l0)
    if _offset_s(end - 1, zone) != off0:
        end = _first_offset_change(start, end - 1, zone)
    return float(start), float(end)


# ---------------------------------------------------------------- dayparts
def daypart(day_type_: str, hour_local: float,
            day_hours: Tuple[int, int] = DEFAULT_DAY_HOURS) -> str:
    """'wd_day' | 'wd_night' | 'nwd_day' | 'nwd_night'."""
    if day_type_ == "workday":
        prefix = "wd_"
    elif day_type_ == "nonworkday":
        prefix = "nwd_"
    else:
        raise ValueError(f"day_type must be one of {DAY_TYPES}, got {day_type_!r}")
    h = float(hour_local)
    if math.isnan(h):
        raise ValueError("daypart: hour_local is NaN")
    a, b = day_hours
    # a > b is a day window that wraps midnight (e.g. a night-shift site)
    is_day = (a <= h < b) if a <= b else (h >= a or h < b)
    return prefix + ("day" if is_day else "night")


def _day_hours(v: Any) -> Tuple[float, float]:
    if v is None:
        return DEFAULT_DAY_HOURS
    if isinstance(v, str):                   # "8-20"
        parts = v.replace("–", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"daypart_day_hours must look like '8-20', got {v!r}")
        v = parts
    try:
        a, b = (float(x) for x in v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"daypart_day_hours must be (start, end) hours, got {v!r}") from exc
    if not (0.0 <= a <= 24.0 and 0.0 <= b <= 24.0):
        raise ValueError(f"daypart_day_hours out of [0, 24]: {v!r}")
    return (int(a) if a.is_integer() else a, int(b) if b.is_integer() else b)


# ---------------------------------------------------------------- tctx
def tctx(ts: float, tz: str = DEFAULT_TZ, calendar: Optional[Calendar] = None,
         day_hours: Tuple[int, int] = DEFAULT_DAY_HOURS, dt: float = 60.0) -> Dict[str, Any]:
    """Full time context of tick `ts` (see module doc for every field). O(1), ~5 us."""
    t = _check_ts(ts)
    loc = _local_epoch_s(t, _zone(tz))
    q, sod = divmod(loc, _DAY_S)             # float floor / mod are exact
    days = int(q)
    if sod >= _DAY_S:
        # % of a tiny negative L rounds up to 86400 (only |L| < ~3e4 s). L is
        # truly just before midnight, which is what slot = L // 900 says too,
        # so stay on that day rather than rolling over to the next one.
        sod = math.nextafter(float(_DAY_S), 0.0)
    hour = int(sod // 3600)
    dow = (days + _EPOCH_DOW) % 7
    dtype = day_type(_dt.date.fromordinal(_EPOCH_ORDINAL + days), calendar)
    hour_local = sod / 3600.0
    if hour_local >= hour + 1:               # division rounded up across the hour
        hour_local = math.nextafter(hour + 1.0, 0.0)
    return {
        "hour_local": hour_local,
        "dow": dow,
        "day_type": dtype,
        "bin48": hour + (24 if dtype == "nonworkday" else 0),
        "bin168": dow * 24 + hour,
        "slot": int(loc // SLOT_S),
        "daypart": daypart(dtype, hour_local, day_hours),
        "cc": cadence_class(dt),
    }


def tctx_from_config(ts: float, config: Mapping[str, Any], dt: float) -> Dict[str, Any]:
    """tctx with tz / calendar / daypart_day_hours taken from ctx.config (defaults above)."""
    cfg = config or {}
    return tctx(ts, tz=cfg.get("tz") or DEFAULT_TZ,
                calendar=parse_calendar(cfg.get("calendar")),
                day_hours=_day_hours(cfg.get("daypart_day_hours")), dt=dt)


def _index(v: Any, names: Tuple[str, ...], what: str) -> float:
    if v is None:
        return math.nan
    if isinstance(v, str):
        try:
            return float(names.index(v))
        except ValueError:
            raise ValueError(f"{what} must be one of {names}, got {v!r}") from None
    x = float(v)                             # already an index (e.g. a decoded row)
    if not math.isnan(x) and not (x.is_integer() and 0 <= x < len(names)):
        raise ValueError(f"{what} index out of range: {v!r}")
    return x


def encode_tctx(t: Mapping[str, Any]) -> np.ndarray:
    """dict -> float32[8] in TCTX_FIELDS order (day_type/daypart as indices)."""
    out = np.empty(len(TCTX_FIELDS), dtype=np.float32)
    for i, f in enumerate(TCTX_FIELDS):
        v = t.get(f)
        if f == "hour_local" and v is not None and math.isfinite(float(v)):
            # float32 has ~3.4 ms resolution at 10 h: 09:59:59.9999 would round
            # to 10.0 and disagree with bin48 / daypart in the stored row, so
            # round down to the last float32 inside the same hour instead.
            h = float(v)
            h32 = np.float32(h)
            if math.floor(float(h32)) > math.floor(h):
                h32 = np.nextafter(np.float32(math.floor(h) + 1), np.float32(0.0))
            out[i] = h32
        elif f == "day_type":
            out[i] = _index(v, DAY_TYPES, f)
        elif f == "daypart":
            out[i] = _index(v, DAYPARTS, f)
        else:
            out[i] = math.nan if v is None else float(v)
    return out


def decode_tctx(v: np.ndarray) -> Dict[str, Any]:
    """Inverse of encode_tctx (ints restored, strings from DAY_TYPES / DAYPARTS)."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != len(TCTX_FIELDS):
        raise ValueError(f"tctx vector must have {len(TCTX_FIELDS)} entries, got {a.size}")
    out: Dict[str, Any] = {}
    for f, x in zip(TCTX_FIELDS, a.tolist()):
        if f == "hour_local":
            out[f] = x
        elif not math.isfinite(x):
            out[f] = None
        elif f in ("day_type", "daypart"):
            # a negative index would silently wrap to the last name
            names = DAY_TYPES if f == "day_type" else DAYPARTS
            i = int(round(x))
            if not (i == x and 0 <= i < len(names)):
                raise ValueError(f"{f} index out of range: {x!r}")
            out[f] = names[i]
        else:
            out[f] = int(round(x))
    return out


# ---------------------------------------------------------------- circular hours
def hour_distance(h1: float, h2: float) -> float:
    """Signed circular hour difference in (-12, 12] (for von Mises smoothing)."""
    d = 12.0 - (12.0 - (float(h1) - float(h2))) % 24.0
    if d <= -12.0:                           # % rounding of a tiny negative to 24.0
        d += 24.0
    return d


def von_mises_weights(hour: float, kappa: float = 4.0, max_dh: int = 2) -> Dict[int, float]:
    """{hour_bin: w} for |dh| <= max_dh with w = exp(kappa (cos(2 pi dh / 24) - 1)),
    dh measured from the bin centre; used to spread a commit over neighbour bins (B03, B07)."""
    h = float(hour)
    if not math.isfinite(h):
        raise ValueError(f"von_mises_weights: hour must be finite, got {hour!r}")
    if max_dh < 0:
        return {}
    h %= 24.0
    k = float(kappa)
    reach = int(min(max_dh, 12)) + 1
    base = int(math.floor(h))
    out: Dict[int, float] = {}
    for b in range(base - reach, base + reach + 1):
        dh = hour_distance(b + 0.5, h)
        if abs(dh) <= max_dh + 1e-9:
            # cos - 1 = -2 sin^2(x/2): keeps small dh free of cancellation
            s = math.sin(math.pi * dh / 24.0)
            out[b % 24] = math.exp(-2.0 * k * s * s)
    return out
