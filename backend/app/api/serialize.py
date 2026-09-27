"""Dataclass / numpy -> JSON-friendly helpers for the API layer.

Starlette renders JSON with allow_nan=False, so a single NaN anywhere in a
payload would turn a read into a 500. Every value therefore goes through
`to_jsonable`: NaN / inf become None (the NaN policy of lib-3: unknown is
null, never a made-up number), numpy scalars and arrays become Python.
"""
from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

try:                                   # numpy is always present in the backend
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        # not dataclasses.asdict: it deep-copies every nested model dict first
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return to_jsonable(obj.tolist())
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            x = float(obj)
            return x if math.isfinite(x) else None
    if hasattr(obj, "_asdict"):        # NamedTuple (e.g. ProfileVersion)
        return to_jsonable(obj._asdict())
    return obj
