"""phier._shape (round 3: one regex pass over alphanumeric stretches and
punctuation / other runs) equals the round-2 per-character implementation."""
import numpy as np

from app.engines.behavior.lib import phier as H


def _ref_shape(s):
    """Round-2 phier._shape, verbatim (per-character _cls loop)."""
    if not s:
        return "E0"
    runs = []
    for ch in s:
        c = H._cls(ch)
        if runs and runs[-1][0] == c and c in "LUDX":
            runs[-1] = (c, runs[-1][1] + 1)
        elif runs and runs[-1][0] == c:
            runs[-1] = (c, runs[-1][1] + 1)
        else:
            runs.append((c, 1))
    out = []
    i = 0
    while i < len(runs):
        j = i
        tot = 0
        kinds = set()
        while j < len(runs) and runs[j][0] in "LUD":
            tot += runs[j][1]
            kinds.add(runs[j][0])
            j += 1
        if j > i and len(kinds) > 1 and tot > 8:
            out.append(f"A{tot}")
            i = j
            continue
        if j > i:
            for c, n in runs[i:j]:
                out.append(f"{c}{n}")
            i = j
            continue
        c, n = runs[i]
        if c == "X":
            out.append(f"X{n}")
        else:
            lit = "SP" if c == " " else c
            out.append(lit if n == 1 else f"{lit}{{{n}}}")
        i += 1
    return " ".join(out)


def test_shape_equals_the_character_loop():
    r = np.random.default_rng(11)
    alpha = list("abcxyzABCXYZ0189") + sorted(H._PUNCT_KEEP) + list("\t\n\x00é中ÄΩ​😀")
    b64 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    cases = ["", "a", "jack", "mike.w", "  ", "--__", "a\\b", "x]y[z^w-v", "Ab3dEf9hIj", "é中",
             "😀😀a", "abcdefgh1", "abcdefg1", "ABCDEFGHI", "12345678901", "a1" * 5]
    for _ in range(3000):
        n = int(r.integers(0, 60))
        cases.append("".join(alpha[int(i)] for i in r.integers(0, len(alpha), n)))
    for _ in range(200):                                # long form paddings (P00's hot case)
        cases.append("".join(b64[int(i)] for i in r.integers(0, len(b64), int(r.integers(8, 4000)))))
    for s in cases:
        assert H._shape.__wrapped__(s) == _ref_shape(s), s
    assert H.shape("jack") == "L4" and H.shape("mike.w") == "L4 . L1" and H.shape("") == "E0"
    assert H.shape("a  b") == "L1 SP{2} L1" and H.shape("Ab3dEf9hIj") == "A10"
