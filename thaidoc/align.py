"""Put accurate VLM text onto Tesseract line positions.

Tesseract knows *where* every line is; Typhoon knows *what* it says. We walk
both lists in reading order and pair them with fuzzy matching on a digit-free
Thai key (Tesseract's Thai digits are unreliable, the letters much less so).
A VLM paragraph may cover several wrapped geometry lines.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from .model import Cell, Line, Seg, Table
from .ocr import Unit
from .thai_text import match_key


def _sim(a: str, b: str) -> float:
    ka, kb = match_key(a), match_key(b)
    if not ka or not kb:
        ca, cb = "".join(a.split()), "".join(b.split())
        from .thai_text import to_arabic_digits
        return float(fuzz.ratio(to_arabic_digits(ca), to_arabic_digits(cb))) if ca and cb else 0.0
    return fuzz.ratio(ka, kb)


def _contains(big: str, small: str) -> float:
    kb, ks = match_key(big), match_key(small)
    if not kb or not ks:
        return 0.0
    return fuzz.partial_ratio(ks, kb) if len(ks) <= len(kb) else 0.0


def _digits(t: str) -> str:
    from .thai_text import to_arabic_digits
    return "".join(c for c in to_arabic_digits(t) if c.isdigit())


def _ltext(l: Line) -> str:
    t = getattr(l, "_text_cache", None)
    if t is None:
        t = l.text()
        l._text_cache = t   # type: ignore[attr-defined]
    return t


def _pair_score(u: Unit, group: list[Line]) -> float:
    """How well a VLM unit fits a run of geometry lines (0-100)."""
    strong = [l for l in group if not getattr(l, "_weak", False)]
    if not strong:                                   # all unreadable to Tesseract
        return 56.0 if len(group) == 1 else 0.0
    text = " ".join(_ltext(l) for l in strong)
    s = _sim(u.text, text)
    if len(group) == 1:
        s = max(s, 0.85 * _contains(text, u.text))   # Tesseract line = unit + noise
    du, dl = _digits(u.text), _digits(text)
    if du and du == dl:
        s += 3                                       # list items differing only in digits
    return s


def align_body(lines: list[Line], units: list[Unit], pitch: float = 60) -> list[tuple[Unit | None, list[Line]]]:
    """Return groups (unit, lines). unit=None -> keep Tesseract text for those lines.

    Order-preserving global alignment (dynamic programming): every unit takes
    0 or 1-4 consecutive lines (a wrapped paragraph), lines may stay unmatched,
    and the pairing with the best total fit wins. Greedy matching fails on
    lists whose items differ only in digits when Tesseract garbles some of them.
    Units that match nothing go on a synthetic line after the previous group,
    so no VLM text is ever lost.
    """
    n, m = len(lines), len(units)
    MIN, MAXK = 55.0, 12
    NEG = float("-inf")
    best = [[NEG] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            v = best[i][j]
            if v == NEG:
                continue
            if j < n and v > best[i][j + 1]:                 # line keeps Tesseract text
                best[i][j + 1], back[i][j + 1] = v, ("line", i, j)
            if i < m and v > best[i + 1][j]:                 # unit unmatched -> synthetic
                best[i + 1][j], back[i + 1][j] = v, ("unit", i, j)
            if i < m:
                for k in range(1, MAXK + 1):
                    if j + k > n:
                        break
                    grp = lines[j:j + k]
                    if k > 1 and grp[-1].baseline - grp[-2].baseline > pitch * 1.6:
                        break
                    sc = _pair_score(units[i], grp)
                    if sc < MIN:
                        continue
                    nv = v + (sc - MIN) - 2 * (k - 1)        # joining lines must pay off
                    if nv > best[i + 1][j + k]:
                        best[i + 1][j + k], back[i + 1][j + k] = nv, ("pair", i, j, k)
    steps = []
    i, j = m, n
    while (i, j) != (0, 0):
        st = back[i][j]
        steps.append(st)
        if st[0] == "line":
            j -= 1
        elif st[0] == "unit":
            i -= 1
        else:
            i, j = i - 1, j - st[3]
    groups: list[tuple[Unit | None, list[Line]]] = []
    for st in reversed(steps):
        if st[0] == "line":
            groups.append((None, [lines[st[2]]]))
        elif st[0] == "unit":
            groups.append((units[st[1]], [_synthetic(groups, pitch)]))
        else:
            _, ui, lj, k = st
            groups.append((units[ui], lines[lj:lj + k]))
    return _resolve_synthetic(groups)


def _raw(t: str) -> str:
    from .thai_text import to_arabic_digits
    return to_arabic_digits("".join(t.split()))


def _resolve_synthetic(groups: list[tuple[Unit | None, list[Line]]]):
    """Clean up VLM units that found no line in reading order.

    VLMs sometimes emit a line twice or out of order. Placing those on a
    synthetic line duplicates text and stacks it on top of real lines, so:
    an exact duplicate of text already placed is dropped; a unit that clearly
    belongs to a real line (better than what that line got) takes that line.
    """
    drop: set[int] = set()
    for si, (u, grp) in enumerate(groups):
        if u is None or not getattr(grp[0], "_synthetic", False) or u.role == "page_number":
            continue
        ru = _raw(u.text)
        if not ru:
            continue
        dup = any(gi != si and gi not in drop and u2 is not None and fuzz.ratio(ru, _raw(u2.text)) >= 92
                  and _digits(u.text) == _digits(u2.text) for gi, (u2, _g) in enumerate(groups))
        if dup:
            drop.add(si)
            continue
        # a line right next to it that kept its (misread) Tesseract text is the
        # likeliest home, e.g. "(ลงชื่อ) ร.ต.อ." read as "(ลงซ้อ)9.91.2." under a signature
        near = [(gi, _sim(u.text, groups[gi][1][0].text())) for gi in range(max(0, si - 2), min(len(groups), si + 3))
                if gi not in drop and groups[gi][0] is None and len(groups[gi][1]) == 1
                and not getattr(groups[gi][1][0], "_synthetic", False)]
        near = [(gi, sc) for gi, sc in near if sc >= (40 if abs(gi - si) == 1 else 45)]
        if near:
            gi = max(near, key=lambda x: x[1])[0]
            groups[gi] = (u, groups[gi][1])
            drop.add(si)
            continue
        best_gi, best = -1, 0.0
        for gi, (u2, g2) in enumerate(groups):
            if gi in drop or getattr(g2[0], "_synthetic", False) or len(g2) != 1:
                continue
            s = _sim(u.text, g2[0].text())
            if s > best:
                best_gi, best = gi, s
        if best < 70:
            continue
        u2, g2 = groups[best_gi]
        if u2 is None:
            groups[best_gi] = (u, g2)
            drop.add(si)
        elif _sim(u2.text, g2[0].text()) + 10 < best:
            groups[best_gi] = (u, g2)
            if fuzz.partial_ratio(_raw(u2.text), ru) >= 90:
                drop.add(si)                  # the displaced unit was just a piece of this one
            else:
                groups[si] = (u2, grp)
    groups = [g for i, g in enumerate(groups) if i not in drop]
    return _absorb_next_lines(groups)


def _absorb_next_lines(groups):
    """A VLM unit matched to one line that also holds the text of the next,
    unmatched line (e.g. a two-line title returned as one line) takes it over,
    instead of that line showing Tesseract's misreading as a duplicate."""
    out = []
    for u, grp in groups:
        if out and u is None and out[-1][0] is not None and len(grp) == 1 \
                and not getattr(grp[0], "_synthetic", False):
            pu, pg = out[-1]
            if not getattr(pg[-1], "_synthetic", False):
                gap = grp[0].baseline - pg[-1].baseline
                pitch = max(1, pg[-1].y1 - pg[-1].y0) * 1.9
                old = _sim(pu.text, " ".join(l.text() for l in pg))
                new = _sim(pu.text, " ".join(l.text() for l in pg + grp))
                if 0 < gap < pitch * 1.6 and new > old + 5 and \
                        split_over(pu.text, [" ".join(l.text() for l in pg), grp[0].text()]):
                    out[-1] = (pu, pg + grp)
                    continue
        out.append((u, grp))
    return out


def _split_to_segs(text: str, gsegs: list[Seg]) -> list[str] | None:
    """Split one VLM line into the tab columns Tesseract saw (by text proportion)."""
    keys = [max(1, g.x1 - g.x0) for g in gsegs]          # ink width ~ characters
    tot = sum(keys)
    n = len(text)
    cuts, acc = [], 0
    for kk in keys[:-1]:
        acc += kk
        target = int(round(acc / tot * n))
        spaces = [i for i, c in enumerate(text) if c == " "]
        if not spaces:
            return None
        cut = min(spaces, key=lambda i: abs(i - target))
        if abs(cut - target) > max(4, n * 0.15):
            return None
        cuts.append(cut)
    if len(set(cuts)) != len(cuts) or cuts != sorted(cuts):
        return None
    parts, prev = [], 0
    for c in cuts:
        parts.append(text[prev:c].strip()); prev = c + 1
    parts.append(text[prev:].strip())
    if not all(parts):
        return None
    # every piece must resemble what Tesseract saw in that column
    for part, g in zip(parts, gsegs):
        kp, kg = match_key(part), match_key(g.text)
        if kp and kg and fuzz.ratio(kp, kg) < 45:
            return None
        if bool(kp) != bool(kg) and len(kp or kg) > 2:
            return None
    return parts


def apply_unit(u: Unit, group: list[Line]):
    """Write unit text onto the first line of the group (segments keep x positions)."""
    first = group[0]
    if len(u.segs) == 1 and len(group) == 1 and len(first.segs) > 1:
        parts = _split_to_segs(u.segs[0], first.segs)
        if parts:
            first.segs = [Seg(t, g.x0, g.x1) for t, g in zip(parts, first.segs)]
            first.text_override = "\t".join(parts)
            return
    if len(u.segs) > 1 and len(group) == 1:
        gs = first.segs or [Seg(first.text(), first.x0, first.x1)]
        if len(gs) == len(u.segs):
            first.segs = [Seg(t, g.x0, g.x1) for t, g in zip(u.segs, gs)]
        elif len(gs) > len(u.segs):
            first.segs = [Seg(t, g.x0, g.x1) for t, g in zip(u.segs, gs)]
        else:
            first.segs = [Seg(" ".join(u.segs), gs[0].x0, gs[-1].x1)]
        first.text_override = "\t".join(s.text for s in first.segs)
    else:
        first.text_override = _bullet(u.text, first)


def _bullet(text: str, line: Line) -> str:
    """Typhoon writes list bullets as Markdown "* "; print what the page shows."""
    import re
    m = re.match(r"^[*•]\s+", text)
    if not m:
        return text
    seen = line.text().lstrip()
    mark = "-" if seen[:1] in "-–—=~" else "•"
    return mark + " " + text[m.end():]


def fill_table(t: Table, rows: list[list[dict]] | None) -> bool:
    """Use VLM cell text for a ruled table when its shape matches the grid."""
    if not rows:
        return False
    grid_rows: dict[int, list[Cell]] = {}
    for c in t.cells:
        grid_rows.setdefault(c.r, []).append(c)
    ordered = [sorted(grid_rows[r], key=lambda c: c.c) for r in sorted(grid_rows)]
    # flatten both into reading order and pair when counts agree
    g_cells = [c for row in ordered for c in row]
    h_cells = [c for row in rows for c in row]
    if len(g_cells) != len(h_cells):
        # tolerate a VLM that drops empty cells
        h_cells = [c for c in h_cells if c["text"].strip()]
        g_cells = [c for c in g_cells if c.lines]
        if len(g_cells) != len(h_cells):
            return False
    for gc, hc in zip(g_cells, h_cells):
        gc.override_lines = [l for l in hc["text"].split("\n")]
    return True


def fill_table_fuzzy(t: Table, rows: list[list[dict]] | None) -> bool:
    """VLM table shaped differently from the ruled grid (e.g. one row per printed
    line): give each geometry line in the cells its best-matching VLM line.
    Applies nothing and returns False when the two clearly aren't the same table."""
    if not rows:
        return False
    vlm = [l.strip() for row in rows for c in row for l in c["text"].split("\n") if l.strip()]
    glines = [l for c in t.cells for l in c.lines if not getattr(l, "_weak", False)]
    if not vlm or not glines:
        return False
    pairs = sorted(((_sim(v, g.text()), gi, vi) for gi, g in enumerate(glines) for vi, v in enumerate(vlm)),
                   reverse=True)
    got: dict[int, int] = {}
    used_v: set[int] = set()
    for s, gi, vi in pairs:
        if s < 50:
            break
        if gi in got or vi in used_v:
            continue
        if len(glines[gi].segs) > 1 and s < 75 and not _split_to_segs(vlm[vi], glines[gi].segs):
            continue              # tab columns: matched column by column below instead
        got[gi] = vi
        used_v.add(vi)
    if len(got) < max(1, 0.4 * len(glines)):
        return False
    texts = {gi: vlm[vi] for gi, vi in got.items()}
    # a cell the VLM split into pieces ("- ความถูกต้อง..." / "☐ ถูกต้อง ☐ ไม่ถูกต้อง"):
    # a piece that won no line of its own joins the line that printed it
    for vi, v in enumerate(vlm):
        if vi in used_v or not match_key(v):
            continue
        cands = [(fuzz.partial_ratio(match_key(v), match_key(glines[gi].text())), gi) for gi in texts]
        cands = [c for c in cands if c[0] >= 80 and _contains(texts[c[1]], v) < 80]
        if cands:
            gi = max(cands)[1]
            kg, kv = match_key(glines[gi].text()), match_key(v)
            pos = fuzz.partial_ratio_alignment(kv, kg)
            at_end = pos is not None and pos.dest_start > len(kg) / 3
            texts[gi] = (texts[gi] + " " + v) if at_end else (v + " " + texts[gi])
            used_v.add(vi)
    # one VLM line often covers two printed lines of a cell: split it between them
    for c in t.cells:
        cl = sorted((gi for gi, g in enumerate(glines) if any(g is l for l in c.lines)),
                    key=lambda gi: glines[gi].baseline)
        for a, b in zip(cl, cl[1:]):
            if a in texts and b not in texts and _contains(texts[a], glines[b].text()) >= 75:
                parts = split_over(texts[a], [glines[a].text(), glines[b].text()])
                if parts:
                    texts[a], texts[b] = parts
    # tab-column lines ("ตำแหน่ง <tab> ผกก.ปค.") that no VLM line fits as a whole:
    # match each column against the VLM's words (the VLM often reads such blocks column-wise)
    words = [w for v in vlm for w in v.split()]
    chunks = {" ".join(words[i:i + n]) for n in (1, 2, 3, 4) for i in range(len(words) - n + 1)}
    for gi, ln in enumerate(glines):
        if gi in texts or len(ln.segs) < 2:
            continue
        cols = []
        for sg in ln.segs:
            best = max(chunks, key=lambda ch: (_sim(ch, sg.text), -abs(len(ch) - len(sg.text))), default="")
            cols.append(best if best and len(match_key(sg.text)) >= 2 and _sim(best, sg.text) >= 70 else sg.text)
        ln.segs = [Seg(x, g.x0, g.x1) for x, g in zip(cols, ln.segs)]
        texts[gi] = "\t".join(cols)
    from .thai_text import is_thai
    for gi, ln in enumerate(glines):
        if gi not in texts and sum(is_thai(ch) or ch.isalpha() for ch in ln.text()) < 6:
            ln._weak = True       # leftover scribble (handwriting residue), not printed text
    for gi, text in texts.items():
        ln = glines[gi]
        if len(ln.segs) > 1:
            parts = _split_to_segs(text, ln.segs)
            if parts:
                ln.segs = [Seg(x, g.x0, g.x1) for x, g in zip(parts, ln.segs)]
                text = "\t".join(parts)
        ln.text_override = text
    return True


_THAI_MARKS = set("\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e"
                  "\u0e30\u0e32\u0e33\u0e45")


def split_over(text: str, refs: list[str]) -> list[str] | None:
    """Split text into len(refs) pieces, each as close as possible to its
    reference (what Tesseract read on that printed line). Thai is often
    written without spaces, so any character boundary may be a cut (spaces
    preferred); never inside a syllable's marks."""
    out, rest = [], text
    for k in range(len(refs) - 1):
        tail = " ".join(refs[k + 1:])
        cuts = [i for i in range(1, len(rest)) if rest[i] not in _THAI_MARKS and rest[i - 1] not in "\u0e40\u0e41\u0e42\u0e43\u0e44"]
        if not cuts:
            return None
        cut = max(cuts, key=lambda i: _sim(rest[:i], refs[k]) + _sim(rest[i:], tail)
                  + (3 if rest[i] == " " or rest[i - 1] == " " else 0))
        out.append(rest[:cut].strip())
        rest = rest[cut:]
    out.append(rest.strip())
    if not all(out) or any(_sim(o, r) < 45 for o, r in zip(out, refs)):
        return None
    return out


def _synthetic(groups, pitch: float) -> Line:
    """An invisible geometry line one pitch below the previous group."""
    from .model import Word
    if groups:
        prev = groups[-1][1][-1]
        base, x0 = prev.baseline + pitch, groups[-1][1][0].x0
    else:
        base, x0 = pitch * 2, int(pitch * 3)
    ln = Line([Word("", x0, int(base - pitch * 0.6), x0 + 10, int(base))], baseline=int(base))
    ln.segs = []
    ln._synthetic = True   # type: ignore[attr-defined]
    return ln
