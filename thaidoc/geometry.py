"""Page geometry: ruled tables, horizontal rules, underlines, checkboxes and
text-line positions (via Tesseract word boxes).

Tesseract is used here for *where* text is. What the text says can come from a
better Thai engine (see ocr.py) and is aligned onto these lines afterwards.
"""
from __future__ import annotations

import statistics

import cv2
import numpy as np
import pytesseract

from .model import Cell, Line, Rule, Seg, Table, Word
from .thai_text import normalize


# --------------------------------------------------------------------------- lines
def _binary(rgb: np.ndarray, thr: int = 150) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (g < thr).astype(np.uint8)


def _ridge_points(patch: np.ndarray, x_off: int, xh: float):
    """(x positions, stroke thickness / x-height) of the strokes' centre-line pixels."""
    if patch.size == 0 or patch.sum() < 20:
        return None
    p = np.pad(patch, 1)
    dt = cv2.distanceTransform(p, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (p > 0)
    ys, xs = np.nonzero(ridge)
    return xs - 1 + x_off, 2 * dt[ys, xs] / max(1.0, xh)


def _bold_runs(ridge, med_w: float, xh: float) -> tuple[list[tuple[int, int]], int]:
    """(bold x-ranges, total width of windows that are clearly regular weight)."""
    if ridge is None or not med_w or ridge[0].size == 0:
        return [], 0
    xs, ws = ridge
    order = np.argsort(xs)
    xs, ws = xs[order], ws[order]
    win, step = max(4, int(xh * 1.6)), max(2, int(xh * 0.5))
    runs: list[list[int]] = []
    regular = 0
    for a in range(int(xs[0]), int(xs[-1]) + 1, step):
        lo, hi = np.searchsorted(xs, a), np.searchsorted(xs, a + win)
        last = a + win >= xs[-1]
        if hi - lo < win * 0.6 and not (last and hi - lo >= win * 0.25):
            continue                         # too little ink in this window to judge
        mw = float(np.median(ws[lo:hi]))
        if mw < med_w * 1.1:
            regular += step
        if mw > med_w * 1.25:
            s0, s1 = int(xs[lo]), int(xs[hi - 1]) if not last else int(xs[-1])
            if runs and s0 - runs[-1][1] < xh:
                runs[-1][1] = max(runs[-1][1], s1)
            else:
                runs.append([s0, s1])
    return [(a, b) for a, b in runs if b - a > xh * 2], regular


def _stroke_width(patch: np.ndarray) -> float | None:
    """Typical stroke thickness in px: twice the distance-to-edge along the
    strokes' centre lines. Independent of letter shapes (dense Thai digits are
    not "bold"), unlike plain ink density."""
    if patch.size == 0 or patch.sum() < 20:
        return None
    p = np.pad(patch, 1)
    dt = cv2.distanceTransform(p, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (p > 0)
    vals = dt[ridge]
    return float(2 * np.median(vals)) if vals.size else None


def detect_rules(rgb: np.ndarray, dpi: int):
    """Return (h_mask, v_mask) of long straight lines (scanned rules are often pale)."""
    b = _binary(rgb, 205)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(dpi * 0.45)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(dpi * 0.3))))
    h = cv2.morphologyEx(b, cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(b, cv2.MORPH_OPEN, vk)
    # bridge small breaks from faint scanning
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(dpi * 0.2), 1)))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(dpi * 0.2))))
    h = cv2.dilate(h, np.ones((3, 3), np.uint8))
    v = cv2.dilate(v, np.ones((3, 3), np.uint8))
    return h.astype(bool), v.astype(bool)


def _cluster(vals: list[int], tol: int) -> list[int]:
    vals = sorted(vals)
    out: list[list[int]] = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(statistics.median(g)) for g in out]


def detect_tables(rgb: np.ndarray, dpi: int) -> tuple[list[Table], list[Rule], np.ndarray]:
    h, v = detect_rules(rgb, dpi)
    grid = cv2.dilate((h | v).astype(np.uint8), np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, 8)
    tables: list[Table] = []
    used = np.zeros_like(h)
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if w < dpi * 1.0 or hh < dpi * 0.3:
            continue
        x, y, w, hh = x + 4, y + 4, w - 8, hh - 8   # undo the dilation margin
        sub_v = v[y:y + hh, x:x + w] & (lab[y:y + hh, x:x + w] == i)
        sub_h = h[y:y + hh, x:x + w] & (lab[y:y + hh, x:x + w] == i)
        if sub_v.sum() < hh or sub_h.sum() < w:      # needs both directions
            continue
        cols = np.nonzero(sub_v.sum(0) > hh * 0.15)[0]
        rows = np.nonzero(sub_h.sum(1) > w * 0.15)[0]
        col_x = _cluster([x + c for c in cols], 6)
        row_y = _cluster([y + r for r in rows], 6)
        if len(col_x) < 2 or len(row_y) < 2:
            continue
        t = Table(x, y, x + w, y + hh, col_x, row_y)
        t.cells = _cells(t, h, v)
        tables.append(t)
        used[y:y + hh, x:x + w] = True

    # free-standing horizontal rules (e.g. under a title)
    rules: list[Rule] = []
    hn, hlab, hstats, _ = cv2.connectedComponentsWithStats((h & ~used).astype(np.uint8), 8)
    W = rgb.shape[1]
    mask = (h | v) & used                      # table borders only
    for i in range(1, hn):
        x, y, w, hh, _ = hstats[i]
        if w > W * 0.3 and hh < dpi * 0.05:
            rules.append(Rule(int(x), int(x + w), int(y + hh // 2)))
            mask[y:y + hh, x:x + w] |= (hlab[y:y + hh, x:x + w] == i)
    return tables, rules, mask


def _cells(t: Table, h: np.ndarray, v: np.ndarray) -> list[Cell]:
    R, C = len(t.row_y) - 1, len(t.col_x) - 1
    parent = {(r, c): (r, c) for r in range(R) for c in range(C)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for r in range(R):
        y0, y1 = t.row_y[r] + 4, t.row_y[r + 1] - 4
        for c in range(C - 1):
            x = t.col_x[c + 1]
            seg = v[y0:y1, max(0, x - 3):x + 4].any(1)
            if seg.mean() < 0.5:
                union((r, c), (r, c + 1))
    for c in range(C):
        x0, x1 = t.col_x[c] + 4, t.col_x[c + 1] - 4
        for r in range(R - 1):
            y = t.row_y[r + 1]
            seg = h[max(0, y - 3):y + 4, x0:x1].any(0)
            if seg.mean() < 0.5:
                union((r, c), (r + 1, c))
    groups: dict = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    cells = []
    for members in groups.values():
        rs = [m[0] for m in members]
        cs = [m[1] for m in members]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cells.append(Cell(r0, c0, t.col_x[c0], t.row_y[r0], t.col_x[c1 + 1], t.row_y[r1 + 1],
                          rowspan=r1 - r0 + 1, colspan=c1 - c0 + 1))
    cells.sort(key=lambda c: (c.r, c.c))
    return cells


def detect_checkboxes(rgb: np.ndarray, dpi: int) -> list[tuple[int, int, int, int]]:
    b = _binary(rgb) * 255
    cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    lo, hi = dpi * 0.07, dpi * 0.22
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (lo < w < hi and lo < h < hi and 0.75 < w / h < 1.33):
            continue
        approx = cv2.approxPolyDP(c, 0.08 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        inner = b[y + 4:y + h - 4, x + 4:x + w - 4]
        if inner.size and (inner > 0).mean() < 0.25:      # hollow
            out.append((x, y, x + w, y + h))
    # de-duplicate inner/outer contours of the same box
    out.sort()
    ded = []
    for bx in out:
        if not any(abs(bx[0] - d[0]) < 6 and abs(bx[1] - d[1]) < 6 for d in ded):
            ded.append(bx)
    return ded


def find_underlines(rgb: np.ndarray, dpi: int, line_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Short thin horizontal strokes (x0, x1, y) — candidates for text underlines."""
    b = _binary(rgb)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(dpi * 0.14)), 1))
    hz = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    hz[line_mask] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(hz, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if h <= max(3, dpi * 0.03):
            out.append((int(x), int(x + w), int(y + h / 2)))
    return out


def strip_strokes(rgb: np.ndarray, dpi: int, line_mask: np.ndarray, unders, boxes=()) -> np.ndarray:
    """Copy of the page with ruled lines, underlines (and checkboxes) painted white."""
    out = rgb.copy()
    out[line_mask] = 255
    for (x0, y0, x1, y1) in boxes:
        out[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = 255
    t = max(2, int(dpi * 0.012))
    for (x0, x1, y) in unders:
        out[y - t:y + t + 1, x0:x1] = 255
    return out


# ------------------------------------------------------------------ OCR geometry
def tesseract_lines(rgb: np.ndarray, dpi: int, lang: str = "tha+eng") -> list[Line]:
    data = pytesseract.image_to_data(rgb, lang=lang, config="--psm 3 --oem 1",
                                     output_type=pytesseract.Output.DICT)
    groups: dict = {}
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        conf = float(data["conf"][i])
        if not txt or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if h > dpi * 0.6 or w > rgb.shape[1] * 0.95:
            continue
        groups.setdefault(key, []).append(Word(normalize(txt), x, y, x + w, y + h, conf))
    lines = [Line(sorted(ws, key=lambda w: w.x0)) for ws in groups.values() if ws]
    return [ln for ln in lines if _plausible(ln, dpi)]


def _plausible(ln: Line, dpi: int) -> bool:
    txt = "".join(w.text for w in ln.words)
    if not txt.strip():
        return False
    # lone punctuation / speckles
    if len(txt) <= 1 and ln.x1 - ln.x0 < dpi * 0.08:
        return False
    if sum(w.conf for w in ln.words) / len(ln.words) < 20 and len(txt) < 4:
        return False
    return True


def merge_same_row(lines: list[Line]) -> list[Line]:
    """Tesseract sometimes puts the columns of one printed row into separate lines."""
    lines = sorted(lines, key=lambda l: (l.cy, l.x0))
    out: list[Line] = []
    for ln in lines:
        for o in out:
            ov = min(o.y1, ln.y1) - max(o.y0, ln.y0)
            hmin = min(o.y1 - o.y0, ln.y1 - ln.y0)
            same_base = o.baseline and ln.baseline and abs(o.baseline - ln.baseline) <= max(2, 0.12 * hmin)
            # overlap with the other line's *words* (its bbox may span gaps between columns)
            hov = sum(max(0, min(w.x1, ln.x1) - max(w.x0, ln.x0)) for w in o.words)
            apart = hov < 0.2 * max(1, ln.x1 - ln.x0)
            if (ov > 0.6 * hmin or same_base) and apart:
                o.words = sorted(o.words + ln.words, key=lambda w: w.x0)
                break
        else:
            out.append(ln)
    return sorted(out, key=lambda l: l.y0)


def drop_junk(lines: list[Line], dpi: int, return_weak: bool = False):
    """Remove OCR noise lines (speckles, stroke fragments read as Latin junk)."""
    from .thai_text import thai_ratio
    if not lines:
        return lines
    hs = sorted(l.y1 - l.y0 for l in lines)
    med_h = hs[len(hs) // 2]
    page_thai = thai_ratio("".join(w.text for l in lines for w in l.words)) > 0.5
    keep, weak = [], []
    for l in lines:
        txt = "".join(w.text for w in l.words)
        conf = sum(w.conf for w in l.words) / len(l.words)
        letters = [c for c in txt if c.isalpha()]
        if l.y1 - l.y0 > med_h * 2.6:
            continue
        if (l.y1 - l.y0 < med_h * 0.55 and conf < 70) or \
                (page_thai and letters and thai_ratio(txt) < 0.35 and conf < 75 and len(txt) < 25):
            if l.y1 - l.y0 >= med_h * 0.55:
                weak.append(l)          # unreadable but real-sized: a VLM may read it
            continue
        keep.append(l)
    keep = _drop_overlapping(keep)
    weak = [w for w in _drop_overlapping(weak + keep) if w in weak]
    return (keep, weak) if return_weak else keep


def _drop_overlapping(lines: list[Line]) -> list[Line]:
    """Two OCR lines on top of each other: keep the more confident one."""
    def conf(l):
        return sum(w.conf for w in l.words) / len(l.words)
    out = sorted(lines, key=conf, reverse=True)
    kept: list[Line] = []
    for l in out:
        clash = False
        for k in kept:
            vo = min(l.y1, k.y1) - max(l.y0, k.y0)
            ho = min(l.x1, k.x1) - max(l.x0, k.x0)
            if vo > 0.5 * min(l.y1 - l.y0, k.y1 - k.y0) and ho > 0.3 * min(l.x1 - l.x0, k.x1 - k.x0):
                clash = True
                break
        if not clash:
            kept.append(l)
    return sorted(kept, key=lambda l: l.y0)


def measure_lines(lines: list[Line], rgb: np.ndarray, dpi: int) -> tuple[float, float]:
    """Fill baseline/size/bold/segments on each line; return (pitch_px, body_pt)."""
    b = _binary(rgb)
    xh_list, dens = [], []
    for ln in lines:
        y0, y1, x0, x1 = ln.y0, ln.y1, ln.x0, ln.x1
        prof = b[y0:y1, x0:x1].sum(1).astype(float)
        if prof.size == 0 or prof.max() == 0:
            ln.baseline = y1
            continue
        thr = prof.max() * 0.45
        idx = np.nonzero(prof >= thr)[0]
        # main band = the longest run of dense rows
        runs, start = [], idx[0]
        for a, c in zip(idx, idx[1:]):
            if c != a + 1:
                runs.append((start, a)); start = c
        runs.append((start, idx[-1]))
        top, bot = max(runs, key=lambda r: r[1] - r[0])
        ln.baseline = y0 + int(bot)
        xh = max(4, bot - top + 1)
        xh_list.append(xh)
        band = b[y0 + top:y0 + bot + 1, x0:x1]
        dens.append(band.mean() if band.size else 0)
        ln._ridge = (y0, y1, x0, x1)   # type: ignore[attr-defined]   # measured once med_xh is known
        ln._xh = xh            # type: ignore[attr-defined]
        ln._dens = dens[-1]    # type: ignore[attr-defined]
    med_xh = statistics.median(xh_list) if xh_list else dpi * 0.1
    med_d = statistics.median(dens) if dens else 0.2
    for ln in lines:
        box = getattr(ln, "_ridge", None)
        if box is not None:
            xh_l = getattr(ln, "_xh", med_xh)
            # small measured x-heights are mostly measurement noise: only bigger text
            # (titles) is judged against its own size
            xh_n = xh_l if xh_l / med_xh > 1.25 else med_xh
            ln._xhn = xh_n   # type: ignore[attr-defined]
            ln._ridge = _ridge_points(b[box[0]:box[1], box[2]:box[3]], box[2], xh_n)   # type: ignore[attr-defined]
    allw = [r[1] for ln in lines if getattr(ln, "_ridge", None) is not None for r in [ln._ridge] if r[1].size]
    med_w = float(np.median(np.concatenate(allw))) if allw else 0.0
    bases = sorted(ln.baseline for ln in lines)
    diffs = [b2 - b1 for b1, b2 in zip(bases, bases[1:]) if dpi * 0.15 < b2 - b1 < dpi * 0.5]
    pitch = statistics.median(diffs) if diffs else med_xh * 2.9
    body_pt = round((pitch / dpi * 72) / 1.31 * 2) / 2
    body_pt = min(22.0, max(10.0, body_pt))
    gap_tab = dpi * 0.2
    H, W = b.shape
    for ln in lines:
        # true horizontal extent of the printed line (OCR may drop a leading "๒.")
        y0, y1 = max(0, ln.baseline - int(med_xh * 1.6)), min(H, ln.baseline + int(med_xh * 0.6))
        cols = b[y0:y1].sum(0) > 0
        x = ln.x0
        reach = int(dpi * 0.22)
        while x > 0 and cols[max(0, x - reach):x].any():
            nz = np.nonzero(cols[max(0, x - reach):x])[0]
            x = max(0, x - reach) + int(nz[0])
        if ln.x0 - x < dpi * 0.8:
            ln.ink_x0 = int(x)
        xh = getattr(ln, "_xh", med_xh)
        ratio = xh / med_xh
        letters = sum(1 for w in ln.words for c in w.text if c.isalpha() or "\u0e01" <= c <= "\u0e2e")
        if 0.85 < ratio < 1.15 or (letters < 4 and ratio < 1) or ratio < 0.72 or \
                (ratio < 1 and letters < 25):
            ln.size_pt = body_pt
        else:
            ln.size_pt = round(body_pt * ratio * 2) / 2
        # bold = thicker strokes than the page's text, measured in a window sliding
        # along the line (Thai has no spaces, so word boxes don't show where bold ends)
        ranges, regular = _bold_runs(getattr(ln, "_ridge", None), med_w, getattr(ln, "_xhn", med_xh))
        span = max(1, ln.x1 - ln.x0)
        covered = sum(b1 - a1 for a1, b1 in ranges)
        rd = getattr(ln, "_ridge", None)
        line_med = float(np.median(rd[1])) if rd is not None and rd[1].size else 0.0
        ln.bold = covered > 0.75 * span or \
            (med_w > 0 and line_med > med_w * 1.3 and regular < 0.15 * span)
        ln.bold_ranges = [] if ln.bold else ranges
        segs: list[list[Word]] = [[ln.words[0]]]
        for prev, w in zip(ln.words, ln.words[1:]):
            blank = w.x0 - prev.x1 > gap_tab and \
                cols[prev.x1 + 2:w.x0 - 2].sum() < 3      # nothing printed in the gap
            if blank:
                segs.append([w])
            else:
                segs[-1].append(w)
        from .model import join_words
        ln.segs = [Seg(join_words(s), s[0].x0, s[-1].x1) for s in segs]
    return pitch, body_pt


def attach_underlines(lines: list[Line], unders: list[tuple[int, int, int]], pitch: float):
    for (x0, x1, y) in unders:
        best, bd = None, 1e9
        for ln in lines:
            d = y - ln.baseline
            if -2 <= d <= pitch * 0.32 and x0 >= ln.x0 - 10 and x1 <= ln.x1 + 30:
                if d < bd:
                    best, bd = ln, d
        if best is not None:
            best.underlines.append((x0, x1))


def insert_checkbox_words(lines: list[Line], boxes, pitch: float):
    for (x0, y0, x1, y1) in boxes:
        cy = (y0 + y1) / 2
        for ln in lines:
            if ln.y0 - pitch * 0.3 <= cy <= ln.baseline + 4 and ln.x0 - pitch * 3 <= x0 <= ln.x1 + pitch * 3:
                ln.words.append(Word("☐", x0, y0, x1, y1, 99))
                ln.words.sort(key=lambda w: w.x0)
                break
