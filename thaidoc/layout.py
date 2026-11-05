"""Turn measured lines into paragraphs (wrapped lines merged, columns as tabs,
underlines mapped to character ranges) and attach marks to paragraphs."""
from __future__ import annotations

import re

from .model import Cell, Line, Mark, Page, Para, Rule, Seg

ASC = 0.78   # where the baseline sits inside a line of exact height `pitch` (Word)


def line_text(ln: Line) -> str:
    if ln.text_override is not None:
        return ln.text_override
    return ln.text()


def _para_from(group: list[Line], page: Page, text: str | None = None) -> Para:
    first = group[0]
    txt = text if text is not None else " ".join(line_text(l) for l in group)
    p = Para(lines=group, text=txt, size_pt=first.size_pt or page.body_pt, bold=first.bold)
    if first.segs and len(first.segs) > 1 and "\t" in txt:
        p.tabs = [s.x0 for s in first.segs[1:]]
    p.pitch_px = max(page.pitch_px, (p.size_pt * 1.25) / 72 * page.dpi)
    _map_underlines(p)
    return p


def _map_underlines(p: Para):
    """Underline / bold x-ranges on each geometry line -> character ranges in p.text."""
    p.underline_ranges = _map_ranges(p, "underlines")
    if not p.bold:
        p.bold_ranges = _map_ranges(p, "bold_ranges")


def _map_ranges(p: Para, attr: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not any(getattr(l, attr) for l in p.lines):
        return out
    text = p.text
    ln0 = p.lines[0]
    parts = text.split("\t")
    if len(p.lines) == 1 and len(ln0.segs) > 1 and len(parts) == len(ln0.segs):
        starts = []
        acc = 0
        for part in parts:
            starts.append(acc); acc += len(part) + 1
        for (ux0, ux1) in getattr(ln0, attr):
            for sg, part, st in zip(ln0.segs, parts, starts):
                lo, hi = max(ux0, sg.x0), min(ux1, sg.x1)
                if hi - lo > 4:
                    span = max(1, sg.x1 - sg.x0)
                    a = st + int(round((lo - sg.x0) / span * len(part)))
                    b = st + int(round((hi - sg.x0) / span * len(part)))
                    a, b = _snap(text, a, b)
                    if a == st and lo - sg.x0 > len(part) and " " in part:   # underline starts after "๘. "
                        a = st + part.index(" ") + 1
                    if b > a:
                        out.append((a, b))
        return out
    n = len(text)
    if n == 0:
        return out
    widths = [max(1, l.x1 - l.x0) for l in p.lines]
    tot = sum(widths)
    starts, acc = [], 0
    for w in widths:
        starts.append(int(round(acc / tot * n)))
        acc += w
    starts.append(n)
    for li, ln in enumerate(p.lines):
        c0, c1 = starts[li], starts[li + 1]
        span = max(1, ln.x1 - ln.x0)
        for (ux0, ux1) in getattr(ln, attr):
            a = c0 + int(round((ux0 - ln.x0) / span * (c1 - c0)))
            b = c0 + int(round((ux1 - ln.x0) / span * (c1 - c0)))
            a, b = _snap(text, a, b)
            if b > a:
                out.append((a, b))
    return out


def _snap(text: str, a: int, b: int) -> tuple[int, int]:
    """Snap an approximate range to nearby word boundaries (space, colon, tab)."""
    bounds = [0, len(text)] + [m.start() for m in re.finditer(r"[ \t:]", text)] + \
             [m.end() for m in re.finditer(r"[ \t:]", text)]
    rng = max(3, (b - a) // 4)
    near_a = [x for x in bounds if abs(x - a) <= rng]
    near_b = [x for x in bounds if abs(x - b) <= rng]
    if near_a:
        a = min(near_a, key=lambda x: abs(x - a))
    if near_b:
        b = min(near_b, key=lambda x: abs(x - b))
    # don't underline leading/trailing spaces or a trailing " :"
    while a < b and text[a] in " \t":
        a += 1
    while b > a and text[b - 1] in " \t:":
        b -= 1
    return a, b


def merge_wrapped(lines: list[Line], page: Page, right_edge: int) -> list[list[Line]]:
    """Group consecutive lines that are one wrapped paragraph (no VLM grouping)."""
    groups: list[list[Line]] = []
    em = page.body_pt / 72 * page.dpi
    for ln in lines:
        if groups:
            prev = groups[-1][-1]
            gap = ln.baseline - prev.baseline
            full = prev.x1 > right_edge - 1.6 * em
            same_style = abs(ln.size_pt - prev.size_pt) < 1 and ln.bold == prev.bold
            single = len(ln.segs) <= 1 and len(prev.segs) <= 1
            indent_ok = ln.x0 <= groups[-1][0].x0 + 0.6 * em
            if full and same_style and single and indent_ok and gap < page.pitch_px * 1.35:
                groups[-1].append(ln)
                continue
        groups.append([ln])
    return groups


def build_paras(page: Page, groups: list[tuple[str | None, list[Line]]], margin_l: int, margin_r_x: int):
    W = page.width_px
    em = page.body_pt / 72 * page.dpi
    paras = [_para_from(g, page, t) for t, g in groups]
    paras.sort(key=lambda p: p.baseline)
    for i, p in enumerate(paras):
        first = p.lines[0]
        cx = (first.x0 + first.x1) / 2
        centred = len(p.lines) == 1 and abs(cx - W / 2) < W * 0.03 and first.x0 > margin_l + 2 * em \
            and len(first.segs) <= 1
        if centred:
            # a line in a left-aligned block (e.g. a signature block) is not centred
            for q in paras[max(0, i - 2):i] + paras[i + 1:i + 3]:
                if abs(q.baseline - p.baseline) < p.pitch_px * 2.5 and abs(q.x0 - first.x0) < 0.5 * em:
                    centred = False
        if getattr(first, "_page_number", False):
            centred = True
        if centred:
            p.align = "center"
        elif len(p.lines) > 1 and _centred_block(p, W, margin_l, margin_r_x, em):
            # e.g. a two-line title: keep it centred, break where the source broke
            p.align = "center"
            p.text = _break_lines(p)
        elif len(p.lines) > 1:
            p.align = "justify"
    # average printed width of one character on this page (for wrap estimates)
    ratios = [(p.lines[0].x1 - p.lines[0].x0) / len(p.text) for p in paras
              if len(p.lines) == 1 and len(p.text) > 15 and "\t" not in p.text]
    page._char_w = sorted(ratios)[len(ratios) // 2] if ratios else page.dpi * 0.07  # type: ignore
    page._right = margin_r_x   # type: ignore[attr-defined]
    # free-standing rules become empty paragraphs with a bottom border
    for r in page.rules:
        rp = Para(lines=[], text="", size_pt=4, rule_below=True)
        rp._rule = r   # type: ignore[attr-defined]
        paras.append(rp)
    paras.sort(key=lambda p: p.baseline if p.lines else p._rule.y)  # type: ignore[attr-defined]
    page.paras = paras


def _centred_block(p: Para, W: int, margin_l: int, margin_r_x: int, em: float) -> bool:
    """Every line centred on the page and none of them a full-width wrapped line."""
    if "\t" in p.text:
        return False
    for ln in p.lines:
        if abs(ln.cx - W / 2) > W * 0.03 or ln.x0 < margin_l + 2 * em or ln.x1 > margin_r_x - 2 * em:
            return False
    return True


def _break_lines(p: Para) -> str:
    """Put line breaks ("\n") where the source lines ended."""
    from .align import split_over
    parts = split_over(p.text, [ln.text() for ln in p.lines])
    if not parts:
        return p.text
    # the separators are single spaces: replace them by \n so lengths don't change
    chars, pos = list(p.text), 0
    for part in parts[:-1]:
        i = p.text.index(part, pos) + len(part)
        if i < len(chars) and chars[i] == " ":
            chars[i] = "\n"
        pos = i
    return "".join(chars)


def cell_paras(cell: Cell, page: Page):
    lines = sorted(cell.lines, key=lambda l: l.baseline)
    if cell.override_lines is not None:
        strong = [l for l in lines if not getattr(l, "_weak", False)]
        if len(strong) >= len(cell.override_lines):
            lines = strong
    else:
        lines = [l for l in lines if not getattr(l, "_weak", False)]
    if cell.override_lines is not None and lines:
        texts = [t for t in cell.override_lines]
        # pair VLM lines with geometry lines in order; extra VLM lines join the last one
        for i, ln in enumerate(lines):
            if i < len(texts):
                ln.text_override = texts[i] if i < len(lines) - 1 else " ".join(texts[i:])
            else:
                ln.text_override = ""
        lines = [l for l in lines if (l.text_override or "").strip()]
        # keep printed gaps (e.g. "วันที่ ____ พ.ย.") as tab columns
        from .align import _split_to_segs
        for ln in lines:
            if len(ln.segs) > 1 and "\t" not in (ln.text_override or ""):
                parts = _split_to_segs(ln.text_override, ln.segs)
                if parts:
                    ln.segs = [Seg(t, g.x0, g.x1) for t, g in zip(parts, ln.segs)]
                    ln.text_override = "\t".join(parts)
    groups = [[l] for l in lines]
    cell.paras = [_para_from(g, page) for g in groups]


# ----------------------------------------------------------------------- marks
def anchor_marks(page: Page):
    """Attach each mark to the paragraph whose first line is nearest above it."""
    def candidates(m: Mark):
        cx, cy = (m.x0 + m.x1) / 2, m.y0 + (m.y1 - m.y0) * 0.75
        for t in page.tables:
            for c in t.cells:
                if c.x0 <= cx <= c.x1 and c.y0 <= cy <= c.y1 and c.paras:
                    return c.paras
        return [p for p in page.paras if p.lines]

    for m in page.marks:
        paras = candidates(m)
        if not paras:
            continue
        # anchor to the line at/above the picture's TOP so its offset is never negative:
        # LibreOffice, Google Docs and phone viewers clamp negative line offsets to 0
        cy = m.y0 + 2

        def top(p: Para):
            return p.baseline - ASC * p.pitch_px

        above = [p for p in paras if top(p) <= cy]
        target = max(above, key=top) if above else min(paras, key=lambda p: abs(top(p) - cy))
        target.anchors.append(m)


def rendered_lines(p: Para, page: Page, right_x: float | None = None) -> int:
    """How many lines Word will need for this paragraph (at least the source count)."""
    if not p.lines:
        return 1
    n = len(p.lines)
    if "\t" in p.text:
        return n
    cw = getattr(page, "_char_w", page.dpi * 0.07)
    right = right_x if right_x is not None else getattr(page, "_right", page.width_px * 0.9)
    avail = max(page.dpi * 0.5, right - min(l.x0 for l in p.lines))
    est = int(len(p.text) * cw * 1.02 / avail) + 1
    return max(n, est)
