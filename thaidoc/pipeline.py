"""End-to-end conversion: PDF -> Document model -> .docx bytes."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

from . import align, fontembed, geometry, ink, layout
from .docx_writer import Writer
from .model import Document, Line, Page, Rule, Seg, join_words
from .ocr import TyphoonEngine, parse_units
from .pdf_input import open_pdf

Progress = Callable[[float, str], None]


@dataclass
class Options:
    engine: str = "auto"            # auto | typhoon | tesseract
    typhoon_api_key: str | None = None
    typhoon_base_url: str | None = None
    typhoon_model: str | None = None
    dpi: int = 250
    keep_ink: bool = True
    ink: ink.InkOptions = field(default_factory=ink.InkOptions)
    font: str = "auto"              # auto = the PDF's font via its portable twin (every device)
                                    # auto-exact = the PDF's font itself (Word/LibreOffice)
                                    # or a font name
    digits: str = "keep"            # keep | thai | arabic
    tess_lang: str = "tha+eng"
    max_pages: int = 60


def analyse(pdf_path: str, opts: Options, progress: Progress = lambda f, m: None,
            on_page_image: Callable | None = None) -> Document:
    engine = None
    if opts.engine in ("auto", "typhoon"):
        if opts.typhoon_base_url or TyphoonEngine.available(opts.typhoon_api_key):
            engine = TyphoonEngine(opts.typhoon_api_key, opts.typhoon_base_url, opts.typhoon_model)
        elif opts.engine == "typhoon":
            raise RuntimeError("Typhoon engine selected but no API key / base URL configured")
    pages: list[Page] = []
    pdf_fonts: dict[str, int] = {}
    raws = list(open_pdf(pdf_path, opts.dpi, opts.max_pages))
    total = len(raws)
    doc = Document(pages=pages, font=opts.font)
    doc.engine = engine.name if engine else "tesseract"
    for raw in raws:
        base = raw.index / total
        step = 1 / total
        say = lambda f, m: progress(base + step * f, f"Page {raw.index + 1}/{total}: {m}")
        say(0.02, "finding pen ink, stamps and photos")
        h, w = raw.rgb.shape[:2]
        page = Page(raw.index, w, h, raw.dpi, raw.kind)
        for k, v in (raw.fonts or {}).items():
            pdf_fonts[k] = pdf_fonts.get(k, 0) + v
        if on_page_image:
            on_page_image(raw.index, raw.rgb)
        regions = raw.image_boxes if raw.kind == "digital" else None
        marks, clean = ([], raw.rgb)
        if opts.keep_ink:
            marks, clean = ink.extract(raw.rgb, raw.dpi, opts.ink, regions, f"p{raw.index + 1}")
        page.marks = marks

        say(0.15, "detecting tables and lines")
        tables, rules, line_mask = geometry.detect_tables(clean, raw.dpi)
        unders = geometry.find_underlines(clean, raw.dpi, line_mask)
        boxes = geometry.detect_checkboxes(clean, raw.dpi)

        weak: list[Line] = []
        if raw.kind == "digital":
            lines = list(raw.text_lines)
            ocr_img = clean
        else:
            say(0.25, "locating text lines")
            ocr_img = geometry.strip_strokes(clean, raw.dpi, line_mask, unders,
                                             boxes if engine is None else ())
            lines = geometry.tesseract_lines(ocr_img, raw.dpi, opts.tess_lang)
            lines, weak = geometry.drop_junk(lines, raw.dpi, return_weak=True)
            if engine is None:
                weak = []
        for wl in weak:
            wl._weak = True   # type: ignore[attr-defined]
        # split by table cells first, then merge printed rows that Tesseract split
        page.tables = tables
        body = geometry.merge_same_row(_split_into_cells(lines + weak, tables))
        for t in tables:
            for c in t.cells:
                c.lines = geometry.merge_same_row(c.lines)
        all_lines = body + [l for t in tables for c in t.cells for l in c.lines]
        if raw.kind == "digital":
            _digital_measure(all_lines, page)
        else:
            strong = [l for l in all_lines if not getattr(l, "_weak", False)]
            page.pitch_px, page.body_pt = geometry.measure_lines(strong, ocr_img, raw.dpi)
            wk = [l for l in all_lines if getattr(l, "_weak", False)]
            if wk:
                geometry.measure_lines(wk, ocr_img, raw.dpi)
                for wl in wk:
                    wl.size_pt = page.body_pt
            if engine is None:
                geometry.insert_checkbox_words(all_lines, boxes, page.pitch_px)
        if not page.pitch_px:
            page.pitch_px = raw.dpi * 0.29

        # rules sitting right under text are underlines, not separators
        for r in list(rules):
            for ln in body:
                if 0 <= r.y - ln.baseline <= page.pitch_px * 0.35 and r.x0 >= ln.x0 - 20:
                    ln.underlines.append((r.x0, min(r.x1, ln.x1 + 10)))
                    rules.remove(r)
                    break
        geometry.attach_underlines(all_lines, unders, page.pitch_px)
        page.rules = rules

        units, html_tables = [], []
        if engine is not None and raw.kind == "scan":
            say(0.45, "reading Thai text (Typhoon OCR)")
            try:
                units, html_tables = parse_units(engine.page_markdown(clean))
            except Exception as e:  # network / quota — fall back per page
                page.warnings.append(f"Typhoon failed on page {raw.index + 1} ({e}); used Tesseract text")
                units = []
        say(0.8, "rebuilding layout")
        right_edge = max([l.x1 for l in body if not getattr(l, "_weak", False)] + [w * 0.85])
        if units:
            used_tables = _match_tables(tables, html_tables)
            body_units = [u for u in units if u.table_id not in used_tables]
            groups = []
            for u, grp in align.align_body(sorted(body, key=lambda l: l.baseline), body_units, page.pitch_px):
                if u is not None:
                    align.apply_unit(u, grp)
                    if u.role == "page_number" and getattr(grp[0], "_synthetic", False):
                        grp[0].words[0].x0 = int(w / 2 - page.pitch_px)
                        grp[0].words[0].x1 = int(w / 2 + page.pitch_px)
                        grp[0].baseline = int(page.dpi * 0.55)
                        grp[0]._page_number = True   # type: ignore[attr-defined]
                    groups.append((layout.line_text(grp[0]), grp))
                elif not getattr(grp[0], "_weak", False) and _substantial(grp[0]):
                    groups.append((None, grp))      # text the VLM didn't return
        else:
            groups = [(None, g) for g in layout.merge_wrapped(sorted(body, key=lambda l: l.baseline),
                                                              page, right_edge)]
        margin_l = min([l.x0 for l in body] + [t.x0 for t in tables] + [w])
        layout.build_paras(page, groups, margin_l, right_edge)
        for t in tables:
            for c in t.cells:
                layout.cell_paras(c, page)
        layout.anchor_marks(page)
        pages.append(page)
        say(1.0, "done")
    _choose_font(doc, opts, pdf_fonts)
    for p in pages:
        doc.warnings += p.warnings
    return doc


def render(doc: Document, opts: Options) -> bytes:
    return Writer(doc, font=doc.font or opts.font, digits=opts.digits).build()


def _choose_font(doc: Document, opts: Options, pdf_fonts: dict[str, int]):
    """Use the PDF's own font (named in digital PDFs, fitted on scans)."""
    if opts.font not in ("auto", "auto-exact"):
        doc.font = fontembed.bundled(opts.font) or opts.font
    elif pdf_fonts:
        # weight by family: "X-Bold" and "X" count together
        fam: dict[str, int] = {}
        for name, n in pdf_fonts.items():
            f = fontembed.from_pdf_font(name)
            fam[f] = fam.get(f, 0) + n
        doc.font = max(fam, key=fam.get)
        if any(_k.lower().replace(" ", "").startswith("thsarabunit") for _k in
               (n.split("+", 1)[-1] for n in pdf_fonts)):
            doc.warnings.append("The PDF uses TH SarabunIT๙; its free twin TH SarabunPSK was used "
                                "(same letters and spacing, slightly different Thai digit shapes)")
    else:
        samples, pitch_pt, dpi = [], 0.0, 250
        for pg in doc.pages:
            if pg.source != "scan":
                continue
            dpi = pg.dpi
            pitch_pt = pitch_pt or pg.pitch_px / pg.dpi * 72
            for p in pg.paras:
                if len(p.lines) == 1 and not p.bold and "\t" not in p.text and len(p.text) > 15 \
                        and abs(p.size_pt - pg.body_pt) < 0.6:
                    samples.append((p.text, p.lines[0].x1 - p.lines[0].x0))
        fam, k = fontembed.detect_scan_font(samples, pitch_pt, 1.31, dpi)
        doc.font = fam
        if abs(k - 1) > 0.03:     # sizes were estimated for Sarabun proportions
            for pg in doc.pages:
                pg.body_pt = round(pg.body_pt * k * 2) / 2
                paras = pg.paras + [q for t in pg.tables for c in t.cells for q in c.paras]
                for q in paras:
                    if q.lines:
                        q.size_pt = round(q.size_pt * k * 2) / 2
    if opts.font in ("auto", "auto-exact"):
        doc.source_font = doc.font
        if opts.font == "auto":
            doc.font, doc.font_scale = fontembed.portable(doc.font)
    else:
        # a chosen font is set at the size that matches the PDF's text width
        src = _detected_font(doc, pdf_fonts)
        doc.source_font = src
        doc.font_scale = fontembed.width_ratio(src, doc.font) if fontembed.bundled(doc.font) else 1.0
    if not fontembed.bundled(doc.font):
        doc.warnings.append(f"Font '{doc.font}' can't be embedded in the Word file; "
                            "devices without it will show a substitute font")


def _detected_font(doc: Document, pdf_fonts: dict[str, int]) -> str:
    if pdf_fonts:
        fam: dict[str, int] = {}
        for name, n in pdf_fonts.items():
            f = fontembed.from_pdf_font(name)
            fam[f] = fam.get(f, 0) + n
        return max(fam, key=fam.get)
    return fontembed.DEFAULT


def convert(pdf_path: str, opts: Options | None = None, progress: Progress = lambda f, m: None):
    opts = opts or Options()
    doc = analyse(pdf_path, opts, progress)
    return doc, render(doc, opts)


# ---------------------------------------------------------------- helpers
def _digital_measure(lines: list[Line], page: Page):
    bases = sorted(l.baseline for l in lines)
    diffs = [b - a for a, b in zip(bases, bases[1:]) if page.dpi * 0.12 < b - a < page.dpi * 0.6]
    page.pitch_px = statistics.median(diffs) if diffs else page.dpi * 0.29
    sizes = [l.size_pt for l in lines if l.size_pt]
    page.body_pt = statistics.median(sizes) if sizes else 16
    gap_tab = page.dpi * 0.2
    for ln in lines:
        segs = [[ln.words[0]]]
        for prev, w in zip(ln.words, ln.words[1:]):
            (segs.append([w]) if w.x0 - prev.x1 > gap_tab else segs[-1].append(w))
        ln.segs = [Seg("".join(x.text if i == 0 else (" " if x.x0 - s[i - 1].x1 > 2 else "") + x.text
                               for i, x in enumerate(s)), s[0].x0, s[-1].x1) for s in segs]


def _split_into_cells(lines: list[Line], tables) -> list[Line]:
    """Move words that fall inside table cells into those cells; return body lines."""
    if not tables:
        return lines
    body = []
    for ln in lines:
        outside = []
        per_cell: dict = {}
        for wd in ln.words:
            cx, cy = (wd.x0 + wd.x1) / 2, (wd.y0 + wd.y1) / 2
            cell = None
            for t in tables:
                if t.x0 <= cx <= t.x1 and t.y0 <= cy <= t.y1:
                    for c in t.cells:
                        if c.x0 <= cx <= c.x1 and c.y0 <= cy <= c.y1:
                            cell = c
                            break
            if cell is None:
                outside.append(wd)
            else:
                per_cell.setdefault(id(cell), (cell, []))[1].append(wd)
        for cell, words in per_cell.values():
            nl = Line(words, baseline=ln.baseline, size_pt=ln.size_pt, bold=ln.bold,
                      underlines=[u for u in ln.underlines if cell.x0 <= u[0] <= cell.x1])
            if getattr(ln, "_weak", False):
                nl._weak = True   # type: ignore[attr-defined]
            nl.segs = [Seg(join_words(words), words[0].x0, words[-1].x1)]
            cell.lines.append(nl)
        if outside:
            if len(outside) != len(ln.words):
                wk = getattr(ln, "_weak", False)
                ln = Line(outside, baseline=ln.baseline, size_pt=ln.size_pt, bold=ln.bold,
                          underlines=ln.underlines)
                if wk:
                    ln._weak = True   # type: ignore[attr-defined]
                ln.segs = [Seg(join_words(outside), outside[0].x0, outside[-1].x1)]
            body.append(ln)
    return body


def _match_tables(tables, html_tables) -> set[int]:
    """Pair ruled tables with VLM html tables; return used html ids.

    Same shape (cell count) -> cell-by-cell. Otherwise match line by line, so the
    VLM's table text never leaks out as duplicate body paragraphs."""
    used: set[int] = set()
    for t in tables:
        order = sorted((i for i in range(len(html_tables)) if i not in used),
                       key=lambda i: abs(sum(len(r) for r in html_tables[i]) - len(t.cells)))
        for i in order:
            n = sum(len(r) for r in html_tables[i])
            if (abs(n - len(t.cells)) <= 2 and align.fill_table(t, html_tables[i])) \
                    or align.fill_table_fuzzy(t, html_tables[i]):
                used.add(i)
                break
    return used


def _substantial(ln: Line) -> bool:
    """Keep an unmatched Tesseract line next to VLM text only if it is clearly real text."""
    from .thai_text import is_thai
    txt = "".join(w.text for w in ln.words)
    conf = sum(w.conf for w in ln.words) / max(1, len(ln.words))
    return sum(is_thai(c) or c.isalpha() for c in txt) >= 8 and conf >= 75
