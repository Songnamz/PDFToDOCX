"""Write the Document model to .docx with python-docx.

Positioning strategy: every paragraph gets an *exact* line height (the page's
measured line pitch) and a space-before that puts its first line where it was
on the source page. Pen marks and photos are floating pictures anchored to the
paragraph at/above them: horizontally at their exact page position,
vertically relative to that paragraph's top (never a negative offset, never
"relative to line": apps disagree on those) — so they move with the text.
"""
from __future__ import annotations

import copy
import io

from docx import Document as Docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt

from .layout import ASC, rendered_lines
from .model import Document, Mark, Page, Para, Table

EMU_PER_IN = 914400

# OOXML requires these children in schema order
_RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike",
              "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden",
              "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
              "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
              "specVanish", "oMath"]
_PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
              "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
              "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
              "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
              "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle",
              "rPr", "sectPr", "pPrChange"]
_TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
                "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd",
                "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]


def _put(parent, child, order: list[str]):
    """Insert child into parent respecting the schema order (replacing a same-tag child)."""
    tag = child.tag.split("}")[1]
    old = parent.find(qn("w:" + tag))
    if old is not None:
        parent.replace(old, child)
        return child
    idx = order.index(tag) if tag in order else len(order)
    for i, el in enumerate(list(parent)):
        t = el.tag.split("}")[1]
        if t in order and order.index(t) > idx:
            parent.insert(i, child)
            return child
    parent.append(child)
    return child
_ALIGN = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
          "justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "right": WD_ALIGN_PARAGRAPH.RIGHT}


class Writer:
    def __init__(self, doc: Document, font: str = "TH SarabunPSK", digits: str = "keep"):
        self.doc = doc
        self.font = font
        self.digits = digits
        self.d = Docx()
        self._pic_id = 100
        self._setup_styles()

    # ------------------------------------------------------------------ helpers
    def px(self, v: float, page: Page) -> Emu:
        return Emu(int(v / page.dpi * EMU_PER_IN))

    def pt(self, v: float, page: Page) -> Pt:
        return Pt(v / page.dpi * 72)

    def _setup_styles(self):
        st = self.d.styles["Normal"]
        st.font.name = self.font
        st.font.size = Pt(16)
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = _put(rpr, OxmlElement("w:rFonts"), _RPR_ORDER)
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(k), self.font)
        szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), "32"); _put(rpr, szcs, _RPR_ORDER)
        lang = OxmlElement("w:lang"); lang.set(qn("w:val"), "en-US"); lang.set(qn("w:bidi"), "th-TH")
        _put(rpr, lang, _RPR_ORDER)
        pf = st.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)

    def _txt(self, t: str) -> str:
        from .thai_text import to_arabic_digits, to_thai_digits
        if self.digits == "thai":
            return to_thai_digits(t)
        if self.digits == "arabic":
            return to_arabic_digits(t)
        return t

    def _run(self, para, text: str, size: float, bold: bool, underline: bool = False):
        r = para.add_run(self._txt(text))
        r.font.size = Pt(size)
        r.font.bold = bold or None
        r.font.underline = underline or None
        rpr = r._r.get_or_add_rPr()
        szcs = OxmlElement("w:szCs"); szcs.set(qn("w:val"), str(int(round(size * 2))))
        _put(rpr, szcs, _RPR_ORDER)
        if bold:
            _put(rpr, OxmlElement("w:bCs"), _RPR_ORDER)
        return r

    # ------------------------------------------------------------------ build
    def build(self) -> bytes:
        pages = self.doc.pages
        p0 = pages[0]
        sec = self.d.sections[0]
        sec.page_width = self.px(p0.width_px, p0)
        sec.page_height = self.px(p0.height_px, p0)
        self.margin_l = self._left_margin_px()
        right_x = self._right_edge_px()
        sec.left_margin = self.px(self.margin_l, p0)
        sec.right_margin = self.px(max(p0.dpi * 0.3, p0.width_px - right_x), p0)
        self.top_px = p0.dpi * 0.3
        sec.top_margin = self.px(self.top_px, p0)
        sec.bottom_margin = self.px(p0.dpi * 0.25, p0)
        sec.header_distance = self.px(p0.dpi * 0.2, p0)
        body = self.d.element.body
        # python-docx starts with one empty paragraph — remove it
        for p in list(body.iterchildren(qn("w:p"))):
            body.remove(p)
        for page in pages:
            self._page(page)
        zoom = self.d.settings.element.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")
        buf = io.BytesIO()
        self.d.save(buf)
        from .fontembed import bundled, embed
        fam = bundled(self.font)
        if fam:
            bold = any(p.bold for pg in pages for p in pg.paras) or \
                any(q.bold for pg in pages for t in pg.tables for c in t.cells for q in c.paras)
            return embed(buf.getvalue(), fam, ("Regular", "Bold") if bold else ("Regular",))
        return buf.getvalue()

    def _left_margin_px(self) -> int:
        xs = []
        for pg in self.doc.pages:
            xs += [p.x0 for p in pg.paras if p.lines]
            xs += [t.x0 for t in pg.tables]
        p0 = self.doc.pages[0]
        m = min(xs) if xs else p0.dpi
        return int(max(p0.dpi * 0.4, m - p0.dpi * 0.02))

    def _right_edge_px(self) -> int:
        xs = []
        for pg in self.doc.pages:
            xs += [max(l.x1 for l in p.lines) for p in pg.paras if p.lines]
            xs += [t.x1 for t in pg.tables]
            xs += [r.x1 for r in pg.rules]
        p0 = self.doc.pages[0]
        return int(max(xs) + p0.dpi * 0.05) if xs else p0.width_px - p0.dpi

    def _page(self, page: Page):
        # body items in vertical order: paragraphs and tables
        items = [(p.baseline - ASC * p.pitch_px if p.lines else p._rule.y - page.dpi * 0.03, "p", p)  # type: ignore
                 for p in page.paras]
        items += [(t.y0, "t", t) for t in page.tables]
        items.sort(key=lambda x: x[0])
        cursor = self.top_px
        first = True
        for top, kind, obj in items:
            gap = max(0.0, top - cursor)
            if kind == "p":
                self._para(self.d, obj, page, gap, self.margin_l)
                if first and page.index > 0:
                    self._first_emitted.paragraph_format.page_break_before = True
                first = False
                n = rendered_lines(obj, page)
                cursor = max(cursor, top) + (n * obj.pitch_px if obj.lines else page.dpi * 0.06)
            else:
                spacer = self.d.add_paragraph()
                pf = spacer.paragraph_format
                pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
                pf.line_spacing = self.pt(max(2.0, gap), page)
                if first and page.index > 0:
                    pf.page_break_before = True
                first = False
                self._table(obj, page)
                cursor = max(cursor, float(obj.y1))
        if first:   # empty page: still emit a page break
            p = self.d.add_paragraph()
            if page.index > 0:
                p.paragraph_format.page_break_before = True

    def _para(self, container, p: Para, page: Page, space_before_px: float, origin_x: float):
        self._first_emitted = None
        if any(m.enabled for m in p.anchors) and self.pt(space_before_px, page).pt >= 1:
            # pictures are positioned from the paragraph's top; apps disagree on whether
            # that includes "space before", so carry the gap in an empty spacer instead
            spacer = container.add_paragraph()
            spf = spacer.paragraph_format
            spf.space_before = spf.space_after = Pt(0)
            spf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            spf.line_spacing = self.pt(space_before_px, page)
            self._first_emitted = spacer
            space_before_px = 0.0
        para = container.add_paragraph()
        self._first_emitted = self._first_emitted or para
        pf = para.paragraph_format
        pf.space_before = self.pt(space_before_px, page)
        pf.space_after = Pt(0)
        if p.rule_below:
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(2)
            r = p._rule  # type: ignore[attr-defined]
            pf.left_indent = self.px(max(0, r.x0 - origin_x), page)
            self._border_bottom(para)
            return para
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = self.pt(p.pitch_px, page)
        pf.alignment = _ALIGN.get(p.align, WD_ALIGN_PARAGRAPH.LEFT)
        if p.align != "center":
            first_x = p.lines[0].x0
            cont_x = p.lines[1].x0 if len(p.lines) > 1 else first_x
            pf.left_indent = self.px(max(0, cont_x - origin_x), page)
            pf.first_line_indent = self.px(first_x - cont_x, page)
        for tx in p.tabs:
            pf.tab_stops.add_tab_stop(self.px(max(0, tx - origin_x), page), WD_TAB_ALIGNMENT.LEFT)
        # anchored pictures go first so they hang off the paragraph's first line
        for m in p.anchors:
            if m.enabled:
                self._float_picture(para, m, page, p)
        self._runs(para, p)
        return para

    def _runs(self, para, p: Para):
        text = p.text
        marks = [False] * len(text)
        for a, b in p.underline_ranges:
            for i in range(max(0, a), min(len(text), b)):
                marks[i] = True
        i = 0
        while i < len(text):
            j = i
            while j < len(text) and marks[j] == marks[i]:
                j += 1
            chunk = text[i:j]
            for li, seg in enumerate(chunk.split("\n")):
                if li:
                    para.add_run().add_break()
                for k, part in enumerate(seg.split("\t")):
                    if k:
                        para.add_run().add_tab()
                    if part:
                        self._run(para, part, p.size_pt, p.bold, marks[i])
            i = j

    def _border_bottom(self, para):
        ppr = para._p.get_or_add_pPr()
        bdr = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "1"), ("w:color", "000000")):
            b.set(qn(k), v)
        bdr.append(b)
        _put(ppr, bdr, _PPR_ORDER)

    # ------------------------------------------------------------------ tables
    def _table(self, t: Table, page: Page):
        R, C = len(t.row_y) - 1, len(t.col_x) - 1
        tbl = self.d.add_table(rows=R, cols=C)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        tbl.autofit = False
        tblPr = tbl._tbl.tblPr
        for tag in ("w:tblStyle",):
            el = tblPr.find(qn(tag))
            if el is not None:
                tblPr.remove(el)
        ind = OxmlElement("w:tblInd"); ind.set(qn("w:w"), str(self._dxa(t.x0 - self.margin_l, page)))
        ind.set(qn("w:type"), "dxa"); _put(tblPr, ind, _TBLPR_ORDER)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); _put(tblPr, lay, _TBLPR_ORDER)
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "0"), ("w:color", "000000")):
                e.set(qn(k), v)
            borders.append(e)
        _put(tblPr, borders, _TBLPR_ORDER)
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "0"); e.set(qn("w:type"), "dxa"); mar.append(e)
        _put(tblPr, mar, _TBLPR_ORDER)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(self._dxa(t.x1 - t.x0, page))); tw.set(qn("w:type"), "dxa")
        _put(tblPr, tw, _TBLPR_ORDER)
        widths = [self._dxa(t.col_x[i + 1] - t.col_x[i], page) for i in range(C)]
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(w))
        for r in range(R):
            row = tbl.rows[r]
            trPr = row._tr.get_or_add_trPr()
            h = OxmlElement("w:trHeight")
            h.set(qn("w:val"), str(self._dxa(t.row_y[r + 1] - t.row_y[r], page)))
            h.set(qn("w:hRule"), "atLeast")
            trPr.append(h)
            for c in range(C):
                row.cells[c].width = Emu(int(widths[c] * 635))
        for cell in t.cells:
            a = tbl.cell(cell.r, cell.c)
            if cell.rowspan > 1 or cell.colspan > 1:
                a = a.merge(tbl.cell(cell.r + cell.rowspan - 1, cell.c + cell.colspan - 1))
            # clear default paragraph(s)
            for p in list(a._tc.iterchildren(qn("w:p"))):
                a._tc.remove(p)
            cursor = cell.y0 + page.dpi * 0.01
            if not cell.paras:
                a.add_paragraph()
                continue
            for p in cell.paras:
                top = p.baseline - ASC * p.pitch_px
                gap = max(0.0, top - cursor)
                self._para(a, p, page, gap, cell.x0)
                cursor = max(cursor, top) + rendered_lines(p, page, cell.x1) * p.pitch_px

    def _dxa(self, px: float, page: Page) -> int:
        return int(round(px / page.dpi * 1440))

    # ------------------------------------------------------------------ pictures
    def _float_picture(self, para, m: Mark, page: Page, p: Para):
        run = para.add_run()
        w, h = self.px(m.x1 - m.x0, page), self.px(m.y1 - m.y0, page)
        run.add_picture(io.BytesIO(m.png), width=w, height=h)
        inline = run._r.find(".//" + qn("wp:inline"))
        line_top = p.baseline - ASC * p.pitch_px
        x_off = int(m.x0 / page.dpi * EMU_PER_IN)
        y_off = int((m.y0 - line_top) / page.dpi * EMU_PER_IN)
        self._pic_id += 1
        anchor = OxmlElement("wp:anchor")
        for k, v in (("distT", "0"), ("distB", "0"), ("distL", "0"), ("distR", "0"),
                     ("simplePos", "0"), ("relativeHeight", str(251660000 + self._pic_id)),
                     ("behindDoc", "0"), ("locked", "0"), ("layoutInCell", "0"),
                     ("allowOverlap", "1")):
            anchor.set(k, v)
        sp = OxmlElement("wp:simplePos"); sp.set("x", "0"); sp.set("y", "0"); anchor.append(sp)
        ph = OxmlElement("wp:positionH"); ph.set("relativeFrom", "page")
        o = OxmlElement("wp:posOffset"); o.text = str(x_off); ph.append(o); anchor.append(ph)
        pv = OxmlElement("wp:positionV"); pv.set("relativeFrom", "paragraph")
        o = OxmlElement("wp:posOffset"); o.text = str(y_off); pv.append(o); anchor.append(pv)
        for tag in ("wp:extent", "wp:effectExtent"):
            el = inline.find(qn(tag))
            if el is not None:
                anchor.append(copy.deepcopy(el))
        anchor.append(OxmlElement("wp:wrapNone"))
        docpr = copy.deepcopy(inline.find(qn("wp:docPr")))
        docpr.set("id", str(self._pic_id))
        docpr.set("name", f"{'Ink' if m.kind == 'ink' else 'Picture'} {m.id}")
        docpr.set("descr", "Handwriting / stamp from scan" if m.kind == "ink" else "Picture from scan")
        anchor.append(docpr)
        anchor.append(copy.deepcopy(inline.find(qn("wp:cNvGraphicFramePr"))))
        anchor.append(copy.deepcopy(inline.find(qn("a:graphic"))))
        inline.getparent().replace(inline, anchor)
