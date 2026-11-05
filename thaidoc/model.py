"""Data model shared by every stage of the pipeline.

All geometry is in *pixels of the rendered page image* (see Page.dpi).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float = 0.0


@dataclass
class Seg:
    """A run of text that starts at a given x (tab-separated columns)."""
    text: str
    x0: int
    x1: int


@dataclass
class Line:
    words: list[Word]
    baseline: int = 0          # y of the text baseline
    size_pt: float = 0.0       # estimated font size
    bold: bool = False
    segs: list[Seg] = field(default_factory=list)
    underlines: list[tuple[int, int]] = field(default_factory=list)  # x-ranges
    bold_ranges: list[tuple[int, int]] = field(default_factory=list)  # x-ranges of bold words
    text_override: Optional[str] = None  # better text from a VLM engine
    ink_x0: Optional[int] = None         # true left/right edge measured from the image
    ink_x1: Optional[int] = None

    @property
    def x0(self):
        x = min(w.x0 for w in self.words)
        return min(x, self.ink_x0) if self.ink_x0 is not None else x
    @property
    def x1(self):
        x = max(w.x1 for w in self.words)
        return max(x, self.ink_x1) if self.ink_x1 is not None else x
    @property
    def y0(self): return min(w.y0 for w in self.words)
    @property
    def y1(self): return max(w.y1 for w in self.words)
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2

    def text(self) -> str:
        if self.segs:
            return "\t".join(s.text for s in self.segs)
        return join_words(self.words)


def join_words(words: list[Word]) -> str:
    """Join OCR words; Thai words are only separated by a space where the gap is real."""
    if not words:
        return ""
    out = words[0].text
    h = max(1, sorted(w.y1 - w.y0 for w in words)[len(words) // 2])
    for prev, w in zip(words, words[1:]):
        gap = w.x0 - prev.x1
        out += (" " if gap > 0.52 * h else "") + w.text
    return out


@dataclass
class Para:
    """A paragraph ready for the DOCX writer (one or more geometry lines)."""
    lines: list[Line]
    text: str                          # may contain \t between columns
    tabs: list[int] = field(default_factory=list)      # x positions (px) of tab stops
    align: str = "left"                # left | center | justify | right
    size_pt: float = 16.0
    bold: bool = False
    underline_ranges: list[tuple[int, int]] = field(default_factory=list)  # char ranges
    bold_ranges: list[tuple[int, int]] = field(default_factory=list)       # char ranges
    page_break_before: bool = False
    rule_below: bool = False           # horizontal rule drawn under the paragraph
    space_before_px: float = 0.0
    pitch_px: float = 0.0
    anchors: list["Mark"] = field(default_factory=list)

    @property
    def x0(self): return self.lines[0].x0 if self.lines else 0
    @property
    def top(self): return self.lines[0].y0 if self.lines else 0
    @property
    def baseline(self): return self.lines[0].baseline if self.lines else 0


@dataclass
class Cell:
    r: int
    c: int
    x0: int
    y0: int
    x1: int
    y1: int
    rowspan: int = 1
    colspan: int = 1
    lines: list[Line] = field(default_factory=list)
    paras: list[Para] = field(default_factory=list)
    override_lines: Optional[list[str]] = None  # text from VLM table


@dataclass
class Table:
    x0: int
    y0: int
    x1: int
    y1: int
    col_x: list[int]
    row_y: list[int]
    cells: list[Cell] = field(default_factory=list)


@dataclass
class Rule:
    x0: int
    x1: int
    y: int


@dataclass
class Mark:
    """Something kept as a picture: pen ink, a stamp, a photo."""
    id: str
    kind: str          # ink | figure
    x0: int
    y0: int
    x1: int
    y1: int
    png: bytes
    color: str = ""    # e.g. "blue", "red"
    enabled: bool = True


@dataclass
class Page:
    index: int
    width_px: int
    height_px: int
    dpi: int
    source: str                     # "digital" | "scan"
    lines: list[Line] = field(default_factory=list)      # body lines (outside tables)
    tables: list[Table] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    pitch_px: float = 0.0
    body_pt: float = 16.0
    paras: list[Para] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Document:
    pages: list[Page]
    font: str = "TH SarabunPSK"     # font written to the .docx
    engine: str = ""
    source_font: str = ""           # font found in / fitted to the PDF
    font_scale: float = 1.0         # point-size factor (width-matched portable twin)
    warnings: list[str] = field(default_factory=list)
