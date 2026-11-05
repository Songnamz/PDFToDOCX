"""Read PDF pages: render to images and, for born-digital pages, extract the
text layer as positioned lines (no OCR needed)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymupdf

from .model import Line, Word
from .thai_text import normalize, text_quality


@dataclass
class RawPage:
    index: int
    rgb: np.ndarray
    dpi: int
    kind: str                        # digital | scan
    text_lines: list[Line]
    image_boxes: list[tuple[int, int, int, int]]
    fonts: dict[str, int] | None = None     # PDF font name -> characters set in it


def open_pdf(path: str, dpi: int = 250, max_pages: int = 60):
    doc = pymupdf.open(path)
    if doc.needs_pass:
        raise ValueError("PDF is password-protected")
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
        s = dpi / 72
        area = page.rect.width * page.rect.height
        img_boxes, img_area = [], 0.0
        for info in page.get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            img_area += max(0, (x1 - x0) * (y1 - y0))
            img_boxes.append((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)))
        raw_text = page.get_text("text")
        q = text_quality(raw_text)
        covered = img_area / area if area else 0
        digital = q > 0.6 and len(raw_text.strip()) > 20 and not (covered > 0.85 and q < 0.85)
        fonts: dict[str, int] = {}
        lines = _text_lines(page, s, fonts) if digital else []
        yield RawPage(i, rgb, dpi, "digital" if digital else "scan", lines, img_boxes, fonts)


def _text_lines(page, s: float, fonts: dict[str, int]) -> list[Line]:
    out: list[Line] = []
    d = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            words = []
            sizes, bolds, base = [], [], []
            for sp in ln["spans"]:
                t = normalize(sp["text"])
                if not t.strip():
                    continue
                fonts[sp["font"]] = fonts.get(sp["font"], 0) + len(t.strip())
                x0, y0, x1, y1 = sp["bbox"]
                words.append(Word(t.strip(), int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s), 99))
                sizes.append(sp["size"])
                bolds.append(bool(sp["flags"] & 16) or "bold" in sp["font"].lower())
                base.append(sp["origin"][1] * s)
            if not words:
                continue
            L = Line(words)
            L.size_pt = round(max(sizes) * 2) / 2
            L.bold = sum(bolds) > len(bolds) / 2
            L.baseline = int(max(base))
            out.append(L)
    return out
