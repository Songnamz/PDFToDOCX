"""Find pen ink, stamps and photos on a page image and lift them out.

Each mark becomes a transparent PNG (only the ink, no paper) plus its position.
The returned `clean` image has the ink painted out, so OCR never transcribes
handwriting as duplicate typed text.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .model import Mark


@dataclass
class InkOptions:
    blue: bool = True          # blue / navy / violet pens & stamps
    red: bool = True           # red pens & stamps
    green: bool = False
    photos: bool = True        # colour photos (ID photos etc.)
    min_mark_in: float = 0.06  # smallest mark kept (inches)


def _png(rgba: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(b, "PNG", optimize=True)
    return b.getvalue()


def _components(mask: np.ndarray, min_px: int):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return lab, []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    objs = ndimage.find_objects(lab)
    return lab, [(i + 1, int(s), o) for i, (s, o) in enumerate(zip(sizes, objs)) if s >= min_px]


def find_photos(rgb: np.ndarray, dpi: int) -> list[tuple[int, int, int, int]]:
    """Large, dense picture regions (photos, coloured logos) — not text, not signatures."""
    a = rgb.astype(np.int16)
    nonwhite = (255 - a.mean(2)) > 28
    sat = a.max(2) - a.min(2)
    colourful = (sat > 40) & nonwhite
    k = max(3, dpi // 30)
    closed = ndimage.binary_closing(nonwhite, iterations=k)
    closed = ndimage.binary_fill_holes(closed)
    closed = ndimage.binary_opening(closed, iterations=k)       # drop text strokes
    lab, comps = _components(closed, int((dpi * 0.5) ** 2))
    boxes = []
    for i, size, (sy, sx) in comps:
        h, w = sy.stop - sy.start, sx.stop - sx.start
        if min(h, w) < dpi * 0.45:
            continue
        fill = nonwhite[sy, sx].mean()
        col = colourful[sy, sx].mean()
        if (fill > 0.35 and col > 0.08) or fill > 0.6:
            boxes.append((sx.start, sy.start, sx.stop, sy.stop))
    return boxes


def _merge_boxes(boxes: list[list[int]], gap: float) -> list[list[int]]:
    """Merge boxes on the same text row whose horizontal gap is small (one written date)."""
    boxes = sorted(boxes, key=lambda b: b[0])
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                vo = min(a[3], b[3]) - max(a[1], b[1])
                hgap = max(a[0], b[0]) - min(a[2], b[2])
                if vo > 0.4 * min(a[3] - a[1], b[3] - b[1]) and hgap < gap:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _ink_masks(rgb: np.ndarray, opts: InkOptions) -> list[tuple[str, np.ndarray, np.ndarray]]:
    a = rgb.astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    dark = 255 - a.mean(2)
    out = []
    if opts.blue:
        bl = B - R
        core = ((bl >= 5) & (dark > 45)) | ((bl >= 12) & (dark > 20))
        out.append(("blue", core, bl))
    if opts.red:
        rd = R - np.maximum(G, B)
        core = ((rd >= 30) & (dark > 25)) | ((rd >= 60) & (dark > 10))
        out.append(("red", core, rd))
    if opts.green:
        gr = G - np.maximum(R, B)
        core = (gr >= 20) & (dark > 25)
        out.append(("green", core, gr))
    return out


def extract(rgb: np.ndarray, dpi: int, opts: InkOptions | None = None,
            regions: list[tuple[int, int, int, int]] | None = None,
            id_prefix: str = "p0") -> tuple[list[Mark], np.ndarray]:
    """Return (marks, clean_rgb).

    regions: if given, only look for ink inside these boxes (used for digital
    PDFs, where coloured *text* must not be mistaken for ink).
    """
    opts = opts or InkOptions()
    h, w = rgb.shape[:2]
    a = rgb.astype(np.int16)
    dark = 255 - a.mean(2)
    neutral_dark = (np.abs(a[..., 2] - a[..., 0]) < 4) & (dark > 90)
    clean = rgb.copy()
    marks: list[Mark] = []
    n = 0

    # --- photos -------------------------------------------------------------
    photo_mask = np.zeros((h, w), bool)
    if opts.photos:
        for (x0, y0, x1, y1) in find_photos(rgb, dpi):
            if regions is not None and not any(_overlap((x0, y0, x1, y1), r) for r in regions):
                continue
            crop = rgb[y0:y1, x0:x1]
            rgba = np.dstack([crop, np.full(crop.shape[:2], 255, np.uint8)])
            marks.append(Mark(f"{id_prefix}-f{n}", "figure", x0, y0, x1, y1, _png(rgba)))
            n += 1
            photo_mask[y0:y1, x0:x1] = True
            clean[y0:y1, x0:x1] = 255

    region_mask = None
    if regions is not None:
        region_mask = np.zeros((h, w), bool)
        for (x0, y0, x1, y1) in regions:
            region_mask[y0:y1, x0:x1] = True

    # --- pen ink / stamps -------------------------------------------------------
    near_text = ndimage.binary_dilation(neutral_dark, iterations=2)
    min_side = opts.min_mark_in * dpi
    for colour, core, strength in _ink_masks(rgb, opts):
        m = core & ~photo_mask
        if region_mask is not None:
            m &= region_mask
        # scanner colour fringes around black text are weakly coloured and hug it
        m &= ~(near_text & (strength < 16))
        lab, comps = _components(m, 30)
        keep = np.zeros_like(m)
        for i, size, (sy, sx) in comps:
            hh, ww = sy.stop - sy.start, sx.stop - sx.start
            if hh <= 7 and ww >= 6:      # bits of ruled lines / box edges
                continue
            keep[sy, sx] |= lab[sy, sx] == i
        if keep.sum() < 60:
            continue
        # cluster strokes that belong to one signature / date / tick
        cluster = ndimage.binary_dilation(keep, iterations=max(4, dpi // 18))
        clab, ccomps = _components(cluster, 1)
        groups = _merge_boxes([[sx.start, sy.start, sx.stop, sy.stop] for _, _, (sy, sx) in ccomps],
                              gap=0.3 * dpi)
        for (gx0, gy0, gx1, gy1) in groups:
            sel = np.zeros_like(keep)
            sel[gy0:gy1, gx0:gx1] = keep[gy0:gy1, gx0:gx1]
            px = int(sel.sum())
            ys, xs = np.nonzero(sel)
            if px < 80 or len(xs) == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if max(x1 - x0, y1 - y0) < min_side:
                continue
            pad = 3
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            sub = sel[y0:y1, x0:x1]
            sub = ndimage.binary_closing(ndimage.binary_dilation(sub), iterations=2)
            d = dark[y0:y1, x0:x1]
            alpha = np.clip(d * 1.7, 0, 255) * sub
            # where ink crosses black text the pixel is black: keep it inked
            alpha = np.where(sub & neutral_dark[y0:y1, x0:x1], 225, alpha)
            # one clean ink colour per mark (median of its strongest pixels)
            crop = a[y0:y1, x0:x1]
            strong = sel[y0:y1, x0:x1] & (d > np.percentile(d[sel[y0:y1, x0:x1]], 50))
            col = np.median(crop[strong], axis=0) if strong.any() else np.array([30, 50, 170])
            col = np.clip(col * 0.85, 0, 255)
            rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            rgba[..., :3] = col.astype(np.uint8)
            rgba[..., 3] = alpha.astype(np.uint8)
            marks.append(Mark(f"{id_prefix}-i{n}", "ink", int(x0), int(y0), int(x1), int(y1),
                              _png(rgba), colour))
            n += 1
            # paint the ink out of the OCR image (black text underneath survives)
            near = ndimage.binary_dilation(sel[y0:y1, x0:x1], iterations=4)
            tint = (strength[y0:y1, x0:x1] >= 3) & (d < 140)
            er = (ndimage.binary_dilation(sel[y0:y1, x0:x1], iterations=2) | (near & tint)) \
                & ~neutral_dark[y0:y1, x0:x1]
            region = clean[y0:y1, x0:x1]
            region[er] = 255
    return marks, clean


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
