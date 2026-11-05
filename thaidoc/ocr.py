"""Text engines.

* TyphoonEngine — SCB 10X Typhoon OCR (Thai-specialised VLM). Works with the
  hosted API (TYPHOON_OCR_API_KEY) or a self-hosted vLLM server
  (TYPHOON_BASE_URL=http://gpu-box:8000/v1). Any OpenAI-compatible vision
  model can be used the same way by changing the model name.
* Tesseract text is the offline fallback (already produced by geometry.py).

The VLM returns Markdown with HTML tables. `parse_units` turns that into a flat
list of Units (one per paragraph / table row) that align.py maps onto lines.
"""
from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

from .thai_text import normalize

TYPHOON_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements briefly. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
"""


@dataclass
class Unit:
    """One paragraph (segs=[text]) or one table row (segs=cells)."""
    segs: list[str]
    table_id: int = -1      # index of the HTML table it came from, -1 = body text
    role: str = ""          # "page_number" for <page_number> content
    row: int = -1
    cells: list[dict] = field(default_factory=list)   # for table rows: text/colspan/rowspan

    @property
    def text(self) -> str:
        return " ".join(self.segs)


class TyphoonEngine:
    name = "typhoon"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, timeout: float = 180):
        self.api_key = (api_key or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("TYPHOON_API_KEY")
                        or os.getenv("OPENAI_API_KEY") or "no-key")
        self.base_url = base_url or os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
        self.model = model or os.getenv("TYPHOON_MODEL", "typhoon-ocr")
        self.timeout = timeout

    @classmethod
    def available(cls, api_key: str | None = None) -> bool:
        return bool(api_key or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("TYPHOON_API_KEY")
                    or os.getenv("TYPHOON_BASE_URL"))

    def page_markdown(self, rgb: np.ndarray) -> str:
        from openai import OpenAI
        img = Image.fromarray(rgb)
        w, h = img.size
        scale = 1800 / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": TYPHOON_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            max_tokens=16384, temperature=0.1,
            extra_body={"repetition_penalty": 1.1, "top_p": 0.6},
        )
        txt = resp.choices[0].message.content or ""
        # some deployments wrap the answer as {"natural_text": "..."}
        m = re.match(r'^\s*\{\s*"natural_text"\s*:\s*"(.*)"\s*\}\s*$', txt, re.S)
        if m:
            import json
            txt = json.loads('"' + m.group(1) + '"')
        return txt


# ------------------------------------------------------------------- markdown
_md_inline = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<!\w)\*(.+?)\*(?!\w)"), r"\1"),
    (re.compile(r"`([^`]*)`"), r"\1"),
]


def _clean_inline(t: str) -> str:
    t = re.sub(r"<page_number>.*?</page_number>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    for rx, rep in _md_inline:
        t = rx.sub(rep, t)
    t = re.sub(r"^#{1,6}\s*", "", t)
    t = t.replace("\\-", "-").replace("\\.", ".").replace("\\(", "(").replace("\\)", ")")
    t = t.replace("&nbsp;", " ")
    return normalize(t.strip())


def parse_units(md: str) -> tuple[list[Unit], list[list[list[dict]]]]:
    """Return (units in reading order, html tables as rows of cells)."""
    md = re.sub(r"<figure>.*?</figure>", "", md, flags=re.S)
    md = re.sub(r"<page_number>(.*?)</page_number>", lambda m: "\n\u2063PN" + m.group(1).strip() + "\n", md, flags=re.S)
    units: list[Unit] = []
    tables: list[list[list[dict]]] = []
    pos = 0
    for m in re.finditer(r"<table.*?</table>", md, flags=re.S | re.I):
        units += _text_units(md[pos:m.start()])
        rows = _html_rows(m.group(0))
        tid = len(tables)
        tables.append(rows)
        for ri, cells in enumerate(rows):
            segs = [c["text"].replace("\n", " ") for c in cells if c["text"].strip()]
            if segs:
                units.append(Unit(segs, tid, ri, cells))
        pos = m.end()
    units += _text_units(md[pos:])
    # markdown pipe tables (some engines emit them)
    out = []
    for u in units:
        if u.table_id == -1 and u.segs and u.segs[0].startswith("|"):
            cells = [c.strip() for c in u.segs[0].strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            u = Unit([c for c in cells if c], -2)
        out.append(u)
    return out, tables


def _text_units(chunk: str) -> list[Unit]:
    units = []
    for raw in chunk.split("\n"):
        if not raw.strip() or re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", raw):
            continue
        role = ""
        if raw.startswith("\u2063PN"):
            raw, role = raw[3:], "page_number"
        t = _clean_inline(raw)
        if t:
            units.append(Unit([t], role=role))
    return units


def _html_rows(html: str) -> list[list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            for br in td.find_all("br"):
                br.replace_with("\n")
            lines = [normalize(x.strip()) for x in td.get_text().split("\n")]
            cells.append({
                "text": "\n".join(l for l in lines if l),
                "colspan": int(td.get("colspan", 1) or 1),
                "rowspan": int(td.get("rowspan", 1) or 1),
            })
        rows.append(cells)
    return rows
