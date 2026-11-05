"""Use the PDF's own font in the .docx and embed it, so the document looks the
same on every device (also where the font isn't installed).

* Digital PDFs name their fonts: the dominant one is used.
* Scans don't: each bundled family is fitted to the page (its line height
  gives the point size from the measured line pitch, its glyph widths must
  then reproduce the measured line widths). TH SarabunPSK wins ties.
* Bundled families (Thai national fonts, free to redistribute; see
  fonts/LICENSES.txt) are embedded as obfuscated TrueType parts, the way Word
  itself embeds fonts. Other fonts are referenced by name only.
"""
from __future__ import annotations

import io
import re
import statistics
import struct
import uuid
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

FONT_DIR = Path(__file__).parent / "fonts"
DEFAULT = "TH SarabunPSK"


@dataclass
class Family:
    name: str
    files: dict[str, Path] = field(default_factory=dict)   # Regular | Bold | Italic | BoldItalic
    line_height: float = 1.31                               # Word's single line height / em


def _tables(b: bytes) -> dict[bytes, tuple[int, int]]:
    n = struct.unpack(">H", b[4:6])[0]
    return {b[12 + 16 * i:16 + 16 * i]: struct.unpack(">II", b[20 + 16 * i:28 + 16 * i]) for i in range(n)}


def _font_info(path: Path) -> tuple[str, str, float]:
    b = path.read_bytes()
    t = _tables(b)
    o = t[b"name"][0]
    cnt, so = struct.unpack(">HH", b[o + 2:o + 6])
    nm = {}
    for i in range(cnt):
        pid, _eid, _lid, nid, ln, off = struct.unpack(">6H", b[o + 6 + 12 * i:o + 18 + 12 * i])
        if pid == 3 and nid in (1, 2):
            nm[nid] = b[o + so + off:o + so + off + ln].decode("utf-16-be")
    upm = struct.unpack(">H", b[t[b"head"][0] + 18:t[b"head"][0] + 20])[0]
    os2 = t[b"OS/2"][0]
    win_asc, win_desc = struct.unpack(">HH", b[os2 + 74:os2 + 78])
    return nm[1], nm[2].replace(" ", ""), (win_asc + win_desc) / upm


@lru_cache(maxsize=1)
def families() -> dict[str, Family]:
    out: dict[str, Family] = {}
    for p in sorted(FONT_DIR.glob("*.ttf")):
        fam, style, lh = _font_info(p)
        f = out.setdefault(fam, Family(fam))
        f.files[style] = p
        if style == "Regular":
            f.line_height = lh
    # TH SarabunPSK first: preferred whenever fits are equal
    return dict(sorted(out.items(), key=lambda kv: kv[0] != DEFAULT))


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9\u0e00-\u0e7f]", "", name.lower())


def bundled(name: str) -> str | None:
    """Exact bundled family name for a (loosely written) font name, or None."""
    k = _key(name)
    for fam in families():
        if _key(fam) == k:
            return fam
    return None


# Portable twins: Cadson Demak's Google Fonts versions of the Thai national fonts.
# The TH originals are drawn very small for their point size (16 pt TH SarabunPSK is
# as wide as ~10.5 pt in any normal Thai font), so on a device that lacks them -
# and Pages, Quick Look, Google Docs and phone viewers ignore embedded fonts - the
# substitute is ~1.5x wider and the layout falls apart. The twins have normal
# proportions: set at the width-matched size they look like the PDF, Google Docs
# has them built in, and any fallback font is within a few percent.
PORTABLE = {
    "TH SarabunPSK": "Sarabun", "TH Sarabun New": "Sarabun", "TH Niramit AS": "Niramit",
    "TH Krub": "Krub", "TH Kodchasal": "Kodchasan", "TH K2D July8": "K2D",
    "TH Mali Grade 6": "Mali", "TH Chakra Petch": "Chakra Petch", "TH Baijam": "Bai Jamjuree",
    "TH KoHo": "KoHo", "TH Fah kwang": "Fahkwang",
}
_SAMPLE = ("๒. ตำแหน่งปัจจุบัน : ผู้บังคับหมวด (สบ ๑) ฝ่ายปกครองและการฝึก ศูนย์ฝึกอบรมตำรวจภูธรภาค ๑ "
           "เลื่อนทั้งปี ๑.๕ ขั้น จำนวน ๗ ครั้ง Report 2026")


@lru_cache(maxsize=64)
def width_ratio(src: str, dst: str) -> float:
    """Point-size factor that makes `dst` text as wide as `src` text."""
    from PIL import ImageFont
    fs = families()
    if src not in fs or dst not in fs or "Regular" not in fs[src].files or "Regular" not in fs[dst].files:
        return 1.0
    w = [ImageFont.truetype(str(fs[f].files["Regular"]), 200).getlength(_SAMPLE) for f in (src, dst)]
    return w[0] / w[1] if w[1] else 1.0


def portable(src: str) -> tuple[str, float]:
    """(font to write, size factor) that reproduces `src` on every device."""
    dst = PORTABLE.get(src)
    if dst and dst in families():
        return dst, width_ratio(src, dst)
    return src, 1.0


# PostScript names found in Thai PDFs -> the family name Word uses
_KNOWN = {
    "thsarabunpsk": "TH SarabunPSK", "thsarabun": "TH SarabunPSK",
    "thsarabunit9": "TH SarabunPSK", "thsarabunit๙": "TH SarabunPSK",   # same design, not redistributable
    "thsarabunnew": "TH Sarabun New",
    "angsananew": "Angsana New", "angsanaupc": "AngsanaUPC", "cordianew": "Cordia New",
    "cordiaupc": "CordiaUPC", "browallianew": "Browallia New", "browalliaupc": "BrowalliaUPC",
    "tahoma": "Tahoma", "leelawadee": "Leelawadee", "leelawadeeui": "Leelawadee UI",
    "arial": "Arial", "arialmt": "Arial", "timesnewroman": "Times New Roman",
    "timesnewromanpsmt": "Times New Roman", "calibri": "Calibri", "cambria": "Cambria",
}
_STYLE = re.compile(r"[-,]?(bold|italic|oblique|regular|roman|book|medium|semibold|light|black|mt|ps)+$", re.I)


def from_pdf_font(ps_name: str) -> str:
    """'ABCDEF+THSarabunPSK-Bold' -> 'TH SarabunPSK'."""
    name = ps_name.split("+", 1)[-1]
    for cand in (name, _STYLE.sub("", name), _STYLE.sub("", _STYLE.sub("", name))):
        k = _key(cand)
        if k in _KNOWN:
            return _KNOWN[k]
        b = bundled(cand)
        if b:
            return b
    base = _STYLE.sub("", name)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base).strip() or DEFAULT


def detect_scan_font(samples: list[tuple[str, float]], pitch_pt: float, assumed_lh: float,
                     dpi: int) -> tuple[str, float]:
    """samples: (text, measured width px) of plain single lines at body size.

    Returns (family, size factor vs the size estimated with `assumed_lh`)."""
    from PIL import ImageFont
    if len(samples) < 3:
        return DEFAULT, 1.0
    scores = []
    for fam in families().values():
        if "Regular" not in fam.files or not 1.15 <= fam.line_height <= 1.5:
            continue          # display fonts with unreliable metrics: not used for running text
        size_pt = pitch_pt / fam.line_height
        font = ImageFont.truetype(str(fam.files["Regular"]), max(4, int(round(size_pt / 72 * dpi))))
        r = statistics.median(font.getlength(t) / w for t, w in samples if w > 0)
        scores.append((abs(r - 1), fam.name, assumed_lh / fam.line_height))
    best = min(scores)
    default = next((s for s in scores if s[1] == DEFAULT), best)
    # only leave TH SarabunPSK when it clearly doesn't fit and another font clearly does
    if best[1] != DEFAULT and default[0] > 0.05 and best[0] < 0.05 and best[0] < default[0] - 0.03:
        return best[1], best[2]
    return DEFAULT, default[2]


# ------------------------------------------------------------------ embedding
_FONT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/font"
_ODTTF = "application/vnd.openxmlformats-officedocument.obfuscatedFont"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _obfuscate(data: bytes, guid: str) -> bytes:
    """ECMA-376 font obfuscation: XOR the first 32 bytes with the GUID key
    (the GUID's 16 bytes in reverse order of the hex string)."""
    key = bytes.fromhex(guid.strip("{}").replace("-", ""))[::-1]
    head = bytes(b ^ key[i % 16] for i, b in enumerate(data[:32]))
    return head + data[32:]


def embed(docx: bytes, family: str, styles: tuple[str, ...] = ("Regular", "Bold")) -> bytes:
    fam = families().get(family)
    if fam is None:
        return docx
    styles = tuple(s for s in styles if s in fam.files)
    zin = zipfile.ZipFile(io.BytesIO(docx))
    parts = {i.filename: zin.read(i.filename) for i in zin.infolist()}

    rels, embeds, n = [], [], 0
    tag = {"Regular": "embedRegular", "Bold": "embedBold", "Italic": "embedItalic", "BoldItalic": "embedBoldItalic"}
    for st in ("Regular", "Bold", "Italic", "BoldItalic"):
        if st not in styles:
            continue
        n += 1
        guid = "{" + str(uuid.uuid4()).upper() + "}"
        parts[f"word/fonts/font{n}.odttf"] = _obfuscate(fam.files[st].read_bytes(), guid)
        rels.append(f'<Relationship Id="rIdF{n}" Type="{_FONT_REL}" Target="fonts/font{n}.odttf"/>')
        embeds.append(f'<w:{tag[st]} r:id="rIdF{n}" w:fontKey="{guid}"/>')
    if not n:
        return docx
    parts["word/_rels/fontTable.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels) + "</Relationships>").encode()

    ft = parts["word/fontTable.xml"].decode("utf-8")
    if f'xmlns:r="{_R}"' not in ft:
        ft = re.sub(r"<w:fonts\b", f'<w:fonts xmlns:r="{_R}"', ft, count=1)
    ft = re.sub(r'<w:font w:name="' + re.escape(fam.name) + r'".*?</w:font>', "", ft, flags=re.S)
    entry = (f'<w:font w:name="{fam.name}"><w:charset w:val="DE"/><w:family w:val="swiss"/>'
             f'<w:pitch w:val="variable"/>{"".join(embeds)}</w:font>')
    ft = re.sub(r"</w:fonts>\s*$", entry + "</w:fonts>", ft)
    parts["word/fontTable.xml"] = ft.encode("utf-8")

    ct = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="odttf"' not in ct:
        ct = ct.replace("<Default ", f'<Default Extension="odttf" ContentType="{_ODTTF}"/><Default ', 1)
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    st_xml = parts["word/settings.xml"].decode("utf-8")
    if "embedTrueTypeFonts" not in st_xml:
        # schema order: (writeProtection, view, zoom, removePersonalInformation, removeDateAndTime,
        # doNotDisplayPageBoundaries, displayBackgroundShape, printPostScriptOverText,
        # printFractionalCharacterWidth, printFormsData) then embedTrueTypeFonts
        before = r"(<w:(?:writeProtection|view|zoom|removePersonalInformation|removeDateAndTime|" \
                 r"doNotDisplayPageBoundaries|displayBackgroundShape|printPostScriptOverText|" \
                 r"printFractionalCharacterWidth|printFormsData)\b[^>]*?(?:/>|>.*?</w:\w+>))"
        last = None
        for m in re.finditer(before, st_xml, flags=re.S):
            last = m
        ins = "<w:embedTrueTypeFonts/>"
        if last:
            st_xml = st_xml[:last.end()] + ins + st_xml[last.end():]
        else:
            st_xml = re.sub(r"(<w:settings\b[^>]*>)", r"\1" + ins, st_xml, count=1)
    parts["word/settings.xml"] = st_xml.encode("utf-8")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for name, data in parts.items():
            z.writestr(name, data)
    return out.getvalue()
