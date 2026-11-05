"""Fast unit tests (no OCR needed):  python -m pytest -q tests"""
import io

import numpy as np
from PIL import Image, ImageDraw

from thaidoc import ink
from thaidoc.thai_text import match_key, normalize, text_quality


def test_sara_am_and_pua():
    assert normalize("ตําแหน่ง") == "ตำแหน่ง"            # nikhahit + sara aa
    assert normalize("นำ") == "น้ำ"                 # PUA tone mark
    assert normalize("เเก้ไข") == "แก้ไข"                 # two sara e
    assert normalize("ก่ั") == "กั่"                       # tone before vowel


def test_doubled_marks_from_broken_pdfs():
    assert normalize("ข้อมูลส่วนที่ี่") == "ข้อมูลส่วนที่"
    assert normalize("ชััยวัฒน์") == "ชัยวัฒน์"


def test_quality_detects_garbage_layer():
    assert text_quality("riufi n aqtJ ru a n r :rJfr rifi r r u JuocAod o. UFI t0") < 0.6
    assert text_quality("บันทึกสรุปผลการปฏิบัติงาน ข้อมูลส่วนที่ ๑ ประวัติการรับราชการ") > 0.8


def test_match_key_ignores_digits_and_spaces():
    assert match_key("๑๓. การอบรม หลักสูตร") == match_key("13.การอบรมหลักสูตร")


def test_blue_signature_is_lifted_and_erased():
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Printed text", fill="black")
    d.line([(100, 200), (180, 150), (260, 210), (340, 140)], fill=(30, 50, 180), width=4)
    rgb = np.asarray(img)
    marks, clean = ink.extract(rgb, 200)
    assert len(marks) == 1 and marks[0].kind == "ink" and marks[0].color == "blue"
    png = Image.open(io.BytesIO(marks[0].png))
    assert png.mode == "RGBA" and np.asarray(png)[..., 3].max() > 200
    # ink gone from the OCR image, black text untouched
    assert (clean[140:215, 100:345] < 200).sum() == 0
    assert (clean[15:40, 15:120] < 100).sum() > 20


def test_pdf_font_names_map_to_word_families():
    from thaidoc.fontembed import from_pdf_font
    assert from_pdf_font("ABCDEF+THSarabunPSK-Bold") == "TH SarabunPSK"
    assert from_pdf_font("THSarabunNew,Bold") == "TH Sarabun New"
    assert from_pdf_font("BCDEEE+AngsanaNew") == "Angsana New"
    assert from_pdf_font("TimesNewRomanPSMT") == "Times New Roman"


def test_embedded_font_roundtrip():
    """The font inside the .docx de-obfuscates back to the bundled TTF."""
    import re, zipfile
    from docx import Document as Docx
    from thaidoc.fontembed import _obfuscate, embed, families
    buf = io.BytesIO(); Docx().save(buf)
    out = zipfile.ZipFile(io.BytesIO(embed(buf.getvalue(), "TH SarabunPSK")))
    ft = out.read("word/fontTable.xml").decode()
    key = re.search(r'w:name="TH SarabunPSK".*?embedRegular r:id="(\w+)" w:fontKey="(\{[^}]+\})"', ft, re.S)
    rels = out.read("word/_rels/fontTable.xml.rels").decode()
    target = re.search(r'Id="%s"[^>]*Target="([^"]+)"' % key.group(1), rels).group(1)
    data = out.read("word/" + target)
    assert _obfuscate(data, key.group(2)) == families()["TH SarabunPSK"].files["Regular"].read_bytes()
    assert "embedTrueTypeFonts" in out.read("word/settings.xml").decode()
    assert 'Extension="odttf"' in out.read("[Content_Types].xml").decode()


def test_split_over_cuts_thai_without_spaces():
    from thaidoc.align import split_over
    parts = split_over("- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตำแหน่งสูงขึ้นตามหลักเกณฑ์ที่กำหนดหรือไม่ ☐ เป็น ☐ ไม่เป็น",
                       ["- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตําแหน่งสูงขึ้นตามหลักเกณฑ์", "ที่กําหนดหรือไม่| | เป็น | |ไม่เป็น"])
    assert parts == ["- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตำแหน่งสูงขึ้นตามหลักเกณฑ์", "ที่กำหนดหรือไม่ ☐ เป็น ☐ ไม่เป็น"]


def test_table_cell_pieces_are_kept():
    """VLM returns a cell in pieces; every piece ends up on its printed line."""
    from thaidoc.align import fill_table_fuzzy
    from thaidoc.model import Cell, Line, Seg, Table, Word
    def line(t, y):
        ln = Line([Word(t, 10, y - 30, 900, y, 90)], baseline=y); ln.segs = [Seg(t, 10, 900)]; return ln
    c = Cell(0, 0, 0, 0, 1000, 300)
    c.lines = [line("- ความถูกต้องเรียบร้อยของข้อมูล | | ถูกต้อง | | ไม่ถูกต้อง", 50),
               line("- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตําแหน่งสูงขึ้นตามหลักเกณฑ์", 120),
               line("ที่กําหนดหรือไม่| | เป็น | |ไม่เป็น", 190)]
    t = Table(0, 0, 1000, 300, [0, 1000], [0, 300], [c])
    rows = [[{"text": "- ความถูกต้องเรียบร้อยของข้อมูล\n☐ ถูกต้อง ☐ ไม่ถูกต้อง\n"
                      "- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตำแหน่งสูงขึ้นตามหลักเกณฑ์ที่กำหนดหรือไม่ ☐ เป็น ☐ ไม่เป็น"}]]
    assert fill_table_fuzzy(t, rows)
    assert [l.text_override for l in c.lines] == [
        "- ความถูกต้องเรียบร้อยของข้อมูล ☐ ถูกต้อง ☐ ไม่ถูกต้อง",
        "- เป็นผู้มีคุณสมบัติครบถ้วนเลื่อนตำแหน่งสูงขึ้นตามหลักเกณฑ์",
        "ที่กำหนดหรือไม่ ☐ เป็น ☐ ไม่เป็น"]


def test_portable_twin_matches_width():
    """TH fonts are written as their normal-proportion Google twin at the size that
    gives the same text width, so a device without the font substitutes ~equal widths."""
    from thaidoc.fontembed import portable
    font, k = portable("TH SarabunPSK")
    assert font == "Sarabun" and 0.62 < k < 0.72
    assert portable("Angsana New") == ("Angsana New", 1.0)
