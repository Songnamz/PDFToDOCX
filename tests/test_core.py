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
