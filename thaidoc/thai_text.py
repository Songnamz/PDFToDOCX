"""Thai text repair.

PDFs made from Word with legacy Thai fonts (TH SarabunPSK, Angsana, Cordia...)
often store shifted vowels/tone marks as Private-Use-Area glyphs, split sara am
(ำ) into nikhahit + sara aa, or emit marks in the wrong order. OCR engines make
some of the same mistakes. Everything here turns that into clean Unicode Thai.
"""
from __future__ import annotations

import re
import unicodedata

# Windows / Apple Thai PUA presentation forms -> base characters
PUA_MAP = {
    0xF700: "ฐ", 0xF701: "ิ", 0xF702: "ี", 0xF703: "ึ", 0xF704: "ื",
    0xF705: "่", 0xF706: "้", 0xF707: "๊", 0xF708: "๋", 0xF709: "์",
    0xF70A: "่", 0xF70B: "้", 0xF70C: "๊", 0xF70D: "๋", 0xF70E: "์",
    0xF70F: "ญ", 0xF710: "ั", 0xF711: "ํ", 0xF712: "็",
    0xF713: "่", 0xF714: "้", 0xF715: "๊", 0xF716: "๋", 0xF717: "์",
    0xF718: "ุ", 0xF719: "ู", 0xF71A: "ฺ",
}
_PUA_TRANS = {k: v for k, v in PUA_MAP.items()}

ABOVE_BELOW_VOWELS = "ัิีึืฺุู็"
TONES = "่้๊๋์ํ"
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"

_re_tone_vowel = re.compile(f"([{TONES}])([{ABOVE_BELOW_VOWELS}])")
_re_dup_marks = re.compile(f"([{ABOVE_BELOW_VOWELS}{TONES}])\\1+")
_re_zw = re.compile("[​‌‍﻿]")


def normalize(text: str) -> str:
    if not text:
        return text
    t = text.translate(_PUA_TRANS)
    t = _re_zw.sub("", t)
    # nikhahit + sara aa  ->  sara am  (also with a tone mark in between)
    t = re.sub(f"ํ([{TONES[:-1]}]?)า", lambda m: m.group(1) + "ำ", t)
    t = t.replace("ํา", "ำ")
    # tone mark typed before an above/below vowel -> canonical order
    for _ in range(2):
        t = _re_tone_vowel.sub(r"\2\1", t)
    t = _re_dup_marks.sub(r"\1", t)
    # a vowel+tone pair emitted twice (ที่ี่ -> ที่), common in broken PDF text layers
    t = re.sub(f"([{ABOVE_BELOW_VOWELS}])([{TONES}])\\1\\2", r"\1\2", t)
    # เเ (two sara e) -> แ
    t = t.replace("เเ", "แ")
    t = unicodedata.normalize("NFC", t)
    return t


def to_thai_digits(t: str) -> str:
    return t.translate(str.maketrans(ARABIC_DIGITS, THAI_DIGITS))


def to_arabic_digits(t: str) -> str:
    return t.translate(str.maketrans(THAI_DIGITS, ARABIC_DIGITS))


def is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def thai_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha() or is_thai(c)]
    if not letters:
        return 0.0
    return sum(is_thai(c) for c in letters) / len(letters)


def text_quality(text: str) -> float:
    """0..1 — how much a text layer looks like real (Thai or English) text.

    Garbage layers from bad OCR/fonts look like 'riufi n aqtJ ru a n r'.
    """
    t = normalize(text)
    letters = [c for c in t if c.isalpha() or is_thai(c)]
    if len(letters) < 20:
        return 0.0
    thai = sum(is_thai(c) for c in letters)
    if thai / len(letters) > 0.3:
        # orphan combining marks at word starts are a sign of broken extraction
        orphans = len(re.findall(f"(^|\\s)[{ABOVE_BELOW_VOWELS}{TONES}ำ]", t))
        return max(0.0, 1.0 - orphans / max(1, thai) * 8)
    # Latin: share of words that look like words (vowels, sensible length)
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return 0.0
    good = sum(1 for w in words if 2 <= len(w) <= 15 and re.search(r"[aeiouyAEIOUY]", w))
    return good / len(words)


def match_key(text: str) -> str:
    """Key used to fuzzy-match the same line coming from two OCR engines.

    Digits are dropped (Tesseract confuses Thai digits) and so is punctuation/space.
    """
    t = normalize(text)
    return "".join(c for c in t if (is_thai(c) and c not in THAI_DIGITS) or c.isalpha())
