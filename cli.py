"""Command line: python cli.py input.pdf [-o output.docx] [--engine tesseract|typhoon|auto]"""
import argparse
import sys
import time

from thaidoc import Options, convert


def main():
    ap = argparse.ArgumentParser(description="Thai PDF -> editable DOCX (keeps pen ink)")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out")
    ap.add_argument("--engine", default="auto", choices=["auto", "typhoon", "tesseract"])
    ap.add_argument("--font", default="auto", help="auto = the PDF's own font (embedded in the .docx)")
    ap.add_argument("--digits", default="keep", choices=["keep", "thai", "arabic"])
    ap.add_argument("--no-ink", action="store_true", help="drop pen marks / stamps / photos")
    ap.add_argument("--dpi", type=int, default=250)
    a = ap.parse_args()
    opts = Options(engine=a.engine, font=a.font, digits=a.digits, keep_ink=not a.no_ink, dpi=a.dpi)
    t = time.time()
    doc, data = convert(a.pdf, opts, lambda f, m: print(f"\r[{f*100:5.1f}%] {m:<60}", end="", file=sys.stderr))
    out = a.out or a.pdf.rsplit(".", 1)[0] + ".docx"
    with open(out, "wb") as f:
        f.write(data)
    marks = sum(len(p.marks) for p in doc.pages)
    print(f"\nWrote {out}  ({len(doc.pages)} pages, engine={doc.engine}, {marks} pen marks/pictures, "
          f"font={doc.font}, {time.time()-t:.1f}s)", file=sys.stderr)
    for w in doc.warnings:
        print("warning:", w, file=sys.stderr)


if __name__ == "__main__":
    main()
