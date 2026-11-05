"""Run the pipeline with a fake VLM engine that returns fixture Markdown per page.
Lets the Typhoon path be tested without network access."""
import sys, pathlib
from thaidoc import pipeline, ocr

FIX = pathlib.Path(__file__).parent / "fixtures"


class FakeEngine:
    name = "fixture"
    def __init__(self, stem): self.stem, self.i = stem, 0
    def page_markdown(self, rgb):
        self.i += 1
        p = FIX / f"{self.stem}_p{self.i}.md"
        return p.read_text() if p.exists() else ""


if __name__ == "__main__":
    pdf, stem, out = sys.argv[1], sys.argv[2], sys.argv[3]
    orig = ocr.TyphoonEngine
    fake = FakeEngine(stem)
    pipeline.TyphoonEngine = type("T", (), {"available": staticmethod(lambda *a: True),
                                             "__new__": lambda cls, *a, **k: fake})
    doc, data = pipeline.convert(pdf, pipeline.Options(engine="auto"))
    open(out, "wb").write(data)
    for pg in doc.pages:
        print(f"--- page {pg.index+1} ({pg.source}) pitch={pg.pitch_px:.0f} body={pg.body_pt}pt marks={len(pg.marks)}")
        for p in pg.paras:
            if p.lines:
                print(f"  y={p.baseline:5d} x={p.x0:5d} {p.size_pt:4.1f}{'B' if p.bold else ' '} {p.align[:1]} "
                      f"ul={p.underline_ranges} anch={[m.id for m in p.anchors]} | {p.text[:70]}")
            else:
                print("  ----- rule")
        for t in pg.tables:
            for c in t.cells:
                for p in c.paras:
                    print(f"  [cell {c.r},{c.c}] y={p.baseline} anch={[m.id for m in p.anchors]} | {p.text[:60]}")
    print("warnings:", doc.warnings)
