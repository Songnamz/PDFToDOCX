"""Web app: upload a PDF, get an editable Thai DOCX with the pen ink kept.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Env:
  TYPHOON_OCR_API_KEY   use Typhoon's hosted OCR (best Thai text)
  TYPHOON_BASE_URL      or point at your own vLLM server (http://gpu:8000/v1)
  APP_PASSWORD          optional: require this password (HTTP Basic, any username)
  JOB_TTL_MINUTES       uploads and results are deleted after this (default 30)
  MAX_UPLOAD_MB         default 30
  MAX_PAGES             default 40
  WORKERS               parallel conversions (default 2)
"""
from __future__ import annotations

import base64
import io
import os
import pickle
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image

from thaidoc import fontembed, pipeline
from thaidoc.ink import InkOptions
from thaidoc.ocr import TyphoonEngine

TTL = int(os.getenv("JOB_TTL_MINUTES", "30")) * 60
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "40"))
PASSWORD = os.getenv("APP_PASSWORD", "")
ROOT = Path(os.getenv("JOB_DIR", tempfile.gettempdir())) / "thaidoc-jobs"
ROOT.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "static"
ICONS = {"/favicon.svg": "image/svg+xml", "/favicon-32.png": "image/png",
         "/apple-touch-icon.png": "image/png"}
PUBLIC = {"/healthz", "/favicon.ico", *ICONS}           # no password needed (browser icons)
OTHER_FONTS = ("Angsana New", "Cordia New", "Tahoma")   # common on Windows/Mac Office, not embeddable here

app = FastAPI(title="ThaiDoc — PDF to editable Word", docs_url=None, redoc_url=None)
pool = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
jobs: dict[str, dict] = {}
lock = threading.Lock()


# ------------------------------------------------------------------ auth
@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if PASSWORD and request.url.path not in PUBLIC:
        h = request.headers.get("authorization", "")
        ok = False
        if h.lower().startswith("basic "):
            try:
                _, pw = base64.b64decode(h[6:]).decode().split(":", 1)
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="ThaiDoc"'})
    resp = await call_next(request)
    if request.url.path not in PUBLIC:
        resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


# ------------------------------------------------------------------ cleanup
def _reaper():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            dead = [k for k, j in jobs.items() if now - j["created"] > TTL]
            for k in dead:
                shutil.rmtree(jobs[k]["dir"], ignore_errors=True)
                jobs.pop(k, None)
        # orphans from a previous run
        for d in ROOT.iterdir():
            if d.is_dir() and d.name not in jobs and now - d.stat().st_mtime > TTL:
                shutil.rmtree(d, ignore_errors=True)


threading.Thread(target=_reaper, daemon=True).start()


# ------------------------------------------------------------------ helpers
def _job(job_id: str) -> dict:
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found or expired")
    return j


def _run(job_id: str, opts: pipeline.Options):
    j = jobs[job_id]
    d: Path = j["dir"]

    def progress(f, msg):
        j["progress"], j["message"] = round(f, 3), msg

    def save_page(i, rgb):
        im = Image.fromarray(rgb)
        im.thumbnail((1100, 1600))
        im.save(d / f"page{i}.jpg", quality=82)

    try:
        j["status"] = "running"
        doc = pipeline.analyse(str(d / "input.pdf"), opts, progress, save_page)
        data = pipeline.render(doc, opts)
        (d / "output.docx").write_bytes(data)
        (d / "doc.pkl").write_bytes(pickle.dumps(doc))
        for pg in doc.pages:
            for m in pg.marks:
                (d / f"mark_{m.id}.png").write_bytes(m.png)
        j.update(status="done", progress=1.0, message="Done", engine=doc.engine, font=doc.font,
                 font_embedded=bool(fontembed.bundled(doc.font)),
                 warnings=doc.warnings, pages=_pages_info(doc))
    except Exception as e:  # noqa: BLE001
        j.update(status="error", message=str(e) or e.__class__.__name__)
    finally:
        (d / "input.pdf").unlink(missing_ok=True)   # the source is never kept


def _pages_info(doc):
    out = []
    for pg in doc.pages:
        out.append({
            "index": pg.index, "width": pg.width_px, "height": pg.height_px, "source": pg.source,
            "tables": len(pg.tables),
            "marks": [{"id": m.id, "kind": m.kind, "color": m.color, "enabled": m.enabled,
                       "box": [m.x0, m.y0, m.x1, m.y1]} for m in pg.marks],
        })
    return out


# ------------------------------------------------------------------ routes
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.svg")
@app.get("/favicon-32.png")
@app.get("/apple-touch-icon.png")
def icon(request: Request):
    path = request.url.path
    return FileResponse(STATIC / path.lstrip("/"), media_type=ICONS[path],
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/favicon.ico")
def favicon_ico():
    return FileResponse(STATIC / "favicon-32.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/config")
def config():
    return {"typhoon": TyphoonEngine.available(), "max_mb": MAX_MB, "max_pages": MAX_PAGES,
            "ttl_minutes": TTL // 60, "fonts": list(fontembed.families()), "other_fonts": list(OTHER_FONTS)}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), engine: str = Form("auto"),
                     keep_ink: bool = Form(True), photos: bool = Form(True), red_ink: bool = Form(True),
                     digits: str = Form("keep"), font: str = Form("auto")):
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF file")
    if engine not in ("auto", "typhoon", "tesseract") or digits not in ("keep", "thai", "arabic"):
        raise HTTPException(400, "Bad option")
    if font != "auto" and font not in fontembed.families() and font not in OTHER_FONTS:
        font = "auto"
    job_id = uuid.uuid4().hex
    d = ROOT / job_id
    d.mkdir(parents=True)
    (d / "input.pdf").write_bytes(data)
    name = Path(file.filename or "document.pdf").stem[:80] or "document"
    opts = pipeline.Options(engine=engine, keep_ink=keep_ink,
                            ink=InkOptions(red=red_ink, photos=photos), digits=digits, font=font,
                            max_pages=MAX_PAGES)
    with lock:
        jobs[job_id] = {"dir": d, "created": time.time(), "status": "queued", "progress": 0.0,
                        "message": "Waiting in queue", "name": name, "opts": opts}
    pool.submit(_run, job_id, opts)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = _job(job_id)
    keys = ("status", "progress", "message", "name", "engine", "font", "font_embedded", "warnings", "pages")
    out = {k: j.get(k) for k in keys}
    out["expires_in"] = max(0, int(TTL - (time.time() - j["created"])))
    return JSONResponse(out)


@app.get("/api/jobs/{job_id}/pages/{n}.jpg")
def page_image(job_id: str, n: int):
    p = _job(job_id)["dir"] / f"page{n}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/marks/{mark_id}.png")
def mark_image(job_id: str, mark_id: str):
    if not mark_id.replace("-", "").isalnum():
        raise HTTPException(400)
    p = _job(job_id)["dir"] / f"mark_{mark_id}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/jobs/{job_id}/marks")
async def set_marks(job_id: str, request: Request):
    """Turn individual pen marks / pictures on or off and rebuild (no OCR rerun)."""
    j = _job(job_id)
    if j.get("status") != "done":
        raise HTTPException(409, "Job not finished")
    body = await request.json()
    disabled = set(body.get("disabled", []))
    d: Path = j["dir"]
    doc = pickle.loads((d / "doc.pkl").read_bytes())
    for pg in doc.pages:
        for m in pg.marks:
            m.enabled = m.id not in disabled
    (d / "output.docx").write_bytes(pipeline.render(doc, j["opts"]))
    (d / "doc.pkl").write_bytes(pickle.dumps(doc))
    j["pages"] = _pages_info(doc)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    j = _job(job_id)
    p = j["dir"] / "output.docx"
    if not p.exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(p, filename=f"{j['name']}.docx",
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str):
    j = _job(job_id)
    with lock:
        shutil.rmtree(j["dir"], ignore_errors=True)
        jobs.pop(job_id, None)
    return {"ok": True}


@app.get("/healthz")
def health():
    return {"ok": True, "jobs": len(jobs)}
