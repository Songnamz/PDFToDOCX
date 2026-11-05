# ThaiDoc — Thai PDF → editable Word, pen ink included

Converts Thai PDFs, especially **scanned government forms** that ordinary converters
turn into garbage, into an editable `.docx` that:

- has **real, correct Thai text** in **the PDF's own font**, embedded in the file so it
  looks the same on every device (Thai numerals kept as in the source)
- keeps the **layout**: indents, tab columns, underlined labels, centred titles,
  horizontal rules, ruled tables with merged cells, checkboxes ☐
- keeps every **signature, handwritten date, tick mark and stamp** as a transparent
  picture, anchored to the line it belongs to, so it moves with the text
- keeps **photos** (ID photos, logos) as pictures

## How it works

```
PDF page ─┬─ digital text layer? ── PyMuPDF text + Thai repair (PUA glyphs, ำ, doubled marks)
          │
          └─ scan ─► 1. lift ink:   blue/navy/red strokes → transparent PNGs, erased from page
                     2. geometry:   ruled tables, rules, underlines, checkboxes (OpenCV)
                                    + line positions (Tesseract word boxes)
                     3. text:       Typhoon OCR (Thai VLM) reads the ink-free page
                     4. align:      Typhoon's text is fuzzy-matched onto Tesseract's lines
                     5. rebuild:    exact line pitch + space-before reproduce positions;
                                    ink anchored per line (page-x, line-relative y)
```

Step 1 is the important trick: removing the handwriting **before** OCR stops it being
transcribed as duplicate typed text, and gives a clean image to place back over the
editable text.

| Engine | Thai accuracy | Needs |
|---|---|---|
| **Typhoon OCR** (default when configured) | Excellent, including numbers and tables | Free API key from [opentyphoon.ai](https://opentyphoon.ai), or your own GPU running the open model |
| Tesseract (`tha+eng`) | Rough, often mixes up Thai digits | Nothing, works offline |
| Digital PDFs | Perfect (no OCR at all) | Nothing |

## Run it

### Docker (recommended)

```bash
cp .env.example .env        # add TYPHOON_OCR_API_KEY and an APP_PASSWORD
docker compose up -d --build
# open http://localhost:8000
```

To put it behind your own domain, use any reverse proxy (Caddy, nginx, Cloudflare Tunnel).
Uploads live in a RAM-backed tmpfs and are deleted automatically.

### Without Docker

```bash
sudo apt install tesseract-ocr tesseract-ocr-tha     # macOS: brew install tesseract tesseract-lang
pip install -r requirements.txt
export TYPHOON_OCR_API_KEY=...                       # optional but strongly recommended
uvicorn app.main:app --port 8000
```

### Command line

```bash
python cli.py scan.pdf -o scan.docx                  # auto engine
python cli.py scan.pdf --engine tesseract            # fully offline
python cli.py scan.pdf --digits arabic --font "TH Sarabun New"   # default: --font auto
python cli.py scan.pdf --no-ink                      # drop signatures/stamps/photos
```

### Self-hosting the Thai OCR model (no data leaves your network)

```bash
pip install vllm
vllm serve scb10x/typhoon-ocr1.5-2b --served-model-name typhoon-ocr --max-model-len 49152
# then in .env:
TYPHOON_BASE_URL=http://<gpu-host>:8000/v1
```

Typhoon OCR 1.5 is a 2B model, so a single consumer GPU with 8 GB or more is enough.

## Deployment on songnamvm (pdftoword.songnam.xyz)

```
Cloudflare Tunnel "main" ──► 127.0.0.1:8089 ──► pdftoword-nginx ──► pdftoword-app:8000
```

- `docker compose up -d --build` starts both containers (`pdftoword-nginx`, `pdftoword-app`).
- Only nginx is published, and only on `127.0.0.1:8089` (next free port after yt on 8088).
- nginx config: `nginx/default.conf` (35 MB upload limit, 300 s timeouts for slow OCR).
- Tunnel `main` is dashboard-managed: Zero Trust → Networks → Tunnels → main →
  Public Hostname → `pdftoword.songnam.xyz` → `HTTP` `localhost:8089`.
- Settings and the site password are in `.env` (not in git).

## Fonts

`--font auto` (the default) uses the PDF's own font: digital PDFs name it; on scans every
bundled family is fitted to the measured line pitch and line widths (TH SarabunPSK unless
another font clearly fits better). The font is **embedded** in the .docx (regular + bold),
so Word on Mac/Windows and LibreOffice show it even where it isn't installed.

Bundled: the Thai national fonts that may be redistributed (TH SarabunPSK, TH Sarabun New,
TH Niramit AS, TH Krub, TH Kodchasal, TH K2D July8, TH Mali Grade 6, TH Chakra Petch,
TH Baijam, TH KoHo, TH Fah kwang), from the official DIP/SIPA package; licenses in
`thaidoc/fonts/LICENSES.txt`. Microsoft fonts (Angsana New, Cordia New, Tahoma) are used by
name only. Apps that ignore embedded fonts (Apple Pages, Quick Look, Google Docs) substitute
a wider Thai font; open the file in Word or LibreOffice for an exact match.

## Web app features

- Drag-and-drop upload with live progress per page
- Page previews with every detected pen mark or photo outlined. **Click a mark to
  exclude it** (or put it back), then *Apply*: the DOCX rebuilds instantly with no re-OCR
- Options: OCR engine, font, digit style, which kinds of ink/photos to keep
- Privacy: the uploaded PDF is deleted the moment conversion finishes; results expire after
  `JOB_TTL_MINUTES` (default 30); "delete now" button; optional password (`APP_PASSWORD`)

## API

| Method | Path | |
|---|---|---|
| POST | `/api/jobs` | multipart `file` (+ `engine`, `font`, `digits`, `keep_ink`, `red_ink`, `photos`) → `{id}` |
| GET | `/api/jobs/{id}` | status, progress, pages and their marks |
| GET | `/api/jobs/{id}/pages/{n}.jpg` | page preview |
| GET | `/api/jobs/{id}/marks/{mark}.png` | one extracted mark |
| POST | `/api/jobs/{id}/marks` | `{"disabled": ["p3-i2", …]}` → rebuild |
| GET | `/api/jobs/{id}/download` | the .docx |
| DELETE | `/api/jobs/{id}` | delete everything now |

## Project layout

```
thaidoc/
  thai_text.py    Thai Unicode repair (PUA → Unicode, ำ, mark order, doubled marks), quality score
  ink.py          pen/stamp/photo detection, transparent PNG extraction, erasing for OCR
  geometry.py     tables, rules, underlines, checkboxes, Tesseract lines, size/bold/baseline
  ocr.py          Typhoon OCR client + Markdown/HTML → units
  align.py        fuzzy alignment of VLM text onto geometry lines; column splitting
  layout.py       paragraphs, underline character ranges, centring, mark anchoring
  docx_writer.py  python-docx output: exact positioning, tables, floating ink anchors
  fontembed.py    PDF font detection (named or fitted) and font embedding in the .docx
  fonts/          bundled Thai national fonts + LICENSES.txt
  pdf_input.py    page rendering, digital vs scanned detection, text-layer lines
  pipeline.py     ties it all together
app/              FastAPI server + single-page UI
tests/            unit tests + fixture test of the VLM path without network
```

## Limits and tips

- Results are best at 200–300 dpi scans. Very skewed scans should be straightened first.
- Pen colours detected: blue/navy/violet and red (green optional in code). **Black-ink**
  signatures can't be told apart from printed text by colour, so they stay as text noise or
  get dropped. Signing in blue is the norm on Thai forms, so this rarely matters.
- Positions are reproduced with exact line spacing for the detected font. If you force a much
  wider font, long lines may wrap differently.
- Always proofread names, ID numbers and dates before using a converted document officially.
