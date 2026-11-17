"""main.py -- FastAPI application for the vibrational mode scorer.

Deliberately stateless. Scoring naphthalene (18 atoms, 54 modes) takes ~11 ms,
so every request is handled synchronously: no job registry, no progress
polling, no results directory, nothing written to disk. That is what lets the
same code run unchanged on a normal server and on a read-only serverless
filesystem.

Downloads (CSV, .vsc) are generated at score time and embedded in the result
page, so the browser can produce them from a Blob without a second round trip
and without the server holding any run state.
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (ALLOW_PASTE, IS_LOCAL, MAX_UPLOAD_BYTES, REQUEST_CAP_BYTES,
                     SCORE_KEYS, SITE_NOTE,
                     STATIC_DIR, TEMPLATES_DIR, VERSION, asset_version)
from .core.parsers import (ParseError, parse_bond_file, parse_connectivity,
                           parse_frequency_file, parse_gaussian_log, parse_vsc,
                           write_vsc)
from .core.classifier import Thresholds
from .core import frame as frame_mod
from .core.pipeline import (AMBIGUITY_RANGE, DEFAULT_DEG_TOL, DEFAULT_DELTA,
                            FRAME_RULES, LINEAR_TURNS, TIE_BREAKS, analyse,
                            to_csv_rows)

# Starlette caps each NON-file form field at 1 MB (MultiPartParser.max_part_size,
# a class attribute FastAPI never overrides). The result page re-posts its own
# .vsc as a text field and the builder posts typed modes the same way, so any
# input between 1 MB and the 4 MB upload cap scored fine once and then failed
# to re-score with a bare JSON 400. Align the two caps.
from starlette.formparsers import MultiPartParser  # noqa: E402
MultiPartParser.max_part_size = MAX_UPLOAD_BYTES

# modes x atoms above which the linear 'scan' turn is refused on the deployment.
SCAN_WORK_CAP = 1000

app = FastAPI(
    title="Vibrational Mode Scorer",
    description=("Scores each normal mode of a molecule against seven reference "
                 "motions (Tx, Ty, Tz, Rx, Ry, Rz, V_S) and classifies it. "
                 + SITE_NOTE),
    version=VERSION,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["version"] = VERSION
templates.env.globals["asset_v"] = asset_version
templates.env.globals["is_local"] = IS_LOCAL
templates.env.globals["allow_paste"] = ALLOW_PASTE
templates.env.globals["site_note"] = SITE_NOTE
templates.env.globals["score_keys"] = SCORE_KEYS
templates.env.globals["request_cap"] = REQUEST_CAP_BYTES
templates.env.globals["defaults"] = {
    "tau_SB": Thresholds.calibrated().tau_SB,
    "delta": DEFAULT_DELTA,
    "deg_tol": DEFAULT_DEG_TOL,
    "ambiguity_range": AMBIGUITY_RANGE,
    "sym_tol": frame_mod.SYM_TOL,
    "frame_rules": FRAME_RULES,
    "tie_breaks": TIE_BREAKS,
    "linear_turns": LINEAR_TURNS,
}


# ----------------------------------------------------------------------
# Upload handling
# ----------------------------------------------------------------------
async def _read(upload: UploadFile | None, what: str) -> str | None:
    """Decode an upload to text, refusing oversized or binary files.

    The size cap matters on serverless hosts, which reject large request
    bodies at the platform layer with an opaque error; failing here gives the
    user a message that says what to do about it.
    """
    if upload is None or not upload.filename:
        return None
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ParseError(
            f"{what} is {len(data)/1e6:.1f} MB, over the {MAX_UPLOAD_BYTES/1e6:.1f} MB "
            "limit. Most of a Gaussian log (SCF iterations) or an ORCA .hess (the "
            "Hessian itself) is never read by the scorer -- convert to a .vsc file "
            "locally with make_vsc.py and upload that instead (typically a few kB).")
    try:
        # utf-8-sig drops the byte-order mark Windows editors prepend; left in,
        # it hides a Molden file's '[Molden Format]' header and turns a bond
        # list's first index into a non-number.
        return data.decode("utf-8-sig", errors="replace")
    except Exception:
        raise ParseError(f"{what} could not be read as text.")


def _build(mode, log_text, com_text, vsc_text, log_name, com_name, vsc_name,
           pasted=None, arb_text=None, arb_name=None,
           qc_text=None, qc_name=None, bond_text=None, bond_name=None,
           opts=None, source=None):
    """Route the three input paths onto one payload.

    Returns ``(payload, raw)``. ``raw`` keeps the geometry and displacements
    exactly as parsed -- BEFORE principal-axis alignment -- because that is
    what the .vsc writer must serialise. ``MIT()`` is not idempotent: its
    sign-fix heuristic negates the rotation when the heaviest atom's projected
    coordinates sum negative, so re-aligning an already-aligned geometry can
    flip axes and invert the T/R scores. Writing the original frame means a
    downloaded .vsc re-scores to the same numbers as the files it came from.

    ``opts`` carries the user's analysis options (tau_sb, delta, deg_tol,
    reject) straight through to ``analyse``. ``source`` overrides the
    provenance line -- a re-score posts the page's own .vsc back, and should
    still say which files it came from.
    """
    opts = opts or {}
    if mode == "paste":
        if not ALLOW_PASTE:
            raise ParseError(
                "Typed-in modes are turned off on this instance. Use the "
                ".vsc upload above instead.")
        # Two ways in, one meaning: whatever arrives here is DECLARED to be a
        # selection of vibrations. mode_set is forced rather than detected, so
        # a count that happens to equal 3N or 3N-6 is still treated as a
        # selection and the ideal T/R references are still constructed to fill
        # the external slots.
        text = arb_text if (arb_text or "").strip() else pasted
        if not (text or "").strip():
            raise ParseError("No modes were entered or uploaded.")
        if len(text) > MAX_UPLOAD_BYTES:
            raise ParseError("That is too large; upload it as a file.")
        v = parse_vsc(text)
        raw = {"atoms": v["atoms"], "coords": v["coords"],
               "bonds": v["bonds"], "modes": v["modes"]}
        return analyse(v["atoms"], v["coords"], v["bonds"], v["modes"],
                       title=v["title"] or "arbitrary modes",
                       source=source or (arb_name or "typed in"),
                       mode_set="arbitrary", **opts), raw

    if mode == "vsc":
        if not vsc_text:
            raise ParseError("No .vsc file was uploaded.")
        if len(vsc_text) > MAX_UPLOAD_BYTES:
            raise ParseError("That .vsc is too large to post as text; upload it as a file.")
        v = parse_vsc(vsc_text)
        raw = {"atoms": v["atoms"], "coords": v["coords"],
               "bonds": v["bonds"], "modes": v["modes"]}
        payload = analyse(v["atoms"], v["coords"], v["bonds"], v["modes"],
                          title=v["title"] or _stem(vsc_name),
                          source=source or vsc_name or "uploaded .vsc", **opts)
        return payload, raw

    if mode == "qc":
        if not qc_text:
            raise ParseError("No ORCA .hess or Molden file was uploaded.")
        if not bond_text:
            raise ParseError(
                "No bond file was uploaded. ORCA .hess and Molden files carry no "
                "connectivity, and the V-score is a sum over bonds, so a bond list "
                "(or a Gaussian .com/.gjf with geom=connectivity) is required.")
        g = parse_frequency_file(qc_text)
        bonds = parse_bond_file(bond_text, len(g["atoms"]), bond_name or "")
        raw = {"atoms": g["atoms"], "coords": g["coords"],
               "bonds": bonds, "modes": g["modes"]}
        payload = analyse(g["atoms"], g["coords"], bonds, g["modes"],
                          title=_stem(qc_name),
                          source=source or f"{qc_name} + {bond_name}", **opts)
        return payload, raw

    if not log_text:
        raise ParseError("No Gaussian output file was uploaded.")
    if not com_text:
        raise ParseError(
            "No .com/.gjf input deck was uploaded. A Gaussian .log carries no "
            "connectivity -- the bond list lives in the input deck -- and the "
            "V-score is a sum over bonds, so both files are required.")
    g = parse_gaussian_log(log_text)
    bonds = parse_connectivity(com_text, len(g["atoms"]))
    raw = {"atoms": g["atoms"], "coords": g["coords"],
           "bonds": bonds, "modes": g["modes"]}
    # A Gaussian frequency job prints 3N-6 (or 3N-5) modes; T/R are projected
    # out at a stationary point. Detection confirms that rather than assuming it.
    payload = analyse(g["atoms"], g["coords"], bonds, g["modes"],
                      title=_stem(log_name),
                      source=source or f"{log_name} + {com_name}", **opts)
    return payload, raw


def _options(tau_sb, delta, deg_tol, reject, frame_rule="", keep_input_frame="",
             tie_break="", linear_turn="", sym_tol=""):
    """Form strings -> analyse() keyword arguments. Blank means default.

    ``reject`` arrives either as repeated form fields (the result page's
    checkboxes) or as one comma/space-separated string (the JSON API).
    """
    def num(v):
        v = (v or "").strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            raise ParseError(f"{v!r} is not a number.")
    opts = {}
    if num(tau_sb) is not None:
        opts["tau_sb"] = num(tau_sb)
    if num(delta) is not None:
        opts["delta"] = num(delta)
    if num(deg_tol) is not None:
        opts["deg_tol"] = num(deg_tol)
    nums = []
    for item in reject or []:
        nums.extend(x for x in str(item).replace(",", " ").split() if x)
    if nums:
        opts["reject"] = nums
    # Frame of reference (src/frame.py upstream). Choices are validated by
    # analyse(), which says what the allowed values are.
    for key, val in (("frame_rule", frame_rule), ("tie_break", tie_break),
                     ("linear_turn", linear_turn)):
        if (val or "").strip():
            opts[key] = val.strip()
    if (keep_input_frame or "").strip().lower() in ("1", "on", "true", "yes"):
        opts["keep_input_frame"] = True
    if num(sym_tol) is not None:
        opts["sym_tol"] = num(sym_tol)
    if not IS_LOCAL:
        # ~4 s of rescoring on a serverless function; see analyse()'s docstring.
        opts["max_scan_work"] = SCAN_WORK_CAP
    return opts


def _stem(name):
    if not name:
        return "molecule"
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _csv_text(payload, include_references=False, incorporate=False):
    rows = to_csv_rows(payload, include_references, incorporate)
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _vsc_text(payload, raw):
    """Serialise the parsed input -- original frame -- to .vsc text.

    Deliberately built from ``raw``, not from the scored payload: the payload
    holds principal-axis-aligned coordinates, and re-aligning those on upload
    can flip axis signs (see ``_build``). Serialising the frame the data
    arrived in makes the round trip exact.
    """
    return write_vsc(raw["atoms"], raw["coords"],
                     [tuple(b) for b in raw["bonds"]], raw["modes"],
                     title=payload["title"],
                     source=payload.get("source", ""))


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/score", response_class=HTMLResponse)
async def score(
    request: Request,
    mode: str = Form("log"),
    logfile: UploadFile | None = File(None),
    comfile: UploadFile | None = File(None),
    vscfile: UploadFile | None = File(None),
    pasted: str = Form(""),
    arbfile: UploadFile | None = File(None),
    qcfile: UploadFile | None = File(None),
    bondfile: UploadFile | None = File(None),
    vsctext: str = Form(""),
    source: str = Form(""),
    tau_sb: str = Form(""),
    delta: str = Form(""),
    deg_tol: str = Form(""),
    reject: list[str] = Form([]),
    frame_rule: str = Form(""),
    keep_input_frame: str = Form(""),
    tie_break: str = Form(""),
    linear_turn: str = Form(""),
    sym_tol: str = Form(""),
):
    try:
        payload, raw = _build(
            mode,
            await _read(logfile, "The Gaussian output file"),
            await _read(comfile, "The input deck"),
            await _read(vscfile, "The .vsc file") or (vsctext if vsctext.strip() else None),
            logfile.filename if logfile else None,
            comfile.filename if comfile else None,
            vscfile.filename if vscfile else None,
            pasted=pasted,
            arb_text=await _read(arbfile, "The .vsc file"),
            arb_name=arbfile.filename if arbfile else None,
            qc_text=await _read(qcfile, "The ORCA/Molden file"),
            qc_name=qcfile.filename if qcfile else None,
            bond_text=await _read(bondfile, "The bond file"),
            bond_name=bondfile.filename if bondfile else None,
            opts=_options(tau_sb, delta, deg_tol, reject, frame_rule,
                          keep_input_frame, tie_break, linear_turn, sym_tol),
            source=source.strip() or None,
        )
    except ParseError as exc:
        return templates.TemplateResponse(
            request, "index.html", {"error": str(exc)}, status_code=400)
    except Exception as exc:                       # unexpected: still readable
        return templates.TemplateResponse(
            request, "index.html",
            {"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return templates.TemplateResponse(request, "result.html", {
        "p": payload,
        # The page's own input, posted back by the settings form to re-score
        # with different thresholds or rejected candidates -- no state is kept
        # server-side, so the input travels with the page.
        "vsc_text": _vsc_text(payload, raw),
        "payload_json": _embed(payload),
        # Both views are shipped so the download follows the toggle without a
        # round trip. The .vsc has no view: it serialises the INPUT (geometry,
        # connectivity, modes), which no labelling decision touches.
        "downloads_json": _embed({
            "csv": _csv_text(payload),
            "csvall": _csv_text(payload, include_references=True),
            "csv_ext": _csv_text(payload, incorporate=True),
            "csvall_ext": _csv_text(payload, include_references=True,
                                    incorporate=True),
            "vsc": _vsc_text(payload, raw),
        }),
    })


def _embed(obj):
    """JSON for inlining in a <script> block.

    Downloadable text goes through JSON rather than straight into the element
    body: Jinja would otherwise HTML-escape it, turning an irrep like A' into
    A&#39; inside the file the user saves. Escaping '</' additionally stops any
    payload content from closing the script element early.
    """
    return json.dumps(obj).replace("</", "<\\/")


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {})


@app.get("/format", response_class=HTMLResponse)
def format_page(request: Request):
    return templates.TemplateResponse(request, "format.html", {})


# ----------------------------------------------------------------------
# JSON API
# ----------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    return {"status": "ok", "version": VERSION, "scores": list(SCORE_KEYS)}


@app.post("/api/score")
async def api_score(
    mode: str = Form("log"),
    logfile: UploadFile | None = File(None),
    comfile: UploadFile | None = File(None),
    vscfile: UploadFile | None = File(None),
    pasted: str = Form(""),
    arbfile: UploadFile | None = File(None),
    qcfile: UploadFile | None = File(None),
    bondfile: UploadFile | None = File(None),
    vsctext: str = Form(""),
    source: str = Form(""),
    tau_sb: str = Form(""),
    delta: str = Form(""),
    deg_tol: str = Form(""),
    reject: list[str] = Form([]),
    frame_rule: str = Form(""),
    keep_input_frame: str = Form(""),
    tie_break: str = Form(""),
    linear_turn: str = Form(""),
    sym_tol: str = Form(""),
):
    try:
        payload, _raw = _build(
            mode,
            await _read(logfile, "The Gaussian output file"),
            await _read(comfile, "The input deck"),
            await _read(vscfile, "The .vsc file") or (vsctext if vsctext.strip() else None),
            logfile.filename if logfile else None,
            comfile.filename if comfile else None,
            vscfile.filename if vscfile else None,
            pasted=pasted,
            arb_text=await _read(arbfile, "The .vsc file"),
            arb_name=arbfile.filename if arbfile else None,
            qc_text=await _read(qcfile, "The ORCA/Molden file"),
            qc_name=qcfile.filename if qcfile else None,
            bond_text=await _read(bondfile, "The bond file"),
            bond_name=bondfile.filename if bondfile else None,
            opts=_options(tau_sb, delta, deg_tol, reject, frame_rule,
                          keep_input_frame, tie_break, linear_turn, sym_tol),
            source=source.strip() or None,
        )
    except ParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(payload)
