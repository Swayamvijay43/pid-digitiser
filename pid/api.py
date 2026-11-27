"""FastAPI backend for the review dashboard.

    uvicorn pid.api:app --port 8000
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pid.graph import ROOT, build_graph, resume_run, sqlite_checkpointer, start_run, thread
from pid.schemas import ReviewDecision

RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
INDEX = RUNS / "index.json"
_lock = threading.Lock()
graph = build_graph(checkpointer=sqlite_checkpointer())

app = FastAPI(title="P&ID digitiser")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


def _index() -> dict:
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def _set(rid: str, **fields) -> None:
    with _lock:
        idx = _index()
        idx.setdefault(rid, {}).update(fields)
        INDEX.write_text(json.dumps(idx, indent=1))


def _launch(run_id: str, pdf: Path, name: str) -> dict:
    _set(run_id, run_id=run_id, filename=name, created=time.time(), status="processing")

    def work():
        try:
            out = start_run(graph, run_id, pdf)
            _set(run_id, status="awaiting_review" if "__interrupt__" in out else "done")
        except Exception as exc:  # surface pipeline errors in the dashboard
            _set(run_id, status="failed", error=repr(exc))

    threading.Thread(target=work, daemon=True).start()
    return _index()[run_id]


@app.post("/api/runs")
async def create_run(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "upload a PDF")
    run_id = uuid.uuid4().hex[:8]
    (RUNS / run_id).mkdir(parents=True, exist_ok=True)
    pdf = RUNS / run_id / "input.pdf"
    pdf.write_bytes(await file.read())
    return _launch(run_id, pdf, file.filename)


@app.get("/api/samples")
def samples():
    return sorted(p.stem for p in (ROOT / "data" / "test" / "scanned").glob("*.pdf"))[:10]


@app.post("/api/runs/sample/{sheet_id}")
def create_sample_run(sheet_id: str, variant: str = "scanned"):
    pdf = ROOT / "data" / "test" / variant / f"{sheet_id}.pdf"
    if variant not in ("scanned", "vector") or not pdf.exists():
        raise HTTPException(404, "no such sample")
    return _launch(uuid.uuid4().hex[:8], pdf, f"{sheet_id} ({variant})")


@app.get("/api/runs")
def list_runs():
    return sorted(_index().values(), key=lambda r: r["created"], reverse=True)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    meta = _index().get(run_id)
    if not meta:
        raise HTTPException(404)
    snap = graph.get_state(thread(run_id))
    v = snap.values or {}
    pending = [i for t in snap.tasks for intr in t.interrupts for i in intr.value["pending"]]
    return {**meta, "pages": v.get("pages", []), "items": v.get("final_items") or v.get("checked") or v.get("items", []),
            "pending": pending, "exports": v.get("exports")}


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str, decisions: list[ReviewDecision]):
    snap = graph.get_state(thread(run_id))
    if not any(t.interrupts for t in snap.tasks):
        raise HTTPException(409, "run is not waiting for review")
    resume_run(graph, run_id, [d.model_dump() for d in decisions])
    _set(run_id, status="done")
    return get_run(run_id)


def _file(path: Path) -> FileResponse:
    if not path.resolve().is_relative_to(RUNS.resolve()) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/runs/{run_id}/pages/{index}.png")
def page_image(run_id: str, index: int):
    return _file(RUNS / run_id / "pages" / f"page_{index}.png")


@app.get("/api/runs/{run_id}/crops/{det_id}.png")
def crop_image(run_id: str, det_id: str):
    return _file(RUNS / run_id / "crops" / f"{det_id}.png")


@app.get("/api/runs/{run_id}/mto.csv")
def mto(run_id: str):
    return _file(RUNS / run_id / "mto.csv")


DIST = ROOT / "dashboard" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="dashboard")
