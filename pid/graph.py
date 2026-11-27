"""The digitisation pipeline as a LangGraph state machine.

    scan -> detect -> [Send fan-out: read_tag x N symbols] -> validate -> review (interrupt) -> export

- Checkpointing: every super-step is saved to SQLite, so a run paused for human
  review survives a server restart and resumes exactly where it stopped.
- Send API: one read_tag branch per detected symbol, run in parallel.
- interrupt(): the review node pauses the graph with the low-confidence items;
  the dashboard resumes it with Command(resume=[ReviewDecision, ...]).
"""
from __future__ import annotations

import operator
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, TypedDict

import cv2
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from pid.detect import DEFAULT_WEIGHTS, detect_symbols
from pid.export import build_exports
from pid.read_tags import LadderConfig, read_tag, save_crop
from pid.scan import scan_pdf
from pid.schemas import Detection, Item, ReviewDecision, ScannedPage, TextSpan

ROOT = Path(__file__).resolve().parent.parent


class RunState(TypedDict, total=False):
    run_id: str
    pdf_path: str
    work_dir: str
    pages: list[dict]
    texts: dict[str, list[dict]]
    detections: list[dict]
    items: Annotated[list[dict], operator.add]
    checked: list[dict]
    final_items: list[dict]
    exports: dict[str, Any]


class ReadTask(TypedDict):
    det: dict
    spans: list[dict]
    page_image: str
    work_dir: str


def build_graph(cfg: LadderConfig | None = None, weights: str | Path = DEFAULT_WEIGHTS,
                checkpointer=None):
    cfg = cfg or LadderConfig.load()

    def scan(state: RunState) -> dict:
        pages, texts = scan_pdf(state["pdf_path"], Path(state["work_dir"]) / "pages")
        return {"pages": [p.model_dump() for p in pages],
                "texts": {str(k): [s.model_dump() for s in v] for k, v in texts.items()}}

    def detect(state: RunState) -> dict:
        dets = []
        for p in state["pages"]:
            dets += detect_symbols(p["image_path"], p["page_index"], weights)
        return {"detections": [d.model_dump() for d in dets]}

    def fan_out(state: RunState):
        if not state["detections"]:
            return "validate"
        pages = {p["page_index"]: p for p in state["pages"]}
        return [Send("read_tag", ReadTask(det=d, spans=state["texts"][str(d["page_index"])],
                                          page_image=pages[d["page_index"]]["image_path"],
                                          work_dir=state["work_dir"]))
                for d in state["detections"]]

    def read_one(task: ReadTask) -> dict:
        det = Detection.model_validate(task["det"])
        spans = [TextSpan.model_validate(s) for s in task["spans"]]
        gray = cv2.imread(task["page_image"], cv2.IMREAD_GRAYSCALE)
        crop = save_crop(det, gray, Path(task["work_dir"]) / "crops")
        return {"items": [read_tag(det, spans, crop, cfg).model_dump()]}

    def validate(state: RunState) -> dict:
        """Sheet-level deterministic checks that no single read_tag branch can see."""
        items = sorted((Item.model_validate(i) for i in state.get("items", [])), key=lambda i: i.det.det_id)
        claimed = Counter(i.tag for i in items if i.status == "auto_accepted" and i.tag)
        for i in items:
            if i.status == "auto_accepted" and i.tag and claimed[i.tag] > 1:
                i.status, i.resolved_by = "needs_review", None
                i.reason = f"tag {i.tag} claimed by {claimed[i.tag]} symbols (tags must be unique)"
        return {"checked": [i.model_dump() for i in items]}

    def review(state: RunState) -> dict:
        items = [Item.model_validate(i) for i in state.get("checked", [])]
        pending = [i for i in items if i.status == "needs_review"]
        if pending:
            answer = interrupt({"pending": [i.model_dump() for i in pending]})
            decisions = {d.det_id: d for d in (ReviewDecision.model_validate(x) for x in answer)}
            for i in items:
                d = decisions.get(i.det.det_id)
                if i.status != "needs_review" or d is None:
                    continue
                if d.action == "reject":
                    i.status, i.reason = "rejected", "rejected by reviewer (false detection)"
                elif d.action == "confirm":
                    i.status, i.resolved_by = "human_confirmed", "human"
                else:
                    i.status, i.tag, i.resolved_by = "human_corrected", d.tag, "human"
                if d.action != "reject":
                    i.confidence = 1.0
        return {"final_items": [i.model_dump() for i in items]}

    def export(state: RunState) -> dict:
        items = [Item.model_validate(i) for i in state["final_items"]]
        spans = {int(k): [TextSpan.model_validate(s) for s in v] for k, v in state["texts"].items()}
        return {"exports": build_exports(items, spans, Path(state["work_dir"]))}

    g = StateGraph(RunState)
    g.add_node("scan", scan)
    g.add_node("detect", detect)
    g.add_node("read_tag", read_one)
    g.add_node("validate", validate)
    g.add_node("review", review)
    g.add_node("export", export)
    g.add_edge(START, "scan")
    g.add_edge("scan", "detect")
    g.add_conditional_edges("detect", fan_out, ["read_tag", "validate"])
    g.add_edge("read_tag", "validate")
    g.add_edge("validate", "review")
    g.add_edge("review", "export")
    g.add_edge("export", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str | Path = ROOT / "runs" / "checkpoints.sqlite") -> SqliteSaver:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "max_concurrency": 6}


def start_run(graph, run_id: str, pdf_path: str | Path) -> dict:
    work = ROOT / "runs" / run_id
    work.mkdir(parents=True, exist_ok=True)
    return graph.invoke({"run_id": run_id, "pdf_path": str(pdf_path), "work_dir": str(work)}, thread(run_id))


def resume_run(graph, run_id: str, decisions: list[dict]) -> dict:
    return graph.invoke(Command(resume=decisions), thread(run_id))


if __name__ == "__main__":
    import argparse
    import json
    import uuid

    ap = argparse.ArgumentParser(description="Digitise a P&ID PDF")
    ap.add_argument("pdf")
    ap.add_argument("--auto-confirm", action="store_true", help="confirm every review item (demo only)")
    args = ap.parse_args()
    graph = build_graph(checkpointer=sqlite_checkpointer())
    rid = uuid.uuid4().hex[:8]
    out = start_run(graph, rid, args.pdf)
    items = [Item.model_validate(i) for i in out.get("checked", [])]
    print(f"run {rid}: {len(items)} symbols, "
          f"{sum(i.status == 'auto_accepted' for i in items)} auto-accepted, "
          f"{sum(i.status == 'needs_review' for i in items)} need review")
    if "__interrupt__" in out and args.auto_confirm:
        pending = out["__interrupt__"][0].value["pending"]
        out = resume_run(graph, rid, [{"det_id": p["det"]["det_id"], "action": "confirm"} for p in pending])
    if "exports" in out:
        print(json.dumps(out["exports"], indent=1))
