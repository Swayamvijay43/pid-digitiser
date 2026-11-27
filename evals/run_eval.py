"""End-to-end evaluation on the held-out test sheets.

Runs the real LangGraph pipeline (scan -> detect -> read_tag ladder -> review
interrupt) on every test sheet, in both forms: the vector PDF (text layer) and
the degraded scan (OCR path). Each run stops at the human-review interrupt, so
what we measure is exactly what the system decides on its own.

    python -m evals.run_eval            # full ladder
    python -m evals.run_eval --no-vlm   # OCR-only ablation
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import replace

from langgraph.checkpoint.memory import InMemorySaver

from evals.common import DATA, ROOT, load_gt, match, sheet_ids
from pid.graph import build_graph, start_run
from pid.read_tags import LadderConfig
from pid.schemas import CLASS_NAMES, Item


def pct(a, b):
    return f"{a / b:.1%}" if b else "n/a"


def evaluate(variant: str, ids: list[str], graph) -> dict:
    det_tp = det_fp = det_fn = cls_wrong = 0
    per_class = defaultdict(Counter)
    skew_err = []
    rung_total, rung_right = Counter(), Counter()
    auto = auto_right = review = review_guess_right = gt_tagged = 0
    line_total = line_right = 0
    mto_exact = 0
    failures = []
    secs = []

    for sid in ids:
        gt = load_gt("test", sid)
        t0 = time.perf_counter()
        out = start_run(graph, f"eval-{variant}-{sid}-{uuid.uuid4().hex[:4]}", DATA / "test" / variant / f"{sid}.pdf")
        secs.append(time.perf_counter() - t0)
        if variant == "scanned":
            skew_err.append(abs(out["pages"][0]["skew_deg"] - gt["scan_skew_deg"]))
        items = [Item.model_validate(i) for i in out.get("checked", [])]  # after sheet-level validation
        gts = gt["symbols"]
        m = match([i.det.box.as_list() for i in items], [i.det.det_conf for i in items], [g["bbox"] for g in gts])
        det_tp += len(m)
        det_fp += len(items) - len(m)
        det_fn += len(gts) - len(m)
        matched_gt = set(m.values())
        for gi, g in enumerate(gts):
            per_class[g["cls"]]["gt"] += 1
            if gi in matched_gt:
                per_class[g["cls"]]["found"] += 1

        pred_mto, gt_mto = Counter(), Counter()
        for g in gts:
            if g["cls"].endswith("_valve") and g["tag"]:
                gt_mto[(g["cls"], g["line_number"].split('"')[0])] += 1

        for pi, item in enumerate(items):
            if pi not in m:
                if item.status == "auto_accepted":
                    auto += 1  # a false detection that slipped through counts as an error
                    failures.append({"sheet": sid, "kind": "false detection auto-accepted",
                                     "cls": item.det.cls, "tag": item.tag})
                else:
                    review += 1
                continue
            g = gts[m[pi]]
            if item.det.cls != g["cls"]:
                cls_wrong += 1
            if not g["tag"]:  # the drawing has no tag here: any accepted tag is borrowed from a neighbour
                if item.status == "auto_accepted":
                    auto += 1
                    failures.append({"sheet": sid, "kind": "tag accepted for untagged symbol",
                                     "cls": g["cls"], "pred": item.tag, "by": item.resolved_by})
                else:
                    review += 1
                continue
            gt_tagged += 1
            right = item.tag == g["tag"] and item.det.cls == g["cls"]
            if item.status == "auto_accepted":
                auto += 1
                auto_right += right
                rung_total[item.resolved_by] += 1
                rung_right[item.resolved_by] += right
                if not right:
                    failures.append({"sheet": sid, "kind": "wrong auto-accept", "cls": g["cls"], "gt": g["tag"],
                                     "pred": item.tag, "by": item.resolved_by,
                                     "attempts": [a.model_dump() for a in item.attempts]})
                if g["line_number"]:
                    line_total += 1
                    line_right += item.line_number == g["line_number"]
                if g["cls"].endswith("_valve") and item.line_number:
                    pred_mto[(item.det.cls, item.line_number.split('"')[0])] += 1
            else:
                review += 1
                review_guess_right += right
                if len(failures) < 400:
                    failures.append({"sheet": sid, "kind": "sent to review", "cls": g["cls"], "gt": g["tag"],
                                     "guess": item.tag, "reason": item.reason,
                                     "attempts": [{"rung": a.rung, "raw": a.raw, "tag": a.tag} for a in item.attempts]})
        mto_exact += pred_mto == gt_mto
        print(f"[{variant}] {sid}: {len(items)} dets, auto {sum(i.status == 'auto_accepted' for i in items)}, "
              f"review {sum(i.status == 'needs_review' for i in items)}, {secs[-1]:.1f}s")

    n_items = det_tp + det_fp
    return {
        "variant": variant, "sheets": len(ids),
        "detection": {"precision": det_tp / max(det_tp + det_fp, 1), "recall": det_tp / max(det_tp + det_fn, 1),
                      "tp": det_tp, "fp": det_fp, "fn": det_fn, "class_errors": cls_wrong,
                      "per_class_recall": {c: per_class[c]["found"] / max(per_class[c]["gt"], 1) for c in CLASS_NAMES}},
        "deskew_mae_deg": sum(skew_err) / len(skew_err) if skew_err else None,
        "tags": {"gt_tagged_matched": gt_tagged, "auto_accepted": auto, "auto_correct": auto_right,
                 "auto_precision": auto_right / max(auto, 1), "review": review,
                 "review_rate": review / max(n_items, 1), "automation_rate": auto / max(n_items, 1),
                 "review_best_guess_correct": review_guess_right,
                 "by_rung": {r: {"accepted": rung_total[r], "correct": rung_right[r]} for r in rung_total}},
        "line_number_accuracy": line_right / max(line_total, 1),
        "mto_exact_sheets": mto_exact,
        "seconds_per_sheet": sum(secs) / len(secs),
        "failures": failures,
    }


def to_markdown(results: list[dict], label: str) -> str:
    out = [f"## {label}", "",
           "| metric | " + " | ".join(r["variant"] for r in results) + " |",
           "|---|" + "---|" * len(results)]

    def row(name, f):
        out.append(f"| {name} | " + " | ".join(f(r) for r in results) + " |")

    row("sheets", lambda r: str(r["sheets"]))
    row("symbol detection precision", lambda r: f"{r['detection']['precision']:.3f}")
    row("symbol detection recall", lambda r: f"{r['detection']['recall']:.3f}")
    row("wrong class on matched symbols", lambda r: str(r["detection"]["class_errors"]))
    row("deskew error (mean, deg)", lambda r: f"{r['deskew_mae_deg']:.3f}" if r["deskew_mae_deg"] is not None else "-")
    row("**auto-accepted tags: precision**", lambda r: f"**{r['tags']['auto_precision']:.4f}** "
                                                       f"({r['tags']['auto_correct']}/{r['tags']['auto_accepted']})")
    row("automation rate (no human)", lambda r: f"{r['tags']['automation_rate']:.1%}")
    row("sent to human review", lambda r: f"{r['tags']['review']} ({r['tags']['review_rate']:.1%})")
    row("review items whose best guess was right", lambda r: f"{r['tags']['review_best_guess_correct']}/{r['tags']['review']}")
    for rung in ("text_layer", "ocr", "vlm_small", "vlm_large"):
        row(f"accepted by {rung} (correct/total)",
            lambda r, rung=rung: (lambda d: f"{d['correct']}/{d['accepted']}" if d else "0")(r["tags"]["by_rung"].get(rung)))
    row("line number linked correctly", lambda r: f"{r['line_number_accuracy']:.1%}")
    row("sheets with an exactly correct valve MTO", lambda r: f"{r['mto_exact_sheets']}/{r['sheets']}")
    row("seconds per sheet", lambda r: f"{r['seconds_per_sheet']:.1f}")
    out += ["", "Per-class detection recall:", "",
            "| class | " + " | ".join(r["variant"] for r in results) + " |", "|---|" + "---|" * len(results)]
    for c in CLASS_NAMES:
        out.append(f"| {c} | " + " | ".join(f"{r['detection']['per_class_recall'][c]:.3f}" for r in results) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--variants", default="vector,scanned")
    ap.add_argument("--start", type=int, default=0, help="first test sheet index (use unseen sheets for a clean run)")
    ap.add_argument("--name", default=None, help="results file suffix")
    args = ap.parse_args()

    cfg = LadderConfig.load()
    if args.no_vlm:
        cfg = replace(cfg, use_vlm=False)
    graph = build_graph(cfg, checkpointer=InMemorySaver())
    ids = sheet_ids("test")[args.start:]
    ids = ids[:args.limit] if args.limit else ids
    results = [evaluate(v, ids, graph) for v in args.variants.split(",")]
    name = args.name or ("ocr_only" if args.no_vlm else "full_ladder")
    (ROOT / "evals" / f"results_{name}.json").write_text(json.dumps(results, indent=1))
    md = to_markdown(results, ("OCR only (VLM rungs disabled)" if args.no_vlm else
                     f"Full ladder (OCR threshold {cfg.ocr_accept:.2f}, {cfg.vlm_small} -> {cfg.vlm_large})")
                     + f" - sheets {ids[0]} to {ids[-1]}")
    (ROOT / "evals" / f"results_{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
