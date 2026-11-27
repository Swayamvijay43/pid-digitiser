"""Calibrate the OCR auto-accept threshold on the validation split.

For every detected symbol on the scanned validation sheets we take the OCR
reading that passes the tag grammar and record (score, correct?). The accept
threshold is the lowest score whose auto-accepted set still reaches the target
precision. Writes models/ladder.json and evals/CALIBRATION.md.

    python -m evals.calibrate --target 0.995
"""
from __future__ import annotations

import argparse
import json
import tempfile

import numpy as np

from evals.common import DATA, ROOT, load_gt, match, sheet_ids
from pid.detect import detect_symbols
from pid.read_tags import candidate_from_spans
from pid.scan import scan_pdf
from pid.schemas import normalise_tag, tag_is_valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.995)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = []  # (score, correct)
    no_valid = 0
    for sid in sheet_ids("val", args.limit):
        gt = load_gt("val", sid)
        with tempfile.TemporaryDirectory() as tmp:
            pages, texts = scan_pdf(DATA / "val" / "scanned" / f"{sid}.pdf", tmp)
            dets = detect_symbols(pages[0].image_path, 0)
        m = match([d.box.as_list() for d in dets], [d.det_conf for d in dets], [s["bbox"] for s in gt["symbols"]])
        for pi, gi in m.items():
            det, g = dets[pi], gt["symbols"][gi]
            if not g["tag"] or det.cls != g["cls"]:
                continue
            raw, score, _ = candidate_from_spans(det, texts[0])
            tag = normalise_tag(raw, det.cls)
            if tag_is_valid(tag, det.cls) and score is not None:
                rows.append((score, tag == g["tag"]))
            else:
                no_valid += 1
        print(f"{sid}: {len(rows)} grammar-valid OCR readings so far")

    scores = np.array([r[0] for r in rows])
    correct = np.array([r[1] for r in rows])
    table, chosen = [], None
    for t in np.round(np.arange(0.50, 1.0, 0.01), 2):
        sel = scores >= t
        if sel.sum() == 0:
            continue
        prec = correct[sel].mean()
        table.append((t, int(sel.sum()), prec, sel.sum() / len(rows)))
        if chosen is None and prec >= args.target:
            chosen = float(t)
    chosen = chosen if chosen is not None else 0.99

    cfg_path = ROOT / "models" / "ladder.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["ocr_accept"] = chosen
    cfg_path.parent.mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1))

    lines = ["# OCR threshold calibration (val split, scanned sheets)", "",
             f"- Grammar-valid OCR readings: {len(rows)} (overall precision {correct.mean():.3f})",
             f"- Symbols with no grammar-valid OCR reading (always escalated): {no_valid}",
             f"- Target precision for auto-accept: {args.target}",
             f"- **Chosen threshold: {chosen:.2f}**", "",
             "| threshold | accepted | precision | share of valid readings |", "|---|---|---|---|"]
    lines += [f"| {t:.2f} | {n} | {p:.4f} | {c:.1%} |" for t, n, p, c in table[::5] + [r for r in table if r[0] == chosen]]
    (ROOT / "evals" / "CALIBRATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
