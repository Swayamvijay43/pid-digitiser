"""Shared helpers for calibration and evaluation: load ground truth, match detections."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_gt(split: str, sheet_id: str) -> dict:
    return json.loads((DATA / split / "gt" / f"{sheet_id}.json").read_text())


def sheet_ids(split: str, limit: int | None = None) -> list[str]:
    ids = sorted(p.stem for p in (DATA / split / "gt").glob("*.json"))
    return ids[:limit] if limit else ids


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(pred_boxes: list[list[float]], pred_scores: list[float], gt_boxes: list[list[float]],
          thr: float = 0.5) -> dict[int, int]:
    """Greedy class-agnostic matching, highest score first. Returns {pred_idx: gt_idx}."""
    used, out = set(), {}
    for p in np.argsort(-np.asarray(pred_scores)):
        best, best_iou = None, thr
        for g, gb in enumerate(gt_boxes):
            if g in used:
                continue
            v = iou(pred_boxes[p], gb)
            if v >= best_iou:
                best, best_iou = g, v
        if best is not None:
            used.add(best)
            out[int(p)] = best
    return out
