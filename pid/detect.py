"""Stage 2a - symbol detection agent.

A 3300 x 2300 px sheet is too large to downscale to YOLO's input without
shrinking a valve to a few pixels, so we slide a 640 px window with overlap,
run YOLO per tile, shift boxes back to page coordinates and merge duplicates
with our own class-aware NMS (IoU plus intersection-over-smaller, so a valve
cut in half by a tile edge doesn't survive next to its full copy).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pid.schemas import CLASS_NAMES, Box, Detection

TILE, STRIDE = 640, 512
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "pid_symbols.pt"


def tile_starts(n: int) -> list[int]:
    starts = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    if starts[-1] + TILE < n:
        starts.append(n - TILE)
    return starts


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
        iou_thr: float = 0.5, ios_thr: float = 0.3, xios_thr: float = 0.6) -> tuple[list[int], np.ndarray]:
    """Greedy class-aware NMS with fragment merging.

    Returns (kept indices highest score first, boxes). A box of the same class is
    suppressed if IoU > iou_thr. If instead a real share of the smaller box lies
    inside the kept one (IoS > ios_thr) it is a *fragment* (a symbol cut by a tile
    edge, or split where a branch pipe touches it), so the kept box grows to the
    union. On a P&ID two genuine symbols of one class never overlap.
    """
    boxes = boxes.copy()
    order = np.argsort(-scores)
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        while True:  # a merge enlarges the box, which can expose more fragments
            same = (classes == classes[idx]) & ~suppressed
            same[idx] = False
            cand = np.where(same)[0]
            if not len(cand):
                break
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            xx0 = np.maximum(boxes[idx, 0], boxes[cand, 0])
            yy0 = np.maximum(boxes[idx, 1], boxes[cand, 1])
            xx1 = np.minimum(boxes[idx, 2], boxes[cand, 2])
            yy1 = np.minimum(boxes[idx, 3], boxes[cand, 3])
            inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
            iou = inter / (areas[idx] + areas[cand] - inter + 1e-9)
            ios = inter / (np.minimum(areas[idx], areas[cand]) + 1e-9)
            suppressed[cand[iou > iou_thr]] = True
            frags = cand[(iou <= iou_thr) & (ios > ios_thr)]
            if not len(frags):
                break
            suppressed[frags] = True
            boxes[idx, :2] = np.minimum(boxes[idx, :2], boxes[frags, :2].min(axis=0))
            boxes[idx, 2:] = np.maximum(boxes[idx, 2:], boxes[frags, 2:].max(axis=0))
    return cross_class_filter(keep, boxes, scores, xios_thr), boxes


def cross_class_filter(keep: list[int], boxes: np.ndarray, scores: np.ndarray, xios_thr: float) -> list[int]:
    """Symbols on a P&ID never overlap, so a box lying mostly inside a higher-scoring box
    of ANOTHER class is a mislabelled fragment (typically a sliver cut by a tile edge)."""
    out: list[int] = []
    for idx in sorted(keep, key=lambda k: -scores[k]):
        b = boxes[idx]
        area = (b[2] - b[0]) * (b[3] - b[1])
        dup = False
        for j in out:
            o = boxes[j]
            inter = max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
            small = min(area, (o[2] - o[0]) * (o[3] - o[1]))
            if small > 0 and inter / small > xios_thr:
                dup = True
                break
        if not dup:
            out.append(idx)
    return out


@lru_cache(maxsize=2)
def _model(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def detect_symbols(image_path: str, page_index: int, weights: str | Path = DEFAULT_WEIGHTS,
                   conf: float = 0.25, device: str | int | None = None) -> list[Detection]:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    model = _model(str(weights))
    tiles, offsets = [], []
    for ty in tile_starts(h):
        for tx in tile_starts(w):
            tiles.append(cv2.cvtColor(gray[ty:ty + TILE, tx:tx + TILE], cv2.COLOR_GRAY2BGR))
            offsets.append((tx, ty))

    all_boxes, all_scores, all_cls = [], [], []
    for start in range(0, len(tiles), 16):
        results = model.predict(tiles[start:start + 16], imgsz=TILE, conf=conf, iou=0.5,
                                verbose=False, device=device)
        for res, (tx, ty) in zip(results, offsets[start:start + 16]):
            if res.boxes is None or not len(res.boxes):
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += tx
            xyxy[:, [1, 3]] += ty
            all_boxes.append(xyxy)
            all_scores.append(res.boxes.conf.cpu().numpy())
            all_cls.append(res.boxes.cls.cpu().numpy().astype(int))
    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_cls)
    keep, boxes = nms(boxes, scores, classes)

    dets = []
    for n, i in enumerate(sorted(keep, key=lambda k: (boxes[k, 1], boxes[k, 0]))):
        x0, y0, x1, y1 = boxes[i].tolist()
        dets.append(Detection(det_id=f"p{page_index}-d{n:03d}", page_index=page_index,
                              cls=CLASS_NAMES[classes[i]], box=Box(x0=x0, y0=y0, x1=x1, y1=y1),
                              det_conf=round(float(scores[i]), 4)))
    return dets
