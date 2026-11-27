"""Stage 1 - scanning agent.

PDF page -> clean, deskewed grayscale image, plus the text on it.

Hybrid text path: a vector PDF (CAD export) already carries exact text with
coordinates, so we read it from the text layer (deterministic, confidence 1.0).
Only raster scans go through OCR.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

from pid.schemas import RENDER_DPI, Box, ScannedPage, TextSpan

SCALE = RENDER_DPI / 72.0


def render_page(page: fitz.Page) -> np.ndarray:
    pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees, measured from long near-horizontal lines (pipes, borders).

    Returns the angle to pass to cv2.getRotationMatrix2D to straighten the page.
    """
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(bw, 1, np.pi / 1800, threshold=150,
                            minLineLength=int(small.shape[1] * 0.08), maxLineGap=6)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if a > 90:
            a -= 180
        if a < -90:
            a += 180
        if abs(a) <= 8:
            angles.append(a)
            weights.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    a, w = np.array(angles)[order], np.array(weights)[order]
    cdf = np.cumsum(w) / w.sum()
    return float(a[np.searchsorted(cdf, 0.5)])  # length-weighted median


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(gray)
    if abs(angle) < 0.05:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bg = int(np.median(gray))
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg), -angle


def clean(gray: np.ndarray) -> np.ndarray:
    """Remove scan noise while keeping thin lines: median filter + contrast stretch."""
    den = cv2.medianBlur(gray, 3)
    lo, hi = np.percentile(den, (1, 99))
    if hi - lo < 10:
        return den
    return np.clip((den.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def text_layer_spans(page: fitz.Page) -> list[TextSpan]:
    spans = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        spans.append(TextSpan(text=word, box=Box(x0=x0 * SCALE, y0=y0 * SCALE, x1=x1 * SCALE, y1=y1 * SCALE),
                              score=1.0, source="text_layer"))
    return spans


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


OCR_TILE, OCR_OVERLAP = 1400, 200


def ocr_spans(gray: np.ndarray) -> list[TextSpan]:
    """Tiled OCR so 6 pt tag text keeps its full resolution; duplicates in overlaps are merged."""
    engine = _ocr_engine()
    h, w = gray.shape
    spans: list[TextSpan] = []
    step = OCR_TILE - OCR_OVERLAP
    for ty in range(0, max(h - OCR_OVERLAP, 1), step):
        for tx in range(0, max(w - OCR_OVERLAP, 1), step):
            tile = gray[ty:ty + OCR_TILE, tx:tx + OCR_TILE]
            # use_cls=False: pages are deskewed and P&ID text is never upside-down; the 180-degree
            # classifier turned "909" into "606" at 0.996 confidence on a held-out sheet
            result, _ = engine(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), use_cls=False)
            for pts, text, score in result or []:
                xs = [p[0] + tx for p in pts]
                ys = [p[1] + ty for p in pts]
                box = Box(x0=min(xs), y0=min(ys), x1=max(xs) + 0.01, y1=max(ys) + 0.01)
                spans.append(TextSpan(text=text.strip(), box=box, score=float(score), source="ocr"))
    return _dedupe(spans)


def _dedupe(spans: list[TextSpan]) -> list[TextSpan]:
    """Text cut by a tile edge appears twice; keep the wider, more confident copy."""
    spans = sorted(spans, key=lambda s: ((s.box.x1 - s.box.x0), s.score), reverse=True)
    kept: list[TextSpan] = []
    for s in spans:
        a = s.box
        dup = False
        for k in kept:
            b = k.box
            ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            smaller = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
            if smaller > 0 and ix * iy / smaller > 0.5:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def scan_pdf(pdf_path: str | Path, work_dir: str | Path) -> tuple[list[ScannedPage], dict[int, list[TextSpan]]]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pages, texts = [], {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            gray = render_page(page)
            spans = text_layer_spans(page)
            has_text = len(spans) >= 5
            if has_text:
                skew = 0.0  # vector page: geometry is exact, never rotate it
                img = gray
            else:
                img, skew = deskew(gray)
                img = clean(img)
                spans = ocr_spans(img)
            path = work / f"page_{i}.png"
            cv2.imwrite(str(path), img)
            pages.append(ScannedPage(page_index=i, image_path=str(path), width=img.shape[1],
                                     height=img.shape[0], has_text_layer=has_text, skew_deg=round(skew, 3)))
            texts[i] = spans
    return pages, texts
