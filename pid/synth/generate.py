"""Synthetic P&ID generator.

Draws ISA-style P&ID sheets as vector PDFs with PyMuPDF and writes exact ground
truth for every symbol, tag and line number. Ground truth is what makes honest
evaluation possible: every number in evals/RESULTS.md is measured against it.

Outputs, per split:
  <out>/<split>/vector/<id>.pdf   vector PDF with a real text layer (CAD export)
  <out>/<split>/scanned/<id>.pdf  raster-only PDF: rotated, noisy, blurred, JPEG'd (a bad scan)
  <out>/<split>/gt/<id>.json      ground truth in pixel coordinates at RENDER_DPI

Usage:
  python -m pid.synth.generate --out data --train 120 --val 30 --test 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pymupdf as fitz
import numpy as np

from pid.schemas import CLASS_NAMES, RENDER_DPI

PAGE_W, PAGE_H = 1191.0, 842.0  # A3 landscape, points
SCALE = RENDER_DPI / 72.0
BLACK = (0, 0, 0)
FONT = "helv"

INSTRUMENT_LETTERS = ["FT", "PT", "TT", "LT", "PI", "TI", "FI", "LG", "PG", "TE"]
CONTROLLER_FOR = {"F": "FIC", "P": "PIC", "L": "LIC", "T": "TIC"}
SERVICES = ["P", "CW", "ST", "HC", "N", "IA", "PW"]
SPECS = ["A1A", "B2C", "C1A", "D3B", "A2B"]
SIZES = ["2", "3", "4", "6", "8", "10", "12"]


@dataclass
class Sheet:
    rng: random.Random
    page: fitz.Page
    symbols: list[dict] = field(default_factory=list)
    line_numbers: list[dict] = field(default_factory=list)
    occupied: list[fitz.Rect] = field(default_factory=list)
    used_numbers: set[int] = field(default_factory=set)

    # ---- bookkeeping -------------------------------------------------
    def free(self, rect: fitz.Rect, pad: float = 4.0) -> bool:
        r = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
        inside = r.x0 > 30 and r.y0 > 40 and r.x1 < PAGE_W - 30 and r.y1 < 700
        return inside and not any(r.intersects(o) for o in self.occupied)

    def occupy(self, rect: fitz.Rect) -> None:
        self.occupied.append(fitz.Rect(rect))

    def number(self, lo: int = 100, hi: int = 999) -> int:
        while True:
            n = self.rng.randint(lo, hi)
            if n not in self.used_numbers:
                self.used_numbers.add(n)
                return n

    # ---- primitives --------------------------------------------------
    def shape(self):
        return self.page.new_shape()

    def text(self, x: float, y: float, s: str, size: float = 7.0, center: bool = True) -> fitz.Rect:
        """Draw text with its baseline at y; x is the centre when center=True. Returns its box."""
        w = fitz.get_text_length(s, fontname=FONT, fontsize=size)
        x0 = x - w / 2 if center else x
        self.page.insert_text((x0, y), s, fontsize=size, fontname=FONT, color=BLACK)
        return fitz.Rect(x0, y - size * 0.8, x0 + w, y + size * 0.25)

    def pipe(self, p1, p2, width: float = 1.2, dashed: bool = False) -> None:
        sh = self.shape()
        sh.draw_line(p1, p2)
        sh.finish(color=BLACK, width=width, dashes="[3 2] 0" if dashed else None)
        sh.commit()

    def add_symbol(self, cls: str, bbox: fitz.Rect, tag: str | None, tag_rect: fitz.Rect | None,
                   line_number: str | None = None) -> None:
        self.symbols.append({"cls": cls, "bbox_pt": list(bbox), "tag": tag,
                             "tag_bbox_pt": list(tag_rect) if tag_rect else None,
                             "line_number": line_number})
        self.occupy(bbox)
        if tag_rect:
            self.occupy(tag_rect)


# ---- symbol drawers: each returns the symbol's bounding box (points) ------

def bowtie(sh, cx, cy, w=18.0, h=11.0, fill_left=False):
    left = [(cx - w / 2, cy - h / 2), (cx - w / 2, cy + h / 2), (cx, cy)]
    right = [(cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx, cy)]
    sh.draw_polyline(left)
    sh.finish(color=BLACK, fill=BLACK if fill_left else None, width=1.1, closePath=True)
    sh.draw_polyline(right)
    sh.finish(color=BLACK, width=1.1, closePath=True)
    return fitz.Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def draw_gate(s: Sheet, cx, cy):
    sh = s.shape(); r = bowtie(sh, cx, cy); sh.commit(); return r


def draw_globe(s: Sheet, cx, cy):
    sh = s.shape(); r = bowtie(sh, cx, cy)
    sh.draw_circle((cx, cy), 2.8); sh.finish(color=BLACK, fill=BLACK); sh.commit(); return r


def draw_check(s: Sheet, cx, cy):
    sh = s.shape(); r = bowtie(sh, cx, cy, fill_left=True)
    sh.draw_line((cx + 9, cy - 8), (cx + 9, cy + 8)); sh.finish(color=BLACK, width=1.1)
    sh.commit(); return r | fitz.Rect(cx + 8, cy - 8, cx + 10, cy + 8)


def draw_control(s: Sheet, cx, cy):
    sh = s.shape(); r = bowtie(sh, cx, cy)
    sh.draw_line((cx, cy), (cx, cy - 11)); sh.finish(color=BLACK, width=1.0)
    arc = [(cx + 8 * math.cos(a), cy - 11 - 8 * math.sin(a)) for a in np.linspace(0, math.pi, 14)]
    sh.draw_polyline(arc); sh.finish(color=BLACK, width=1.0, closePath=True)
    sh.commit()
    return fitz.Rect(cx - 9, cy - 19.5, cx + 9, cy + 5.5)


def draw_pump(s: Sheet, cx, cy):
    sh = s.shape()
    sh.draw_circle((cx, cy), 12); sh.finish(color=BLACK, width=1.2)
    sh.draw_line((cx, cy - 12), (cx + 16, cy - 12)); sh.finish(color=BLACK, width=1.2)
    sh.draw_polyline([(cx - 9, cy + 16), (cx + 9, cy + 16), (cx, cy + 8)])
    sh.finish(color=BLACK, width=1.0, closePath=True)
    sh.commit()
    return fitz.Rect(cx - 12, cy - 13, cx + 16, cy + 16.5)


def draw_exchanger(s: Sheet, cx, cy):
    sh = s.shape()
    sh.draw_circle((cx, cy), 15); sh.finish(color=BLACK, width=1.2)
    zz = [(cx - 15, cy)] + [(cx - 10 + 5 * i, cy + (-7 if i % 2 == 0 else 7)) for i in range(5)] + [(cx + 15, cy)]
    sh.draw_polyline(zz); sh.finish(color=BLACK, width=1.0)
    sh.commit()
    return fitz.Rect(cx - 15, cy - 15, cx + 15, cy + 15)


def draw_vessel(s: Sheet, cx, cy):
    sh = s.shape()
    sh.draw_rect(fitz.Rect(cx - 17, cy - 40, cx + 17, cy + 40), radius=0.45)
    sh.finish(color=BLACK, width=1.3)
    sh.commit()
    return fitz.Rect(cx - 17, cy - 40, cx + 17, cy + 40)


def draw_bubble(s: Sheet, cx, cy, letters: str, number: str, panel: bool):
    sh = s.shape()
    sh.draw_circle((cx, cy), 12.5); sh.finish(color=BLACK, width=1.0)
    if panel:
        sh.draw_line((cx - 12.5, cy), (cx + 12.5, cy)); sh.finish(color=BLACK, width=0.9)
    sh.commit()
    if panel:
        s.text(cx, cy - 2.6, letters, 6.0); s.text(cx, cy + 7.6, number, 6.0)
    else:
        s.text(cx, cy - 1.2, letters, 6.0); s.text(cx, cy + 6.6, number, 6.0)
    return fitz.Rect(cx - 12.5, cy - 12.5, cx + 12.5, cy + 12.5)


INLINE = {"gate_valve": draw_gate, "globe_valve": draw_globe, "check_valve": draw_check,
          "control_valve": draw_control, "centrifugal_pump": draw_pump, "heat_exchanger": draw_exchanger}
INLINE_WEIGHTS = {"gate_valve": 0.26, "globe_valve": 0.16, "check_valve": 0.14, "control_valve": 0.22,
                  "centrifugal_pump": 0.11, "heat_exchanger": 0.11}
HALF_WIDTH = {"gate_valve": 9, "globe_valve": 9, "check_valve": 10, "control_valve": 9,
              "centrifugal_pump": 16, "heat_exchanger": 15, "vessel": 17}


def tag_for(s: Sheet, cls: str) -> str:
    r = s.rng
    if cls in ("gate_valve", "globe_valve"):
        return f"HV-{s.number(1000, 9999)}"
    if cls == "check_valve":
        return f"NRV-{s.number(100, 999)}"
    if cls == "centrifugal_pump":
        return f"P-{s.number(100, 999)}{r.choice(['', 'A', 'B'])}"
    if cls == "heat_exchanger":
        return f"E-{s.number(100, 999)}"
    if cls == "vessel":
        return f"V-{s.number(100, 999)}"
    raise ValueError(cls)


def place_label(s: Sheet, cx: float, bbox: fitz.Rect, tag: str, prefer_below: bool) -> fitz.Rect | None:
    order = [True, False] if prefer_below else [False, True]
    for below in order:
        y = bbox.y1 + 9.5 if below else bbox.y0 - 4.0
        w = fitz.get_text_length(tag, fontname=FONT, fontsize=7.0)
        rect = fitz.Rect(cx - w / 2, y - 5.6, cx + w / 2, y + 1.75)
        if s.free(rect, pad=2):
            return s.text(cx, y, tag, 7.0)
    return None


def build_sheet(rng: random.Random, doc: fitz.Document, sheet_id: str) -> dict:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    s = Sheet(rng=rng, page=page)

    # Border, title block and notes: distractor text the pipeline must ignore.
    sh = s.shape()
    sh.draw_rect(fitz.Rect(20, 20, PAGE_W - 20, PAGE_H - 20)); sh.finish(color=BLACK, width=1.5)
    tb = fitz.Rect(PAGE_W - 360, PAGE_H - 120, PAGE_W - 20, PAGE_H - 20)
    sh.draw_rect(tb); sh.finish(color=BLACK, width=1.2)
    sh.draw_line((tb.x0, tb.y0 + 40), (tb.x1, tb.y0 + 40)); sh.finish(color=BLACK, width=0.8)
    sh.commit()
    s.text(tb.x0 + 10, tb.y0 + 25, "PIPING & INSTRUMENTATION DIAGRAM", 11, center=False)
    s.text(tb.x0 + 10, tb.y0 + 62, f"DWG NO. PID-{sheet_id.upper()}   REV {rng.choice('ABC')}", 9, center=False)
    s.text(tb.x0 + 10, tb.y0 + 82, "SYNTHETIC DATA - NOT FOR CONSTRUCTION", 8, center=False)
    s.text(40, PAGE_H - 95, "NOTES:", 8, center=False)
    s.text(40, PAGE_H - 82, "1. ALL DIMENSIONS IN MM UNLESS NOTED.", 7, center=False)
    s.text(40, PAGE_H - 70, "2. INSTRUMENT TAGS PER ISA 5.1.", 7, center=False)
    s.occupy(fitz.Rect(20, 705, PAGE_W - 20, PAGE_H - 20))

    n_lines = rng.choice([3, 3, 4])
    ys = np.linspace(140, 610, n_lines) + np.array([rng.uniform(-12, 12) for _ in range(n_lines)])
    line_rows = []

    for y in ys:
        y = float(y)
        size, service, spec = rng.choice(SIZES), rng.choice(SERVICES), rng.choice(SPECS)
        line_no = f'{size}"-{service}-{s.number(1000, 9999)}-{spec}'
        x0, x1 = rng.uniform(80, 140), rng.uniform(990, 1110)

        items: list[tuple[str, float]] = [("vessel", x0)]
        x = x0 + rng.uniform(95, 140)
        while x < x1 - 70:
            cls = rng.choices(list(INLINE_WEIGHTS), weights=list(INLINE_WEIGHTS.values()))[0]
            items.append((cls, x))
            x += rng.uniform(85, 150)
        if rng.random() < 0.5:
            items.append(("vessel", x1))

        extents = []
        for cls, cx in items:
            if cls == "vessel":
                bbox = draw_vessel(s, cx, y)
                tag = tag_for(s, cls)
                lab = place_label(s, cx, bbox, tag, prefer_below=False)
                s.add_symbol("vessel", bbox, tag if lab else None, lab)
                extents.append((cx - 17, cx + 17))
                continue
            bbox = INLINE[cls](s, cx, y)
            if cls == "control_valve":
                var = rng.choice("FPLT")
                loop = s.number(100, 999)
                tag = f"{var}V-{loop}"
                lab = place_label(s, cx, bbox, tag, prefer_below=True)
                s.add_symbol(cls, bbox, tag if lab else None, lab, line_no)
                # controller bubble above the actuator, joined by a dashed signal line
                by = y - rng.uniform(52, 62)
                brect = fitz.Rect(cx - 12.5, by - 12.5, cx + 12.5, by + 12.5)
                if s.free(brect, pad=3):
                    s.pipe((cx, by + 12.5), (cx, bbox.y0), width=0.8, dashed=True)
                    panel = rng.random() < 0.6
                    b = draw_bubble(s, cx, by, CONTROLLER_FOR[var], str(loop), panel)
                    s.add_symbol("instrument_panel" if panel else "instrument_field", b,
                                 f"{CONTROLLER_FOR[var]}-{loop}", b)
                    s.occupy(fitz.Rect(cx - 2, by, cx + 2, bbox.y0))
            else:
                tag = tag_for(s, cls)
                lab = place_label(s, cx, bbox, tag, prefer_below=rng.random() < 0.3)
                s.add_symbol(cls, bbox, tag if lab else None, lab, line_no)
            extents.append((cx - HALF_WIDTH[cls], cx + HALF_WIDTH[cls]))

        end_x = x1 if items[-1][0] == "vessel" else x1 + 30
        # pipe segments between symbols
        bounds = sorted(extents)
        for (a0, a1), (b0, _) in zip(bounds, bounds[1:]):
            if b0 - a1 > 1:
                s.pipe((a1, y), (b0, y))
        if items[-1][0] != "vessel":
            s.pipe((bounds[-1][1], y), (end_x, y))
            sh = s.shape()
            sh.draw_polyline([(end_x, y - 6), (end_x + 22, y - 6), (end_x + 30, y), (end_x + 22, y + 6), (end_x, y + 6)])
            sh.finish(color=BLACK, width=1.0, closePath=True); sh.commit()
            s.text(end_x + 13, y + 16, f"TO U-{rng.randint(100, 900)}", 6.0)

        # standalone transmitters/gauges tapped off the pipe
        for _ in range(rng.randint(1, 3)):
            tx = rng.uniform(x0 + 60, x1 - 60)
            above = rng.random() < 0.6
            by = y - 48 if above else y + 48
            brect = fitz.Rect(tx - 12.5, by - 12.5, tx + 12.5, by + 12.5)
            stem = fitz.Rect(tx - 2, min(by, y) + (12.5 if above else 0), tx + 2, max(by, y) - (0 if above else 12.5))
            if s.free(brect, pad=4) and s.free(stem, pad=1) and not any(a - 6 < tx < b + 6 for a, b in extents):
                letters = rng.choice(INSTRUMENT_LETTERS)
                num = s.number(100, 999)
                s.pipe((tx, y), (tx, by + (12.5 if above else -12.5)), width=0.8)
                b = draw_bubble(s, tx, by, letters, str(num), panel=False)
                s.add_symbol("instrument_field", b, f"{letters}-{num}", b)
                s.occupy(stem)

        # line number label in the widest free gap along the pipe
        gaps = [(a1, b0) for (_, a1), (b0, _) in zip(bounds, bounds[1:])]
        gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
        for g0, g1 in gaps:
            w = fitz.get_text_length(line_no, fontname=FONT, fontsize=6.5)
            if g1 - g0 < w + 16:
                continue
            cx = (g0 + g1) / 2
            rect = fitz.Rect(cx - w / 2, y - 9.6, cx + w / 2, y - 2.5)
            if s.free(rect, pad=1):
                r = s.text(cx, y - 4.5, line_no, 6.5)
                s.line_numbers.append({"text": line_no, "bbox_pt": list(r)})
                s.occupy(r)
                break
        line_rows.append({"y": y, "x0": x0, "x1": end_x, "line_number": line_no})
        s.occupy(fitz.Rect(x0, y - 2, end_x, y + 2))

    # vertical tie-ins between neighbouring process lines
    for a, b in zip(line_rows, line_rows[1:]):
        for _ in range(8):
            vx = rng.uniform(max(a["x0"], b["x0"]) + 50, min(a["x1"], b["x1"]) - 50)
            col = fitz.Rect(vx - 2, a["y"] + 8, vx + 2, b["y"] - 8)
            if s.free(col, pad=3):
                s.pipe((vx, a["y"]), (vx, b["y"]))
                s.occupy(col)
                break

    def px(r):
        return [round(v * SCALE, 1) for v in r] if r else None

    return {
        "sheet_id": sheet_id,
        "dpi": RENDER_DPI,
        "symbols": [{**sym, "bbox": px(sym["bbox_pt"]), "tag_bbox": px(sym["tag_bbox_pt"])} for sym in s.symbols],
        "line_numbers": [{"text": ln["text"], "bbox": px(ln["bbox_pt"])} for ln in s.line_numbers],
    }


# ---- scan degradation ---------------------------------------------------

def render(page: fitz.Page) -> np.ndarray:
    pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def degrade(img: np.ndarray, rng: random.Random, max_angle: float) -> tuple[np.ndarray, float]:
    """Make a clean render look like a scan. Returns the image and the skew angle applied (deg)."""
    out = img.astype(np.float32)
    out = out * rng.uniform(0.72, 0.92) + rng.uniform(15, 45)            # grey paper, faded ink
    if rng.random() < 0.85:
        out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.6, 1.5))
    out += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(5, 16), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    speck = np.random.default_rng(rng.randint(0, 2**31)).random(out.shape) < rng.uniform(2e-5, 1.5e-4)
    out[speck] = rng.randint(20, 80)
    angle = rng.uniform(-max_angle, max_angle)
    if abs(angle) > 1e-3:
        h, w = out.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        bg = int(np.median(out))
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 65)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), angle


def rotate_boxes(boxes: list[list[float]], angle: float, w: int, h: int) -> list[list[float]]:
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = []
    for x0, y0, x1, y1 in boxes:
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]).T
        t = m @ pts
        out.append([float(t[0].min()), float(t[1].min()), float(t[0].max()), float(t[1].max())])
    return out


def image_to_pdf(img: np.ndarray, path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    page.insert_image(page.rect, stream=enc.tobytes())
    doc.save(path, garbage=3, deflate=True)


# ---- YOLO tiles ---------------------------------------------------------

TILE, STRIDE = 640, 512


def tile_starts(n: int) -> list[int]:
    starts = list(range(0, max(n - TILE, 0) + 1, STRIDE))
    if starts[-1] + TILE < n:
        starts.append(n - TILE)
    return starts


def write_tiles(img: np.ndarray, boxes: list[list[float]], classes: list[int], stem: str,
                img_dir: Path, lbl_dir: Path, rng: random.Random) -> int:
    h, w = img.shape
    n = 0
    for ty in tile_starts(h):
        for tx in tile_starts(w):
            lines = []
            for (x0, y0, x1, y1), c in zip(boxes, classes):
                cx0, cy0 = max(x0, tx), max(y0, ty)
                cx1, cy1 = min(x1, tx + TILE), min(y1, ty + TILE)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < 0.6 * (x1 - x0) * (y1 - y0):
                    continue  # mostly cut off: leave it to the neighbouring tile
                lines.append(f"{c} {((cx0 + cx1) / 2 - tx) / TILE:.6f} {((cy0 + cy1) / 2 - ty) / TILE:.6f} "
                             f"{(cx1 - cx0) / TILE:.6f} {(cy1 - cy0) / TILE:.6f}")
            if not lines and rng.random() > 0.1:
                continue  # keep a few empty tiles as negatives
            name = f"{stem}_{ty}_{tx}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img[ty:ty + TILE, tx:tx + TILE], [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    yolo = out / "yolo"
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        rng = random.Random(f"{args.seed}-{split}")
        for sub in ("vector", "scanned", "gt"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        if split != "test":
            (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
        tiles = 0
        for i in range(count):
            sid = f"{split}-{i:03d}"
            doc = fitz.open()
            gt = build_sheet(rng, doc, sid)
            doc.save(out / split / "vector" / f"{sid}.pdf", garbage=3, deflate=True)
            clean = render(doc[0])
            # scanned copy: up to 3 deg skew for val/test; train scans keep a tiny residual skew
            scan, angle = degrade(clean, rng, max_angle=3.0 if split != "train" else 0.4)
            image_to_pdf(scan, out / split / "scanned" / f"{sid}.pdf")
            gt["scan_skew_deg"] = round(angle, 3)
            (out / split / "gt" / f"{sid}.json").write_text(json.dumps(gt, indent=1))
            if split != "test":
                boxes = [sym["bbox"] for sym in gt["symbols"]]
                classes = [CLASS_NAMES.index(sym["cls"]) for sym in gt["symbols"]]
                h, w = clean.shape
                # train on a lightly degraded, unrotated copy plus the clean one; the pipeline deskews first
                tiles += write_tiles(degrade(clean, rng, 0.0)[0], boxes, classes, f"{sid}_d",
                                     yolo / "images" / split, yolo / "labels" / split, rng)
                if split == "train":
                    rb = rotate_boxes(boxes, angle, w, h)
                    tiles += write_tiles(scan, rb, classes, f"{sid}_s",
                                         yolo / "images" / split, yolo / "labels" / split, rng)
        print(f"{split}: {count} sheets, {tiles} tiles")

    (yolo / "data.yaml").write_text(
        f"path: {yolo.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))


if __name__ == "__main__":
    main()
