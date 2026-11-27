"""Stage 2b - tag reading agent with a per-agent escalation ladder.

    text layer  ->  OCR  ->  small VLM (qwen3-vl:2b)  ->  larger VLM (qwen3-vl:4b)  ->  human

Each rung is only tried when the cheaper rung could not produce a reading
that passes a deterministic check. A model's answer alone never auto-accepts:
an OCR reading needs the class's tag grammar plus a calibrated score, and a
VLM reading must also agree with an independent source (OCR or the other VLM).
Everything else goes to the human review queue with the reason attached.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pid.schemas import (INSTRUMENT_CLASSES, LINE_NUMBER, line_number_is_valid, Attempt, Detection, Item, TagReading, TextSpan,
                         normalise_line_number, normalise_tag, tag_is_valid)

HUMAN_NAMES = {
    "gate_valve": "gate valve", "globe_valve": "globe valve", "check_valve": "check valve",
    "control_valve": "control valve", "centrifugal_pump": "centrifugal pump", "heat_exchanger": "heat exchanger",
    "vessel": "vessel", "instrument_field": "field instrument bubble", "instrument_panel": "panel instrument bubble",
}
FORMAT_HINT = {
    "gate_valve": "HV-1234", "globe_valve": "HV-1234", "check_valve": "NRV-123", "control_valve": "FV-123",
    "centrifugal_pump": "P-123 or P-123A", "heat_exchanger": "E-123", "vessel": "V-123",
    "instrument_field": "FT-123 (letters on the top line, number on the bottom line)",
    "instrument_panel": "FIC-123 (letters above the line, number below it)",
}


@dataclass
class LadderConfig:
    ocr_accept: float = 0.90       # calibrated on the val split by evals/calibrate.py
    det_accept: float = 0.50       # below this the symbol class itself is in doubt
    use_vlm: bool = True
    vlm_small: str = "qwen3-vl:2b"
    vlm_large: str = "qwen3-vl:4b"
    ollama_host: str | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LadderConfig":
        p = Path(path) if path else Path(__file__).resolve().parent.parent / "models" / "ladder.json"
        cfg = cls()
        if p.exists():
            for k, v in json.loads(p.read_text()).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


def _centre(b):
    return (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2


def tag_region(det: Detection):
    b = det.box
    if det.cls in INSTRUMENT_CLASSES:
        return b.x0 - 4, b.y0 - 4, b.x1 + 4, b.y1 + 4
    return b.x0 - 70, b.y0 - 45, b.x1 + 70, b.y1 + 45


def candidate_from_spans(det: Detection, spans: list[TextSpan]) -> tuple[str | None, float | None, str]:
    """Pick the text that belongs to this symbol. Returns (raw text, score, source)."""
    x0, y0, x1, y1 = tag_region(det)
    inside = [s for s in spans if x0 <= _centre(s.box)[0] <= x1 and y0 <= _centre(s.box)[1] <= y1]
    inside = [s for s in inside if not LINE_NUMBER.match(normalise_line_number(s.text))]
    if not inside:
        return None, None, "none"
    source = inside[0].source
    if det.cls in INSTRUMENT_CLASSES:
        inside.sort(key=lambda s: _centre(s.box)[1])
        raw = "-".join(s.text for s in inside[:2])
        return raw, min(s.score for s in inside[:2]), source
    cx, cy = _centre(det.box)
    valid = [s for s in inside if tag_is_valid(normalise_tag(s.text, det.cls), det.cls)]
    pool = valid or inside
    best = min(pool, key=lambda s: math.hypot(_centre(s.box)[0] - cx, _centre(s.box)[1] - cy))
    return best.text, best.score, best.source


def line_number_for(det: Detection, spans: list[TextSpan]) -> str | None:
    if det.cls in INSTRUMENT_CLASSES or det.cls == "vessel":
        return None
    cx, cy = _centre(det.box)
    best, best_dx = None, 1e9
    for s in spans:
        txt = normalise_line_number(s.text)
        if not line_number_is_valid(txt):
            continue
        sx, sy = _centre(s.box)
        if abs(sy - cy) < 60 and abs(sx - cx) < best_dx:
            best, best_dx = txt, abs(sx - cx)
    return best


def save_crop(det: Detection, gray: np.ndarray, out_dir: Path) -> str:
    x0, y0, x1, y1 = (int(round(v)) for v in tag_region(det))
    h, w = gray.shape
    crop = gray[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    scale = max(1.0, 220.0 / max(crop.shape[0], 1))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{det.det_id}.png"
    cv2.imwrite(str(path), crop)
    return str(path)


def ask_vlm(model: str, crop_path: str, cls: str, host: str | None = None) -> tuple[str | None, str]:
    """Ask a local VLM for the tag. Output is constrained to TagReading's JSON schema
    and then validated again with Pydantic. Returns (tag or None, raw response)."""
    import ollama

    client = ollama.Client(host=host) if host else ollama.Client()
    where = "inside the circle" if cls in INSTRUMENT_CLASSES else "printed next to the symbol (above or below it)"
    prompt = (f"This is a crop from a P&ID engineering drawing showing a {HUMAN_NAMES[cls]} "
              f"in the centre. Read the tag {where}. Expected format: {FORMAT_HINT[cls]}. "
              "Copy the characters exactly; do not guess digits you cannot see. "
              "Ignore pipe line numbers that contain an inch mark (\"). "
              'If no tag is visible, return {"tag": null, "legible": false}.')
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt, "images": [crop_path]}],
                  format=TagReading.model_json_schema(), options={"temperature": 0, "num_predict": 1536})
    try:
        resp = client.chat(think=False, **kwargs)
    except Exception:  # model without a thinking switch
        resp = client.chat(**kwargs)
    raw = resp["message"]["content"]
    try:
        reading = TagReading.model_validate_json(raw)
    except Exception:
        return None, raw
    return (reading.tag if reading.legible else None), raw


def consistent_with_ocr(vlm_tag: str | None, ocr_tag: str | None) -> bool:
    """True if the VLM tag contains everything OCR did read: the same number, and the OCR
    prefix letters in order (OCR's known failure is dropping letters, e.g. 'F-388' for 'FI-388').
    OCR must have read at least one letter and the full number, so the VLM only fills gaps."""
    if not vlm_tag or not ocr_tag or "-" not in ocr_tag:
        return False
    o_pre, _, o_num = ocr_tag.partition("-")
    v_pre, _, v_num = vlm_tag.partition("-")
    if not o_pre or not o_num.isdigit() or len(o_num) < 3 or o_num != v_num.rstrip("AB"):
        return False
    it = iter(v_pre)
    return all(ch in it for ch in o_pre)  # subsequence test


def read_tag(det: Detection, spans: list[TextSpan], crop_path: str, cfg: LadderConfig) -> Item:
    attempts: list[Attempt] = []
    raw, score, source = candidate_from_spans(det, spans)
    first = normalise_tag(raw, det.cls)
    first_valid = tag_is_valid(first, det.cls)
    rung = "text_layer" if source == "text_layer" else "ocr"
    attempts.append(Attempt(rung=rung, raw=raw, tag=first, valid=first_valid, score=score))
    line_no = line_number_for(det, spans)

    def done(tag, status, by, reading_conf, reason):
        conf = round(det.det_conf * reading_conf, 4)
        if status == "auto_accepted" and det.det_conf < cfg.det_accept:
            status, reason = "needs_review", f"low detection confidence ({det.det_conf:.2f}); " + reason
        return Item(det=det, tag=tag, status=status, resolved_by=by if status == "auto_accepted" else None,
                    confidence=conf, reason=reason, attempts=attempts, line_number=line_no, crop_path=crop_path)

    if first_valid and rung == "text_layer":
        return done(first, "auto_accepted", "text_layer", 1.0, "exact text from the PDF text layer")
    if first_valid and score is not None and score >= cfg.ocr_accept:
        return done(first, "auto_accepted", "ocr", score, f"OCR {score:.2f} >= {cfg.ocr_accept:.2f} and tag grammar ok")
    if not cfg.use_vlm:
        return done(first if first_valid else None, "needs_review", None, (score or 0) * 0.5,
                    "OCR below threshold or failed tag grammar (VLM disabled)")

    small, raw_s = ask_vlm(cfg.vlm_small, crop_path, det.cls, cfg.ollama_host)
    small = normalise_tag(small, det.cls)
    small_valid = tag_is_valid(small, det.cls)
    attempts.append(Attempt(rung="vlm_small", raw=raw_s, tag=small, valid=small_valid))
    if small_valid and first_valid and small == first:
        return done(small, "auto_accepted", "vlm_small", 0.92, "small VLM agrees with OCR")
    if small_valid and consistent_with_ocr(small, first):
        return done(small, "auto_accepted", "vlm_small", 0.85, f"small VLM {small} fills characters OCR dropped ({first})")

    large, raw_l = ask_vlm(cfg.vlm_large, crop_path, det.cls, cfg.ollama_host)
    large = normalise_tag(large, det.cls)
    large_valid = tag_is_valid(large, det.cls)
    attempts.append(Attempt(rung="vlm_large", raw=raw_l, tag=large, valid=large_valid))
    if large_valid and ((small_valid and large == small) or (first_valid and large == first)):
        partner = "small VLM" if large == small else "OCR"
        return done(large, "auto_accepted", "vlm_large", 0.90, f"large VLM agrees with {partner}")
    if large_valid and consistent_with_ocr(large, first):
        return done(large, "auto_accepted", "vlm_large", 0.85, f"large VLM {large} fills characters OCR dropped ({first})")

    guesses = [t for t, ok in ((large, large_valid), (small, small_valid), (first, first_valid)) if ok]
    if not guesses and not any([first, small, large]):
        reason = "no tag found by OCR or either VLM"
    elif not guesses:
        reason = "no reading passes the tag grammar"
    else:
        reason = "readings disagree: " + ", ".join(f"{a.rung}={a.tag}" for a in attempts if a.tag)
    return done(guesses[0] if guesses else None, "needs_review", None, 0.4 if guesses else 0.1, reason)
