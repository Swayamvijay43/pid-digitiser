"""Agent-level data contracts.

Every agent in the pipeline reads and writes these Pydantic models, so a bad
payload fails loudly at the boundary instead of leaking into the export. The
regexes are the deterministic checks that decide whether a tag reading is
plausible; an LLM's answer is never accepted on its own say-so.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RENDER_DPI = 200

CLASS_NAMES = [
    "gate_valve", "globe_valve", "check_valve", "control_valve",
    "centrifugal_pump", "heat_exchanger", "vessel",
    "instrument_field", "instrument_panel",
]
SymbolClass = Literal[
    "gate_valve", "globe_valve", "check_valve", "control_valve",
    "centrifugal_pump", "heat_exchanger", "vessel",
    "instrument_field", "instrument_panel",
]

INSTRUMENT_CLASSES = {"instrument_field", "instrument_panel"}
VALVE_CLASSES = {"gate_valve", "globe_valve", "check_valve", "control_valve"}

# Tag grammar per symbol class (after normalisation).
TAG_PATTERNS: dict[str, re.Pattern] = {
    "gate_valve": re.compile(r"^HV-\d{4}$"),
    "globe_valve": re.compile(r"^HV-\d{4}$"),
    "check_valve": re.compile(r"^NRV-\d{3}$"),
    "control_valve": re.compile(r"^[FPLT]V-\d{3}$"),
    "centrifugal_pump": re.compile(r"^P-\d{3}[AB]?$"),
    "heat_exchanger": re.compile(r"^E-\d{3}$"),
    "vessel": re.compile(r"^V-\d{3}$"),
    # ISA 5.1 identification letters: first = measured variable (Flow, Pressure, Level,
    # Temperature), then a readout/output function (Transmitter, Indicator, Glass,
    # Element, Indicating Controller). "LUC" is a string of capitals, not an ISA tag.
    "instrument_field": re.compile(r"^[FPLT](T|I|G|E|IC)-\d{3}$"),
    "instrument_panel": re.compile(r"^[FPLT](T|I|G|E|IC)-\d{3}$"),
}
LINE_NUMBER = re.compile(r'^\d{1,2}"-[A-Z]{1,2}-\d{4}-[A-Z]\d[A-Z]$')

# Project line-designation table: the finite set of service codes and piping
# classes on this job. Real EPC projects publish this list; here it matches the generator.
LINE_SERVICES = {"P", "CW", "ST", "HC", "N", "IA", "PW"}
PIPE_SPECS = {"A1A", "B2C", "C1A", "D3B", "A2B"}
_CONFUSABLE = [set("0ODQ"), set("1IL|"), set("5S"), set("8B"), set("2Z"), set("6G")]

_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
_ALPHA_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"})


def normalise_tag(raw: str | None, cls: str) -> str | None:
    """Deterministic clean-up of an OCR/VLM tag string. Returns None if empty.

    Fixes only what the grammar makes unambiguous: letters in the prefix,
    digits in the number, one hyphen between them.
    """
    if not raw:
        return None
    s = raw.upper().replace(" ", "").replace("_", "-").replace("—", "-").replace("–", "-").replace(".", "")
    s = re.sub(r"-+", "-", s.strip("-:;,"))
    if "-" in s:
        prefix, _, num = s.partition("-")
    else:  # hyphen lost by OCR: split leading letters from the number
        m = re.match(r"^([A-Z]{1,3})([0-9OQDILSBZG|]{3,4}[AB]?)$", s)
        if not m:
            return s or None
        prefix, num = m.groups()
    if not (1 <= len(prefix) <= 3 and 3 <= len(num) <= 5):
        return s or None
    prefix = prefix.translate(_ALPHA_FIX)
    suffix = ""
    if cls == "centrifugal_pump" and len(num) == 4 and num[-1] in "AB":
        num, suffix = num[:-1], num[-1]
    return f"{prefix}-{num.translate(_DIGIT_FIX)}{suffix}"


def _confusable(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in _CONFUSABLE)


def repair_token(token: str, vocab: set[str], allow_dropped: bool = False) -> str | None:
    """Map an OCR token onto a closed vocabulary. Accept only a UNIQUE match where every
    character is identical or OCR-confusable (0/O/D, 1/I/L, 5/S, 8/B, ...). With
    allow_dropped, the token may also be the vocabulary word with one character missing.
    Ambiguous or no match -> None (the caller leaves it unresolved)."""
    if token in vocab:
        return token
    hits = {v for v in vocab if len(v) == len(token) and all(_confusable(x, y) for x, y in zip(token, v))}
    if allow_dropped:
        for v in vocab:
            if len(v) == len(token) + 1 and any(v[:i] + v[i + 1:] == token for i in range(len(v))):
                hits.add(v)
    return hits.pop() if len(hits) == 1 else None


def normalise_line_number(raw: str) -> str:
    s = (raw.upper().replace(" ", "").replace("''", '"').replace("”", '"').replace("“", '"')
         .replace("'", '"').replace("°", '"').replace("º", '"'))
    m = re.match(r'^(\d{1,2})"?-?([A-Z0-9]{1,2})-?(\w{4})-?(\w{3})$', s)
    if m:  # OCR often drops the inch mark or a hyphen; the grammar tells us where they go
        size, svc, num, spec = m.groups()
        svc = repair_token(svc, LINE_SERVICES, allow_dropped=True) or svc
        spec = repair_token(spec, PIPE_SPECS) or spec
        s = f'{size}"-{svc}-{num.translate(_DIGIT_FIX)}-{spec}'
    return s


def line_number_is_valid(s: str) -> bool:
    if not LINE_NUMBER.match(s):
        return False
    _, svc, _, spec = s.split("-")
    return svc in LINE_SERVICES and spec in PIPE_SPECS


def tag_is_valid(tag: str | None, cls: str) -> bool:
    return bool(tag) and bool(TAG_PATTERNS[cls].match(tag))


class Box(BaseModel):
    model_config = ConfigDict(frozen=True)
    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1")
    @classmethod
    def _x(cls, v, info):
        if v <= info.data["x0"]:
            raise ValueError("x1 must be > x0")
        return v

    @field_validator("y1")
    @classmethod
    def _y(cls, v, info):
        if v <= info.data["y0"]:
            raise ValueError("y1 must be > y0")
        return v

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


# ---- Stage 1: scanning agent ---------------------------------------------
class ScannedPage(BaseModel):
    page_index: int
    image_path: str
    width: int
    height: int
    has_text_layer: bool
    skew_deg: float = Field(description="skew detected and removed, degrees")


class TextSpan(BaseModel):
    text: str
    box: Box
    score: float = Field(ge=0, le=1)
    source: Literal["text_layer", "ocr"]


# ---- Stage 2: engineering agents ------------------------------------------
class Detection(BaseModel):
    det_id: str
    page_index: int
    cls: SymbolClass
    box: Box
    det_conf: float = Field(ge=0, le=1)


class TagReading(BaseModel):
    """The exact JSON a VLM must emit. Passed to Ollama as `format=` (constrained decoding)."""
    tag: str | None = Field(description="the tag text exactly as printed, or null if no tag is visible")
    legible: bool = Field(description="false if the text is too degraded to read")


Rung = Literal["text_layer", "ocr", "vlm_small", "vlm_large", "human"]


class Attempt(BaseModel):
    rung: Rung
    raw: str | None
    tag: str | None
    valid: bool
    score: float | None = None


class Item(BaseModel):
    det: Detection
    tag: str | None
    status: Literal["auto_accepted", "needs_review", "human_corrected", "human_confirmed", "rejected"]
    resolved_by: Rung | None
    confidence: float = Field(ge=0, le=1)
    reason: str
    attempts: list[Attempt] = []
    line_number: str | None = None
    crop_path: str | None = None


class ReviewDecision(BaseModel):
    det_id: str
    action: Literal["confirm", "correct", "reject"]
    tag: str | None = None


# ---- Stage 3: generation ---------------------------------------------------
class MTORow(BaseModel):
    item_type: str
    size_in: str | None
    line_number: str | None
    quantity: int
    tags: list[str]
