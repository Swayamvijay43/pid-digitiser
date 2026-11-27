"""Unit tests for the deterministic checks. No model, GPU or Ollama needed."""
import numpy as np
import pytest
from pydantic import ValidationError

from pid.detect import nms
from pid.read_tags import candidate_from_spans, line_number_for
from pid.scan import estimate_skew
from pid.schemas import (Box, Detection, ReviewDecision, TagReading, TextSpan, normalise_line_number,
                         normalise_tag, tag_is_valid)


@pytest.mark.parametrize("raw,cls,expected", [
    ("HV-1234", "gate_valve", "HV-1234"),
    ("hv 1234", "gate_valve", "HV-1234"),
    ("HV1234", "gate_valve", "HV-1234"),          # hyphen lost by OCR
    ("HV-12O4", "gate_valve", "HV-1204"),         # O read for 0 in the number
    ("F1C-101", "instrument_panel", "FIC-101"),   # 1 read for I in the letters
    ("P-101A", "centrifugal_pump", "P-101A"),
    ("P-1018", "centrifugal_pump", "P-1018"),     # stays invalid; not silently "fixed"
    ("NRV-0S7", "check_valve", "NRV-057"),
    ("", "vessel", None),
])
def test_normalise_tag(raw, cls, expected):
    assert normalise_tag(raw, cls) == expected


def test_grammar_is_per_class():
    assert tag_is_valid("HV-1234", "gate_valve")
    assert not tag_is_valid("HV-1234", "check_valve")
    assert not tag_is_valid("P-1018", "centrifugal_pump")
    assert not tag_is_valid(None, "vessel")


def test_line_number_normalisation():
    assert normalise_line_number("6''-P-1001-A1A") == '6"-P-1001-A1A'
    assert normalise_line_number("6-CW-1O01-A1A") == '6"-CW-1001-A1A'


def test_vlm_contract_rejects_bad_json():
    assert TagReading.model_validate_json('{"tag": "FT-101", "legible": true}').tag == "FT-101"
    with pytest.raises(ValidationError):
        TagReading.model_validate_json('{"tag": "FT-101"}')
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate({"det_id": "x", "action": "approve"})
    with pytest.raises(ValidationError):
        Box(x0=10, y0=0, x1=5, y1=3)


def test_nms_merges_tile_duplicates_but_keeps_neighbours():
    boxes = np.array([[100, 100, 150, 130],   # full valve
                      [100, 100, 125, 130],   # same valve cut at a tile edge (IoS = 1)
                      [300, 100, 350, 130],   # a different valve
                      [102, 101, 151, 131]],  # same valve from the overlapping tile
                     dtype=float)
    scores = np.array([0.9, 0.8, 0.85, 0.7])
    classes = np.array([0, 0, 0, 0])
    keep, _ = nms(boxes, scores, classes)
    assert sorted(keep) == [0, 2]
    # a sliver labelled as another class inside a confident symbol is dropped (real case: y=1024 tile edge)
    sliver = np.array([[1078, 1000, 1129, 1032], [1078, 1024, 1129, 1032]], dtype=float)
    keep, _ = nms(sliver, np.array([0.93, 0.49]), np.array([2, 0]))
    assert keep == [0]
    # but separate symbols of different classes both survive
    keep, _ = nms(np.array([[0, 0, 50, 30], [60, 0, 110, 30]], dtype=float), np.array([0.9, 0.3]), np.array([0, 7]))
    assert sorted(keep) == [0, 1]


def test_nms_merges_split_symbol_into_union():
    # one valve detected as two overlapping halves where a branch pipe touches it
    # (real case from test-003: IoU 0.38, IoS 0.61)
    boxes = np.array([[2669, 1033, 2707, 1063], [2657, 1032, 2688, 1063]], dtype=float)
    keep, merged = nms(boxes, np.array([0.975, 0.955]), np.array([0, 0]))
    assert keep == [0]
    assert merged[0].tolist() == [2657, 1032, 2707, 1063]


def _span(text, x0, y0, x1, y1, score=0.95):
    return TextSpan(text=text, box=Box(x0=x0, y0=y0, x1=x1, y1=y1), score=score, source="ocr")


def _det(cls, x0, y0, x1, y1):
    return Detection(det_id="d", page_index=0, cls=cls, box=Box(x0=x0, y0=y0, x1=x1, y1=y1), det_conf=0.9)


def test_instrument_text_joined_top_to_bottom():
    det = _det("instrument_field", 100, 100, 170, 170)
    spans = [_span("101", 120, 140, 150, 155), _span("FT", 125, 112, 145, 128, 0.9)]
    raw, score, _ = candidate_from_spans(det, spans)
    assert raw == "FT-101" and score == 0.9


def test_valve_ignores_line_number_and_other_class_tags():
    det = _det("gate_valve", 500, 500, 550, 530)
    spans = [_span('6"-P-1001-A1A', 480, 470, 600, 490),   # line number right above
             _span("FV-123", 440, 540, 500, 560),          # control valve tag nearby
             _span("HV-4321", 505, 535, 560, 555)]
    raw, _, _ = candidate_from_spans(det, spans)
    assert raw == "HV-4321"
    assert line_number_for(det, spans) == '6"-P-1001-A1A'


def test_skew_estimate_recovers_rotation():
    import cv2
    img = np.full((1200, 1800), 255, np.uint8)
    for y in range(200, 1100, 150):
        cv2.line(img, (100, y), (1700, y), 0, 3)
    m = cv2.getRotationMatrix2D((900, 600), 2.0, 1.0)
    rot = cv2.warpAffine(img, m, (1800, 1200), borderValue=255)
    # straightening needs the opposite rotation of the one applied
    assert abs(estimate_skew(rot) + 2.0) < 0.15


@pytest.mark.parametrize("vlm,ocr,ok", [
    ("FI-388", "F-388", True),     # OCR dropped a letter; VLM fills it
    ("TT-108", "T-108", True),
    ("TT-109", "T-108", False),    # VLM changed a digit OCR did read
    ("TI-352", "352", False),      # OCR read no letters: VLM would be guessing them alone
    ("PC-530", "PIC", False),      # OCR read no number
    ("FT-101", "TF-101", False),   # letter order must be preserved
])
def test_vlm_must_be_consistent_with_partial_ocr(vlm, ocr, ok):
    from pid.read_tags import consistent_with_ocr
    assert consistent_with_ocr(vlm, ocr) is ok


def test_isa_instrument_grammar():
    assert tag_is_valid("LIC-932", "instrument_panel")
    assert not tag_is_valid("LUC-932", "instrument_panel")   # real OCR error that slipped through the loose grammar
    assert tag_is_valid("FT-101", "instrument_field") and tag_is_valid("LG-450", "instrument_field")


@pytest.mark.parametrize("raw,expected", [
    ("2-PW-5927-03B", '2"-PW-5927-D3B'),   # D read as 0: unique confusable match in the spec list
    ("2-A-7808-A1A", '2"-IA-7808-A1A'),    # I dropped from service code: unique match
    ("8°-N-4826-D3B", '8"-N-4826-D3B'),    # inch mark read as degree sign
])
def test_line_number_repaired_against_project_dictionary(raw, expected):
    from pid.schemas import line_number_is_valid
    assert normalise_line_number(raw) == expected
    assert line_number_is_valid(expected)


def test_repair_refuses_ambiguous_matches():
    from pid.schemas import repair_token
    assert repair_token("C", {"CW", "CS"}, allow_dropped=True) is None
    assert repair_token("X3B", {"D3B"}) is None
