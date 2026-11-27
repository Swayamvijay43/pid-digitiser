"""Stage 3 - generation agent: turn reviewed items into deliverables.

- mto.csv           material take-off: valves grouped by type and pipe size
- instrument_index  every instrument tag with its measured variable and loop
- equipment_list    pumps, exchangers, vessels
- line_list         pipe line numbers found on the sheet
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pid.schemas import (INSTRUMENT_CLASSES, line_number_is_valid, VALVE_CLASSES, Item, MTORow, TextSpan,
                         normalise_line_number)

VARIABLE = {"F": "flow", "P": "pressure", "L": "level", "T": "temperature"}


def build_exports(items: list[Item], spans: dict[int, list[TextSpan]], out_dir: Path) -> dict:
    keep = [i for i in items if i.status in ("auto_accepted", "human_confirmed", "human_corrected") and i.tag]

    groups: dict[tuple, list[str]] = defaultdict(list)
    for i in keep:
        if i.det.cls in VALVE_CLASSES:
            size = i.line_number.split('"')[0] if i.line_number else None
            groups[(i.det.cls, size, i.line_number)].append(i.tag)
    mto = [MTORow(item_type=cls, size_in=size, line_number=ln, quantity=len(tags), tags=sorted(tags))
           for (cls, size, ln), tags in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2] or ""))]

    instruments = [{"tag": i.tag, "type": "panel / DCS" if i.det.cls == "instrument_panel" else "field",
                    "variable": VARIABLE.get(i.tag[0], "other"), "loop": i.tag.split("-")[-1],
                    "page": i.det.page_index, "resolved_by": i.resolved_by}
                   for i in keep if i.det.cls in INSTRUMENT_CLASSES]
    equipment = [{"tag": i.tag, "type": i.det.cls, "page": i.det.page_index, "resolved_by": i.resolved_by}
                 for i in keep if i.det.cls in ("centrifugal_pump", "heat_exchanger", "vessel")]
    lines = sorted({normalise_line_number(s.text) for page in spans.values() for s in page
                    if line_number_is_valid(normalise_line_number(s.text))})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_type", "size_in", "line_number", "quantity", "tags"])
        for r in mto:
            w.writerow([r.item_type, r.size_in or "", r.line_number or "", r.quantity, " ".join(r.tags)])
    doc = {"mto": [r.model_dump() for r in mto], "instrument_index": instruments,
           "equipment_list": equipment, "line_list": lines,
           "counts": {"symbols": len(items), "exported": len(keep),
                      "rejected": sum(i.status == "rejected" for i in items)}}
    (out_dir / "digitised.json").write_text(json.dumps(doc, indent=1))
    return {"mto_csv": str(out_dir / "mto.csv"), "json": str(out_dir / "digitised.json"), **doc["counts"]}
