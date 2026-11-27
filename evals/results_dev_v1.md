## Full ladder (OCR threshold 0.83, qwen3-vl:2b -> qwen3-vl:4b)

| metric | vector | scanned |
|---|---|---|
| sheets | 15 | 15 |
| symbol detection precision | 0.960 | 0.962 |
| symbol detection recall | 1.000 | 1.000 |
| wrong class on matched symbols | 0 | 0 |
| deskew error (mean, deg) | - | 0.032 |
| **auto-accepted tags: precision** | **1.0000** (533/533) | **0.9981** (521/522) |
| automation rate (no human) | 95.9% | 94.1% |
| sent to human review | 23 (4.1%) | 33 (5.9%) |
| review items whose best guess was right | 0/23 | 3/33 |
| accepted by text_layer (correct/total) | 533/533 | 0 |
| accepted by ocr (correct/total) | 0 | 503/504 |
| accepted by vlm_small (correct/total) | 0 | 4/4 |
| accepted by vlm_large (correct/total) | 0 | 14/14 |
| line number linked correctly | 100.0% | 85.6% |
| sheets with an exactly correct valve MTO | 15/15 | 10/15 |
| seconds per sheet | 29.1 | 70.5 |

Per-class detection recall:

| class | vector | scanned |
|---|---|---|
| gate_valve | 1.000 | 1.000 |
| globe_valve | 1.000 | 1.000 |
| check_valve | 1.000 | 1.000 |
| control_valve | 1.000 | 1.000 |
| centrifugal_pump | 1.000 | 1.000 |
| heat_exchanger | 1.000 | 1.000 |
| vessel | 1.000 | 1.000 |
| instrument_field | 1.000 | 1.000 |
| instrument_panel | 1.000 | 1.000 |
