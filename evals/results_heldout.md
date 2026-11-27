## Full ladder (OCR threshold 0.83, qwen3-vl:2b -> qwen3-vl:4b) - sheets test-015 to test-029

| metric | vector | scanned |
|---|---|---|
| sheets | 15 | 15 |
| symbol detection precision | 0.986 | 0.979 |
| symbol detection recall | 1.000 | 1.000 |
| wrong class on matched symbols | 0 | 0 |
| deskew error (mean, deg) | - | 0.031 |
| **auto-accepted tags: precision** | **1.0000** (549/549) | **0.9981** (536/537) |
| automation rate (no human) | 98.2% | 95.4% |
| sent to human review | 10 (1.8%) | 26 (4.6%) |
| review items whose best guess was right | 0/10 | 9/26 |
| accepted by text_layer (correct/total) | 549/549 | 0 |
| accepted by ocr (correct/total) | 0 | 522/523 |
| accepted by vlm_small (correct/total) | 0 | 4/4 |
| accepted by vlm_large (correct/total) | 0 | 10/10 |
| line number linked correctly | 100.0% | 91.4% |
| sheets with an exactly correct valve MTO | 15/15 | 11/15 |
| seconds per sheet | 14.3 | 62.5 |

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
