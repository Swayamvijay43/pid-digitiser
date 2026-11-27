# OCR threshold calibration (val split, scanned sheets)

- Grammar-valid OCR readings: 588 (overall precision 0.991)
- Symbols with no grammar-valid OCR reading (always escalated): 9
- Target precision for auto-accept: 0.995
- **Chosen threshold: 0.83**

| threshold | accepted | precision | share of valid readings |
|---|---|---|---|
| 0.50 | 588 | 0.9915 | 100.0% |
| 0.55 | 588 | 0.9915 | 100.0% |
| 0.60 | 585 | 0.9932 | 99.5% |
| 0.65 | 585 | 0.9932 | 99.5% |
| 0.70 | 583 | 0.9931 | 99.1% |
| 0.75 | 580 | 0.9948 | 98.6% |
| 0.80 | 577 | 0.9948 | 98.1% |
| 0.85 | 571 | 0.9965 | 97.1% |
| 0.90 | 564 | 0.9982 | 95.9% |
| 0.95 | 521 | 0.9981 | 88.6% |
| 0.83 | 573 | 0.9965 | 97.4% |
