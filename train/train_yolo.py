"""Train the P&ID symbol detector on 640 px tiles of the synthetic sheets.

    python -m train.train_yolo --epochs 40

Copies the best weights to models/pid_symbols.pt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data" / "yolo" / "data.yaml"),
        imgsz=640, epochs=args.epochs, batch=args.batch, device=0, workers=2,
        project=str(ROOT / "runs" / "train"), name="symbols", exist_ok=True,
        # drawings are line art: colour jitter is meaningless, and flipping a
        # check valve still gives a check valve, so horizontal flips stay on
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.25,
        mosaic=1.0, close_mosaic=10, patience=15, plots=True, verbose=False,
    )
    best = ROOT / "runs" / "train" / "symbols" / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "pid_symbols.pt")
    metrics = YOLO(str(best)).val(data=str(ROOT / "data" / "yolo" / "data.yaml"), imgsz=640, device=0,
                                  project=str(ROOT / "runs" / "train"), name="val", exist_ok=True, verbose=False)
    print(f"val mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
