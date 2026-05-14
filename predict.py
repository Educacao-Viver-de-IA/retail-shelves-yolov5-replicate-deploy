"""
Retail Shelves YOLOv5 — detecção de produtos em prateleiras de varejo.
Base: Jonathancasjar/Retail_Shelves (yolov5 bestv2.pt).
"""
import io
import json
import os
import sys
import time

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()

import torch
print(f"[module] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
print(f"[module] imports OK", flush=True)
sys.stdout.flush()

WEIGHTS = "/src/weights/retail-shelves/bestv2.pt"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.setup_error = None
        try:
            print(f"[setup] dir contents: {sorted(os.listdir('/src/weights/retail-shelves'))[:10]}", flush=True)
        except Exception as e:
            print(f"[setup] err list: {e}", flush=True)

        try:
            import yolov5
            print(f"[setup] yolov5 imported, loading weights {WEIGHTS}", flush=True)
            sys.stdout.flush()
            self.model = yolov5.load(WEIGHTS)
            print(f"[setup] model loaded (t={time.time()-t0:.1f}s)", flush=True)
            if torch.cuda.is_available():
                self.model.cuda()
                print(f"[setup] model moved to CUDA", flush=True)
            # Default thresholds (override per-predict)
            self.model.conf = 0.25
            self.model.iou = 0.45
            self.model.agnostic = False
            self.model.multi_label = False
            self.model.max_det = 1000
            print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

    def predict(
        self,
        image: Path = Input(description="Foto de prateleira de varejo (jpg/png)."),
        conf_threshold: float = Input(default=0.25, ge=0.0, le=1.0,
            description="Confidence threshold mínimo pra detecção."),
        iou_threshold: float = Input(default=0.45, ge=0.0, le=1.0,
            description="IoU threshold pra NMS."),
        image_size: int = Input(default=640, ge=320, le=1280,
            description="Tamanho da imagem pra inferência (múltiplo de 32)."),
        augment: bool = Input(default=False,
            description="Test-time augmentation (TTA) — mais lento, mais preciso."),
    ) -> dict:
        if self.model is None:
            return {"error": f"Modelo não carregou: {getattr(self, 'setup_error', 'unknown')}"}

        t0 = time.time()
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        img_path = str(image)
        results = self.model(img_path, size=image_size, augment=augment)

        # parse results
        preds = results.pred[0]  # tensor [N, 6] = x1,y1,x2,y2,conf,cls
        if preds is None or len(preds) == 0:
            return {
                "n_detections": 0,
                "detections": [],
                "predict_time_s": round(time.time() - t0, 3),
            }

        boxes = preds[:, :4].cpu().numpy()
        scores = preds[:, 4].cpu().numpy()
        classes = preds[:, 5].cpu().numpy().astype(int)

        # Class names do modelo
        names = self.model.names if hasattr(self.model, 'names') else {}

        detections = []
        for b, s, c in zip(boxes, scores, classes):
            detections.append({
                "xmin": float(b[0]), "ymin": float(b[1]),
                "xmax": float(b[2]), "ymax": float(b[3]),
                "width": float(b[2] - b[0]), "height": float(b[3] - b[1]),
                "class_id": int(c),
                "label": names.get(c, f"class_{c}") if isinstance(names, dict) else (names[c] if isinstance(names, list) and c < len(names) else f"class_{c}"),
                "score": float(s),
            })
        detections.sort(key=lambda d: -d["score"])

        # Per-class summary
        from collections import Counter
        class_counts = Counter(d["label"] for d in detections)

        return {
            "n_detections": len(detections),
            "class_counts": dict(class_counts),
            "image_size": image_size,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "augment": augment,
            "detections": detections,
            "predict_time_s": round(time.time() - t0, 3),
        }
