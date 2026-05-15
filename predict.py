"""Retail Shelves YOLOv5 — detecta produtos em prateleira (Jonathancasjar/Retail_Shelves)."""
import json
import os
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")

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
        self.names = {}
        self.setup_error = None
        try:
            print(f"[setup] WEIGHTS exists: {os.path.exists(WEIGHTS)}", flush=True)
            print(f"[setup] dir: {os.listdir('/src/weights/retail-shelves')[:10]}", flush=True)
        except Exception as e:
            print(f"[setup] dir err: {e}", flush=True)

        try:
            # ultralytics 8.x carrega yolov5 .pt nativamente — converte internamente
            from ultralytics import YOLO
            print(f"[setup] ultralytics imported, loading {WEIGHTS}", flush=True)
            sys.stdout.flush()
            self.model = YOLO(WEIGHTS, task="detect")
            self.names = getattr(self.model, 'names', {}) or {}
            print(f"[setup] DONE (t={time.time()-t0:.1f}s) | classes={self.names}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

    def predict(
        self,
        image: Path = Input(description="Foto de prateleira de varejo."),
        conf_threshold: float = Input(default=0.25, ge=0.0, le=1.0),
        iou_threshold: float = Input(default=0.45, ge=0.0, le=1.0),
        image_size: int = Input(default=640, ge=320, le=1280),
    ) -> dict:
        if self.model is None:
            return {"error": f"Modelo não carregou: {getattr(self, 'setup_error', '?')}"}

        t0 = time.time()
        # ultralytics rejeita tempfile sem extensão — passa via ndarray
        pil = Image.open(str(image)).convert("RGB")
        arr = np.array(pil)

        results = self.model.predict(
            source=arr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        r = results[0] if results else None
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return {
                "n_detections": 0, "detections": [],
                "class_counts": {}, "image_size": image_size,
                "predict_time_s": round(time.time() - t0, 3),
            }

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        detections = []
        for b, s, c in zip(boxes, scores, classes):
            detections.append({
                "xmin": float(b[0]), "ymin": float(b[1]),
                "xmax": float(b[2]), "ymax": float(b[3]),
                "class_id": int(c),
                "label": names.get(int(c), f"class_{int(c)}"),
                "score": float(s),
            })
        detections.sort(key=lambda d: -d["score"])

        from collections import Counter
        class_counts = Counter(d["label"] for d in detections)

        return {
            "n_detections": len(detections),
            "class_counts": dict(class_counts),
            "image_size": image_size,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "detections": detections,
            "predict_time_s": round(time.time() - t0, 3),
        }
