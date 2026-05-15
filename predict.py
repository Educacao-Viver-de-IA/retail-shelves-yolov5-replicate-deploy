"""Retail Shelves YOLOv5 — detecta produtos em prateleira (Jonathancasjar/Retail_Shelves)."""
import json
import os
import sys
import time
import types

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()

# Monkey-patch huggingface_hub.utils._errors (removido em hf>=0.27 mas yolov5 lib espera)
import huggingface_hub.utils
if not hasattr(huggingface_hub.utils, '_errors'):
    import huggingface_hub.errors as _hf_errors
    _shim = types.ModuleType('huggingface_hub.utils._errors')
    # Cria attrs com classes reais de huggingface_hub.errors
    for name in dir(_hf_errors):
        if not name.startswith('_'):
            setattr(_shim, name, getattr(_hf_errors, name))
    sys.modules['huggingface_hub.utils._errors'] = _shim
    huggingface_hub.utils._errors = _shim
    print(f"[module] monkey-patched huggingface_hub.utils._errors", flush=True)

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
            import yolov5
            print(f"[setup] yolov5 imported (with hf monkey-patch)", flush=True)
            sys.stdout.flush()
            self.model = yolov5.load(WEIGHTS)
            if torch.cuda.is_available():
                self.model.cuda()
            self.model.conf = 0.25
            self.model.iou = 0.45
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
        # yolov5 lib aceita PIL diretamente
        pil = Image.open(str(image)).convert("RGB")

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

        results = self.model(pil, size=image_size)
        preds = results.pred[0]  # [N, 6] x1,y1,x2,y2,conf,cls
        if preds is None or len(preds) == 0:
            return {
                "n_detections": 0, "detections": [],
                "class_counts": {}, "image_size": image_size,
                "predict_time_s": round(time.time() - t0, 3),
            }
        boxes = preds[:, :4].cpu().numpy()
        scores = preds[:, 4].cpu().numpy()
        classes = preds[:, 5].cpu().numpy().astype(int)
        names = self.names

        detections = []
        for b, s, c in zip(boxes, scores, classes):
            label = names.get(int(c), f"class_{int(c)}") if isinstance(names, dict) else (names[int(c)] if int(c) < len(names) else f"class_{int(c)}")
            detections.append({
                "xmin": float(b[0]), "ymin": float(b[1]),
                "xmax": float(b[2]), "ymax": float(b[3]),
                "class_id": int(c), "label": label, "score": float(s),
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
