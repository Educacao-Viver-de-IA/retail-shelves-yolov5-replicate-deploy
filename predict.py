"""Retail Shelves YOLOv5 — detecta produtos em prateleira (Jonathancasjar/Retail_Shelves)."""
import json
import os
import sys
import time

# Pre-clean offline flags pra ultralytics nao bloquear
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
        self.setup_error = None
        try:
            print(f"[setup] WEIGHTS exists: {os.path.exists(WEIGHTS)}", flush=True)
            print(f"[setup] dir: {os.listdir('/src/weights/retail-shelves')[:10]}", flush=True)
        except Exception as e:
            print(f"[setup] dir err: {e}", flush=True)

        # Skip yolov5 lib (importa huggingface_hub._errors que tá quebrado).
        # Carrega via torch direct + ultralytics standalone hub
        try:
            print(f"[setup] loading via torch.load (state_dict approach)", flush=True)
            sys.stdout.flush()
            # yolov5 .pt salva como model dict — carrega direto via torch.load
            ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
            print(f"[setup] checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt,dict) else type(ckpt)}", flush=True)

            # YOLOv5 .pt geralmente contém 'model' (DetectionModel)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model = ckpt['model']
            else:
                self.model = ckpt

            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda().float()  # important pra yolov5: convert pra fp32
            else:
                self.model = self.model.float()

            # Class names
            self.names = getattr(self.model, 'names', {}) or {}
            print(f"[setup] DONE (t={time.time()-t0:.1f}s) classes={list(self.names.values())[:5] if isinstance(self.names,dict) else self.names[:5]}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

    def _preprocess(self, pil, image_size):
        """yolov5 preprocessing: letterbox resize to image_size, normalize 0-1, NCHW."""
        from torchvision.transforms.functional import to_tensor
        # Simple resize (não letterbox por simplicidade)
        pil_r = pil.resize((image_size, image_size), Image.BILINEAR)
        t = to_tensor(pil_r).unsqueeze(0)  # [1, 3, H, W]
        return t

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
        pil = Image.open(str(image)).convert("RGB")
        W, H = pil.size
        device = next(self.model.parameters()).device
        tensor = self._preprocess(pil, image_size).to(device, dtype=torch.float32)

        with torch.inference_mode():
            output = self.model(tensor)
        # output formato yolov5: list ou tensor [B, N, 6] (x,y,w,h,conf,cls) ou variantes
        if isinstance(output, (list, tuple)):
            output = output[0]

        # Apply NMS manual usando torchvision
        import torchvision
        # output shape: [B, N, num_classes+5] em format raw — precisa pos-process
        # yolov5 retorna [B, N, 5+nc] onde 5 = xc,yc,w,h,obj_conf, depois nc class scores
        if output.ndim == 3:
            preds = output[0]
        else:
            preds = output

        # Filter por confidence (assumindo formato yolov5 padrão: 5 + nc)
        if preds.shape[-1] > 5:  # já tem class scores
            obj_conf = preds[:, 4]
            class_scores = preds[:, 5:]
            cls_idx = class_scores.argmax(dim=-1)
            cls_conf = class_scores.max(dim=-1).values
            scores = obj_conf * cls_conf
            mask = scores > conf_threshold
            boxes_xywh = preds[mask, :4]
            scores = scores[mask]
            cls_idx = cls_idx[mask]
        else:
            boxes_xywh = preds[:, :4]
            scores = preds[:, 4]
            cls_idx = torch.zeros(len(scores), dtype=torch.long, device=device)
            mask = scores > conf_threshold
            boxes_xywh = boxes_xywh[mask]
            scores = scores[mask]
            cls_idx = cls_idx[mask]

        if len(boxes_xywh) == 0:
            return {
                "n_detections": 0, "detections": [],
                "class_counts": {}, "image_size": image_size,
                "predict_time_s": round(time.time() - t0, 3),
            }

        # Convert xywh -> xyxy (center format pra corner)
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # NMS
        keep = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold)
        boxes = boxes_xyxy[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        cls_idx = cls_idx[keep].cpu().numpy()

        # Scale boxes back to original image size
        scale_x = W / image_size
        scale_y = H / image_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        detections = []
        for b, s, c in zip(boxes, scores, cls_idx):
            label = self.names.get(int(c), f"class_{int(c)}") if isinstance(self.names, dict) else (self.names[int(c)] if int(c) < len(self.names) else f"class_{int(c)}")
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
            "image_original_size": [W, H],
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "detections": detections,
            "predict_time_s": round(time.time() - t0, 3),
        }
