# Retail Shelves YOLOv5

Detector de produtos em prateleiras de varejo usando [Jonathancasjar/Retail_Shelves](https://huggingface.co/Jonathancasjar/Retail_Shelves) (YOLOv5 multi-classe).

- **Versão**: YOLOv5 7.0.13
- **Hardware Replicate**: T4
- **Input**: foto de prateleira (jpg/png)
- **Output**: lista de detecções com bounding box + label + score

## Uso

```python
import replicate
output = replicate.run(
    "csviverdeia/retail-shelves-yolov5:<VERSION>",
    input={
        "image": "https://example.com/prateleira.jpg",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "image_size": 640,
    }
)
print(output["n_detections"], output["class_counts"])
```
