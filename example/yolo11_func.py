import PIL.Image
from ultralytics import YOLO

import cvat_sdk.auto_annotation as cvataa

_model = YOLO("yolo11n.pt")

spec = cvataa.DetectionFunctionSpec(
    labels=[cvataa.label_spec(name, id) for id, name in _model.names.items()],
)

def _yolo_to_cvat(results):
    for result in results:
        for box, label in zip(result.boxes.xyxy, result.boxes.cls):
            yield cvataa.rectangle(int(label.item()), [p.item() for p in box])

def detect(context, image):
    conf_threshold = 0.5 if context.conf_threshold is None else context.conf_threshold
    return list(_yolo_to_cvat(_model.predict(
        source=image, verbose=False, conf=conf_threshold)))
‍

