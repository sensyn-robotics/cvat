import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
import PIL.Image
from ultralytics import YOLO

_model = YOLO("yolo11n-pose.pt")

spec = cvataa.DetectionFunctionSpec(
    labels=[
        cvataa.skeleton_label_spec(
            name,
            id,
            [
                cvataa.keypoint_spec(kp_name, kp_id)
                for kp_id, kp_name in enumerate(
                    [
                        "Nose",
                        "Left Eye",
                        "Right Eye",
                        "Left Ear",
                        "Right Ear",
                        "Left Shoulder",
                        "Right Shoulder",
                        "Left Elbow",
                        "Right Elbow",
                        "Left Wrist",
                        "Right Wrist",
                        "Left Hip",
                        "Right Hip",
                        "Left Knee",
                        "Right Knee",
                        "Left Ankle",
                        "Right Ankle",
                    ]
                )
            ],
        )
        for id, name in _model.names.items()
    ],
)


def detect(
    context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
) -> list[models.LabeledShapeRequest]:
    conf_threshold = 0.5 if context.conf_threshold is None else context.conf_threshold

    return [
        cvataa.skeleton(
            int(label.item()),
            [
                cvataa.keypoint(kp_index, kp.tolist(), outside=kp_conf.item() < 0.5)
                for kp_index, (kp, kp_conf) in enumerate(zip(kps, kp_confs))
            ],
        )
        for result in _model.predict(source=image, conf=conf_threshold)
        for label, kps, kp_confs in zip(
            result.boxes.cls, result.keypoints.xy, result.keypoints.conf
        )
    ]
