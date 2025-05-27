import cv2  # OpenCV for contour finding
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models  # For the return type hint (as in your example)
import numpy as np
import PIL.Image
import torch
from transformers import (
    AutoConfig,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)

# --- Configuration ---
# You can choose different OneFormer models from Hugging Face.
# COCO models are good for general instance segmentation.
MODEL_NAME = "shi-labs/oneformer_coco_swin_large"
# Other options:
# MODEL_NAME = "shi-labs/oneformer_cityscapes_swin_large" # For Cityscapes (street scenes)
# MODEL_NAME = "shi-labs/oneformer_ade20k_swin_large" # For ADE20K (broader semantic/instance)

CONFIDENCE_THRESHOLD = 0.5  # Threshold for keeping detected instances
POLYGON_AREA_THRESHOLD = 10  # Minimum area (pixels) for a polygon to be kept
POLYGON_SIMPLIFICATION_EPSILON_FACTOR = 0.002  # Factor for cv2.approxPolyDP

_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Global Model Loading (can be slow) ---
print(
    f"INFO: Attempting to load OneFormer model and processor globally: {MODEL_NAME} on device: {_device}"
)
print(
    "INFO: This may take some time, especially on the first run or with large models..."
)
_model = None
_processor = None
_model_config = None  # To store config for spec and later use if model loads

try:
    # Try to load config first for spec, as it's lighter
    _model_config = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"INFO: Successfully loaded model configuration for {MODEL_NAME}.")

    _processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
    _model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()  # Set model to evaluation mode
    print(
        "INFO: OneFormer model and processor loaded successfully and set to eval mode."
    )
except Exception as e:
    print(f"ERROR: Failed to load OneFormer model/processor globally: {e}")
    print("ERROR: Subsequent 'detect' calls will likely fail or return empty results.")
    # _model and _processor will remain None

# --- 2. Global `spec` Definition ---
print("INFO: Defining global 'spec' for CVAT auto annotation...")
_labels_for_spec = []
if _model_config and hasattr(_model_config, "id2label"):
    for id_str, name_str in _model_config.id2label.items():
        try:
            # Use the model's own integer class ID for the CVAT spec
            cvat_id = int(id_str)
            _labels_for_spec.append(cvataa.label_spec(name=name_str, id=cvat_id))
        except ValueError:
            print(
                f"WARNING (spec): Could not convert model label ID '{id_str}' to int for label '{name_str}'. Skipping."
            )
    print(
        f"INFO (spec): Generated labels for spec: {[l.name for l in _labels_for_spec]}"
    )
else:
    print(
        "WARNING (spec): Model config or id2label not available. Spec will have no labels."
    )
    print(
        "WARNING (spec): This might cause issues with 'cvat-cli create-native' or UI."
    )

spec = cvataa.DetectionFunctionSpec(labels=_labels_for_spec)
print("INFO: Global 'spec' defined.")


# --- 3. `detect()` Function ---
def detect(
    context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
) -> list[models.LabeledShapeRequest]:
    if not _model or not _processor:
        print(
            "ERROR (detect): OneFormer model or processor was not loaded globally. Cannot perform detection."
        )
        return []

    # Get confidence threshold from context if available, otherwise use default
    active_conf_threshold = CONFIDENCE_THRESHOLD
    if hasattr(context, "conf_threshold") and context.conf_threshold is not None:
        # Ensure context.conf_threshold is valid (e.g. float)
        try:
            active_conf_threshold = float(context.conf_threshold)
            print(
                f"INFO (detect): Using confidence threshold from context: {active_conf_threshold}"
            )
        except (ValueError, TypeError):
            print(
                f"WARNING (detect): Invalid conf_threshold in context ('{context.conf_threshold}'). Using default: {CONFIDENCE_THRESHOLD}"
            )
            active_conf_threshold = CONFIDENCE_THRESHOLD
    else:
        print(
            f"INFO (detect): Using default confidence threshold: {active_conf_threshold}"
        )

    original_size_hw = image.size[::-1]  # (height, width)
    generated_shapes = []

    try:
        # Preprocess image
        # For OneFormer, "instance" task_inputs is typically used for instance segmentation.
        # Some model variants might prefer "panoptic" for instance-style outputs.
        inputs = _processor(
            images=image, task_inputs=["instance"], return_tensors="pt"
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        # Post-process for instance segmentation
        # target_sizes expects a list of (height, width) tuples
        instance_seg_outputs = _processor.post_process_instance_segmentation(
            outputs, target_sizes=[original_size_hw], threshold=active_conf_threshold
        )
        # Expected output: list of [dict_per_image]. Each dict_per_image has:
        # - "segmentation": 2D torch.Tensor (segment_id map for the image)
        # - "segments_info": list of dicts (one per detected instance/segment)
        #   - each segment_info_dict: {"id": segment_id, "label_id": model_class_id, "score": ...}

        if not instance_seg_outputs:
            print("INFO (detect): No instance segmentation outputs from processor.")
            return []

        # Assuming batch size of 1 was processed
        output_data = instance_seg_outputs[0]
        segmentation_map_tensor = output_data.get("segmentation")
        segments_info_list = output_data.get("segments_info")

        if segmentation_map_tensor is None or not segments_info_list:
            print("INFO (detect): Segmentation map or segments_info is missing/empty.")
            return []

        segmentation_map_np = segmentation_map_tensor.cpu().numpy()

        for segment_info in segments_info_list:
            # The 'threshold' in post_process_instance_segmentation should have already filtered by score.
            # segment_info['score'] is available if further filtering is needed.

            segment_id_in_map = segment_info["id"]
            model_internal_label_id = segment_info[
                "label_id"
            ]  # This is the model's own integer class ID

            # Create a binary mask for the current segment/instance
            instance_mask_np = (segmentation_map_np == segment_id_in_map).astype(
                np.uint8
            )

            if (
                np.sum(instance_mask_np) < POLYGON_AREA_THRESHOLD
            ):  # Skip if mask is too small or empty
                continue

            # Convert binary mask to polygon(s) using OpenCV
            contours, _ = cv2.findContours(
                instance_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) < POLYGON_AREA_THRESHOLD:
                    continue

                # Convert contour to a polygon shape
                # epsilon = POLYGON_SIMPLIFICATION_EPSILON_FACTOR * cv2.arcLength(
                #     contour, True
                # )
                # approx_polygon_points = cv2.approxPolyDP(contour, epsilon, True)

                # # Flatten polygon points for CVAT: [x1, y1, x2, y2, ...]
                # points_flat_list = approx_polygon_points.reshape(-1).tolist()

                # if (
                #     len(points_flat_list) >= 6
                # ):  # A polygon needs at least 3 points (6 values)
                #     generated_shapes.append(
                #         cvataa.polygon(
                #             label_id=int(
                #                 model_internal_label_id
                #             ),  # Use model's label ID, should map to spec
                #             points=points_flat_list,
                #         )
                #     )

                # instead of creaing a polygon, calculate the bounding box
                np_contour_points = np.array(contour).reshape(
                    -1, 2
                )  # Ensure contour is Nx2
                x_min, y_min = np.min(np_contour_points, axis=0)
                x_max, y_max = np.max(np_contour_points, axis=0)

                # Create a rectangle shape
                generated_shapes.append(
                    cvataa.rectangle(  # <--- CHANGE HERE
                        label_id=int(model_internal_label_id),
                        points=[
                            float(x_min),
                            float(y_min),
                            float(x_max),
                            float(y_max),
                        ],  # Ensure points are floats
                    )
                )

    except Exception as e:
        print(
            f"ERROR (detect): Exception during OneFormer inference or processing: {e}"
        )
        import traceback

        print(traceback.format_exc())  # Print full traceback for debugging
        return []  # Return empty list on error

    print(f"INFO (detect): Found {len(generated_shapes)} polygon shapes.")
    return generated_shapes


# No init_context() or get_spec() as per the user's working example structure.
print("INFO: OneFormer segmentation script definition complete. Ready for CVAT agent.")
