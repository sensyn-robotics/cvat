import cv2  # OpenCV for contour finding
import cvat_sdk.auto_annotation as cvataa
import numpy as np
import PIL.Image
import torch
from transformers import (
    AutoConfig,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)

# --- Configuration ---
MODEL_NAME = "shi-labs/oneformer_coco_swin_large"  # Or your chosen OneFormer model
CONFIDENCE_THRESHOLD = 0.5
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Variables for Model (loaded by init_context for the agent) ---
_model = None
_processor = None


# --- Helper function to create the spec ---
# This logic needs to run when the script is imported by cvat-cli create-native
def _generate_spec_object():
    """
    Generates the DetectionFunctionSpec object.
    This is called directly when the script is imported to define the global 'spec'.
    """
    print(f"Attempting to load configuration for {MODEL_NAME} to generate spec...")
    try:
        # Load only the configuration, not the full model
        config = AutoConfig.from_pretrained(MODEL_NAME)

        if not hasattr(config, "id2label"):
            raise ValueError(
                f"Model config for {MODEL_NAME} does not have 'id2label' attribute."
            )

        current_labels = []
        for id_str, name_str in config.id2label.items():
            try:
                # CVAT spec expects integer IDs for labels
                cvat_id = int(id_str)
                current_labels.append(cvataa.label_spec(name=name_str, id=cvat_id))
            except ValueError:
                print(
                    f"Warning in _generate_spec_object: Could not convert model label ID '{id_str}' to int for label '{name_str}'. Skipping."
                )
                pass

        if not current_labels:
            print(
                "Warning in _generate_spec_object: No labels could be generated from model config."
            )

        print(
            f"_generate_spec_object: Generated labels for spec: {[l.name for l in current_labels]}"
        )
        return cvataa.DetectionFunctionSpec(labels=current_labels)

    except Exception as e:
        print(f"Error in _generate_spec_object(): {e}")
        # Re-raise to ensure create-native fails clearly if spec cannot be generated.
        raise RuntimeError(
            f"Failed to generate spec in _generate_spec_object() for model {MODEL_NAME}: {e}"
        )


# --- Define `spec` globally ---
# This line is executed when cvat-cli imports this file for 'create-native'
spec = _generate_spec_object()


# --- Optional: get_spec() function ---
# Some versions or parts of CVAT might still look for this.
# It simply returns the already defined global spec.
def get_spec():
    global spec
    # print("get_spec() called, returning globally defined spec.") # For debugging
    return spec


# --- Initialization (called by CVAT agent when it starts) ---
def init_context(context):
    global _model, _processor
    context.logger.info(
        f"init_context: Initializing OneFormer model: {MODEL_NAME} on device: {_device}"
    )

    try:
        _processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
        _model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(
            _device
        )
        _model.eval()
        context.logger.info("init_context: OneFormer model initialized successfully.")
    except Exception as e:
        context.logger.error(f"init_context: Error initializing OneFormer model: {e}")
        raise


# --- Output Conversion ---
def _oneformer_instance_to_cvat_shapes(
    panoptic_outputs, original_image_size, model_id2label_mapping
):
    shapes = []
    if not panoptic_outputs:  # Check if list is empty
        return shapes

    # Assuming batch size of 1 from processor output structure
    if (
        not panoptic_outputs[0].get("segments_info")
        or panoptic_outputs[0].get("segmentation") is None
    ):
        return shapes

    segmentation_map = panoptic_outputs[0]["segmentation"]
    segments_info = panoptic_outputs[0]["segments_info"]

    segmentation_map_np = segmentation_map.cpu().numpy()

    for segment in segments_info:
        if segment["score"] < CONFIDENCE_THRESHOLD:
            continue

        segment_id = segment["id"]
        model_label_id = segment["label_id"]

        instance_mask = (segmentation_map_np == segment_id).astype(np.uint8)

        if np.sum(instance_mask) == 0:
            continue

        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 5:  # Filter small areas
                continue

            epsilon = 0.005 * cv2.arcLength(contour, True)  # Adjust simplification
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            points = approx_polygon.reshape(-1).tolist()

            if len(points) >= 6:  # Need at least 3 points for a polygon
                shapes.append(
                    cvataa.polygon(label_id=int(model_label_id), points=points)
                )
    return shapes


# --- Main Detection Function ---
def detect(context, image: PIL.Image.Image):
    global _model, _processor

    if _model is None or _processor is None:
        # This error should ideally be logged via context.logger if available
        print(
            "Error in detect: Model or processor not initialized. Ensure init_context was called by the agent."
        )
        return []

    original_size_hw = image.size[::-1]  # (height, width)

    inputs = _processor(images=image, task_inputs=["instance"], return_tensors="pt").to(
        _device
    )

    try:
        with torch.no_grad():
            outputs = _model(**inputs)

        instance_seg_outputs = _processor.post_process_instance_segmentation(
            outputs, target_sizes=[original_size_hw], threshold=CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        print(f"Error during OneFormer inference or post-processing: {e}")
        return []

    cvat_shapes = _oneformer_instance_to_cvat_shapes(
        instance_seg_outputs, original_size_hw, _model.config.id2label
    )

    return cvat_shapes
