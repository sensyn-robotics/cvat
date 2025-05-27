import cv2
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
MODEL_NAME = "shi-labs/oneformer_coco_swin_large"
CONFIDENCE_THRESHOLD = 0.5
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Variables for Model ---
_model = None
_processor = None


# --- Helper function to create the spec (remains the same) ---
def _generate_spec_object():
    print(f"Attempting to load configuration for {MODEL_NAME} to generate spec...")
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        if not hasattr(config, "id2label"):
            raise ValueError(
                f"Model config for {MODEL_NAME} does not have 'id2label' attribute."
            )
        current_labels = []
        for id_str, name_str in config.id2label.items():
            try:
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
        raise RuntimeError(
            f"Failed to generate spec in _generate_spec_object() for model {MODEL_NAME}: {e}"
        )


spec = _generate_spec_object()


def get_spec():
    global spec
    return spec


# --- Initialization (called by CVAT agent when it starts) ---
def init_context(context):
    global _model, _processor

    # Use print for initial, critical logging, then try context.logger
    print(
        "PROGRESS: init_context called."
    )  # Print immediately when function is entered

    log_info = lambda msg: print(f"INFO (init_context): {msg}")
    log_error = lambda msg: print(f"ERROR (init_context): {msg}")

    if hasattr(context, "logger") and context.logger is not None:
        log_info = context.logger.info
        log_error = context.logger.error
    else:
        print("WARNING (init_context): context.logger not available, using print.")

    log_info(f"Initializing OneFormer model: {MODEL_NAME} on device: {_device}")

    try:
        log_info("Loading processor...")
        _processor = OneFormerProcessor.from_pretrained(MODEL_NAME)
        log_info("Processor loaded.")

        log_info("Loading model...")
        _model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(
            _device
        )
        _model.eval()  # Set model to evaluation mode
        log_info("Model loaded and set to eval mode.")

        log_info("OneFormer model initialized SUCCESSFULLY.")

    except Exception as e:
        log_error(f"Error during OneFormer model initialization: {e}")
        # Re-raise the exception to ensure the agent knows init failed critically
        raise RuntimeError(f"Failed to initialize model in init_context: {e}")


# --- Output Conversion (remains the same) ---
def _oneformer_instance_to_cvat_shapes(
    panoptic_outputs, original_image_size, model_id2label_mapping
):
    # ... (same as previous version)
    shapes = []
    if not panoptic_outputs:
        return shapes
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
            if cv2.contourArea(contour) < 5:
                continue
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            points = approx_polygon.reshape(-1).tolist()
            if len(points) >= 6:
                shapes.append(
                    cvataa.polygon(label_id=int(model_label_id), points=points)
                )
    return shapes


# --- Main Detection Function ---
def detect(context, image: PIL.Image.Image):
    global _model, _processor

    # Basic logging, try context.logger if available
    log_info_detect = lambda msg: print(f"INFO (detect): {msg}")
    log_error_detect = lambda msg: print(f"ERROR (detect): {msg}")
    if hasattr(context, "logger") and context.logger is not None:
        log_info_detect = context.logger.info
        log_error_detect = context.logger.error
    else:
        print("WARNING (detect): context.logger not available, using print.")

    log_info_detect("Function called.")

    if _model is None or _processor is None:
        log_error_detect(
            "Model or processor not initialized. Ensure init_context was called and completed successfully."
        )
        return []  # Return empty list on error as per CVAT spec

    original_size_hw = image.size[::-1]

    try:
        log_info_detect("Preprocessing image with OneFormer processor...")
        inputs = _processor(
            images=image, task_inputs=["instance"], return_tensors="pt"
        ).to(_device)
        log_info_detect("Image preprocessed. Performing inference...")

        with torch.no_grad():
            outputs = _model(**inputs)
        log_info_detect("Inference complete. Post-processing...")

        instance_seg_outputs = _processor.post_process_instance_segmentation(
            outputs, target_sizes=[original_size_hw], threshold=CONFIDENCE_THRESHOLD
        )
        log_info_detect("Post-processing complete.")

    except Exception as e:
        log_error_detect(f"Error during OneFormer inference or post-processing: {e}")
        return []

    try:
        log_info_detect("Converting OneFormer output to CVAT shapes...")
        cvat_shapes = _oneformer_instance_to_cvat_shapes(
            instance_seg_outputs, original_size_hw, _model.config.id2label
        )
        log_info_detect(f"Conversion complete. Found {len(cvat_shapes)} shapes.")
    except Exception as e:
        log_error_detect(f"Error during shape conversion: {e}")
        return []

    return cvat_shapes
