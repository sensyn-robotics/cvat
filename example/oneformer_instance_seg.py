import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models  # For the return type hint (as in your example)
import numpy as np
import PIL.Image
import torch
from cvat_sdk.masks import encode_mask as cvat_sdk_encode_mask
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
MIN_MASK_AREA_PIXELS = 10  # Minimum number of pixels in a mask to be considered
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
            _labels_for_spec.append(
                cvataa.label_spec(name=name_str, id=cvat_id, type="mask")
            )
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


def _oneformer_instance_to_cvat_shapes(
    panoptic_outputs, original_image_size_wh, model_id2label_mapping
):
    generated_shapes = []
    if not panoptic_outputs:
        return generated_shapes

    output_data = panoptic_outputs[0]
    segmentation_map_tensor = output_data.get("segmentation")
    segments_info_list = output_data.get("segments_info")

    if segmentation_map_tensor is None or not segments_info_list:
        return []

    segmentation_map_np = segmentation_map_tensor.cpu().numpy()  # This is (H, W)

    for segment_info in segments_info_list:
        segment_id_in_map = segment_info["id"]
        model_internal_label_id = segment_info["label_id"]

        # instance_mask_np is a (height, width) 2D NumPy array (0 or 1)
        instance_mask_np_uint8 = (segmentation_map_np == segment_id_in_map).astype(
            np.uint8
        )

        if np.sum(instance_mask_np_uint8) < MIN_MASK_AREA_PIXELS:
            continue

        # Calculate tight bounding box [xtl, ytl, xbr, ybr] for the current instance mask
        rows, cols = np.where(instance_mask_np_uint8)
        if rows.size == 0 or cols.size == 0:  # Empty mask after all
            continue

        xtl = float(np.min(cols))
        ytl = float(np.min(rows))
        xbr = float(np.max(cols) + 1.0)  # Make it exclusive for bottom-right
        ybr = float(np.max(rows) + 1.0)  # Make it exclusive for bottom-right

        width = xbr - xtl
        height = ybr - ytl
        if width <= 0 or height <= 0:
            continue

        # Ensure the bitmap is boolean for cvat_sdk_encode_mask
        boolean_bitmap = instance_mask_np_uint8.astype(bool)

        # The bbox for encode_mask is [x1, y1, x2, y2] (inclusive for x1,y1; exclusive for x2,y2 often)
        # The doc says "limited to points between (x1,y1) and (x2,y2)" and "(0,0) <= (x1,y1) < (x2,y2) <= (W,H)"
        # This implies x2,y2 are exclusive limits.
        # Our xbr, ybr are already exclusive.
        bbox_for_encode_mask = [xtl, ytl, xbr, ybr]

        try:
            # Use cvat_sdk.masks.encode_mask
            # bitmap should be the full image sized boolean mask for the instance
            # bbox defines the area to consider within that bitmap.
            rle_float_list = cvat_sdk_encode_mask(
                boolean_bitmap, bbox=bbox_for_encode_mask
            )
        except Exception as e:
            print(
                f"ERROR (shapes): cvat_sdk_encode_mask failed for label {model_internal_label_id}: {e}"
            )
            import traceback

            print(traceback.format_exc())
            continue

        if not rle_float_list:  # If encode_mask returns empty list for some reason
            print(
                f"WARNING (shapes): cvat_sdk_encode_mask returned empty RLE list for label {model_internal_label_id}. Skipping."
            )
            continue

        generated_shapes.append(
            cvataa.mask(
                label_id=int(model_internal_label_id),
                points=rle_float_list,  # This is now List[float] from cvat_sdk_encode_mask
                left=xtl,
                top=ytl,
                right=xbr,  # Use xbr directly (exclusive)
                bottom=ybr,  # Use ybr directly (exclusive)
            )
        )

    return generated_shapes


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

    # original_size_hw は (height, width) だが、Pillowのimage.sizeは (width, height)
    original_size_wh = image.size  # (width, height)
    generated_shapes = []

    try:
        inputs = _processor(
            images=image, task_inputs=["instance"], return_tensors="pt"
        ).to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)

        instance_seg_outputs = _processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[original_size_wh[::-1]],  # (height, width) を渡す
            threshold=active_conf_threshold,
        )

        if not instance_seg_outputs:
            print("INFO (detect): No instance segmentation outputs from processor.")
            return []

        # _model.config.id2label を渡すが、現在の _oneformer_instance_to_cvat_shapes は未使用
        generated_shapes = _oneformer_instance_to_cvat_shapes(
            instance_seg_outputs, original_size_wh, {}
        )

    except Exception as e:
        print(
            f"ERROR (detect): Exception during OneFormer inference or processing: {e}"
        )
        import traceback

        print(traceback.format_exc())
        return []

    print(f"INFO (detect): Found {len(generated_shapes)} RLE mask shapes.")
    return generated_shapes


# No init_context() or get_spec() as per the user's working example structure.
print("INFO: OneFormer segmentation script definition complete. Ready for CVAT agent.")
