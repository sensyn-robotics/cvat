import os  # Ensure os is imported if not already

import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
import numpy as np
import PIL.Image
import pytorch_lightning as pl  # Or: from lightning.pytorch import LightningModule as pl_LightningModule
import torch
import yaml
from cvat_sdk.masks import encode_mask as cvat_sdk_encode_mask
from transformers import (
    AutoConfig,
    OneFormerForUniversalSegmentation,  # We might instantiate this or a custom model
    OneFormerProcessor,
)

# --- YAML Configuration Loading ---
YAML_CONFIG_PATH = "example/tree-logging-survey.yaml"  # USER ACTION: Verify this path
yaml_config = None
try:
    with open(YAML_CONFIG_PATH, "r") as f:
        yaml_config = yaml.safe_load(f)
    print(f"INFO: Successfully loaded configuration from {YAML_CONFIG_PATH}")
except Exception as e_yaml:
    print(f"ERROR: Failed to load YAML configuration from {YAML_CONFIG_PATH}: {e_yaml}")
    print(
        "ERROR: Proceeding with script defaults, but this may lead to issues if checkpoint relies on YAML config."
    )
    # Provide some defaults or raise an error if YAML is critical
    # For now, we'll let it proceed and potentially fail later if critical values are missing.


# --- Configuration ---

# USER ACTION: Choose how to load the model:
# Option 1: Load from Hugging Face Hub (original behavior)
# LOAD_FROM_CHECKPOINT = False
MODEL_NAME_OR_PATH = "shi-labs/oneformer_coco_swin_large"  # Default Hugging Face model

# Option 2: Load from a local .ckpt file
LOAD_FROM_CHECKPOINT = True  # <<< SET TO True TO LOAD FROM CHECKPOINT
# USER ACTION: If LOAD_FROM_CHECKPOINT is True, set these:
CHECKPOINT_PATH = "example/tree_logging_v6.ckpt"  # <<< UPDATE THIS PATH
# This HF model ID will be used for:
# 1. Loading the processor (assuming your custom model uses compatible preprocessing).
# 2. Loading a base config (for id2label, and potentially for instantiating your model architecture if needed).
BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR = "shi-labs/oneformer_coco_swin_large"  # Adjust if your ckpt is based on a different OneFormer variant

# Override with YAML config if available
if yaml_config:
    if "backbone" in yaml_config:
        BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR = yaml_config["backbone"]
        MODEL_NAME_OR_PATH = yaml_config[
            "backbone"
        ]  # Also set this for non-checkpoint loading consistency
        print(
            f"INFO: Using 'backbone' from YAML for model/config/processor base: {BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR}"
        )
    else:
        print(
            f"WARNING: 'backbone' not found in {YAML_CONFIG_PATH}. Using default: {BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR}"
        )

    # Extract class information for later use if loading from checkpoint
    if "classes" in yaml_config and isinstance(yaml_config["classes"], list):
        _yaml_class_names = yaml_config["classes"]
        _yaml_num_classes = len(_yaml_class_names)
        _yaml_id2label = {i: label for i, label in enumerate(_yaml_class_names)}
        _yaml_label2id = {label: i for i, label in enumerate(_yaml_class_names)}
        print(
            f"INFO: Loaded {_yaml_num_classes} classes from YAML: {_yaml_class_names}"
        )
    else:
        _yaml_class_names = None
        _yaml_num_classes = None
        _yaml_id2label = None
        _yaml_label2id = None
        print(
            f"WARNING: 'classes' not found or not a list in {YAML_CONFIG_PATH}. Cannot determine class info from YAML."
        )
else:  # yaml_config is None (failed to load)
    _yaml_class_names = None
    _yaml_num_classes = None
    _yaml_id2label = None
    _yaml_label2id = None

# --- PyTorch Lightning Module Definition ---
# USER ACTION REQUIRED:
# If your checkpoint was saved from a custom PyTorch Lightning module,
# you MUST define that class (or a compatible version for inference) here.
# Its __init__ method must be callable with the arguments you provide to
# `load_from_checkpoint` (or arguments found in the checkpoint's hparams).
# The module should have an attribute (e.g., `self.model`) that holds the
# actual Hugging Face model (OneFormerForUniversalSegmentation or compatible).


# Example Minimal LightningModule for wrapping a Hugging Face model:
# Adjust this class to match the structure of the LightningModule used for training.
class OneFormerPLModule(pl.LightningModule):
    def __init__(
        self, model_config_hf, **kwargs
    ):  # Arguments must match how it was trained/saved or passed to load_from_checkpoint
        super().__init__()
        # If you used self.save_hyperparameters() in your training module,
        # PL will try to load them. Explicitly passed args to load_from_checkpoint
        # (like model_config_hf here) will override saved hparams if names clash.
        # self.save_hyperparameters() # Uncomment if you used this during training and want to rely on it.

        # It's common to pass the Hugging Face model config to initialize the model
        self.model = OneFormerForUniversalSegmentation(model_config_hf)
        # If your PL module had other components (e.g., criterion, metrics), define them here
        # if they are needed for the model structure, though not necessarily for inference.

        # Store any other necessary args if your model needs them.
        # For example, if your __init__ took other parameters:
        # self.some_other_param = kwargs.get('some_other_param_name')

    def forward(self, **inputs):
        # This forward is for the LightningModule.
        # The actual inference will likely call self.model.forward() or self.model(**inputs)
        return self.model(**inputs)

    # Other PL methods (training_step, configure_optimizers, etc.) are not strictly needed
    # for loading the model for inference if they are not part of the model's core structure.


# USER ACTION: If LOAD_FROM_CHECKPOINT is True and your model class is custom (not directly OneFormerForUniversalSegmentation),
# you need to define or import your model class here. For example:
#
# class MyCustomModel(torch.nn.Module):
#     def __init__(self, config): # Takes a Hugging Face config object
#         super().__init__()
#         # Example: Use the config to build a OneFormer-like model
#         self.core_model = OneFormerForUniversalSegmentation(config)
#         # Or define your custom layers...
#
#     def forward(self, **inputs):
#         # Your model's forward pass
#         # Ensure output is compatible with OneFormerProcessor.post_process_instance_segmentation
#         return self.core_model(**inputs) # Example if wrapping OneFormerForUniversalSegmentation
#
# If your .ckpt is for a standard OneFormerForUniversalSegmentation fine-tuned, you might not need a custom class.

CONFIDENCE_THRESHOLD = 0.5
MIN_MASK_AREA_PIXELS = 10
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Global Model Loading (can be slow) ---
_model = None
_processor = None
# _model_config = None

print(f"INFO: Device set to: {_device}")
if LOAD_FROM_CHECKPOINT:
    print(f"INFO: Attempting to load model from CHECKPOINT: {CHECKPOINT_PATH}")
    print(
        f"INFO: Using BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR: {BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR}"
    )
else:
    print(f"INFO: Attempting to load model from HUGGING_FACE_HUB: {MODEL_NAME_OR_PATH}")

print(
    "INFO: This may take some time, especially on the first run or with large models..."
)

try:
    # Load config and processor from the base Hugging Face model ID in both cases
    # The config provides id2label, and the processor handles image preprocessing.
    _model_config = AutoConfig.from_pretrained(BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR)
    print("INFO: Successfully loaded model configuration for spec/processor base.")

    # USER ACTION: If loading from a checkpoint trained with a different number of classes
    # than the BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR, adjust num_labels here.
    # We will use the class information from the YAML file if available.
    if LOAD_FROM_CHECKPOINT:
        if (
            _yaml_num_classes is not None
        ):  # Check if class info was successfully loaded from YAML
            num_classes_for_checkpoint = _yaml_num_classes
            id2label_for_checkpoint = _yaml_id2label
            label2id_for_checkpoint = _yaml_label2id

            if (
                hasattr(_model_config, "num_labels")
                and _model_config.num_labels != num_classes_for_checkpoint
            ):
                print(
                    f"INFO: Overriding _model_config.num_labels from {_model_config.num_labels} to {num_classes_for_checkpoint} (from YAML)."
                )
                _model_config.num_labels = num_classes_for_checkpoint

            # Also update id2label and label2id in the config to match the YAML classes
            # This is important for the model's classification head and for the spec generation.
            if id2label_for_checkpoint:
                _model_config.id2label = id2label_for_checkpoint
                print(
                    f"INFO: Updated _model_config.id2label with {len(id2label_for_checkpoint)} labels from YAML."
                )
            if label2id_for_checkpoint:
                _model_config.label2id = label2id_for_checkpoint
                print("INFO: Updated _model_config.label2id from YAML.")

        else:
            # Fallback if YAML class info wasn't available - this part might need manual adjustment
            # or will rely on the previous hardcoded logic if you re-add it.
            print(
                "WARNING: Class information not available from YAML. Model config might not match checkpoint if num_labels differs from the base model."
            )
            print(
                "WARNING: You might need to manually set num_labels, id2label, and label2id if errors occur."
            )
            # Example of previous logic (you might need to re-enable/adjust if YAML fails):
            # num_classes_in_checkpoint = 6 # Default or previously determined number
            # if hasattr(_model_config, "num_labels") and _model_config.num_labels != num_classes_in_checkpoint:
            #     print(f"INFO: Overriding _model_config.num_labels from {_model_config.num_labels} to {num_classes_in_checkpoint} (fallback).")
            #     _model_config.num_labels = num_classes_in_checkpoint

    _processor = OneFormerProcessor.from_pretrained(
        BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR
    )
    print("INFO: Successfully loaded processor.")

    if LOAD_FROM_CHECKPOINT:
        # Load model from local checkpoint
        if (
            not CHECKPOINT_PATH
            or CHECKPOINT_PATH
            == "path/to/your/tree_logging_v6.ckpt"  # Default placeholder
        ):
            # Check if the path is still the placeholder if it's not an empty string
            if CHECKPOINT_PATH == "example/tree_logging_v6.ckpt" and not os.path.exists(
                CHECKPOINT_PATH
            ):
                print(
                    f"WARNING: CHECKPOINT_PATH '{CHECKPOINT_PATH}' does not exist. Please ensure it's correct."
                )
            elif not CHECKPOINT_PATH:
                raise ValueError(
                    "USER ACTION REQUIRED: CHECKPOINT_PATH is not set. Please update it."
                )
            # If it's set but doesn't exist, load_from_checkpoint will raise FileNotFoundError

        print(
            f"INFO: Loading model using PyTorch Lightning from checkpoint: {CHECKPOINT_PATH}"
        )

        # USER ACTION REQUIRED:
        # 1. Ensure `OneFormerPLModule` (or your custom PL module class) is defined correctly above.
        #    Its name must match the class used here in `load_from_checkpoint`.
        # 2. Adjust the arguments passed to `load_from_checkpoint` if your PL module's
        #    `__init__` method requires different or additional arguments.
        #    The `_model_config` is passed here as `model_config_hf`, assuming the
        #    `OneFormerPLModule.__init__` expects an argument with this name.
        #    If your PL module saved hyperparameters (using self.save_hyperparameters()),
        #    `load_from_checkpoint` might pick them up automatically. Explicit arguments
        #    passed here will override saved hyperparameters if the names match.

        try:
            # Ensure the class name `OneFormerPLModule` matches the one defined above.
            # Pass arguments that your LightningModule's __init__ expects.
            # `model_config_hf` is an example; your PL module might need different/more args.
            loaded_pl_module = OneFormerPLModule.load_from_checkpoint(
                checkpoint_path=CHECKPOINT_PATH,
                map_location=torch.device("cpu"),  # Load to CPU first
                model_config_hf=_model_config,  # This kwarg name must match an arg in OneFormerPLModule.__init__
            )
            print(
                f"INFO: Successfully loaded PyTorch Lightning module from {CHECKPOINT_PATH}."
            )

            # The actual model for inference is usually an attribute of the LightningModule
            if hasattr(loaded_pl_module, "model"):
                _model = loaded_pl_module.model.to(
                    _device
                )  # Move the extracted model to the target device
                print(
                    f"INFO: Extracted underlying Hugging Face model and moved to {_device}."
                )
            else:
                # Fallback: if the PL module itself is the torch.nn.Module for inference
                print(
                    "WARNING: loaded_pl_module does not have a 'model' attribute. Attempting to use the PL module itself as the model."
                )
                _model = loaded_pl_module.to(_device)

        except Exception as e_pl_load:
            print(
                f"ERROR: Failed to load model from checkpoint using PyTorch Lightning: {e_pl_load}"
            )
            import traceback

            print(traceback.format_exc())
            print("Please ensure that:")
            print(
                "  1. `pytorch-lightning` (or `lightning`) is installed in your environment."
            )
            print(
                "  2. The PyTorch Lightning module class (e.g., `OneFormerPLModule`) is correctly defined in this script."
            )
            print(
                "  3. The `__init__` signature of your PL class matches the arguments passed to `load_from_checkpoint` OR that necessary hparams are saved in the checkpoint."
            )
            print(
                f"  4. The checkpoint file '{CHECKPOINT_PATH}' is a valid PyTorch Lightning checkpoint saved from an instance of that class."
            )
            _model = None  # Ensure model is None if loading failed

    else:
        # Load model from Hugging Face Hub
        _model = OneFormerForUniversalSegmentation.from_pretrained(
            MODEL_NAME_OR_PATH
        ).to(_device)
        print(
            f"INFO: Successfully loaded model from Hugging Face Hub: {MODEL_NAME_OR_PATH}"
        )

    if _model:  # Check if model loading was successful before setting to eval mode
        _model.eval()
        print("INFO: Model loaded successfully and set to eval mode.")
    else:
        print("ERROR: Model could not be loaded. Subsequent operations will fail.")


except Exception as e:
    print(f"ERROR: Failed to load OneFormer model/processor: {e}")
    import traceback

    print(traceback.format_exc())
    print("ERROR: Subsequent 'detect' calls will likely fail or return empty results.")
    _model = None  # Ensure model is None if loading failed
    _processor = None
    _model_config = None


# --- 2. Global `spec` Definition ---
# This section should work as is, using _model_config loaded above.
print("INFO: Defining global 'spec' for CVAT auto annotation...")
_labels_for_spec = []
if _model_config and hasattr(_model_config, "id2label"):
    for id_str, name_str in _model_config.id2label.items():
        try:
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
# ... (rest of your script: spec definition, _oneformer_instance_to_cvat_shapes, detect, and local testing section) ...
# ... existing code ...
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
            # USER ACTION REQUIRED (if loading from checkpoint with custom model):
            # Ensure your loaded _model's forward pass is compatible with `**inputs`
            # from the OneFormerProcessor. If not, you may need to adapt how `inputs`
            # are passed or how `outputs` are structured.
            outputs = _model(**inputs)

        # Assuming the output structure of your custom model is compatible with
        # what OneFormerProcessor.post_process_instance_segmentation expects.
        # If not, you'll need to adapt this post-processing step.
        instance_seg_outputs = _processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[original_size_wh[::-1]],  # (height, width) を渡す
            threshold=active_conf_threshold,
        )

        if not instance_seg_outputs:
            print("INFO (detect): No instance segmentation outputs from processor.")
            return []

        # _model.config.id2label を渡すが、現在の _oneformer_instance_to_cvat_shapes は未使用
        # Pass the actual id2label mapping from _model_config
        id2label_map_for_shapes = {}
        if _model_config and hasattr(_model_config, "id2label"):
            id2label_map_for_shapes = _model_config.id2label

        generated_shapes = _oneformer_instance_to_cvat_shapes(
            instance_seg_outputs, original_size_wh, id2label_map_for_shapes
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


# --- Local Testing Section ---
if __name__ == "__main__":
    import argparse  # Import argparse
    import os
    from types import SimpleNamespace  # For a simple mock context

    import cv2  # For visualization (ensure opencv-python is installed)

    print("\n--- RUNNING LOCAL TEST ---")

    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Local test runner for OneFormer instance segmentation script."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the test image file (e.g., /home/mas/test_images/my_image.png)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,  # Default confidence threshold if not provided
        help="Confidence threshold for detection (default: 0.3)",
    )
    # Add an argument to specify if loading from checkpoint for local testing
    parser.add_argument(
        "--load-from-ckpt",
        action="store_true",  # Makes it a flag, True if present
        help="Force loading from checkpoint for local test (overrides script's LOAD_FROM_CHECKPOINT for testing only).",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=CHECKPOINT_PATH,  # Default to script's global CHECKPOINT_PATH
        help="Path to the checkpoint file if --load-from-ckpt is used.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="shi-labs/oneformer_coco_swin_large"
        if LOAD_FROM_CHECKPOINT
        else MODEL_NAME_OR_PATH,  # Default based on script config
        help="Hugging Face model ID to use if not loading from checkpoint, or as base for ckpt.",
    )

    args = parser.parse_args()

    # --- Override global config for local testing if flags are set ---
    if args.load_from_ckpt:
        LOAD_FROM_CHECKPOINT = True
        CHECKPOINT_PATH = args.ckpt_path
        BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR = (
            args.hf_model
        )  # Use this as base for ckpt
        print("INFO (Local Test Override): Forcing LOAD_FROM_CHECKPOINT=True")
        print(f"INFO (Local Test Override): Using CKPT_PATH='{CHECKPOINT_PATH}'")
        print(
            f"INFO (Local Test Override): Using BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR='{BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR}'"
        )
    elif not LOAD_FROM_CHECKPOINT:  # If script is set to load from HF and no override
        MODEL_NAME_OR_PATH = args.hf_model
        print(
            f"INFO (Local Test Override): Using MODEL_NAME_OR_PATH='{MODEL_NAME_OR_PATH}' for HF loading."
        )

    # --- 2. Configuration for Local Test (from arguments) ---
    TEST_IMAGE_PATH = args.image_path
    LOCAL_TEST_CONF_THRESHOLD = args.conf_threshold

    print(f"INFO (Local Test): Test image path from argument: {TEST_IMAGE_PATH}")
    print(
        f"INFO (Local Test): Confidence threshold from argument: {LOCAL_TEST_CONF_THRESHOLD}"
    )

    if not os.path.exists(TEST_IMAGE_PATH):
        print(
            f"ERROR (Local Test): Test image not found at '{TEST_IMAGE_PATH}'. Please set the correct path."
        )
        exit()

    # --- 3. Ensure Global Model is Loaded (it should be if script is run directly) ---
    # The global loading logic at the top of the script will run first.
    # We check _model here to ensure it completed.
    if not _model or not _processor:
        print(
            "ERROR (Local Test): Model or processor not loaded globally. Exiting local test."
        )
        print(
            "Ensure the script is run in an environment where the model can be downloaded or found, and paths are correct."
        )
        exit()

    print("INFO (Local Test): Using globally loaded model setup.")
    if LOAD_FROM_CHECKPOINT:
        print(
            f"INFO (Local Test): Model source: Checkpoint '{CHECKPOINT_PATH}' (Base: '{BASE_HF_MODEL_ID_FOR_CONFIG_PROCESSOR}')"
        )
    else:
        print(
            f"INFO (Local Test): Model source: HuggingFace Hub '{MODEL_NAME_OR_PATH}'"
        )

    # --- 4. Create a Mock CVAT Context ---
    mock_context = SimpleNamespace(
        conf_threshold=LOCAL_TEST_CONF_THRESHOLD,
    )
    print(
        f"INFO (Local Test): Mock context created with conf_threshold = {mock_context.conf_threshold}"
    )

    # --- 5. Load the Test Image ---
    try:
        pil_image = PIL.Image.open(TEST_IMAGE_PATH).convert("RGB")
        print(
            f"INFO (Local Test): Successfully loaded test image: {TEST_IMAGE_PATH} (Size: {pil_image.size})"
        )
    except Exception as e:
        print(f"ERROR (Local Test): Failed to load test image '{TEST_IMAGE_PATH}': {e}")
        exit()

    # --- 6. Call the detect Function ---
    print("INFO (Local Test): Calling detect function...")

    detected_shapes = detect(mock_context, pil_image)
    print(f"INFO (Local Test): detect function returned {len(detected_shapes)} shapes.")

    # --- 7. Inspect and Visualize Results ---
    if not detected_shapes:
        print("INFO (Local Test): No shapes were detected.")
    else:
        print("\n--- Detected Shapes (Summary) ---")
        for i, shape in enumerate(detected_shapes):
            label_name = "Unknown"
            # Use the globally loaded _model_config for label names
            if (
                _model_config
                and hasattr(_model_config, "id2label")
                and str(shape.label_id)
                in _model_config.id2label  # id2label keys are often strings
            ):
                label_name = _model_config.id2label[str(shape.label_id)]
            elif (  # Fallback if label_id is int
                _model_config
                and hasattr(_model_config, "id2label")
                and shape.label_id in _model_config.id2label
            ):
                label_name = _model_config.id2label[shape.label_id]

            print(
                f"Shape {i + 1}: Label ID={shape.label_id} (Name: {label_name}), "
                f"Type={shape.type}, Points Length={len(shape.points)}, "
                f"BBox=[{shape.left:.1f}, {shape.top:.1f}, {shape.right:.1f}, {shape.bottom:.1f}]"
            )

        # --- Optional: Visualize on Image using OpenCV ---
        try:
            cv_image = np.array(pil_image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            label_colors = {}
            if _model_config and hasattr(_model_config, "id2label"):
                for label_id_key in _model_config.id2label.keys():
                    try:  # Handle if key is int or str
                        label_id_int = int(label_id_key)
                        label_colors[label_id_int] = (
                            (label_id_int * 50) % 255,
                            (label_id_int * 90) % 255,
                            (label_id_int * 120) % 255,
                        )
                    except ValueError:
                        pass

            output_image_path = "local_test_output.png"
            for shape in detected_shapes:
                color = label_colors.get(shape.label_id, (0, 0, 255))
                label_name_viz = "Unknown"
                if _model_config and hasattr(_model_config, "id2label"):
                    label_name_viz = _model_config.id2label.get(
                        str(shape.label_id),
                        _model_config.id2label.get(
                            shape.label_id, f"ID:{shape.label_id}"
                        ),
                    )

                xtl, ytl, xbr, ybr = (
                    int(shape.left),
                    int(shape.top),
                    int(shape.right),
                    int(shape.bottom),
                )
                cv2.rectangle(cv_image, (xtl, ytl), (xbr, ybr), color, 2)
                cv2.putText(
                    cv_image,
                    label_name_viz,
                    (xtl, ytl - 10 if ytl > 20 else ytl + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            cv2.imwrite(output_image_path, cv_image)
            print(f"\nINFO (Local Test): Visualization saved to {output_image_path}")
        except Exception as e:
            print(f"ERROR (Local Test): Failed during visualization: {e}")
            import traceback

            print(traceback.format_exc())
    print("\n--- LOCAL TEST COMPLETE ---")
