# check_sdk_contents.py
import cvat_sdk.masks

print("-" * 50)
print("CVAT SDK Masks Module Inspector")
print("-" * 50)

try:
    print(f"Path to cvat_sdk.masks module: {cvat_sdk.masks.__file__}")
except AttributeError:
    print(
        "Could not determine path for cvat_sdk.masks (it might be a built-in or unusually structured module)."
    )

print("\n--- Public Contents of cvat_sdk.masks ---")
public_names_found = False
for name in dir(cvat_sdk.masks):
    if not name.startswith("_"):
        print(name)
        public_names_found = True
if not public_names_found:
    print("(No public names found or module is empty/not as expected)")

print("\n--- Specific Function Checks ---")

# Check for mask_to_bbox
try:
    from cvat_sdk.masks import mask_to_bbox

    print("RESULT: 'mask_to_bbox' was successfully imported.")
    # Additionally, verify with hasattr as a double-check
    if hasattr(cvat_sdk.masks, "mask_to_bbox"):
        print("CONFIRMED: hasattr(cvat_sdk.masks, 'mask_to_bbox') is True.")
    else:
        print(
            "WARNING: hasattr(cvat_sdk.masks, 'mask_to_bbox') is False despite import success (unusual)."
        )
except ImportError:
    print("RESULT: 'mask_to_bbox' could NOT be imported (ImportError).")
    if hasattr(cvat_sdk.masks, "mask_to_bbox"):
        print(
            "WARNING: hasattr(cvat_sdk.masks, 'mask_to_bbox') is True despite ImportError (very unusual)."
        )
    else:
        print("CONFIRMED: hasattr(cvat_sdk.masks, 'mask_to_bbox') is False.")

# Check for mask_to_rle
try:
    from cvat_sdk.masks import mask_to_rle

    print("RESULT: 'mask_to_rle' was successfully imported.")
    if hasattr(cvat_sdk.masks, "mask_to_rle"):
        print("CONFIRMED: hasattr(cvat_sdk.masks, 'mask_to_rle') is True.")
    else:
        print(
            "WARNING: hasattr(cvat_sdk.masks, 'mask_to_rle') is False despite import success (unusual)."
        )
except ImportError:
    print("RESULT: 'mask_to_rle' could NOT be imported (ImportError).")
    if hasattr(cvat_sdk.masks, "mask_to_rle"):
        print(
            "WARNING: hasattr(cvat_sdk.masks, 'mask_to_rle') is True despite ImportError (very unusual)."
        )
    else:
        print("CONFIRMED: hasattr(cvat_sdk.masks, 'mask_to_rle') is False.")

print("-" * 50)
