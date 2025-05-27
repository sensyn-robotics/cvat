import inspect

import cvat_sdk.masks

print("--- Inspecting cvat_sdk.masks.encode_mask ---")
try:
    print(f"Help on encode_mask: {help(cvat_sdk.masks.encode_mask)}")
    print("\n--- Signature ---")
    print(inspect.signature(cvat_sdk.masks.encode_mask))
except AttributeError:
    print("cvat_sdk.masks.encode_mask does not exist or cannot be inspected.")
except Exception as e:
    print(f"Error inspecting encode_mask: {e}")
exit()
