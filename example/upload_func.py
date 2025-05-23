import requests

sessionid = "45j7a7va0am9dikad7npj1qk45z5zp84"
task_id = 1386300  # sample in sensyn
function_file_path = "yolo11_func.py"

cookies = {"sessionid": sessionid}
headers = {"Accept": "application/json"}

# Upload the function file
with open(function_file_path, "rb") as f:
    files = {"function_file": f}
    resp = requests.post(
        f"https://app.cvat.ai/api/tasks/{task_id}/auto_annotate",
        cookies=cookies,
        headers=headers,
        files=files,
        data={"allow_unmatched_labels": "true"}
    )
    print(resp.status_code, resp.text)
