import requests
import json

url = "http://localhost:8000/api/analyze"
payload = {
    "answers": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
    "loss_limit_value": 9500000
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Response Data:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
