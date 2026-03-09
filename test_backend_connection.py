import requests
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://34.70.252.201:8000")

def test_health():
    print(f"Testing health check at {BACKEND_URL}/health...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate():
    print(f"\nTesting generation at {BACKEND_URL}/generate...")
    payload = {
        "prompt": "Hello, who are you?",
        "max_new_tokens": 50,
        "temperature": 0.7
    }
    try:
        response = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Success! Response received.")
            # print(f"Preview: {response.json()['generated_text'][:100]}...")
            return True
        else:
            print(f"Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    health_ok = test_health()
    if health_ok:
        test_generate()
    else:
        print("Skipping generation test since health check failed.")
