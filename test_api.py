import httpx
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(response.json())

def test_embeddings():
    payload = {"text": "This is a legal contract for software services."}
    response = httpx.post(f"{BASE_URL}/embeddings", json=payload)
    print(f"Embeddings: {response.status_code}")
    if response.status_code == 200:
        print(f"Embedding length: {len(response.json()['embedding'])}")

def test_generate():
    payload = {
        "prompt": "Summarize the following clause: 'The parties agree to keep all information confidential during the term of this agreement.'",
        "max_new_tokens": 50
    }
    response = httpx.post(f"{BASE_URL}/generate", json=payload)
    print(f"Generate: {response.status_code}")
    if response.status_code == 200:
        print(f"Generated text: {response.json()['generated_text']}")

if __name__ == "__main__":
    print("Testing API (make sure uvicorn is running)...")
    try:
        test_health()
        test_embeddings()
        # test_generate() # Generation might fail locally without GPU/RAM
    except Exception as e:
        print(f"Error testing API: {e}")
