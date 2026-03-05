# 🚀 Model Deployment API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow-blue?style=for-the-badge)](https://huggingface.co/)

A high-performance FastAPI service for deploying specialized LLMs and embedding models. This API serves a fine-tuned **Contract LoRA** model for legal text generation and a **Sentence Transformer** for high-quality text embeddings.

---

## ✨ Features

- **Legal Text Generation**: Specialized Llama 3 model fine-tuned for legal and contract-related tasks.
- **Text Embeddings**: Fast and reliable embeddings using `all-MiniLM-L6-v2`.
- **Hardware Optimized**: Automatic GPU/CUDA acceleration with CPU fallback.
- **Quantization Support**: Efficient model loading for optimized memory usage.
- **Docker Ready**: Fully containerized for seamless deployment.

---

## 🛠️ Models Used

| Task | Model ID | Description |
| :--- | :--- | :--- |
| **Generation** | `shibinsha02/contract-lora` | LoRA Adapter for Legal/Contract tasks |
| **Base Model** | `StevenChen16/llama3-8b-Lawyer` | Base Lawyer Llama 3 model |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight embedding model |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (optional, but recommended for generation)
- Docker (optional)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Model_Deploy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

---

## 🐳 Docker Deployment

Build and run the containerized application:

```bash
docker build -t model-deploy-api .
docker run -p 8000:8000 model-deploy-api
```

---

## 📡 API Endpoints

### 🩺 Health Check
`GET /health`
Verifies if the API and models are loaded correctly.

### 📝 Text Generation
`POST /generate`
Generates legal text based on a prompt.

**Request Body:**
```json
{
  "prompt": "Draft a termination clause for a software service agreement.",
  "max_new_tokens": 200,
  "temperature": 0.7
}
```

### 🔢 Text Embeddings
`POST /embeddings`
Generates vector embeddings for a given text.

**Request Body:**
```json
{
  "text": "This is a sample legal text for embedding."
}
```

---

## 📄 License

[MIT](LICENSE)

---

Built with ❤️ by [Shibinsha](https://github.com/shibinsha02)
