# Use NVIDIA CUDA base image for GPU support
# If you don't have a GPU, use python:3.10-slim
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
