import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import time

app = FastAPI(
    title="Model Deployment API",
    description="API for contract LoRA generation and text embeddings",
    version="1.0.0"
)

# --- Configuration ---
LORA_MODEL_ID = "shibinsha02/contract-lora"
BASE_MODEL_ID = "StevenChen16/llama3-8b-Lawyer"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Global variables for models
generation_pipeline = None
embedding_model = None

# --- Models ---
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str

# --- Startup Event ---
@app.on_event("startup")
async def load_models():
    global generation_pipeline, embedding_model
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    
    print("Loading generation model (this might take a while)...")
    # Setting up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Load with 4-bit quantization if possible for Llama 3 on typical GPUs
    # Otherwise fallback to float16 or float32
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
        
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        print(f"Error loading generation model: {e}")
        # Placeholder/Mock for local testing if hardware is insufficient
        generation_pipeline = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embeddings_loaded": embedding_model is not None,
        "generation_loaded": generation_pipeline is not None
    }

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    embedding = embedding_model.encode(request.text).tolist()
    return EmbeddingResponse(
        embedding=embedding,
        model=EMBEDDING_MODEL_ID
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if generation_pipeline is None:
        raise HTTPException(status_code=503, detail="Generation model not loaded or hardware insufficient")
    
    start_time = time.time()
    
    outputs = generation_pipeline(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True if request.temperature > 0 else False
    )
    
    generated_text = outputs[0]["generated_text"]
    end_time = time.time()
    
    return GenerateResponse(
        generated_text=generated_text,
        generation_time=round(end_time - start_time, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
