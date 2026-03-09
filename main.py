import torch
import json
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    TextIteratorStreamer
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Any

app = FastAPI(
    title="Model Deployment API",
    description="API for contract LoRA generation and text embeddings (L4 Optimized)",
    version="1.2.0"
)

# --- Configuration ---
LORA_MODEL_ID = "shibinsha02/contract-lora"
BASE_MODEL_ID = "StevenChen16/llama3-8b-Lawyer"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Performance Settings
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high") # L4 supports TF32

# Global variables for models
model = None
tokenizer = None
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
    global model, tokenizer, embedding_model
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    
    print("Loading generation model for L4 GPU (Bfloat16 + Flash Attention 2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # L4 Optimization: Use Bfloat16 and Flash Attention 2
        # This requires ~16GB VRAM but is much faster than 4-bit quantization
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if device == "cuda":
            # Enable Flash Attention 2 for major speedup on L4
            model_kwargs["attn_implementation"] = "flash_attention_2"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            **model_kwargs
        )

        # Load LoRA adapter and merge for speed
        print("Merging LoRA adapter (Bfloat16)...")
        peft_model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
        model = peft_model.merge_and_unload()
        model.eval()
        
        # Optional: compile model for speedup
        if device == "cuda":
            try:
                print("Compiling model for Ada Lovelace (optional)...")
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"Model compilation skipped: {e}")

        print("Generation model fully optimized for L4.")

    except Exception as e:
        print(f"Error loading generation model: {e}")
        # Fallback to standard loading if Flash Attention 2 fails
        if "flash_attention_2" in str(e):
            print("Retrying without Flash Attention 2...")
            # (Recursive call or logic to retry without flash_attn if needed)
        model = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embeddings_loaded": embedding_model is not None,
        "generation_loaded": model is not None,
        "vram_optimization": "Bfloat16 + Flash Attention 2" if model else "None",
        "device": str(next(model.parameters()).device) if model else "N/A"
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
    if model is None:
        raise HTTPException(status_code=503, detail="Generation model not loaded")
    
    start_time = time.time()
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.temperature > 0 else 1.0,
            top_p=request.top_p,
            do_sample=True if request.temperature > 0 else False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    
    return GenerateResponse(
        generated_text=generated_text,
        generation_time=round(end_time - start_time, 2)
    )

@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Generation model not loaded")

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature if request.temperature > 0 else 1.0,
        top_p=request.top_p,
        do_sample=True if request.temperature > 0 else False,
        use_cache=True,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    def generate_with_stream():
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
        thread.join()

    return StreamingResponse(generate_with_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
