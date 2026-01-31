import os
import json
import time
import requests
import numpy as np
import faiss
import uvicorn
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# --- Configuration ---
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"

# --- Globals ---
class GlobalState:
    index = None
    meta_chunks = [] # List of chunk dicts
    embedding_client = None

state = GlobalState()

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    q: str
    k: int = 5

class SearchHit(BaseModel):
    id: str
    title: str
    text: str
    url: str
    score: float

class SearchResponse(BaseModel):
    answer: str
    hits: List[SearchHit]
    query_time_ms: float

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading index and metadata...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print(f"WARNING: {INDEX_PATH} or {META_PATH} not found. Search will fail until ingested.")
    else:
        try:
            state.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                state.meta_chunks = meta.get('chunks', [])
            print(f"Loaded index with {state.index.ntotal} vectors and {len(state.meta_chunks)} metadata entries.")
        except Exception as e:
            print(f"Error loading index: {e}")

    # Setup Embedding Client (Local Only)
    if SentenceTransformer:
        print("Using SentenceTransformer for embeddings.")
        state.embedding_client = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    else:
        print("CRITICAL: sentence-transformers not installed.")

    yield
    # Shutdown
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Helpers ---
def get_query_embedding(text: str) -> np.ndarray:
    text = text.replace("\n", " ")
    # Local Embedding
    vec = state.embedding_client.encode([text])[0]
    arr = np.array([vec], dtype='float32')
    
    faiss.normalize_L2(arr)
    return arr

def generate_llm_answer(query: str, hits: List[dict]) -> str:
    # 1. Build Context
    seen_ids = set()
    context_lines = []
    
    for h in hits:
        pid = h['paper_id']
        if pid not in seen_ids:
            seen_ids.add(pid)
            title = h['title']
            text = h['content']
            context_lines.append(f"Paper ID: {pid}\nTitle: {title}\nSummary snippet: {text}\n---")
    
    if len(seen_ids) > 5:
        context_lines = context_lines[:5]
    
    context_str = "\n".join(context_lines)
    
    # 2. Construct Prompt (Requirement: Structure & Citations)
    prompt = f"""You are a research assistant. Answer the question based ONLY on the context below.
Provide a structured answer with categorized points if applicable.
Cite sources using EXACTLY this format: "Title (Paper ID)".
Keep it concise (3-6 sentences).

Context:
{context_str}

Question: {query}
Answer:"""

    # 3. Call Ollama (Local)
    try:
        # Assumes Ollama is running on default port 11434
        model_name = "mistral:7b" 
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        res = requests.post("http://localhost:11434/api/generate", json=payload)
        
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        else:
            print(f"Ollama Error: {res.status_code} - {res.text}")
            return fallback_summary(hits, error_msg="Ollama service returned error.")
            
    except Exception as e:
        print(f"Ollama Connection Error: {e}")
        return fallback_summary(hits, error_msg="Local AI service unavailable.")

def fallback_summary(hits: List[dict], error_msg: str = "") -> str:
    # Simple frequent word or just concatenation of top hit titles
    titles = set(h['title'] for h in hits[:3])
    msg = "LLM synthesis unavailable. "
    if error_msg: 
        msg = f"{error_msg} "
    msg += "Top relevant papers found: " + "; ".join(titles) + "."
    return msg

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    start_time = time.time()
    
    if not state.index:
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingest.py first or download from Colab.")

    # 1. Embed
    q_vec = get_query_embedding(req.q)
    
    # 2. Search
    k = req.k
    # FAISS search
    scores, indices = state.index.search(q_vec, k)
    
    hits_data = []
    found_chunks = []
    
    # 3. Retrieve
    # indices[0] is the list of neighbor indices
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # No enough neighbors
        chunk = state.meta_chunks[idx]
        score = float(scores[0][i])
        
        found_chunks.append(chunk)
        
        hits_data.append(SearchHit(
            id=chunk['paper_id'],
            title=chunk['title'],
            text=chunk['content'],
            url=chunk['arxiv_url'],
            score=score
        ))

    # 4. Synthesize
    answer = generate_llm_answer(req.q, found_chunks)
    
    duration = (time.time() - start_time) * 1000
    
    return SearchResponse(
        answer=answer,
        hits=hits_data,
        query_time_ms=duration
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
