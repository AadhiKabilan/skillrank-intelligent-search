import os
import json
import time
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

# Optional imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# --- Configuration ---
INDEX_PATH = "index.faiss"
META_PATH = "meta.json"
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"

# --- Globals ---
class GlobalState:
    index = None
    meta_chunks = [] # List of chunk dicts
    embedding_client_type = None
    embedding_client = None
    openai_client = None # For ChatCompletion

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
    meta_embedding_model = None
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print(f"WARNING: {INDEX_PATH} or {META_PATH} not found. Search will fail until ingested.")
    else:
        try:
            state.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                state.meta_chunks = meta.get('chunks', [])
                meta_embedding_model = meta.get('embedding_model')
            print(f"Loaded index with {state.index.ntotal} vectors and {len(state.meta_chunks)} metadata entries. Model: {meta_embedding_model}")
        except Exception as e:
            print(f"Error loading index: {e}")

    # Setup Embedding Client
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if we should force local based on index metadata
    force_local = (meta_embedding_model == "local")
    
    if api_key and OpenAI and not force_local:
        print("Using OpenAI for embeddings and synthesis.")
        state.embedding_client_type = "openai"
        state.embedding_client = OpenAI(api_key=api_key)
        state.openai_client = OpenAI(api_key=api_key)
    elif SentenceTransformer:
        print("Using SentenceTransformer for embeddings. LLM synthesis unavailable.")
        state.embedding_client_type = "local"
        state.embedding_client = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    else:
        print("CRITICAL: No embedding model available.")

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
    if state.embedding_client_type == "openai":
        try:
            # Must match ingest model
            res = state.embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL_OPENAI)
            vec = res.data[0].embedding
            arr = np.array([vec], dtype='float32')
        except Exception as e:
            print(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")
    else:
        # Local
        vec = state.embedding_client.encode([text])[0]
        arr = np.array([vec], dtype='float32')
    
    faiss.normalize_L2(arr)
    return arr

def generate_llm_answer(query: str, hits: List[dict]) -> str:
    if not state.openai_client:
        return fallback_summary(hits)
    
    # Dedup papers for context
    seen_ids = set()
    context_lines = []
    
    citation_map = {} # id -> title shortcut

    for h in hits:
        pid = h['paper_id']
        if pid not in seen_ids:
            seen_ids.add(pid)
            title = h['title']
            text = h['content']
            context_lines.append(f"Paper ID: {pid}\nTitle: {title}\nSummary snippet: {text}\n---")
            citation_map[pid] = title
    
    if len(seen_ids) > 6:
        # Cap context
        context_lines = context_lines[:6]
    
    prompt = f"""You are a research assistant. Answer the user's question based ONLY on the following paper snippets.
Perform cross-document reasoning if applicable.
Cite your sources using the format: "1) Title — Paper ID".
Keep the answer concise (3-6 sentences).
If the context doesn't answer the question, say so.

Context:
{"\n".join(context_lines)}

Question: {query}
Answer:"""

    try:
        completion = state.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return fallback_summary(hits, error_msg="LLM service unavailable.")

def fallback_summary(hits: List[dict], error_msg: str = "") -> str:
    # Simple frequent word or just concatenation of top hit titles
    titles = set(h['title'] for h in hits[:3])
    msg = "LLM synthesis unavailable (No API key or error). "
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
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingest.py first.")

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
