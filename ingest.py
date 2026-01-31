import json
import argparse
import os
import sys
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Optional imports for embeddings
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: Failed to import sentence_transformers: {e}")
    SentenceTransformer = None

# --- Configuration & Defaults ---
DEFAULT_CHUNK_SIZE = 800
DEFAULT_BATCH_SIZE = 100
DEFAULT_LIMIT = None
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"

# --- Data Structures ---
@dataclass
class TextChunk:
    paper_id: str
    title: str
    content: str
    chunk_index: int
    arxiv_url: str

@dataclass
class IndexMetadata:
    chunks: List[Dict[str, Any]]
    index_size: int
    embedding_model: str
    created_at: str
    paper_count: int

# --- Helper Functions ---

def get_embedding_client(force_local: bool = False):
    if not force_local:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI:
            return "openai", OpenAI(api_key=api_key)
    
    if SentenceTransformer:
        return "local", SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    print("Error: neither OpenAI key present nor sentence-transformers installed.")
    sys.exit(1)

# ... (rest of code)

def main():
    parser = argparse.ArgumentParser(description="ArXiv Semantic Search Ingestion")
    parser.add_argument("--input", default="arxivData.json", help="Path to input JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers (for demo)")
    parser.add_argument("--chunk_chars", type=int, default=DEFAULT_CHUNK_SIZE, help="Max characters per chunk")
    parser.add_argument("--output", default=".", help="Output directory for index and meta")
    parser.add_argument("--local", action="store_true", help="Force use of local sentence-transformers")
    
    args = parser.parse_args()
    
    # 1. Load Data
    papers = load_arxiv_data(args.input, args.limit)
    print(f"Loaded {len(papers)} papers.")
    
    # 2. Chunk
    chunks = create_chunks(papers, args.chunk_chars)
    print(f"Created {len(chunks)} chunks.")
    if not chunks:
        print("No chunks created. Exiting.")
        return

    # 3. Vectorize
    client_type, client = get_embedding_client(args.local)
    texts = [c.content for c in chunks]
    vectors = generate_embeddings(texts, client_type, client)
    
    # 4. Build Index
    d = vectors.shape[1]
    print(f"Building FAISS index (dim={d})...")
    # IndexFlatIP is inner product; since vectors are normalized, it equals cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    
    # 5. Save
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    index_path = os.path.join(output_dir, "index.faiss")
    meta_path = os.path.join(output_dir, "meta.json")
    
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {meta_path}...")
    # Convert chunks to dicts for JSON serialization
    chunks_data = [asdict(c) for c in chunks]
    
    meta_obj = IndexMetadata(
        chunks=chunks_data,
        index_size=len(chunks),
        embedding_model=client_type,
        created_at=datetime.utcnow().isoformat(),
        paper_count=len(papers)
    )
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(meta_obj), f, indent=2)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
