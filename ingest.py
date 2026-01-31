import json
import argparse
import os
import sys
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: Failed to import sentence_transformers: {e}")
    SentenceTransformer = None

# --- Configuration & Defaults ---
DEFAULT_CHUNK_SIZE = 800
DEFAULT_BATCH_SIZE = 100
DEFAULT_LIMIT = None
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

def generate_embeddings(texts: List[str], client: Any, batch_size=100) -> np.ndarray:
    embeddings = []
    total = len(texts)
    
    print(f"Generating embeddings for {total} chunks using local SentenceTransformer...")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        # SentenceTransformer
        batch_embs = client.encode(batch)
        embeddings.extend(batch_embs)
        
        sys.stdout.write(f"\rProcessed {min(i + batch_size, total)}/{total}")
        sys.stdout.flush()
    
    print("\nNormalization...")
    emb_array = np.array(embeddings, dtype='float32')
    # L2 Normalize for Cosine Similarity
    faiss.normalize_L2(emb_array)
    return emb_array

def create_chunks(papers: List[Dict], chunk_size: int) -> List[TextChunk]:
    chunks = []
    for paper in papers:
        # Extract fields safely
        p_id = paper.get('id', 'unknown')
        title = paper.get('title', 'No Title')
        summary = paper.get('summary', '') or paper.get('abstract', '') # Handle both summary/abstract keys
        
        # Helper to get first link
        links = paper.get('arxiv_links', [])
        # Sometimes links are strings, sometimes dicts depending on dataset version
        url = ""
        if isinstance(links, list) and len(links) > 0:
            first = links[0]
            if isinstance(first, dict):
                url = first.get('href', '')
            elif isinstance(first, str):
                url = first
        if not url:
             # Fallback if id looks like an arxiv id
             url = f"https://arxiv.org/abs/{p_id}"

        # Combine proper content
        full_text = f"{title}\n{summary}"
        
        # Simple chunking
        # Design says ~800 chars. We'll do a simple split or window.
        # Given abstracts are usually short, often 1 chunk is enough.
        # But let's check length.
        
        start = 0
        text_len = len(full_text)
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Try to break on space
            if end < text_len:
                last_space = full_text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            segment = full_text[start:end].strip()
            if segment:
                chunks.append(TextChunk(
                    paper_id=p_id,
                    title=title,
                    content=segment,
                    chunk_index=chunk_idx,
                    arxiv_url=url
                ))
                chunk_idx += 1
            
            start = end
            
    return chunks

def load_arxiv_data(filepath: str, limit: Optional[int]) -> List[Dict]:
    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            # Sometimes wrapped in a key
            keys = list(data.keys())
            if len(keys) == 1 and isinstance(data[keys[0]], list):
                data = data[keys[0]]
                
        if not isinstance(data, list):
            print("Error: PDF JSON root is not a list.")
            sys.exit(1)
            
        if limit:
            return data[:limit]
        return data
        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in {filepath}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ArXiv Semantic Search Ingestion (Local Utility)")
    parser.add_argument("--input", default="arxivData.json", help="Path to input JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers (for demo)")
    parser.add_argument("--chunk_chars", type=int, default=DEFAULT_CHUNK_SIZE, help="Max characters per chunk")
    parser.add_argument("--output", default=".", help="Output directory for index and meta")
    
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

    # 3. Vectorize (Local Only)
    if not SentenceTransformer:
        print("Error: sentence-transformers not installed.")
        sys.exit(1)
        
    print("Loading local SentenceTransformer model...")
    client = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
    
    texts = [c.content for c in chunks]
    vectors = generate_embeddings(texts, client)
    
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
        embedding_model="local",
        created_at=datetime.utcnow().isoformat(),
        paper_count=len(papers)
    )
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(meta_obj), f, indent=2)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
