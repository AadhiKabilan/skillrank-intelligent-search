# 🚀 Colab GPU Ingestion Script

# 1. Open Google Colab: https://colab.research.google.com/
# 2. Runtime -> Change runtime type -> T4 GPU
# 3. Paste this code into a cell and Run.

# --- Install Dependencies ---
!pip install sentence-transformers faiss-gpu kagglehub

import kagglehub
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import glob

# --- Download Data (Kaggle) ---
print("Downloading Dataset...")
path = kagglehub.dataset_download("neelshah18/arxivdataset")
print("Path to dataset files:", path)

# Find the JSON file
json_files = glob.glob(f"{path}/*.json")
if not json_files:
    raise FileNotFoundError("No JSON file found in dataset path")
INPUT_FILE = json_files[0]
print(f"Using input file: {INPUT_FILE}")

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2" # Fast, efficient, matches local ingest

# --- Load Data ---
print("Loading JSON into memory...")
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

# Ensure data is a list
if isinstance(data, dict):
    keys = list(data.keys())
    # Often wrapped in a key like "data" or "papers" or just root dict
    if len(keys) == 1 and isinstance(data[keys[0]], list):
        data = data[keys[0]]
    else:
        # Some datasets are newline delimited JSON
        # If the load worked, it might be a list already. 
        pass

if not isinstance(data, list):
    print(f"Warning: Data root is {type(data)}. Trying to fix...")
    # Add custom fix logic if needed, but usually it's list or dict-wrapper
    
print(f"Loaded {len(data)} papers.")

# --- Chunking ---
chunks = []
MAX_CHUNKS = 100000 # Safety cap to prevent RAM explosion if 24k papers are massive
print(f"Chunking papers (Target ~800 chars)...")

for i, paper in enumerate(data):
    if len(chunks) >= MAX_CHUNKS: 
        break
        
    # Robust extraction (handle missing keys)
    title = paper.get('title', 'No Title')
    summary = paper.get('summary', '') or paper.get('abstract', '')
    p_id = paper.get('id', str(i))
    
    # Handle links
    links = paper.get('arxiv_links', [])
    url = ""
    if isinstance(links, list) and len(links) > 0:
         first = links[0]
         if isinstance(first, dict): url = first.get('href', '')
         elif isinstance(first, str): url = first
    if not url: url = f"https://arxiv.org/abs/{p_id}"

    full_text = f"{title}\n{summary}"
    
    # Simple chunk loop
    start = 0
    text_len = len(full_text)
    
    while start < text_len:
        end = min(start + 800, text_len)
        # Try to break on space
        if end < text_len:
            last_space = full_text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        segment = full_text[start:end].strip()
        if segment:
            chunks.append({
                "paper_id": p_id,
                "title": title,
                "content": segment,
                "arxiv_url": url
            })
        start = end

print(f"Generated {len(chunks)} chunks.")

# --- Embedding (GPU) ---
print(f"Loading Model {MODEL_NAME} to GPU...")
model = SentenceTransformer(MODEL_NAME, device='cuda')

print("Generating Embeddings...")
# Encode in batches
vectors = model.encode([c['content'] for c in chunks], batch_size=64, show_progress_bar=True)

# --- Indexing ---
print("Normalizing & Indexing...")
faiss.normalize_L2(vectors)
d = vectors.shape[1]
index = faiss.IndexFlatIP(d)
index.add(vectors)

# --- Saving ---
print("Saving artifacts...")
faiss.write_index(index, "index.faiss")

meta_data = {
    "chunks": chunks,
    "embedding_model": "local" # Signals app.py to use SentenceTransformers
}
with open("meta.json", "w") as f:
    json.dump(meta_data, f) # Standard json dump

print("✅ DONE!")
print("Download 'index.faiss' and 'meta.json' from the files tab on the left.")
