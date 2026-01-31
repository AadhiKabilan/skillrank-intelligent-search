# 🔍 ArXiv Semantic Search System

A complete, production-ready intelligent search system for 24K+ research papers. Built for hackathon performance with **Local LLM-powered semantic search**, data-grounded synthesis, and sub-second response times.

## 🎯 Problem Overview

Traditional keyword search fails to capture semantic meaning. This system solves that by:
- **Semantic Search**: Uses local vector embeddings to find conceptually related papers.
- **RAG Architecture**: Retrieval-Augmented Generation for grounded answers.
- **Local AI**: 100% Privacy-focused using Ollama (running Mistral 7B).
- **Scalable**: Indexing performed on high-performance GPUs (via Colab) and served locally.

## 📸 Screenshots

![Search Interface](screenshots/ArXiv%20Semantic%20Search-mh.png)
_Semantic Search Interface with Complex Queries_

![Search Results](screenshots/ArXiv%20Semantic%20Search-mh%20(1).png)
_Synthesized Answer with Citations_

![Results Detail](screenshots/ArXiv%20Semantic%20Search-mh%20(2).png)
_Detailed Paper Hits and Citations_

## 🚀 Quick Start (30 seconds)

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed & running (`ollama serve`).

### Setup & Run

```bash
# 1. Start Ollama and pull the model (Mistral 7B)
ollama pull mistral:7b
ollama serve

# 2. Setup Python Environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Process Papers (24k Full Dataset)
# Option A: Run the `colab_ingest.py` script on Google Colab (Fastest - 5 mins).
# Option B (Local Demo):
python ingest.py --limit 1000

# 4. Start the Application
uvicorn app:app --reload

# 5. Open Browser
# Navigate to http://localhost:8000
```

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Data Ingest   │      │   Search API     │      │    Frontend     │
│   (Colab GPU)   │      │   (FastAPI)      │      │ (index.html)    │
├─────────────────┤      ├──────────────────┤      ├─────────────────┤
│ • Load JSON     │─────▶│ • FAISS Index    │◀─────│ • Search Box    │
│ • Chunking      │      │ • Semantic       │      │ • Results View  │
│ • Embeddings    │      │   Search         │      │ • Real-time     │
│  (SentenceTx)   │      │ • Local LLM      │      │                 │
└─────────────────┘      │   (Ollama)       │      └─────────────────┘
                         └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Ollama / API   │
                         │   (Mistral 7B)   │
                         └──────────────────┘
```

## Implementation Details

### Design Decisions
1.  **Architecture**: Split ingestion (Heavy GPU work) from Serving (Fast CPU Work).
2.  **Vector Store**: FAISS for millisecond-level similarity search.
3.  **Local LLM**: Switched to Ollama (Mistral 7B) to ensure 0% reliance on paid APIs and 100% offline capability.
4.  **Prompt Engineering**: Optimized system prompts to enforce strict citation formats `Title (ID)` required by the problem statement.

## ✅ Task Status

- [x] **Ingestion**: Scaled to 24k papers using Colab GPU acceleration.
- [x] **Vector Search**: Local FAISS index integration.
- [x] **Synthesis**: Replaced Mock/Cloud APIs with robust Local RAG (Ollama).
- [x] **Frontend**: Clean, responsive UI.

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3 | Core logic |
| **Backend** | FastAPI | High-performance API |
| **Vector DB** | FAISS | Similarity Search |
| **LLM** | Ollama (Mistral) | Answer Synthesis |
| **Embeddings** | all-MiniLM-L6-v2 | Vector generation |

---
*Built for the SkillRank Hackathon.*