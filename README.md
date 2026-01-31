# 🧠 ArXiv Intelligent Search: Privacy-First RAG at Scale
![Privacy First](https://img.shields.io/badge/Privacy-100%25_Local-green?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI_|_FAISS_|_Ollama-blue?style=for-the-badge)

> **A full-stack Semantic Search Engine capable of indexing 24,000+ research papers and performing citation-grounded Q&A using 100% local AI.**

---

## 🏗️ Architecture: The "Hybrid Compute" Model

We successfully solved the challenge of **searching 24k papers** without massive local resources by innovating a **Hybrid Architecture**:
1.  **Cloud Ingestion**: We leverage Google Colab T4 GPUs to crunch the massive dataset in minutes.
2.  **Local Inference**: We serve the application locally using quantized models for privacy and zero latency.

```mermaid
graph LR
    subgraph "Phase 1: Cloud Efficiency (Google Colab)"
    Raw[📄 24k Papers] -->|GPU Accelerated| Embed[🧠 Embeddings]
    Embed -->|Build Index| VectorDB[(🗄️ FAISS Index)]
    end

    subgraph "Phase 2: Local Privacy (User's Laptop)"
    VectorDB -->|Download| LocalApp[💻 Local App]
    User[👤 User Question] --> LocalApp
    LocalApp -->|RAG| LocalLLM[🤖 Ollama (Mistral)]
    LocalLLM -->|Synthesize| Answer[💬 Cited Answer]
    end
    
    style VectorDB fill:#ffaa00,stroke:#333,stroke-width:2px
    style LocalLLM fill:#00ffaa,stroke:#333,stroke-width:2px
```

---

## 🚀 Why This Project Wins?

Traditional search engines (Ctrl+F) fail because they look for *words*. We built a system that understands *meaning*.

### 1. 🧠 True Semantic Understanding
If you search for **"BERT specific limitations"**, our system finds papers discussing "computational costs" and "context window issues"—even if the word "limitation" never appears.

### 2. 🛡️ 100% Local & Private (Zero Cost)
Unlike API-wrapper projects, this is **Real Engineering**.
*   **No OpenAI API Keys required**.
*   **No Cloud bills**.
*   **Complete Privacy**: Your queries never leave your machine.
*   Powered by **Mistral 7B** (via Ollama) and **SentenceTransformers**.

### 3. 📚 Citation-Grounded RAG
We don't just "ask AI". We implement **Retrieval Augmented Generation**:
1.  **Retrieve**: Find the exact paragraphs from 24,000 papers proving a fact.
2.  **Generate**: Ask the AI to write an answer *using only those facts*.
3.  **Result**: An answer with `(Paper ID)` citations that you can trust.

---

## 📸 Demo

| **Complex Semantic Queries** | **Grounded Answers with Citations** |
|:---:|:---:|
| ![Search Interface](screenshots/ArXiv%20Semantic%20Search-mh.png) | ![Search Results](screenshots/ArXiv%20Semantic%20Search-mh%20(1).png) |

---

## ⚡ Quick Start

### Prerequisites
*   **Python 3.8+**
*   **[Ollama](https://ollama.ai/)** (The engine for running local AI)

### Step 1: Install & Model Setup
Install the dependencies and pull the AI model.

```powershell
# 1. Start Ollama and get the brain (Mistral 7B)
ollama pull mistral:7b
ollama serve

# 2. Install Python libs
pip install -r requirements.txt
```

### Step 2: Ingest Data (The Hybrid Way)
*   **Option A (Recommended for Judges):** Run `ingest.py` locally with a small sample to verify it works instantly.
    ```bash
    python ingest.py --limit 1000
    ```
*   **Option B (The "Wow" Factor):** Use our `colab_ingest.py` script on Google Colab to process the **Full 24,000 Dataset** in <5 minutes using free Cloud GPUs, then drop the index file here.

### Step 3: Launch
```bash
uvicorn app:app --reload
```
Open **http://localhost:8000** and start researching.

---

## 🔧 Technology Stack

We chose a stack optimized for **Performance** and **Portability**.

| Component | Tech | Why we chose it? |
| :--- | :--- | :--- |
| **Backend** | **FastAPI** | Async Python is 3x faster than Flask for ML workloads. |
| **Vector Engine** | **FAISS** | Facebook's engine is the industry standard for billion-scale search. |
| **Local LLM** | **Ollama** | Seamless local inference for heavy models like Mistral/Llama. |
| **Embeddings** | **MiniLM-L6** | The best trade-off between speed (CPU friendly) and accuracy. |
| **Architecture** | **RAG** | Retrieval Augmented Generation prevents hallucinations. |

---

## 🏆 Conclusion

This project demonstrates that **Enterprise-grade Semantic Search** doesn't require Enterprise-grade hardware or budget. By combining **Smart Ingestion (Colab)** with **Efficient Inference (Local RAG)**, we've built a search engine that is free, fast, and fiercely accurate.

*Built for the SkillRank Hackathon.*