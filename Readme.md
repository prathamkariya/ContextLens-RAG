# █ ContextLens RAG

> **An Enterprise-Grade Retrieval-Augmented Generation Engine.**
> Bridging the gap between powerful Large Language Models and private enterprise data with uncompromising factual grounding, auditability, and semantic integrity.

---

## System Architecture Blueprint

A poorly structured README translates to poorly understood software. This document serves as your architectural blueprint. ContextLens RAG moves beyond basic AI wrappers by implementing a sophisticated **Orchestration Loop** that enforces factual precision and deep context preservation.

### The Architectural Triad

ContextLens RAG is engineered upon three foundational pillars designed to eliminate hallucinations and preserve meaning:

#### 1. Intelligent Ingestion & Indexing
* **Core Technology:** `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
* **The Challenge:** Traditional chunking arbitrarily cuts sentences in half, inevitably causing "Context Loss". 
* **The Solution:** We implement a 1000-token chunk size with a 15% overlap rule (150 tokens) to rigidly preserve semantic boundaries (Paragraph -> Sentence -> Word). This ensures the retrieval engine never loses context mid-thought.

#### 2. Parallel Hybrid Search Retrieval
* **Core Technology:** HuggingFace `all-MiniLM-L6-v2` local embeddings paired with **FAISS** (Dense vector search) and **BM25** (Sparse keyword search).
* **The Challenge:** Semantic search misses exact technical jargon (like product codes or alphanumerics), while pure keyword search fails to capture broader conceptual intent.
* **The Solution:** ContextLens RAG executes both searches in parallel and fuses the results. BM25 filters using exact term matching, while FAISS refines by identifying deeper semantic connections. A manual Reciprocal Rank Fusion (RRF) and deduplication logic extracts the optimal intersection of context.

#### 3. High-Speed Grounded Generation
* **Core Technology:** `langchain-groq` utilizing the `Llama-3.3-70b-versatile` model.
* **The Challenge:** LLMs inherently risk hallucinating or inventing facts when not strictly constrained by bounds.
* **The Solution:** Operated on the Groq LPU, inference runs with ultra-low latency reasoning. We enforce a strict "Prompt Stack" that compels the model to derive answers exclusively from the retrieved text. It is explicitly required to cite its sources (e.g., `[Source X]`), guaranteeing 100% downstream auditability.

---

## Technical Stack

* **Frontend:** Custom HTML5, CSS3, Vanilla JavaScript (Decoupled, responsive UI).
* **Backend Framework:** Flask (REST API orchestration).
* **Orchestration Engine:** LangChain v1.x (Stable Ecosystem).
* **Vector Store:** FAISS (CPU-optimized for high-speed local vector storage).
* **Sparse Index:** Rank-BM25.
* **Embeddings:** Sentence-Transformers (Local execution for absolute privacy and zero external footprint).
* **LLM Engine:** Groq API.

---

## Getting Started

### Prerequisites
* Python 3.9+
* A valid Groq API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ChatWithPDF
   ```

2. **Install dependencies:**
   It is highly recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration:**
   Create a `.env` file in the root directory and securely add your Groq API key:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   ```

### Execution

**To run the Web UI:**
```bash
python app.py
```
Then navigate to `http://127.0.0.1:5000` in your preferred browser.

**To run the CLI diagnostic interface:**
Ensure a sample PDF named `rag_wp.pdf` is located in the root directory, then execute:
```bash
python main.py
```