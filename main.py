"""
main.py — ContextLens RAG pipeline

Architecture (#27): Three single-responsibility classes composed by GroqRAGSystem.
  DocumentIngester  → load & split a PDF into metadata-rich chunks
  HybridRetriever   → dense + sparse search fused with RRF, then reranked
  AnswerGenerator   → prompt, LLM call, answer extraction
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Layer 1 & 2: Ingestion & Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Layer 4 & 6: Retrieval (Dense + Sparse)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Layer 8 & 9: Augmentation & Generation
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Cross-encoder reranker (#11)
from sentence_transformers import CrossEncoder

# ── Startup validation (#3 already done, kept here for module import safety) ──
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key_here"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once, reused across uploads) ──────────────

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_reranker: Optional[CrossEncoder] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """#4 FIX: Cached at module level — never reloaded per upload."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading dense embedding model (one-time)…")
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Dense embedding model ready.")
    return _embedding_model


def _get_reranker() -> CrossEncoder:
    """#11: Cross-encoder reranker cached at module level."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker (one-time)…")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker ready.")
    return _reranker


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 & 2 — Document Ingestion
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentIngester:
    """
    Single responsibility: load a PDF and split it into metadata-rich chunks.

    #8  FIX: corrected separators order — "." before " " caused mid-sentence
        breaks on abbreviations like "Fig. 3". Now uses ["\n\n", "\n", " ", ""].
    #9  FIX: page number and source filename preserved in chunk.metadata so the
        UI can display "From page 4 of report.pdf" rather than a raw text blob.
    """

    # Safer separator order: paragraph → line → word → character
    _SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self._SEPARATORS,
        )

    def ingest(self, pdf_path: str) -> list[Document]:
        """Load *pdf_path* and return a list of Document chunks with metadata."""
        logger.info("Ingesting PDF: %s", pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()                # Each page is a Document

        # #9: PyPDFLoader already sets page_number in metadata; we enrich it
        # with the source filename so the UI can surface it later.
        source_name = Path(pdf_path).name
        for page in pages:
            page.metadata.setdefault("source", source_name)

        chunks = self._splitter.split_documents(pages)
        logger.info("Split into %d chunks (size=%d, overlap=%d).",
                    len(chunks), self._splitter._chunk_size,
                    self._splitter._chunk_overlap)
        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Layers 4 & 6 — Hybrid Retrieval + Reranking
# ═══════════════════════════════════════════════════════════════════════════════

FAISS_INDEX_DIR = "faiss_index"


class HybridRetriever:
    """
    Single responsibility: retrieve the most relevant chunks for a query.

    #10 FIX: BM25 k set at construction time, not mutated per-call (race-safe).
    #11    : Cross-encoder reranker rescores candidates before final truncation.
    #15    : FAISS index persisted to disk; reloaded on next startup if present.
    """

    def __init__(self, chunks: list[Document], fetch_k: int = 6) -> None:
        self._fetch_k = fetch_k
        embeddings = _get_embedding_model()

        # Dense index — #15: persist to disk so restarts don't lose the index
        self._vectorstore = FAISS.from_documents(chunks, embeddings)
        self._vectorstore.save_local(FAISS_INDEX_DIR)
        logger.info("FAISS index saved to '%s'.", FAISS_INDEX_DIR)

        # Sparse index — #10: k fixed at construction
        self._bm25 = BM25Retriever.from_documents(chunks, k=fetch_k)

    @classmethod
    def from_saved_index(cls, chunks: list[Document], fetch_k: int = 6) -> "HybridRetriever":
        """
        Reload a persisted FAISS index if it exists; otherwise build from chunks.
        #15: Called at startup so the user doesn't have to re-upload after a restart.
        """
        instance = object.__new__(cls)
        instance._fetch_k = fetch_k
        embeddings = _get_embedding_model()

        if Path(FAISS_INDEX_DIR).exists():
            logger.info("Loading persisted FAISS index from '%s'.", FAISS_INDEX_DIR)
            instance._vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
        else:
            instance._vectorstore = FAISS.from_documents(chunks, embeddings)
            instance._vectorstore.save_local(FAISS_INDEX_DIR)

        instance._bm25 = BM25Retriever.from_documents(chunks, k=fetch_k)
        return instance

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        """
        Hybrid RRF retrieval followed by cross-encoder reranking.

        Steps:
          1. Dense similarity search (fetch_k results)
          2. Sparse BM25 keyword search (fetch_k results)
          3. Reciprocal Rank Fusion → deduplicated candidate list
          4. Cross-encoder reranker rescores candidates against query
          5. Return top_k highest-scoring documents
        """
        # 1. Dense
        dense_docs: list[Document] = self._vectorstore.similarity_search(
            query, k=self._fetch_k
        )
        # 2. Sparse
        sparse_docs: list[Document] = self._bm25.invoke(query)

        # 3. RRF fusion (constant C=60 is the standard)
        rrf_scores: dict[str, float] = {}
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        for rank, doc in enumerate(sparse_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)

        all_docs: dict[str, Document] = {
            d.page_content: d for d in dense_docs + sparse_docs
        }
        candidates: list[Document] = [
            all_docs[key]
            for key, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # 4. Rerank with cross-encoder (#11)
        if len(candidates) > top_k:
            reranker = _get_reranker()
            pairs = [(query, doc.page_content) for doc in candidates]
            scores: list[float] = reranker.predict(pairs).tolist()
            candidates = [
                doc for _, doc in sorted(
                    zip(scores, candidates), key=lambda x: x[0], reverse=True
                )
            ]

        return candidates[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Layers 8 & 9 — Answer Generation
# ═══════════════════════════════════════════════════════════════════════════════

_PROMPT_TEMPLATE = """SYSTEM INSTRUCTION:
You are a grounded assistant. Use ONLY the provided context to answer.
If the answer is not present in the context, say you do not know.
Cite the relevant [Source X] tag(s) in your response to support your answer.

CONTEXT:
{context}

USER QUERY:
{question}

ANSWER:"""


class AnswerGenerator:
    """
    Single responsibility: given a query and retrieved documents, generate an answer.

    #13 FIX: Prompt compiled once in __init__, not on every call.
    #14 FIX: Early return when context is empty — no LLM call, no hallucination.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        self._llm = ChatGroq(model_name=model_name, temperature=0)
        # #13: Parse template once at construction, not per query
        self._prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
        self._chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | self._prompt
            | self._llm
        )

    def generate(self, query: str, context_docs: list[Document]) -> str:
        """Return a grounded answer string using *context_docs* as evidence."""
        # #14: Guard against empty retrieval — don't hallucinate on empty context
        if not context_docs:
            logger.warning("No context documents retrieved for query: %r", query)
            return (
                "I could not find relevant content in the document to answer your question. "
                "Try rephrasing, or check that the PDF contains information on this topic."
            )

        # Build context block with source tags and metadata (#9)
        context_parts: list[str] = []
        for i, doc in enumerate(context_docs):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "document")
            context_parts.append(
                f"[Source {i}] (page {page}, {source}):\n{doc.page_content}"
            )
        context_text = "\n\n".join(context_parts)

        response = self._chain.invoke({"context": context_text, "question": query})

        # Safe attribute access (#9 from previous bug list)
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        return answer or "The model returned an empty response. Please try rephrasing."


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator — backward-compatible public API
# ═══════════════════════════════════════════════════════════════════════════════

class GroqRAGSystem:
    """
    Thin orchestrator that composes DocumentIngester, HybridRetriever,
    and AnswerGenerator. app.py imports only this class.

    #27: Each concern is now independently testable via its own class.
    """

    def __init__(self, pdf_path: str) -> None:
        ingester = DocumentIngester()
        self._chunks: list[Document] = ingester.ingest(pdf_path)
        self._retriever = HybridRetriever(self._chunks)
        self._generator = AnswerGenerator()
        logger.info("GroqRAGSystem ready (%d chunks indexed).", len(self._chunks))

    def ask(self, query: str) -> tuple[str, list[Document]]:
        """
        Single public entry point. Returns (answer_string, source_documents).
        The source_documents are the exact chunks the LLM received.
        """
        docs = self._retriever.retrieve(query)
        answer = self._generator.generate(query, docs)
        return answer, docs


# ── CLI for quick smoke-testing ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    PDF_FILE = "rag_wp.pdf"
    if not os.path.exists(PDF_FILE):
        logger.error("PDF not found: %s", PDF_FILE)
        sys.exit(1)

    rag = GroqRAGSystem(PDF_FILE)
    print("\n--- ContextLens RAG  (type 'exit' to quit) ---\n")
    while True:
        user_q = input("Ask: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            break
        if not user_q:
            continue

        ans, sources = rag.ask(user_q)
        print(f"\nANSWER:\n{ans}\n")
        print(f"[{len(sources)} source chunk(s) retrieved]\n")