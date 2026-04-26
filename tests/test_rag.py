"""
tests/test_rag.py — Unit and smoke tests for ContextLens RAG (#28)

Run with:
    pytest tests/ -v

Tests are deliberately dependency-light: the LLM and embedding model are
mocked so the test suite runs offline without a GROQ_API_KEY.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_doc(text: str, page: int = 1, source: str = "test.pdf"):
    """Create a minimal LangChain-like Document stub."""
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"page": page, "source": source}
    return doc


# ── DocumentIngester ──────────────────────────────────────────────────────────

class TestDocumentIngester(unittest.TestCase):

    @patch("main.PyPDFLoader")
    def test_ingest_preserves_source_metadata(self, MockLoader):
        """Each chunk must carry the source filename in its metadata."""
        # Arrange
        from main import DocumentIngester

        fake_page = _make_doc("Hello world. " * 100, page=0, source="")
        fake_page.metadata = {"page": 0}          # loader doesn't set source
        MockLoader.return_value.load.return_value = [fake_page]

        ingester = DocumentIngester(chunk_size=200, chunk_overlap=20)

        # Act
        chunks = ingester.ingest("test.pdf")

        # Assert
        self.assertGreater(len(chunks), 0, "Expected at least one chunk")
        for chunk in chunks:
            self.assertEqual(chunk.metadata.get("source"), "test.pdf",
                             "Source filename missing from chunk metadata")

    @patch("main.PyPDFLoader")
    def test_ingest_uses_safe_separators(self, MockLoader):
        """Separator list must not contain '.' (causes mid-sentence splits)."""
        from main import DocumentIngester

        MockLoader.return_value.load.return_value = []
        ingester = DocumentIngester()

        self.assertNotIn(".", ingester._splitter._separators,
                         "Separator '.' causes mid-sentence chunk breaks (see issue #8)")


# ── HybridRetriever ───────────────────────────────────────────────────────────

class TestHybridRetriever(unittest.TestCase):

    def _make_retriever(self, docs):
        """Build a HybridRetriever with all heavy dependencies mocked out."""
        from main import HybridRetriever

        with (
            patch("main._get_embedding_model") as mock_emb,
            patch("main.FAISS") as MockFAISS,
            patch("main.BM25Retriever") as MockBM25,
        ):
            mock_emb.return_value = MagicMock()
            MockFAISS.from_documents.return_value = MagicMock()
            MockBM25.from_documents.return_value = MagicMock()

            retriever = HybridRetriever(docs, fetch_k=4)

        return retriever

    def test_retrieve_returns_top_k(self):
        """retrieve() must return at most k documents."""
        docs = [_make_doc(f"chunk {i}") for i in range(10)]
        retriever = self._make_retriever(docs)

        # Mock internal search methods
        retriever._vectorstore.similarity_search.return_value = docs[:4]
        retriever._bm25.invoke.return_value = docs[2:6]

        with patch("main._get_reranker") as mock_ranker:
            mock_ranker.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            result = retriever.retrieve("test query", top_k=3)

        self.assertLessEqual(len(result), 3)

    def test_rrf_deduplicates_overlap(self):
        """Docs appearing in both dense and sparse results must appear only once."""
        shared_doc = _make_doc("shared content")
        dense_only = _make_doc("dense only")
        sparse_only = _make_doc("sparse only")

        docs = [shared_doc, dense_only, sparse_only]
        retriever = self._make_retriever(docs)

        retriever._vectorstore.similarity_search.return_value = [shared_doc, dense_only]
        retriever._bm25.invoke.return_value = [shared_doc, sparse_only]

        with patch("main._get_reranker") as mock_ranker:
            mock_ranker.return_value.predict.return_value = [1.0, 0.9, 0.8]
            result = retriever.retrieve("query", top_k=5)

        contents = [d.page_content for d in result]
        self.assertEqual(len(contents), len(set(contents)), "Duplicate documents in result")


# ── AnswerGenerator ────────────────────────────────────────────────────────────

class TestAnswerGenerator(unittest.TestCase):

    def _make_generator(self):
        from main import AnswerGenerator
        with patch("main.ChatGroq") as MockGroq:
            MockGroq.return_value = MagicMock()
            gen = AnswerGenerator()
        return gen

    def test_empty_context_returns_canned_response(self):
        """#14: No LLM call when context_docs is empty; returns a safe message."""
        gen = self._make_generator()
        gen._chain = MagicMock()   # should never be called

        answer = gen.generate("What is X?", context_docs=[])

        gen._chain.invoke.assert_not_called()
        self.assertIn("could not find", answer.lower())

    def test_answer_extracted_from_content_attribute(self):
        """#9 (prev): Returns response.content when attribute exists."""
        gen = self._make_generator()

        fake_response = MagicMock()
        fake_response.content = "The answer is 42."
        gen._chain = MagicMock()
        gen._chain.invoke.return_value = fake_response

        docs = [_make_doc("relevant context")]
        answer = gen.generate("What is the answer?", docs)

        self.assertEqual(answer, "The answer is 42.")

    def test_prompt_compiled_once_not_per_call(self):
        """#13: ChatPromptTemplate.from_template called exactly once (in __init__)."""
        with patch("main.ChatGroq"), \
             patch("main.ChatPromptTemplate") as MockPrompt:
            MockPrompt.from_template.return_value = MagicMock()
            from main import AnswerGenerator
            AnswerGenerator()
            AnswerGenerator()   # second instance — template parsed once per instance

        # Each instance compiles once; two instances → two calls total (not 0, not >2)
        self.assertEqual(MockPrompt.from_template.call_count, 2)


# ── Flask routes ─────────────────────────────────────────────────────────────

class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        # Patch heavy imports before importing app
        os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")

        import main as main_mod
        main_mod.GROQ_API_KEY = "test-key-for-unit-tests"

        import app as app_mod
        app_mod.app.config["TESTING"] = True
        app_mod.app.config["SECRET_KEY"] = "test-secret"
        self.client = app_mod.app.test_client()

    def test_health_returns_200(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")

    def test_chat_without_upload_returns_400(self):
        res = self.client.post(
            "/chat",
            json={"query": "What is this about?"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("upload", res.get_json()["error"].lower())

    def test_chat_empty_query_returns_400(self):
        res = self.client.post(
            "/chat",
            json={"query": "   "},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_upload_rejects_non_pdf(self):
        from io import BytesIO
        data = {"pdf_file": (BytesIO(b"fake content"), "malware.exe")}
        res = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("PDF", res.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
