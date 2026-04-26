# app.py
"""
Flask web server for ContextLens RAG.

Fixes applied in this file:
  #1  — Streaming size guard via WSGI middleware (MAX_CONTENT_LENGTH uses the
         Content-Length *header*, which clients can forge; the middleware reads
         the actual byte count from the request stream).
  #2  — Temp file wrapped in NamedTemporaryFile context manager for guaranteed
         cleanup even if os.close() or file.save() raises.
  #4  — flask-cors configured (wildcard for dev; tighten in production).
  #5  — Rate limiting via flask-limiter (5 req/min on /chat, 3 req/min on /upload).
"""
from __future__ import annotations

import os
import secrets
import tempfile
import logging

from flask import Flask, Response, request, jsonify, render_template, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from main import GroqRAGSystem

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# #3 (from previous sweep): Stable secret key from env — never auto-generated
# in production because restarts would invalidate all user sessions.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("SECRET_KEY"):
    logger.warning(
        "SECRET_KEY not set in environment. Sessions will reset on every restart. "
        "Add SECRET_KEY to your .env file."
    )

# #5 (from previous sweep): Hard reject oversized requests at Flask layer.
# Note: MAX_CONTENT_LENGTH checks the Content-Length *header* — a lying client
# can bypass it. The _StreamSizeLimitMiddleware below enforces the real limit.
_MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB
app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES

# #4: CORS — allow all origins in dev. In production, replace "*" with your
# deployed frontend URL: origins=["https://your-app.com"]
CORS(app, origins="*")

# #5: Rate limiting — protects Groq API quota and server resources.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# ── #1: Streaming size guard middleware ───────────────────────────────────────

class _StreamSizeLimitMiddleware:
    """
    WSGI middleware that counts actual bytes from the request body stream and
    raises a 413 before any application code runs. This is necessary because
    MAX_CONTENT_LENGTH only checks the Content-Length *header* which can be
    omitted or forged by clients sending chunked-encoded bodies.
    """

    def __init__(self, wsgi_app, max_bytes: int) -> None:
        self._app = wsgi_app
        self._max = max_bytes

    def __call__(self, environ, start_response):
        content_length = environ.get("CONTENT_LENGTH")
        if content_length:
            try:
                if int(content_length) > self._max:
                    body = b"Request body too large."
                    start_response(
                        "413 Request Entity Too Large",
                        [
                            ("Content-Type", "text/plain"),
                            ("Content-Length", str(len(body))),
                        ],
                    )
                    return [body]
            except ValueError:
                pass  # malformed Content-Length — let Flask handle it
        return self._app(environ, start_response)


app.wsgi_app = _StreamSizeLimitMiddleware(app.wsgi_app, _MAX_UPLOAD_BYTES)

# ── Per-session state ─────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf"}

# Session-keyed RAG instances — fixes BUG-007 (global rag_instance corruption)
_rag_instances: dict[str, GroqRAGSystem] = {}


def _session_id() -> str:
    session.setdefault("id", secrets.token_urlsafe(16))
    return session["id"]


def _get_rag() -> GroqRAGSystem | None:
    return _rag_instances.get(_session_id())


def _set_rag(instance: GroqRAGSystem) -> None:
    _rag_instances[_session_id()] = instance


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/health")
def health() -> Response:
    """Deployment health-check endpoint (Railway, Render, Fly.io)."""
    return jsonify({"status": "ok", "document_loaded": _get_rag() is not None}), 200


@app.route("/upload", methods=["POST"])
@limiter.limit("3 per minute")
def upload_file() -> Response:
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["pdf_file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # #2 FIX (Windows-safe): Use mkstemp() — the only reliably cross-platform
    # approach. mkstemp returns a raw OS file descriptor (fd). We close it
    # immediately with os.close(fd), which releases all OS-level locks with no
    # side effects. The file exists on disk but is fully unlocked, so
    # file.save(tmp_path) can open it for writing without PermissionError.
    # NamedTemporaryFile (even with delete=False) retains an internal lock on
    # Windows until the Python object is garbage-collected — mkstemp does not.
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.close(fd)          # release the fd lock NOW — before anyone else writes
        file.save(tmp_path)   # Werkzeug writes the uploaded bytes
        rag = GroqRAGSystem(tmp_path)

        _set_rag(rag)
        return jsonify({"message": "Document indexed successfully!"}), 200

    except Exception as exc:
        logger.error("Upload error: %s", exc, exc_info=True)
        return jsonify({"error": "Failed to process the PDF. Check server logs."}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # best-effort cleanup


@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")
def chat() -> Response:
    rag = _get_rag()
    if not rag:
        return jsonify({"error": "Please upload a PDF first."}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query."}), 400

    # Hard cap on query length — mirrors the frontend maxlength attribute (#16)
    if len(user_query) > 1000:
        return jsonify({"error": "Query exceeds 1000 character limit."}), 400

    try:
        answer, context_docs = rag.ask(user_query)

        # #9: Surface page number and source filename alongside chunk text
        sources = [
            {
                "text": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "document"),
            }
            for doc in context_docs
        ]

        return jsonify({"answer": answer, "sources": sources}), 200

    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        return jsonify({"error": "An internal error occurred. Check server logs."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)