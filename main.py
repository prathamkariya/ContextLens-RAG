import os
import tempfile
from dotenv import load_dotenv


# Layer 1 & 2: Ingestion & Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Layer 4 & 6: Retrieval (Dense + Sparse)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Layer 8 & 9: Augmentation & Generation
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (GROQ_API_KEY)
load_dotenv()

class GroqRAGSystem:
    def __init__(self, pdf_path):
        """
        Initializes the Triad architecture: Indexing, Retrieval, and Generation.
        """
        print(f"Layer 1 & 2: Processing {pdf_path}")
        self.chunks = self._ingest_and_split(pdf_path)
        
        print("Layer 4 & 6: Initializing Hybrid Index ")
        self.vectorstore, self.bm25 = self._setup_retrieval(self.chunks)
        
        # Layer 9: Initialize Groq Llama 3.3
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    def _ingest_and_split(self, path):
        # Extraction (Layer 1) [cite: 564, 579]
        loader = PyPDFLoader(path)
        
        # Recursive Splitting (Layer 2) [cite: 598, 618]
        # Prioritizes paragraph/sentence boundaries and uses 15% overlap [cite: 605, 609]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        return splitter.split_documents(loader.load())

    def _setup_retrieval(self, chunks):
        # Dense Vectors (Layer 4): Conceptual Meaning [cite: 658, 690]
        # Running locally via HuggingFace for speed and privacy
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Sparse Vectors (Layer 6): Keyword Matching [cite: 715, 717, 721]
        bm25 = BM25Retriever.from_documents(chunks)
        return vectorstore, bm25

    def manual_hybrid_retrieve(self, query, k=3):
        """
        Combines Semantic (Dense) and Lexical (Sparse) search results.
        Captures both 'Intent' and 'Technical Jargon'[cite: 637, 726].
        """
        # A. Get Semantic Matches
        dense_docs = self.vectorstore.similarity_search(query, k=k)
        
        # B. Get Keyword Matches
        self.bm25.k = k
        sparse_docs = self.bm25.invoke(query)
        
        # C. Fusion and Deduplication
        combined = []
        seen = set()
        for doc in (dense_docs + sparse_docs):
            if doc.page_content not in seen:
                combined.append(doc)
                seen.add(doc.page_content)
        return combined[:k]

    def ask(self, query):
        # 1. Retrieve relevant evidence (Layer 8) [cite: 749, 752]
        context_docs = self.manual_hybrid_retrieve(query)
        context_text = "\n\n".join([f"[Source {i}]: {d.page_content}" for i, d in enumerate(context_docs)])
        
        # 2. Build the Prompt Stack [cite: 757, 760]
        template = """
        SYSTEM INSTRUCTION: 
        You are a grounded assistant. Use ONLY the provided context to answer. 
        If the answer is not present, say you do not know. 
        Enforce auditability by citing [Source X] in your response. [cite: 751, 755]

        CONTEXT: 
        {context}

        USER QUERY: 
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # 3. Execution via Orchestration Loop [cite: 533]
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt 
            | self.llm
        )
        return chain.invoke(query)

# CLI INTERFACE FOR TESTING 
if __name__ == "__main__":
    # Point this to your actual PDF file in VS Code
    PDF_FILE = "rag_wp.pdf" 
    
    if not os.path.exists(PDF_FILE):
        print(f"Error: {PDF_FILE} not found. Please place it in the project folder.")
    else:
        rag = GroqRAGSystem(PDF_FILE)
        print("\n--- RAG System Active ---")
        while True:
            user_q = input("\nAsk (or 'exit'): ")
            if user_q.lower() == "exit": break
            
            response = rag.ask(user_q)
            print(f"\nAI ANSWER:\n{response.content}")