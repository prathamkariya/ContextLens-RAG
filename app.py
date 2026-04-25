# app.py
import os
import tempfile
from flask import Flask, request, jsonify, render_template
from main import GroqRAGSystem # Your existing backend

app = Flask(__name__)

# Global variable to hold our RAG instance (for single-user prototype)
rag_instance = None

@app.route('/')
def index():
    # Serves the index.html from the templates folder
    return render_template('index.html')

#Upload files and process them with the RAG system.
# The uploaded PDF is saved temporarily, processed, and then deleted to free up space. The response includes a success message or an error if something goes wrong during the upload or processing.

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag_instance
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save temp file
        fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        file.save(temp_path)
        
        try:
            # Initialize Layer 1-6: Ingestion and Indexing
            rag_instance = GroqRAGSystem(temp_path)
            return jsonify({'message': ' Document indexed successfully!'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.remove(temp_path)

@app.route('/chat', methods=['POST'])
def chat():
    global rag_instance
    if not rag_instance:
        return jsonify({'error': 'Please upload a PDF first.'}), 400
        
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({'error': 'Empty query'}), 400

    try:
        # Execute Layer 8-9: Retrieval and Generation
        context_docs = rag_instance.manual_hybrid_retrieve(user_query)
        sources = [doc.page_content for doc in context_docs]
        
        response = rag_instance.ask(user_query)
        
        return jsonify({
            'answer': response.content,
            'sources': sources
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)