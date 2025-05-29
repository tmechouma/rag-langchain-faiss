# Import necessary libraries and modules
from flask import Flask, render_template, request, jsonify  # Flask for web server, rendering HTML templates, handling requests, and returning JSON
from rag_pipeline import RAGPipeline  # Custom RAG (Retrieval-Augmented Generation) pipeline for document-based QA
import os  # OS module for file and directory operations
from werkzeug.utils import secure_filename  # Utility to safely handle filenames
import threading  # For thread-safe access to shared resources

# Initialize the Flask application
app = Flask(__name__)

# Configuration: set upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

# Initialize a thread-safe lock for pipeline access
pipeline_lock = threading.Lock()

# Instantiate the RAG pipeline, pointing to the directory with uploaded documents
rag_pipeline = RAGPipeline(data_dir="data")

# Helper function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the homepage
@app.route('/')
def home():
    files = []
    # If upload folder exists, list all files in it
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
    # Render the homepage with the list of uploaded files
    return render_template('index.html', files=files)

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed and process it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Sanitize the filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Save the file to the upload folder
        
        # Reinitialize the RAG pipeline with the new documents (thread-safe)
        with pipeline_lock:
            rag_pipeline.initialize()
        
        return jsonify({"success": True, "filename": filename})
    
    return jsonify({"error": "Invalid file type"}), 400

# Route to handle question answering
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json  # Parse the incoming JSON
    question = data.get('question', '')  # Extract the question
    max_tokens = data.get('max_tokens', 512)  # Optional: max tokens for generation
    temperature = data.get('temperature', 0.3)  # Optional: temperature for sampling
    
    try:
        # Use the RAG pipeline to get an answer (thread-safe)
        with pipeline_lock:
            answer = rag_pipeline.query(
                question=question,
                max_tokens=max_tokens,
                temperature=temperature
            )
        return jsonify({"answer": answer})
    except Exception as e:
        # Handle any exception and return an error message
        return jsonify({"error": str(e)}), 500

# Entry point: ensures upload folder exists and runs the Flask server
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)  # Run the app on all interfaces, port 5000, with debug enabled
