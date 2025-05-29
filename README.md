# rag-langchain-faiss
A privacy-focused RAG pipeline that answers questions about your documents - 100% offline using Zephyr-7B and FAISS*

# 🧠 Local Document Intelligence Agent

*A privacy-focused RAG pipeline that answers questions about your documents - 100% offline using Zephyr-7B and FAISS*

![RAG Workflow Diagram](https://github.com/tmechouma/rag-langchain-faiss/blob/main/gui.png)  

![RAG Workflow Diagram](https://github.com/tmechouma/rag-langchain-faiss/blob/main/workflow.png)  



## ⚠️ Important Setup Note
**Before running:**  
You must manually download the 4GB Zephyr-7B model file and put it in the model folder. Also you need to update the model's path (rag_pipeline.py) depending to where you put the repository locally. Line 66: model_path = os.path.join("D:/rag-langchain-app", "model", "zephyr-7b-beta.Q4_0.gguf").
## 🔥 Key Features
- **Complete local execution** - No API calls, no data leaves your machine
- **Multi-format support** - PDFs, TXT files (with clean text extraction)
- **Optimized for efficiency** - 4-bit quantized Zephyr-7B + FAISS HNSW indexing
- **Self-healing pipeline** - Auto-reinitializes when documents change
- **Privacy by design** - Ideal for legal/medical/confidential documents

## 🛠️ Tech Stack
| Component           | Technology                          |
|---------------------|-------------------------------------|
| LLM                 | Zephyr-7B-beta (4-bit quantized)    |
| Embeddings          | all-MiniLM-L6-v2                    |
| Vector Store        | FAISS with HNSW indexing            |
| Framework           | LangChain + Flask                   |
| Hardware Acceleratio| CUDA (optional)                    |

## 🚀 Getting Started

### Prerequisites
```bash
python>=3.9
pip install -r requirements.txt
```
### Project structure.
   ``` rag-langchain-app
├── app.py                 # Flask backend
├── rag_pipeline.py        # Core RAG logic
├── models/                # LLM storage
│   └── zephyr-7b-beta.Q4_K_M.gguf
├── data/                  # Document uploads
├── templates/             # Web UI
│   └── index.html
└── requirements.txt     ```
