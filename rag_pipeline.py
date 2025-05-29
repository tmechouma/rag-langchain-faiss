# Import necessary modules from LangChain and other libraries
from langchain_community.llms import LlamaCpp  # Interface for using LlamaCpp models
from langchain_community.vectorstores import FAISS  # FAISS for vector similarity search
from langchain.chains import RetrievalQA  # Retrieval-based QA chain
from langchain_core.callbacks import BaseCallbackHandler  # Base class for custom callback handlers
from langchain_huggingface import HuggingFaceEmbeddings  # Embeddings using HuggingFace models
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Loaders for PDF and text files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits documents into manageable chunks
import torch  # For checking GPU availability
import logging  # For logging events and errors
from pydantic import ConfigDict  # Configuration for data models
import os  # OS operations for file handling

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom handler to log when LLM starts processing
class CustomHandler(BaseCallbackHandler):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow any types in config
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info(f"LLM query started with prompts: {prompts[:60]}...")

# Main class for Retrieval-Augmented Generation pipeline
class RAGPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir  # Directory containing uploaded documents
        self.vectorstore = None  # Placeholder for the vector database
        self.llm = None  # Placeholder for the language model
        self.qa_chain = None  # Placeholder for the retrieval QA chain

    def load_documents(self):
        """Load all PDF/TXT files from the data directory into document objects"""
        documents = []
        for filename in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, filename)
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(path)  # Load PDFs
                    documents.extend(loader.load())
                elif filename.endswith('.txt'):
                    loader = TextLoader(path)  # Load plain text files
                    documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
        return documents

    def initialize(self):
        """Initialize the embedding model, document vectorstore, LLM, and QA chain"""
        try:
            # 1. Load sentence-transformer embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Use CPU
                encode_kwargs={'normalize_embeddings': True}  # Normalize vectors
            )
            print('embeddings:')

            # 2. Load and split documents into chunks
            documents = self.load_documents()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,       # Size of each text chunk
                chunk_overlap=200      # Overlap between chunks
            )
            print('documents:', documents)
            splits = text_splitter.split_documents(documents)
            print('splits:', splits)

            # 3. Create a FAISS vector store from the chunks
            self.vectorstore = FAISS.from_documents(splits, embeddings)

            # 4. Load the local LlamaCpp model from disk
            model_path = os.path.join("D:/rag-langchain-app", "model", "zephyr-7b-beta.Q4_0.gguf")
            print(model_path)

            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.3,
                n_ctx=2048,  # Context window size
                n_gpu_layers=20 if torch.cuda.is_available() else 0,  # Use GPU layers if available
                callbacks=[CustomHandler()],  # Custom callback for logging
                verbose=False
            )

            # 5. Create a QA chain using the retriever and LLM
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),  # Convert vectorstore into retriever
                callbacks=[CustomHandler()]
            )

            logger.info("RAG pipeline initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def query(self, question: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Execute a query against the QA chain with configurable parameters"""
        try:
            self.ensure_initialized()
            max_tokens = int(max_tokens)
            temperature = float(temperature)

            # Set generation parameters dynamically if supported by LLM
            if hasattr(self.llm, 'temperature'):
                self.llm.temperature = temperature
            if hasattr(self.llm, 'max_tokens'):
                self.llm.max_tokens = max_tokens

            # Invoke the QA chain with the question
            result = self.qa_chain.invoke({"query": question})
            return result["result"]

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error: {str(e)}"

    def is_initialized(self):
        """Check whether all components (vectorstore, llm, chain) are initialized"""
        return all([
            self.vectorstore is not None,
            self.llm is not None,
            self.qa_chain is not None
        ])

    def ensure_initialized(self):
        """Reinitialize the pipeline if it's not already initialized"""
        if not self.is_initialized():
            self.initialize()
