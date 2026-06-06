import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ignoring langchain-community deprecation warning
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Logs Config
logging.basicConfig(level=logging.os.getenv("LOGGING"), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project Config
CONFIG = {
    "pdf_path": input("Enter the path of the PDF file: "), # UPDATE PATH
    "db_path": "./chroma_db",
    "embedding_model": os.getenv("EMBEDDING_MODEL"), # Embedding model
    "chunk_size": 1000,
    "chunk_overlap": 200   
}

def load_document(path):
    """load document and return the pages"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading document from {path}")
    loader = PyPDFLoader(path)
    return loader.load()


def process_chuncks(documents):
    """split documents into logical pieces"""
    logger.info("Creating chunks of text")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def index_in_vector(chunks):
    """Index the chunks into the vector database"""
    logger.info(f"starting embedding if model {CONFIG['embedding_model']}")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])

    logger.info(f"saving in {CONFIG['db_path']}")  
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CONFIG["db_path"]
    )
    return vectorstore
    
def main():
    try:
        docs = load_document(CONFIG["pdf_path"])
        chunks = process_chuncks(docs)
        vectorstore = index_in_vector(chunks)

        logger.info(f"\nSuccess! Indexed {len(chunks)} chunks in the vector database\n")

    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")

if __name__ == "__main__":
    main()
