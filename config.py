import os
from dotenv import load_dotenv

load_dotenv()

# Paths
PDF_PATH = r'C:\Users\Acer\OneDrive\Desktop\Intern\A2A_Hospital\Phần 11.pdf'
VNCORENLP_SAVE_DIR = r"C:\Users\Acer\OneDrive\Desktop\Intern\A2A_Hospital\models\vncorenlp"  
OUTPUT_LOG = r"C:\Users\Acer\OneDrive\Desktop\Intern\A2A_Hospital\logs.txt"
OUTPUT_JSON_CLEAN = r"C:\Users\Acer\OneDrive\Desktop\Intern\A2A_Hospital\pages_clean_mode.json"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "namdp-ptit/ViRanker"
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Qdrant
QDRANT_URL = "https://72249cbf-3fe1-4881-b625-d4f1cb122aea.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "test_4_collection"
VECTOR_SIZE = 1024  # Kích thước dense vector từ BGE-M3

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")