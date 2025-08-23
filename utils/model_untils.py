import os
from dotenv import load_dotenv
import google.generativeai as genai


def load_gemini_model():
    """
    Load the Gemini model for text generation.
    """
    load_dotenv()  # Load environment variables from .env file
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel(os.getenv("GEMINI_MODEL"))

def get_gemini_embedding(targets: list[str]):
    """
    Get the embedding vector for a given text using Gemini embeddings.

    Args:
        target (str): The text to be embedded.

    Returns:
        np.array: The embedding vector.
    """
    result = genai.embed_content(
        model="models/embedding-001",  # or "models/text-embedding-004" for the newer version
        content=targets,
    )
    embeddings = result["embedding"]
    return embeddings