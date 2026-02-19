import os
from dotenv import load_dotenv

# Charge les variables depuis le fichier .env
load_dotenv()

class Settings:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "synapse_knowledge")
    
    # Modèles spécifiques
    EMBEDDING_MODEL = "voyage-large-2" # Excellent pour le retrieval
    LLM_MODEL = "mistral-large-latest" # Ou "open-mixtral-8x22b"

settings = Settings()
