from langchain_mistralai import ChatMistralAI
from langchain_voyageai import VoyageAIEmbeddings
from src.config import settings

class AIFactory:
    """
    Factory pattern pour instancier les composants IA.
    Permet de centraliser la configuration et facilite les tests (mocking).
    """
    
    @staticmethod
    def get_embedding_model() -> VoyageAIEmbeddings:
        """Retourne le modèle d'embedding VoyageAI configuré."""
        if not settings.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY n'est pas définie.")
            
        return VoyageAIEmbeddings(
            voyage_api_key=settings.VOYAGE_API_KEY,
            model=settings.EMBEDDING_MODEL,
            batch_size=24 # Optimisation pour l'ingestion
        )

    @staticmethod
    def get_llm(temperature: float = 0) -> ChatMistralAI:
        """Retourne le LLM Mistral configuré."""
        if not settings.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY n'est pas définie.")

        return ChatMistralAI(
            api_key=settings.MISTRAL_API_KEY,
            model=settings.LLM_MODEL,
            temperature=temperature,
            max_retries=2
        )

# Test rapide (à supprimer plus tard ou déplacer dans un fichier de test)
if __name__ == "__main__":
    try:
        embeddings = AIFactory.get_embedding_model()
        llm = AIFactory.get_llm()
        print(f"✅ Stack initialisée : {settings.EMBEDDING_MODEL} & {settings.LLM_MODEL}")
    except Exception as e:
        print(f"❌ Erreur de configuration : {e}")
