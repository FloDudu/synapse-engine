# src/core/qa_service.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.infrastructure.ai_factory import AIFactory
from src.infrastructure.vector_store_manager import VectorStoreManager

# Définition du prompt par défaut
DEFAULT_RAG_PROMPT = """
Vous êtes un assistant IA expert, conçu pour répondre aux questions en vous basant sur un contexte fourni.
Restez strictement factuel et utilisez uniquement les informations du contexte ci-dessous pour formuler votre réponse.
Si l'information n'est pas dans le contexte, dites "Je ne trouve pas l'information dans les documents fournis."
Citez vos sources en utilisant le nom du fichier d'origine si disponible dans les métadonnées.

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE FACTUELLE:
"""

class QAService:
    """
    Service pour gérer la logique de questions-réponses (RAG).
    """

    def __init__(self):
        """
        Initialise le service en chargeant les composants nécessaires.
        """
        try:
            self.llm = AIFactory.get_llm()
            embedding_model = AIFactory.get_embedding_model()
            self.vector_store_manager = VectorStoreManager(embedding_model)
            self.retriever = self.vector_store_manager.get_retriever()
        except ValueError as e:
            # Cette erreur est levée par l'AIFactory si les clés API manquent
            raise RuntimeError(f"Erreur d'initialisation du service: {e}")
        except Exception as e:
            raise RuntimeError(f"Une erreur inattendue est survenue: {e}")

    def get_rag_chain(self):
        """
        Crée et retourne la chaîne LangChain pour le RAG.
        """
        # Template pour le prompt
        prompt = PromptTemplate.from_template(DEFAULT_RAG_PROMPT)

        # Fonction pour formater les documents récupérés
        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Inconnue')}\nContenu: {doc.page_content}"
                for doc in docs
            )

        # Création de la chaîne RAG
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def ask_question(self, question: str) -> str:
        """
        Pose une question et retourne la réponse générée par la chaîne RAG.
        """
        if not hasattr(self, 'rag_chain'):
            self.rag_chain = self.get_rag_chain()
        
        return self.rag_chain.invoke(question)
