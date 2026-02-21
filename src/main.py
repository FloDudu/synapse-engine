# src/main.py

import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from src.core.qa_service import QAService

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Synapse Engine - Second Cerveau",
    description="Une API pour discuter avec vos documents.",
    version="1.0.0"
)

# Modèle de données pour la requête
class QuestionRequest(BaseModel):
    question: str

# Modèle de données pour la réponse
class AnswerResponse(BaseModel):
    answer: str

# Initialisation du service QA
# On le place ici pour qu'il soit initialisé une seule fois au démarrage.
try:
    qa_service = QAService()
except RuntimeError as e:
    logger.error(f"Erreur critique au démarrage: {e}")
    # Si le service ne peut pas démarrer (ex: clés API manquantes), on empêche l'application de fonctionner.
    qa_service = None 

@app.on_event("startup")
async def startup_event():
    if qa_service is None:
        logger.warning("Le service de QA n'est pas disponible. L'API ne pourra pas répondre aux questions.")

@app.post("/ask", 
          response_model=AnswerResponse,
          summary="Posez une question à vos documents",
          tags=["Question-Réponse"])
async def ask_question(request: QuestionRequest):
    """
    Recevez une question et retournez une réponse basée sur les documents ingérés.
    """
    if qa_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service n'est pas configuré correctement. Vérifiez les clés API et la configuration."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La question ne peut pas être vide."
        )
        
    logger.info(f"Question reçue: '{request.question}'")
    
    try:
        answer = qa_service.ask_question(request.question)
        logger.info(f"Réponse générée: '{answer}'")
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors du traitement de votre question."
        )

if __name__ == "__main__":
    import uvicorn
    # Cet endpoint est utile pour un test rapide, mais en production, utilisez un serveur Gunicorn/Uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
