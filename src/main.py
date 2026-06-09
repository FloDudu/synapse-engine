import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from src.core.qa_service import QAService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    expected_key = os.getenv("APP_SECRET_KEY")
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : Clé d'API invalide ou manquante."
        )


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("APP_SECRET_KEY"):
        raise RuntimeError("APP_SECRET_KEY environment variable is not set.")

    logger.info("Starting QA service...")
    try:
        app.state.qa_service = QAService()
        logger.info("QA service started successfully.")
    except Exception as e:
        logger.error(f"Critical error during QA service startup: {e}")
        app.state.qa_service = None

    yield


app = FastAPI(
    title="Synapse Engine - Second Cerveau",
    description="Une API pour discuter avec vos documents.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/ask",
          response_model=AnswerResponse,
          summary="Posez une question à vos documents",
          tags=["Question-Réponse"],
          dependencies=[Depends(verify_api_key)])
async def ask_question(request: QuestionRequest, http_request: Request):
    qa_service = http_request.app.state.qa_service

    if qa_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service QA n'a pas pu démarrer. Vérifiez les logs et les variables d'environnement."
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
