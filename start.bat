@echo off
echo Lancement de Synapse Engine...

REM Lancer l'API dans une nouvelle fenêtre
start "Synapse API" cmd /k "uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"

REM Attendre 3 secondes pour laisser le temps à l'API de démarrer
timeout /t 3 /nobreak >nul

REM Lancer Streamlit dans une nouvelle fenêtre
start "Synapse Frontend" cmd /k "streamlit run src/interface/app.py --server.port 8501"