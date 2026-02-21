#!/bin/bash

# Démarrer l'API en arrière-plan
uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Démarrer le Frontend Streamlit
streamlit run src/interface/app.py --server.port 8501 --server.address 0.0.0.0
