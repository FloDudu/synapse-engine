# Utiliser une image Python légère
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ src/
# Copier le dossier de base de données s'il existe (optionnel, mieux vaut utiliser un volume)
# COPY chroma_db/ chroma_db/ 

# Variables d'environnement par défaut
ENV PYTHONPATH=/app

# Exposer les ports (8000 pour FastAPI, 8501 pour Streamlit)
EXPOSE 8000
EXPOSE 8501

# Script de démarrage pour lancer à la fois l'API et le Frontend
# Note: Dans une vraie prod, on utiliserait docker-compose pour séparer les services.
# Ici, on utilise un script shell simple pour la démo.
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
