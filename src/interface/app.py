import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration de la page
st.set_page_config(page_title="Synapse Engine", page_icon="🧠")

st.title("🧠 Synapse Engine")
st.markdown("### Votre Second Cerveau Numérique")

# Configuration de l'API (à adapter selon l'environnement, ex: Docker)
API_URL = os.getenv("API_URL", "http://localhost:8000")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "synapse_secret_dev")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input("Posez une question à vos documents..."):
    # Affichage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Appel à l'API Backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Analyse des documents en cours..."):
                # On ajoute le header d'authentification
                headers = {"X-API-Key": APP_SECRET_KEY}
                response = requests.post(f"{API_URL}/ask", json={"question": prompt}, headers=headers)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "Pas de réponse.")
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Erreur API: {response.status_code} - {response.text}"
                    message_placeholder.error(error_msg)
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Impossible de contacter le serveur Synapse Engine. Vérifiez qu'il est lancé.")

st.sidebar.markdown("---")
st.sidebar.info("Synapse Engine v1.0\nPowered by Mistral & VoyageAI")
