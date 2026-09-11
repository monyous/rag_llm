import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from auth import init_db, register_user, login_user
from database import (
    init_chunks_db, add_document, add_chunk, get_all_chunks,
    get_user_documents, save_message, get_chat_history
)
from embedding import embed_text, embed_texts, vector_to_blob, chunk_text, find_relevant_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

init_db()
init_chunks_db()

st.set_page_config(page_title="Assistant RAG", page_icon="🤖", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None


def show_login_page():
    st.title("🔐 Connexion / Inscription")
    tab_login, tab_register = st.tabs(["Se connecter", "Créer un compte"])

    with tab_login:
        username = st.text_input("Nom d'utilisateur", key="login_user")
        password = st.text_input("Mot de passe", type="password", key="login_pass")
        if st.button("Se connecter"):
            success, message = login_user(username, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error(message)

    with tab_register:
        new_username = st.text_input("Choisir un nom d'utilisateur", key="reg_user")
        new_password = st.text_input("Choisir un mot de passe", type="password", key="reg_pass")
        if st.button("Créer le compte"):
            if len(new_username) < 3:
                st.error("Le nom d'utilisateur doit faire au moins 3 caractères.")
            elif len(new_password) < 4:
                st.error("Le mot de passe doit faire au moins 4 caractères.")
            else:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message + " Tu peux maintenant te connecter.")
                else:
                    st.error(message)


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def process_uploaded_file(uploaded_file, username: str):
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    if not text.strip():
        st.error("Impossible d'extraire du texte de ce fichier.")
        return

    doc_id = add_document(username, uploaded_file.name)
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    with st.spinner(f"Traitement de {len(chunks)} morceaux de texte..."):
        embeddings = embed_texts(chunks)
        for chunk_content, embedding in zip(chunks, embeddings):
            blob = vector_to_blob(embedding)
            add_chunk(doc_id, chunk_content, blob)

    st.success(f"'{uploaded_file.name}' traité : {len(chunks)} chunks ajoutés.")


def show_chat_page():
    username = st.session_state.username

    with st.sidebar:
        st.header(f"👤 {username}")
        if st.button("Se déconnecter"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("📄 Ajouter un document")
        uploaded_file = st.file_uploader("Choisir un fichier", type=["pdf", "txt"])
        if uploaded_file is not None:
            if st.button("Traiter ce fichier"):
                process_uploaded_file(uploaded_file, username)

        st.divider()
        st.subheader("📚 Tes documents")
        docs = get_user_documents(username)
        if docs:
            for filename, uploaded_at in docs:
                st.text(f"• {filename}")
        else:
            st.caption("Aucun document ajouté pour le moment.")

    st.title("🤖 Assistant RAG")

    if "messages" not in st.session_state:
        history = get_chat_history(username)
        st.session_state.messages = [{"role": r, "content": c} for r, c in history]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pose ta question sur tes documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(username, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            all_chunks = get_all_chunks(username)

            if all_chunks:
                relevant = find_relevant_chunks(prompt, all_chunks, top_k=3)
                context = "\n\n---\n\n".join(relevant)
                system_prompt = (
                    "Tu es un assistant qui répond aux questions en te basant "
                    "uniquement sur le contexte fourni ci-dessous. Si la réponse "
                    "n'est pas dans le contexte, dis-le clairement.\n\n"
                    f"Contexte:\n{context}"
                )
            else:
                system_prompt = (
                    "Tu es un assistant. Aucun document n'a encore été ajouté, "
                    "réponds avec tes connaissances générales et précise-le."
                )

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_message(username, "assistant", answer)


if st.session_state.logged_in:
    show_chat_page()
else:
    show_login_page()