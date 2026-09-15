import streamlit as st
import os
import base64
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from auth import init_db, register_user, login_user
from database import (
    init_chunks_db, add_document, add_chunk, get_all_chunks,
    get_user_documents, create_session, get_sessions,
    update_session_title, delete_session, save_message, get_session_messages
)
from embedding import embed_texts, vector_to_blob, chunk_text, find_relevant_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

init_db()
init_chunks_db()

st.set_page_config(page_title="Assistant RAG_llms", page_icon="🤖", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None


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


def encode_image_to_base64(uploaded_image) -> str:
    return base64.b64encode(uploaded_image.getvalue()).decode("utf-8")


def ensure_active_session(username: str):
    if st.session_state.current_session_id is None:
        sessions = get_sessions(username)
        if sessions:
            st.session_state.current_session_id = sessions[0][0]
        else:
            st.session_state.current_session_id = create_session(username)


def start_new_chat(username: str):
    new_id = create_session(username)
    st.session_state.current_session_id = new_id
    st.session_state.pending_image = None
    st.rerun()


def switch_session(session_id: int):
    st.session_state.current_session_id = session_id
    st.session_state.pending_image = None
    st.rerun()


def show_chat_page():
    username = st.session_state.username
    ensure_active_session(username)

    with st.sidebar:
        st.header(f"👤 {username}")
        if st.button("Se déconnecter"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.current_session_id = None
            st.rerun()

        st.divider()

        if st.button("➕ Nouvelle conversation", use_container_width=True):
            start_new_chat(username)

        st.subheader("🕓 Historique")
        sessions = get_sessions(username)
        for session_id, title, created_at in sessions:
            col1, col2 = st.columns([5, 1])
            with col1:
                is_current = session_id == st.session_state.current_session_id
                label = f"**{title}**" if is_current else title
                if st.button(label, key=f"session_{session_id}", use_container_width=True):
                    switch_session(session_id)
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}"):
                    delete_session(session_id)
                    if st.session_state.current_session_id == session_id:
                        st.session_state.current_session_id = None
                    st.rerun()

        st.divider()
        st.subheader("📄 Ajouter un document")
        uploaded_file = st.file_uploader("Fichier (PDF/TXT)", type=["pdf", "txt"], key="doc_uploader")
        if uploaded_file is not None:
            if st.button("Traiter ce fichier"):
                process_uploaded_file(uploaded_file, username)

        st.subheader("📚 Tes documents")
        docs = get_user_documents(username)
        if docs:
            for filename, uploaded_at in docs:
                st.text(f"• {filename}")
        else:
            st.caption("Aucun document ajouté.")

    st.title("🤖 Assistant RAG")

    messages = get_session_messages(st.session_state.current_session_id)
    for role, content in messages:
        with st.chat_message(role):
            st.markdown(content)

    uploaded_image = st.file_uploader(
        "🖼️ Joindre une image (optionnel)", type=["png", "jpg", "jpeg","avif"], key="image_uploader"
    )
    if uploaded_image is not None:
        st.session_state.pending_image = uploaded_image
        st.image(uploaded_image, width=200, caption="Image prête à être envoyée")

    prompt = st.chat_input("Pose ta question...")
    if prompt:
        session_id = st.session_state.current_session_id

        current_sessions = {s[0]: s[1] for s in get_sessions(username)}
        if current_sessions.get(session_id) == "Nouvelle conversation":
            new_title = prompt[:40] + ("..." if len(prompt) > 40 else "")
            update_session_title(session_id, new_title)

        save_message(session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
            if st.session_state.pending_image:
                st.image(st.session_state.pending_image, width=200)

        with st.chat_message("assistant"):
            if st.session_state.pending_image is not None:
                image_b64 = encode_image_to_base64(st.session_state.pending_image)
                mime = st.session_state.pending_image.type

                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {
                            "role": "system",
                            "content": "Tu réponds TOUJOURS en français, quelle que soit la langue de l'image ou du contenu analysé.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                                },
                            ],
                        },
                    ],
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.pending_image = None

            else:
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

        save_message(session_id, "assistant", answer)
        st.rerun()


if st.session_state.logged_in:
    show_chat_page()
else:
    show_login_page()