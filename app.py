import streamlit as st
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from auth import init_db, register_user, login_user
from database import (
    init_chunks_db, add_document, add_chunk, get_all_chunks,
    get_user_documents, create_session, get_sessions,
    update_session_title, delete_session, save_message, get_session_messages,
    create_project, get_projects, delete_project
)
from embedding import embed_texts, vector_to_blob, chunk_text, find_relevant_chunks

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

init_db()
init_chunks_db()

st.set_page_config(page_title="Assistant RAG", page_icon="🤖", layout="wide")

MODEL_OPTIONS = {
    "🧠 Texte — gpt-oss-120b (rapide, pour RAG et discussion)": "openai/gpt-oss-120b",
    "👁️ Vision — qwen3.8-27b (texte + images)": "qwen/qwen3.8-27b",
}

DEFAULT_IMAGE_PROMPT = "Décris cette image en détail."

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


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


def process_uploaded_file(uploaded_file, username: str, project_id: int = None):
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    if not text.strip():
        st.error("Impossible d'extraire du texte de ce fichier.")
        return

    doc_id = add_document(username, uploaded_file.name, project_id=project_id)
    chunks = chunk_text(text, chunk_size=200, overlap=30)

    with st.spinner(f"Traitement de {len(chunks)} morceaux de texte..."):
        embeddings = embed_texts(chunks)
        for chunk_content, embedding in zip(chunks, embeddings):
            blob = vector_to_blob(embedding)
            add_chunk(doc_id, chunk_content, blob)

    st.success(f"'{uploaded_file.name}' traité : {len(chunks)} chunks ajoutés.")


def encode_image_to_base64(uploaded_image) -> str:
    image = Image.open(uploaded_image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def transcribe_audio(audio_value) -> str:
    transcription = client.audio.transcriptions.create(
        file=("audio.wav", audio_value.getvalue()),
        model="whisper-large-v3-turbo",
        language="fr",
        response_format="json",
    )
    return transcription.text


def ensure_active_session(username: str):
    if st.session_state.current_session_id is None:
        sessions = get_sessions(username)
        if sessions:
            st.session_state.current_session_id = sessions[0][0]
        else:
            st.session_state.current_session_id = create_session(username)


def start_new_chat(username: str, project_id: int = None):
    new_id = create_session(username, project_id=project_id)
    st.session_state.current_session_id = new_id
    st.session_state.pending_image = None
    st.session_state.uploader_key += 1
    st.rerun()


def switch_session(session_id: int):
    st.session_state.current_session_id = session_id
    st.session_state.pending_image = None
    st.session_state.uploader_key += 1
    st.rerun()


def render_session_row(session_id, title, is_current):
    col1, col2 = st.columns([5, 1])
    with col1:
        label = f"**{title}**" if is_current else title
        if st.button(label, key=f"session_{session_id}", use_container_width=True):
            switch_session(session_id)
    with col2:
        if st.button("🗑️", key=f"delete_{session_id}"):
            delete_session(session_id)
            if st.session_state.current_session_id == session_id:
                st.session_state.current_session_id = None
            st.rerun()


def get_current_project(all_sessions, all_projects, current_session_id):
    for session_id, title, created_at, project_id in all_sessions:
        if session_id == current_session_id:
            if project_id is None:
                return None, None
            for pid, pname in all_projects:
                if pid == project_id:
                    return pid, pname
            return project_id, "Projet inconnu"
    return None, None


def answer_with_text_model(prompt, username, project_id, model_id):
    all_chunks = get_all_chunks(username, project_id=project_id)
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
            "Tu es un assistant. Aucun document n'a encore été ajouté dans ce contexte, "
            "réponds avec tes connaissances générales et précise-le."
        )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def answer_with_vision_model(prompt, image_file):
    image_b64 = encode_image_to_base64(image_file)
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
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content


def show_chat_page():
    username = st.session_state.username
    ensure_active_session(username)

    all_sessions = get_sessions(username)
    all_projects = get_projects(username)
    current_project_id, current_project_name = get_current_project(
        all_sessions, all_projects, st.session_state.current_session_id
    )
    session_id = st.session_state.current_session_id

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

        st.divider()
        st.subheader("📁 Projets")
        with st.expander("➕ Créer un nouveau projet"):
            new_project_name = st.text_input("Nom du projet", key="new_project_name")
            if st.button("Créer", key="create_project_btn"):
                if new_project_name.strip():
                    create_project(username, new_project_name.strip())
                    st.rerun()

        for project_id, project_name in all_projects:
            project_sessions = [s for s in all_sessions if s[3] == project_id]
            is_active_project = project_id == current_project_id
            expander_label = f"📁 {project_name} ({len(project_sessions)})"
            if is_active_project:
                expander_label = f"➡️ {expander_label} — actif"
            with st.expander(expander_label, expanded=is_active_project):
                if st.button("➕ Nouvelle conversation ici", key=f"new_in_project_{project_id}"):
                    start_new_chat(username, project_id=project_id)
                for s_id, title, created_at, s_project_id in project_sessions:
                    render_session_row(s_id, title, s_id == st.session_state.current_session_id)
                if st.button("🗑️ Supprimer ce projet", key=f"delete_project_{project_id}"):
                    delete_project(project_id)
                    st.rerun()

        st.divider()
        st.subheader("🕓 Conversations générales")
        no_project_sessions = [s for s in all_sessions if s[3] is None]
        if no_project_sessions:
            for s_id, title, created_at, s_project_id in no_project_sessions:
                render_session_row(s_id, title, s_id == st.session_state.current_session_id)
        else:
            st.caption("Aucune conversation hors projet.")

        st.divider()
        doc_scope_label = f"projet « {current_project_name} »" if current_project_id else "conversation générale"
        st.subheader(f"📄 Documents ({doc_scope_label})")
        uploaded_doc = st.file_uploader("Fichier (PDF/TXT)", type=["pdf", "txt"], key="doc_uploader")
        if uploaded_doc is not None:
            if st.button("Traiter ce fichier"):
                process_uploaded_file(uploaded_doc, username, project_id=current_project_id)
        docs = get_user_documents(username, project_id=current_project_id)
        if docs:
            for filename, uploaded_at in docs:
                st.text(f"• {filename}")
        else:
            st.caption("Aucun document dans ce contexte.")

    if current_project_id:
        st.info(f"📁 Tu travailles actuellement dans le projet **{current_project_name}**")
    else:
        st.caption("💬 Conversation générale (hors projet)")

    col_title, col_model = st.columns([3, 2])
    with col_title:
        st.title("🤖 Assistant RAG")
    with col_model:
        selected_label = st.selectbox("Modèle utilisé", list(MODEL_OPTIONS.keys()), key="model_choice")
    model_id = MODEL_OPTIONS[selected_label]

    messages = get_session_messages(session_id)
    for role, content in messages:
        with st.chat_message(role):
            st.markdown(content)

    def handle_send(prompt: str, image_file=None):
        display_text = prompt if prompt.strip() else "🖼️ (image envoyée sans texte)"

        current_titles = {s[0]: s[1] for s in get_sessions(username)}
        if current_titles.get(session_id) == "Nouvelle conversation":
            title_source = prompt.strip() if prompt.strip() else "Image envoyée"
            new_title = title_source[:40] + ("..." if len(title_source) > 40 else "")
            update_session_title(session_id, new_title)

        save_message(session_id, "user", display_text)
        with st.chat_message("user"):
            st.markdown(display_text)
            if image_file:
                st.image(image_file, width=200)

        with st.chat_message("assistant"):
            if image_file is not None:
                final_prompt = prompt.strip() if prompt.strip() else DEFAULT_IMAGE_PROMPT
                answer = answer_with_vision_model(final_prompt, image_file)
            else:
                answer = answer_with_text_model(prompt, username, current_project_id, model_id)
            st.markdown(answer)

        save_message(session_id, "assistant", answer)
        st.session_state.pending_image = None
        st.session_state.uploader_key += 1
        st.rerun()

    col_attach, col_mic, col_preview = st.columns([1, 1, 6])

    with col_attach:
        with st.popover("📎"):
            st.caption("Joindre une image")
            new_image = st.file_uploader(
                "Image", type=["png", "jpg", "jpeg"],
                key=f"image_attach_{st.session_state.uploader_key}",
                label_visibility="collapsed"
            )
            if new_image is not None:
                st.session_state.pending_image = new_image

    with col_mic:
        with st.popover("🎤"):
            st.caption("Message vocal" + (" à propos de l'image jointe" if st.session_state.pending_image else ""))
            audio_value = st.audio_input("Micro", key="audio_input", label_visibility="collapsed")
            if audio_value is not None:
                if st.button("Transcrire et envoyer"):
                    with st.spinner("Transcription en cours..."):
                        text = transcribe_audio(audio_value)
                    handle_send(text, image_file=st.session_state.pending_image)

    with col_preview:
        if st.session_state.pending_image is not None:
            p1, p2, p3 = st.columns([1, 1, 4])
            with p1:
                st.image(st.session_state.pending_image, width=60)
            with p2:
                if st.button("✖ Retirer"):
                    st.session_state.pending_image = None
                    st.session_state.uploader_key += 1
                    st.rerun()
            with p3:
                if st.button("📤 Envoyer l'image seule (sans texte)"):
                    handle_send("", image_file=st.session_state.pending_image)

    typed_prompt = st.chat_input("Pose ta question...")
    if typed_prompt:
        handle_send(typed_prompt, image_file=st.session_state.pending_image)


if st.session_state.logged_in:
    show_chat_page()
else:
    show_login_page()