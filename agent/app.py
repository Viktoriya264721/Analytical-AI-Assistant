"""Streamlit web application for the AI-CFO agent.

Three modes:
- Dashboard: financial KPIs and charts for a selected month.
- Report: AI-generated monthly narrative report.
- Chat: interactive dialogue with the ReAct AI assistant.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

from agent.context import fetch_available_months, build_dashboard_data, resolve_names_in_text, anonymize_names_in_text
from agent.report import generate_monthly_report
from agent.pdf_export import markdown_to_html_bytes
from agent.dashboard import render_dashboard
from agent.persistence import (
    new_session_id,
    save_message,
    load_conversation,
    list_conversations,
)
from agent.graph import build_graph, chat, generate_welcome, generate_follow_up_questions, restore_session
from agent.auth import login_user, create_user, delete_user, list_users
from agent.llm import DEFAULT_MODEL

st.set_page_config(
    page_title="AI-CFO",
    page_icon="",
    layout="wide",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Reset & Base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container {
    max-width: 920px;
    padding: 1.5rem 1.2rem 4rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #334663 0%, #283852 50%, #1e2d42 100%);
    border-right: 1px solid rgba(163,197,241,0.08);
}
section[data-testid="stSidebar"] * {
    color: #C8DAF0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #8BA4C8 !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #E0EAF5 !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(163,197,241,0.12) !important;
    margin: 0.8rem 0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* ---- Sidebar brand ---- */
.sidebar-brand {
    text-align: center;
    padding: 0.3rem 0 0.6rem;
}
.sidebar-brand-icon {
    font-size: 28px;
    margin-bottom: 2px;
}
.sidebar-brand-title {
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: 1px;
}
.sidebar-brand-sub {
    font-size: 11px;
    color: #6582AA !important;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ---- Sidebar conversation buttons ---- */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stButton button span,
section[data-testid="stSidebar"] .stButton button p {
    background: rgba(163,197,241,0.14) !important;
    border: 1px solid rgba(163,197,241,0.28) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 8px 12px !important;
    transition: all 0.15s ease !important;
}
section[data-testid="stSidebar"] .stButton button span,
section[data-testid="stSidebar"] .stButton button p {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(163,197,241,0.25) !important;
    border-color: rgba(163,197,241,0.45) !important;
}

/* ---- Main area buttons ---- */
.stButton button,
.stButton button span,
.stButton button p {
    color: #FFFFFF !important;
}
.stButton button {
    background: linear-gradient(135deg, #5A7BAA 0%, #3D5A80 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 2px 10px rgba(51,70,99,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #6D90BF 0%, #5A7BAA 100%) !important;
    box-shadow: 0 4px 14px rgba(51,70,99,0.4) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton button,
.stDownloadButton button span,
.stDownloadButton button p {
    color: #FFFFFF !important;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #5A7BAA 0%, #3D5A80 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 2px 10px rgba(101,130,170,0.25) !important;
}
.stDownloadButton button:hover {
    background: linear-gradient(135deg, #6D90BF 0%, #5A7BAA 100%) !important;
}

/* ---- Chat messages ---- */
.stChatMessage {
    border-radius: 14px !important;
    margin-bottom: 6px !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* ---- Chat input ---- */
.stChatInput {
    border-radius: 12px !important;
}
.stChatInput > div {
    border-radius: 12px !important;
    border: 1px solid #D6E0ED !important;
    box-shadow: 0 1px 4px rgba(51,70,99,0.06) !important;
}
.stChatInput textarea {
    font-size: 14px !important;
}

/* ---- Spinner ---- */
.stSpinner > div {
    border-top-color: #4B6387 !important;
}

/* ---- Section title (used in dashboard) ---- */
.section-title {
    font-size: 15px;
    font-weight: 600;
    color: #334663;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 5px;
    border-bottom: 2px solid #83A2CD;
    display: inline-block;
}

/* ---- Chat header ---- */
.chat-header {
    padding: 1.8rem 0 1rem;
}
.chat-header-title {
    font-size: 22px;
    font-weight: 700;
    color: #334663;
    margin-bottom: 2px;
}
.chat-header-sub {
    font-size: 13px;
    color: #6582AA;
    font-weight: 400;
}

/* ---- History label ---- */
.history-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6582AA !important;
    font-weight: 600;
    margin: 10px 0 6px;
}

/* ---- Mobile ---- */
@media (max-width: 768px) {
    .block-container { padding: 0.8rem 0.6rem 3rem; }
    section[data-testid="stSidebar"] { min-width: 200px !important; max-width: 240px !important; }
    .stChatInput textarea { font-size: 16px !important; } /* prevent iOS zoom */
    .chat-header-title { font-size: 18px; }
    [data-testid="stChatMessageContent"] { font-size: 14px !important; }
}
</style>
""", unsafe_allow_html=True)



def _get_secret(key: str) -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


@st.cache_resource
def get_supabase_client():
    env_path = os.getenv("ENV_PATH")
    if env_path:
        load_dotenv(env_path)

    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        st.error("Supabase credentials not found.")
        st.stop()

    return create_client(url, key)


@st.cache_resource
def get_cfo_graph(_supabase, model_name: str):
    """Build and cache the compiled LangGraph CFO agent."""
    return build_graph(_supabase, model_name)



for key, default in {
    "last_report": None,
    "last_month": None,
    "chat_messages": [],
    "chat_session_id": None,
    "welcome_generated": False,
    "session_restored": False,
    "welcome_questions": [],
    "pending_question": None,
    "follow_up_questions": [],
    "show_follow_up": False,
    "authenticated": False,
    "auth_username": None,
    "auth_role": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default



if not st.session_state.authenticated:
    supabase_client = get_supabase_client()

    st.markdown(
        '<div style="max-width:360px;margin:80px auto 0;">'
        '<div style="text-align:center;margin-bottom:24px;">'
        '<div style="font-size:30px;font-weight:700;color:#334663;">ШІ-асистент</div>'
        '<div style="font-size:16px;color:#6582AA;">Аналітика та інсайти для вашого бізнесу</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    username = st.text_input("Логін", key="login_username")
    password = st.text_input("Пароль", type="password", key="login_password")
    if st.button("Увійти", use_container_width=True):
        if username and password:
            ok, msg, role = login_user(supabase_client, username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.auth_username = username
                st.session_state.auth_role = role
                st.rerun()
            else:
                st.error(msg)
        else:
            st.warning("Введіть логін і пароль.")

    st.stop()



with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-title">ШІ-асистент</div>'
        '<div class="sidebar-brand-sub">Аналітика та інсайти</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    role_label = "Адміністратор" if st.session_state.auth_role == "admin" else ""
    label_text = f"{role_label} · {st.session_state.auth_username}" if role_label else st.session_state.auth_username
    st.caption(label_text)
    if st.button("Вийти", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_username = None
        st.session_state.auth_role = None
        st.session_state.chat_session_id = None
        st.session_state.chat_messages = []
        st.session_state.welcome_generated = False
        st.session_state.session_restored = False
        st.session_state.welcome_questions = []
        st.session_state.pending_question = None
        st.session_state.follow_up_questions = []
        st.session_state.show_follow_up = False
        st.rerun()

    st.markdown("---")

    supabase_client = get_supabase_client()

    mode_options = ["Дашборд", "Чат з асистентом"]
    if st.session_state.auth_role == "admin":
        mode_options.append("Користувачі")
    mode = st.radio("Режим", options=mode_options, index=0)

    available_months = fetch_available_months(supabase_client)
    latest_month = available_months[0] if available_months else ""

    if mode == "Дашборд":
        st.markdown("---")
        if not available_months:
            st.warning("Немає даних у monthly_metrics.")
            st.stop()
        target_month = st.selectbox("Місяць", options=available_months, index=0)

    elif mode == "Чат з асистентом":
        st.markdown("---")

        if st.button("✦  Нова розмова", use_container_width=True):
            st.session_state.chat_session_id = new_session_id()
            st.session_state.chat_messages = []
            st.session_state.welcome_generated = False
            st.session_state.session_restored = False
            st.session_state.welcome_questions = []
            st.session_state.pending_question = None
            st.session_state.follow_up_questions = []
            st.session_state.show_follow_up = False
            st.rerun()

        try:
            conversations = list_conversations(supabase_client, st.session_state.auth_username)
        except Exception as e:
            st.caption(f"⚠️ Не вдалося завантажити історію: {e}")
            conversations = []

        if conversations:
            st.markdown('<div class="history-label">Історія</div>', unsafe_allow_html=True)
            for conv in conversations[:8]:
                if st.button(
                    conv["preview"],
                    key=f"c_{conv['session_id']}",
                    use_container_width=True,
                ):
                    st.session_state.chat_session_id = conv["session_id"]
                    st.session_state.chat_messages = load_conversation(
                        supabase_client, conv["session_id"]
                    )
                    st.session_state.welcome_generated = True
                    st.session_state.session_restored = False
                    st.rerun()



if mode == "Дашборд":
    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 2])

    with btn_col1:
        if st.button("Згенерувати AI-звіт", type="primary", use_container_width=True):
            with st.spinner("Генерую звіт..."):
                try:
                    report_md = generate_monthly_report(
                        supabase=supabase_client,
                        target_month=target_month,
                        model_name=DEFAULT_MODEL,
                    )
                    st.session_state.last_report = report_md
                    st.session_state.last_month = target_month
                    st.rerun()
                except Exception as e:
                    st.error(f"Помилка генерації звіту: {e}")

    with btn_col2:
        if st.session_state.last_report and st.session_state.last_month == target_month:
            try:
                html_bytes = markdown_to_html_bytes(st.session_state.last_report)
                st.download_button(
                    label="Завантажити звіт",
                    data=html_bytes,
                    file_name=f"report_{target_month}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Помилка: {e}")

    with st.spinner("Завантаження даних..."):
        dashboard_data = build_dashboard_data(supabase_client, target_month)
    render_dashboard(dashboard_data)

    if st.session_state.last_report and st.session_state.last_month == target_month:
        st.markdown("---")
        st.markdown('<div class="section-title">AI-звіт</div>', unsafe_allow_html=True)
        st.markdown(st.session_state.last_report)



elif mode == "Чат з асистентом":
    st.markdown(
        '<div class="chat-header">'
        '<div class="chat-header-title">AI-CFO</div>'
        '<div class="chat-header-sub">Ваш фінансовий директор. Запитуйте будь-що про бізнес.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cfo_graph = get_cfo_graph(supabase_client, DEFAULT_MODEL)

    if not st.session_state.chat_session_id:
        st.session_state.chat_session_id = new_session_id()

    if st.session_state.chat_messages and not st.session_state.session_restored:
        restore_session(
            cfo_graph,
            st.session_state.chat_session_id,
            st.session_state.chat_messages,
            available_months,
            latest_month,
        )
        st.session_state.session_restored = True

    if not st.session_state.chat_messages and not st.session_state.welcome_generated:
        with st.spinner("CFO аналізує дані..."):
            try:
                welcome, questions = generate_welcome(
                    graph=cfo_graph,
                    available_months=available_months,
                    latest_month=latest_month,
                )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": welcome}
                )
                st.session_state.welcome_questions = questions
                save_message(
                    supabase_client,
                    st.session_state.chat_session_id,
                    "assistant",
                    welcome,
                    st.session_state.auth_username,
                )
                st.session_state.welcome_generated = True
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")
                st.session_state.welcome_generated = True

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            content = (
                resolve_names_in_text(msg["content"], supabase_client)
                if msg["role"] == "assistant"
                else msg["content"]
            )
            st.markdown(content)

    user_messages = [m for m in st.session_state.chat_messages if m["role"] == "user"]
    if st.session_state.welcome_questions and not user_messages:
        cols = st.columns(len(st.session_state.welcome_questions))
        for i, (col, question) in enumerate(zip(cols, st.session_state.welcome_questions)):
            with col:
                if st.button(question, key=f"wq_{i}", use_container_width=True):
                    st.session_state.pending_question = question
                    st.session_state.welcome_questions = []
                    st.rerun()

    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        user_input = None

    typed_input = st.chat_input("Запитайте щось...")
    if typed_input:
        user_input = typed_input

    if user_input:
        st.session_state.follow_up_questions = []
        st.session_state.show_follow_up = False
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        save_message(
            supabase_client,
            st.session_state.chat_session_id,
            "user",
            user_input,
            st.session_state.auth_username,
        )

        with st.chat_message("assistant"):
            with st.spinner("Аналізую..."):
                try:
                    user_input_anon = anonymize_names_in_text(user_input, supabase_client)
                    answer_raw = chat(
                        graph=cfo_graph,
                        session_id=st.session_state.chat_session_id,
                        user_input=user_input_anon,
                        available_months=available_months,
                        latest_month=latest_month,
                    )
                    st.markdown(resolve_names_in_text(answer_raw, supabase_client))
                except Exception as e:
                    answer_raw = f"Помилка: {e}"
                    st.error(answer_raw)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer_raw})
        save_message(
            supabase_client,
            st.session_state.chat_session_id,
            "assistant",
            answer_raw,
            st.session_state.auth_username,
        )
        st.rerun()

    has_exchange = any(m["role"] == "user" for m in st.session_state.chat_messages)
    if has_exchange:
        if st.session_state.show_follow_up and st.session_state.follow_up_questions:
            cols = st.columns(len(st.session_state.follow_up_questions))
            for i, (col, question) in enumerate(
                zip(cols, st.session_state.follow_up_questions)
            ):
                with col:
                    if st.button(question, key=f"fq_{i}", use_container_width=True):
                        st.session_state.pending_question = question
                        st.session_state.follow_up_questions = []
                        st.session_state.show_follow_up = False
                        st.rerun()
        elif not st.session_state.show_follow_up:
            if st.button("Ще теми", key="show_follow_up_btn"):
                with st.spinner("Підбираю теми..."):
                    try:
                        st.session_state.follow_up_questions = generate_follow_up_questions(
                            graph=cfo_graph,
                            chat_messages=st.session_state.chat_messages,
                            available_months=available_months,
                            latest_month=latest_month,
                        )
                        st.session_state.show_follow_up = True
                    except Exception:
                        pass
                st.rerun()



elif mode == "Користувачі" and st.session_state.auth_role == "admin":
    st.markdown(
        '<div class="chat-header">'
        '<div class="chat-header-title">Управління користувачами</div>'
        '<div class="chat-header-sub">Додавайте та видаляйте акаунти.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Новий користувач</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_u = st.text_input("Логін", key="admin_new_username")
    with col2:
        new_p = st.text_input("Пароль", type="password", key="admin_new_password")
    with col3:
        new_role = st.selectbox("Роль", ["user", "admin"], key="admin_new_role")

    if st.button("Створити", use_container_width=True):
        if new_u and new_p:
            ok, msg = create_user(supabase_client, new_u, new_p, new_role)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        else:
            st.warning("Введіть логін і пароль.")

    st.markdown('<div class="section-title">Список користувачів</div>', unsafe_allow_html=True)
    users = list_users(supabase_client)

    if not users:
        st.info("Немає користувачів.")
    else:
        for u in users:
            col_name, col_role, col_del = st.columns([3, 1, 1])
            with col_name:
                st.write(u["username"])
            with col_role:
                st.caption(u.get("role", "user"))
            with col_del:
                if u["username"] != st.session_state.auth_username:
                    if st.button("Видалити", key=f"del_{u['username']}"):
                        delete_user(supabase_client, u["username"])
                        st.rerun()
                else:
                    st.caption("(ви)")
