import streamlit as st

# Page config
st.set_page_config(
    page_title="FakeGPT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - Sidebar flex layout으로 프로필 하단 고정
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #1b1b1b;
        --bg-2: #202020;
        --panel: #242424;
        --panel-2: #2a2a2a;
        --text: #e7e7e7;
        --muted: #a9a9a9;
        --accent: #12b981;
        --border: rgba(255,255,255,0.08);
    }

    html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(1200px 800px at 20% -10%, #2a2a2a 0%, #1b1b1b 45%, #191919 100%);
        color: var(--text);
    }
    
    /* Main content */
    .main .block-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem 1rem 6rem 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* 사이드바 버튼 */
    section[data-testid="stSidebar"] .stButton > button {
        background: none !important;
        border: none !important;
        color: var(--muted) !important;
        padding: 0.65rem 0.75rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #232323 !important;
        color: var(--text) !important;
    }
    
    /* 섹션 라벨 */
    .section-label {
        font-size: 0.6rem;
        font-weight: 700;
        color: #6f6f6f;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.4rem 0.5rem 0.25rem;
        text-align: center;
    }
    
    /* Divider */
    section[data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
        border-top: 1px solid var(--border) !important;
    }
    
    /* 스크롤 영역 (Recent) */
    .scroll-container {
        flex: 1 !important;
        overflow-y: auto !important;
        min-height: 0 !important;
    }
    
    /* Spacer */
    .sidebar-spacer {
        flex: 1 !important;
    }
    
    /* Chat input */
    .stChatInput > div {
        background-color: #2a2a2a !important;
        border: 1px solid var(--border) !important;
        border-radius: 28px !important;
        box-shadow: 0 12px 25px rgba(0,0,0,0.35);
    }

    .stChatInput textarea {
        background: transparent !important;
        color: var(--text) !important;
    }

    .stChatInput ::placeholder {
        color: #8a8a8a !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessageContent"] p {
        color: var(--text) !important;
    }
    
    /* Header */
    .header-text {
        text-align: center;
        padding: 1rem 0 0.5rem;
        font-size: 0.95rem;
        color: var(--muted);
        font-weight: 500;
    }

    .header-text span {
        color: var(--text);
    }

    .topbar {
        display: none;
    }
    
    /* Welcome */
    .welcome-container {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 62vh;
    }

    .welcome-text {
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    
    /* Dialog Modal */
    div[data-testid="stDialog"] > div {
        border-radius: 16px !important;
        max-width: 480px !important;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25) !important;
    }
    
    /* Profile header */
    .profile-header {
        background: linear-gradient(135deg, #2f9f79 0%, #1d6e53 100%);
        padding: 2rem 1.5rem;
        text-align: center;
        border-radius: 16px 16px 0 0;
        margin: -1rem -1rem 0 -1rem;
    }
    
    .profile-avatar {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        border: 3px solid white;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 2rem;
        margin: 0 auto 0.75rem;
    }
    
    .profile-name { font-size: 1.5rem; font-weight: 600; color: white; }
    .profile-email { font-size: 0.9rem; color: rgba(255,255,255,0.85); margin: 0.25rem 0 0.75rem; }
    .profile-badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 1rem;
        font-size: 0.8rem;
    }
    
    .profile-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1.25rem;
        background: #f3f3f3;
        margin: 0 -1rem;
    }
    
    .stat-item { text-align: center; }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #333; }
    .stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
    
    /* Modal headers */
    .modal-header {
        padding: 1.5rem;
        text-align: center;
        border-radius: 16px 16px 0 0;
        margin: -1rem -1rem 1rem -1rem;
        color: white;
    }
    .modal-header.settings { background: linear-gradient(135deg, #0f8e6e, #2ed67b); }
    .modal-header.theme { background: linear-gradient(135deg, #ff7b0f, #ff3c6a); }
    .modal-header.usage { background: linear-gradient(135deg, #1f8ef1, #18c0c8); }
    .modal-header.help { background: linear-gradient(135deg, #f5a623, #fce38a); }
    
    .modal-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    
    /* Hide defaults */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        "Testchat",
        "Testchat",
        "Testchat",
        "Testchat",
        "Testchat",
        "Testchat"
    ]

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

for key in ["show_settings", "show_theme", "show_usage", "show_help", "show_profile"]:
    if key not in st.session_state:
        st.session_state[key] = False

# ============ MODALS ============

@st.dialog("My Profile", width="large")
def show_profile_modal():
    st.markdown("""
    <div class="profile-header">
        <div class="profile-avatar">LC</div>
        <div class="profile-name">Lee Chloe</div>
        <div class="profile-email">lee.chloe@example.com</div>
        <div class="profile-badge">✨ Plus Member</div>
    </div>
    <div class="profile-stats">
        <div class="stat-item"><div class="stat-value">128</div><div class="stat-label">Chats</div></div>
        <div class="stat-item"><div class="stat-value">1.2K</div><div class="stat-label">Messages</div></div>
        <div class="stat-item"><div class="stat-value">45</div><div class="stat-label">Days</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚙️ Settings", use_container_width=True, key="m_set"):
            st.session_state.show_settings = True
            st.rerun()
    with c2:
        if st.button("🎨 Theme", use_container_width=True, key="m_thm"):
            st.session_state.show_theme = True
            st.rerun()
    
    c3, c4 = st.columns(2)
    with c3:
        if st.button("📊 Usage", use_container_width=True, key="m_usg"):
            st.session_state.show_usage = True
            st.rerun()
    with c4:
        if st.button("❓ Help", use_container_width=True, key="m_hlp"):
            st.session_state.show_help = True
            st.rerun()
    
    if st.button("🚪 Log out", use_container_width=True, key="m_out"):
        st.session_state.messages = []
        st.toast("Logged out!", icon="👋")
        st.rerun()

@st.dialog("Settings", width="large")
def show_settings_modal():
    st.markdown('<div class="modal-header settings"><div class="modal-icon">⚙️</div><h3 style="margin:0">Settings</h3></div>', unsafe_allow_html=True)
    st.toggle("Enable notifications", value=True, key="s1")
    st.toggle("Auto-save conversations", value=True, key="s2")
    st.toggle("Save chat history", value=True, key="s3")
    if st.button("Save", type="primary", use_container_width=True):
        st.toast("Saved!", icon="✅")

@st.dialog("Theme", width="large")
def show_theme_modal():
    st.markdown('<div class="modal-header theme"><div class="modal-icon">🎨</div><h3 style="margin:0">Theme</h3></div>', unsafe_allow_html=True)
    st.radio("Color scheme", ["Light", "Dark", "System"], horizontal=True, key="t1")
    st.color_picker("Accent", "#10a37f", key="t2")
    if st.button("Apply", type="primary", use_container_width=True):
        st.toast("Applied!", icon="🎨")

@st.dialog("Usage", width="large")
def show_usage_modal():
    st.markdown('<div class="modal-header usage"><div class="modal-icon">📊</div><h3 style="margin:0">Usage</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Messages", "1,247", "+12%")
    c2.metric("Tokens", "45.2K", "+8%")
    c3.metric("Avg", "1.2s", "-0.3s")
    st.progress(0.65, "65% used")

@st.dialog("Help", width="large")
def show_help_modal():
    st.markdown('<div class="modal-header help"><div class="modal-icon">❓</div><h3 style="margin:0">Help</h3></div>', unsafe_allow_html=True)
    with st.expander("How to start a new chat?"):
        st.write("Click '✏️ New chat' in sidebar.")
    with st.expander("How to delete chat?"):
        st.write("Click 🗑️ next to the chat.")
    if st.button("📧 Contact", type="primary", use_container_width=True):
        st.toast("Opening email...", icon="📧")

# ============ MODAL TRIGGERS ============
if st.session_state.show_profile:
    st.session_state.show_profile = False
    show_profile_modal()
if st.session_state.show_settings:
    st.session_state.show_settings = False
    show_settings_modal()
if st.session_state.show_theme:
    st.session_state.show_theme = False
    show_theme_modal()
if st.session_state.show_usage:
    st.session_state.show_usage = False
    show_usage_modal()
if st.session_state.show_help:
    st.session_state.show_help = False
    show_help_modal()

# ============ MAIN CONTENT ============
st.markdown('<div class="header-text">New conversation</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown('<div class="welcome-container"><h1 class="welcome-text">Where should we begin?</h1></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = f"Hello! Responding to: '{prompt}'\n\nThis is a demo. Connect OpenAI API for real responses."
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    if len(st.session_state.messages) == 2:
        title = prompt[:25] + "..." if len(prompt) > 25 else prompt
        st.session_state.chat_history.insert(0, title)
