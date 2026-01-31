import streamlit as st
import os

# Ensure OpenAI API key is available before importing agents.
if not os.getenv("OPENAI_API_KEY"):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

from langchain_core.messages import HumanMessage, AIMessage
from main import build_langgraph_app


@st.cache_resource
def _get_langgraph_app():
    # Compile the LangGraph once per Streamlit server process.
    return build_langgraph_app()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": [], "route": "math_agent"}


st.set_page_config(
    page_title="Chat",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    :root {
        --bg: #343541;
        --panel: #40414f;
        --text: #ececf1;
        --muted: #9ca3af;
        --border: #565869;
        --input: #40414f;
    }

    .stApp {
        background-color: var(--bg);
    }

    .main .block-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1.5rem 1rem 6rem 1rem;
    }

    .stChatInput > div {
        background-color: var(--input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
    }

    .stChatInput textarea {
        background: transparent !important;
        color: var(--text) !important;
    }

    [data-testid="stChatMessageContent"] p {
        color: var(--text) !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #10a37f !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


if not st.session_state.messages:
    st.markdown("<h2 style='text-align:center;color:#ececf1;'>How can I help?</h2>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                app = _get_langgraph_app()
                state = st.session_state.agent_state
                state["messages"].append(HumanMessage(content=prompt))

                final_text = ""
                for s in app.stream(state, stream_mode="values"):
                    msg = s["messages"][-1]
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        final_text = msg.content
                    state = s

                st.session_state.agent_state = state

                if not final_text:
                    final_text = "I could not produce a response."

                st.markdown(final_text)
                response = final_text
            except Exception as e:
                st.error("Agent failed to run. Check that OPENAI_API_KEY is set and the agent modules import correctly.")
                with st.expander("Error details"):
                    st.exception(e)
                response = "Sorry - the agent crashed. See details above."

    st.session_state.messages.append({"role": "assistant", "content": response})
