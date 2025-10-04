# ================= Streamlit Frontend =================
import streamlit as st
from backend import chatbot, retrieve_all_threads, generate_conversation_name_with_llm
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ---------------- Session Management ----------------
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ---------------- Session Defaults ----------------

st.session_state.setdefault("message_history", [])
st.session_state.setdefault("thread_name", "New Chat")
st.session_state.setdefault("thread_id", generate_thread_id())
st.session_state.setdefault("chat_threads", retrieve_all_threads())
add_thread(st.session_state["thread_id"])

# ---------------- Sidebar ----------------
st.sidebar.title("ChatBot")
if st.sidebar.button("New Chat"):
    
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    state=chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    display_name = state.values.get("Name", thread_id)
    if st.sidebar.button(display_name):
        st.session_state["thread_name"] = display_name
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]

# ---------------- Chat Messages ----------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_run"
        }
    }

    # ---------------- AI Streaming ----------------
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_text = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Done!", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_text})
    

