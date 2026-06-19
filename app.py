import streamlit as st

st.set_page_config(page_title="AI Chat System", layout="wide")

# ---------------- MEMORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- SIMPLE AI ----------------
def bot_response(text):
    text = text.lower()

    if "hello" in text or "hi" in text:
        return "Hello! I am your AI assistant."

    if "name" in text:
        return "I am a simple chatbot built using Streamlit."

    if "study" in text:
        return "Study daily in small chunks. Consistency beats pressure."

    if "python" in text:
        return "Python is used for AI, web apps, automation, and data science."

    return "I understand. Tell me more."

# ---------------- TITLE ----------------
st.title("🤖 AI Chat System")

# ---------------- SHOW CHAT ----------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- INPUT ----------------
user_input = st.chat_input("Type your message...")

if user_input:

    # user message
    st.session_state.chat.append({
        "role": "user",
        "content": user_input
    })

    # bot response
    reply = bot_response(user_input)

    st.session_state.chat.append({
        "role": "assistant",
        "content": reply
    })

    st.rerun()
