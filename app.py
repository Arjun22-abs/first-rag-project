# app.py

import streamlit as st
from main import answer_query   # 🔗 RAG backend connection

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

# ---------------------------
# TITLE
# ---------------------------
st.title("RAG Document Chatbot - javascript and Python")
st.write("Ask questions strictly based on the uploaded document.")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------
# CHAT INPUT
# ---------------------------
user_input = st.chat_input("Ask your query here...")

# ---------------------------
# HANDLE USER QUERY
# ---------------------------
if user_input:
    # Store & show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").write(user_input)

    # Call RAG backend
    with st.spinner("Searching document..."):
        try:
            answer, sources = answer_query(user_input)
        except Exception as e:
            answer = "Something went wrong while processing your query."
            sources = []
            st.error(str(e))

    # Store & show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.chat_message("assistant").write(answer)

    # Show sources (if any)
    if sources:
        with st.expander("Sources"):
            for src in sources:
                st.write(src)