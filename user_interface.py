import os
import streamlit as st
from dotenv import load_dotenv
from chinook_musicstore_agent import build_agent

# Page configuration
st.set_page_config(
    page_title="Music Store Assistant", page_icon="🎵", layout="centered"
)
st.title("🎵 Music Store AI Assistant")
st.markdown("Ask me anything about songs, artists, and albums!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_with_chat_history" not in st.session_state:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.agent_with_chat_history = build_agent(openai_api_key)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask about music..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent_with_chat_history.invoke(
                {"input": user_input}, config={"configurable": {"session_id": "<foo>"}}
            )
            response = result["output"]
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with clear chat button
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
