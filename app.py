import streamlit as st
from rag_engine import load_index_and_chunks, get_answer


st.set_page_config(page_title="Chat with my thesis", layout="wide")

# Load RAG components
@st.cache_resource
def init_rag():
    return load_index_and_chunks()

chunks, index, model = init_rag()

st.title("Chat with my thesis")
st.markdown("This app uses Mistral free tier LLM with rate limitations, which might make the app unavailable at times. Thank you for your patience!\n\nVoit kokeilla myös suomeksi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# User input (automatically placed at bottom)
if prompt := st.chat_input("Ask something about the thesis..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "❓"})
    with st.chat_message("user", avatar="❓"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant", avatar="👩‍🎓"):
        with st.spinner("Thinking..."):
            response = get_answer(prompt, index, chunks, model)
            st.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "👩‍🎓"})