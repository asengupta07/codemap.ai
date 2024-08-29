import streamlit as st
import asyncio
from helpers import generate

st.set_page_config(page_title="CodeMap.AI", page_icon=":material/code:", layout="centered")

st.title("CodeMap.AI")

with st.sidebar:
    st.title("About Cody")
    st.write("Your own personal AI assistant, Cody, that helps you get started with coding and advance in various technical niches! With a knowledge base of over 1,000 documents curated by professionals out of expert articles and roadmaps, CodeMap.AI is here to help you with your coding journey.")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    with st.expander("How to Use"):
        st.write("1. Type your question in the chat window.")
        st.write("2. Cody will respond with expert advice.")
        st.write("3. Cody will use the conversation history to provide more accurate responses.")
        st.write("4. Cody will provide expert advice on how to start learning coding and advance in various technical niches like web development, machine learning, blockchain, cybersecurity, and more.")

    with st.expander("About the AI"):
        st.write("Cody is an RAG AI assistant trained on a knowledge base of over 100,000 documents curated by professionals out of expert articles and roadmaps. Cody is designed to provide expert advice on how to start learning coding and advance in various technical niches like web development, machine learning, blockchain, cybersecurity, and more. Cody uses the conversation history to provide more accurate responses.")

    with st.expander("About the Data"):
        st.write("The data used to train Cody consists of expert articles and roadmaps on various technical niches like web development, machine learning, blockchain, cybersecurity, and more. The data is curated by professionals to ensure that Cody provides accurate and expert-level responses.")

    with st.expander("About the Model"):
        st.write("Cody is powered by the RAG (Retrieval-Augmented Generation) architecture over Llama3.1 400B base model, which combines the strengths of retrieval-based and generation-based models to provide accurate and comprehensive responses. The model is trained on a knowledge base of over 100,000 documents curated by professionals to ensure that Cody provides expert advice on how to start learning coding and advance in various technical niches.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Cody Anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[-10:]
    formatted_history = ""
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        formatted_history += f"{role}: {content}\n"

    history_text = formatted_history

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(generate(prompt, formatted_history))
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Ask Cody anything about coding, including roadmaps and guidance! 🤖")
