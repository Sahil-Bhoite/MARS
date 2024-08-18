import streamlit as st
import os
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GooglePalm
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, use_ollama):
    embeddings = OllamaEmbeddings(model="llama2") if use_ollama else GooglePalmEmbeddings()
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_conversation_chain(vector_store, use_ollama):
    llm = ChatOllama(model="llama2") if use_ollama else GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

def main():
    st.set_page_config(page_title="Chat with Word Documents", layout="wide")
    st.header("Chat with Word Documents")

    use_ollama = st.sidebar.checkbox("Use Ollama (offline) instead of Google Palm")
    docx_file = st.file_uploader("Upload your Word document", type="docx")

    if docx_file:
        text = extract_text_from_docx(docx_file)
        text_chunks = get_text_chunks(text)
        vector_store = get_vector_store(text_chunks, use_ollama)
        conversation = get_conversation_chain(vector_store, use_ollama)

        user_question = st.text_input("Ask a question about your Word document:")
        if user_question:
            response = conversation({'question': user_question})
            st.write("Response:", response['answer'])

if __name__ == "__main__":
    main()
