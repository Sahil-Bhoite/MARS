# Import necessary libraries
# Make sure to install required libraries using: pip install streamlit PyPDF2 langchain google palm-microservice langchain-googlepalm

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Set Google API key (replace 'put your google api here' with your actual API key)
os.environ['GOOGLE_API_KEY'] = 'put your google api here'

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks using Google Palm embeddings
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create a conversational retrieval chain
def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

# Function to handle user input and generate responses
def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("MARS: ", message.content)

# Main function
def main():
    # Set page configuration
    st.set_page_config("Multi-model AI Research System")
    st.header("Chat with Multiple PDFs 📁")
    
    # Get user's question input
    user_question = st.text_input("Ask a Question from the PDFs")
    
    # Initialize session state variables if not present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    # Process user input and handle PDF uploads
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("MARS ")
        st.subheader("Upload your PDFS here")
        
        # File uploader for PDF documents
        pdf_docs = st.file_uploader("Upload your PDFs and Click on the NEXT Button", accept_multiple_files=True)
        
        # Process PDFs and create a vector store when the "NEXT" button is clicked
        if st.button("NEXT"):
            with st.spinner("Let me process your PDFs"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

# Run the main function
if __name__ == "__main__":
    main()
