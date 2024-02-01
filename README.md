# Multi-model AI Research System (MARS)

## Overview

MARS is a Streamlit-based application that allows users to interactively query information from multiple PDF documents using conversational AI powered by Google Palm embeddings. The system supports uploading PDFs, processing the text, and responding to user queries in a conversational manner.

## Features

- **Conversational AI:** Utilizes Google Palm embeddings and a conversational retrieval chain to provide responses to user queries.
- **PDF Processing:** Extracts text from uploaded PDF documents and processes it into chunks for efficient querying.
- **User Interaction:** Allows users to ask questions and receives responses in a conversational format.

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install streamlit PyPDF2 langchain google palm-microservice langchain-googlepalm
