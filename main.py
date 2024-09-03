import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import json
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("GOOGLE_API_KEY environment variable is not set.")
    logging.error("GOOGLE_API_KEY environment variable is not set.")
    st.stop()
else:
    logging.info("Google API Key successfully loaded.")

# Configure the generative AI API
genai.configure(api_key=api_key)
logging.info("Google Generative AI configured with API key.")

st.set_page_config(page_title="DocsGenie Ai", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

# API key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Step 1: Document Processing
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_pdf_text(pdf_docs):
    text = ""
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_pdf, pdf_docs)
        for result in results:
            text += result
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=10000)
    chunks = text_splitter.split_text(text)
    return chunks

# Step 2: Information Extraction and Tagging
def extract_information(text_chunk):
    try:
        equipment_name = re.findall(r'(?i)equipment name:\s*([^\n]+)', text_chunk)
        domain = re.findall(r'(?i)domain:\s*([^\n]+)', text_chunk)
        model_numbers = re.findall(r'(?i)model number:\s*([^\n]+)', text_chunk)
        manufacturer = re.findall(r'(?i)manufacturer:\s*([^\n]+)', text_chunk)
        
        tags = {
            'equipment_name': equipment_name[0] if equipment_name else 'N/A',
            'domain': domain[0] if domain else 'N/A',
            'model_numbers': model_numbers[0] if model_numbers else 'N/A',
            'manufacturer': manufacturer[0] if manufacturer else 'N/A',
        }
        return tags
    except Exception as e:
        logging.error(f"Error extracting information: {e}")
        return {
            'equipment_name': 'N/A',
            'domain': 'N/A',
            'model_numbers': 'N/A',
            'manufacturer': 'N/A',
        }

# Step 3: Vector Database Integration
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    text_embeddings = []
    metadata = []

    def process_chunk(chunk):
        tags = extract_information(chunk)
        vector = embeddings.embed_query(chunk)
        return vector, {"page_content": chunk, "metadata": tags}

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_chunk, text_chunks)
        for vector, meta in results:
            text_embeddings.append(vector)
            metadata.append(meta)

    # Convert text embeddings to numpy array
    vectors = np.array(text_embeddings).astype(np.float32)
    
    if len(vectors) == 0:
        st.error("No text chunks were generated for vectorization. Please check your PDF content.")
        return

    dimension = vectors.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save the index and metadata
    try:
        faiss.write_index(index, "faiss_index.index")
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
    except Exception as e:
        logging.error(f"Error saving FAISS index or metadata: {e}")
        st.error("An error occurred while saving the vector store. Please try again.")

# Step 4: Query Processing
def parse_query(query):
    equipment = re.findall(r'(?i)equipment:\s*([^\n]+)', query)
    model = re.findall(r'(?i)model:\s*([^\n]+)', query)
    manufacturer = re.findall(r'(?i)manufacturer:\s*([^\n]+)', query)
    
    parsed_query = {
        'equipment': equipment[0] if equipment else '',
        'model': model[0] if model else '',
        'manufacturer': manufacturer[0] if manufacturer else ''
    }
    return parsed_query

# Enhanced User Query Handling
def user_input(user_question, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error("An error occurred while processing your query. Please try again.")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Main function
def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Processing complete.")
                else:
                    st.error("No text could be extracted from the uploaded PDFs.")

if __name__ == "__main__":
    main()
