import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Medical Chatbot", layout="centered")

st.markdown(
    """
    <h1 style="color: #008000; text-align: center;">Medical Chatbot</h1>
    <div style="text-align: center;">
        <img src="https://www.pngfind.com/pngs/m/126-1269385_chatbots-builder-pricing-crozdesk-chat-bot-png-transparent.png" alt="Chatbot Logo" width="100" height="100">
    </div>
    """, unsafe_allow_html=True
)

#<img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png" alt="Chatbot Logo" width="80" height="80">

@st.cache_resource
def load_or_create_faiss():
    if os.path.exists("faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings)
        return db
    else:
        loader = PyPDFLoader('Medical_book.pdf')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index") 
        return db

db = load_or_create_faiss()

model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
"""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
)

parser = StrOutputParser()

st.markdown(
    "<h4 style='color:red;'>Ask a medical question related to the content of the PDF:</h4>", 
    unsafe_allow_html=True
)

user_question = st.text_input("Your Question", "")

if user_question:
    with st.spinner('Processing your question...'):
        context_docs = db.similarity_search(user_question, k=2)  
        context = "\n".join([doc.page_content for doc in context_docs])

        chain = prompt | model | parser
        response = chain.invoke({
            "context": context,
            "question": user_question,
        })

        st.write("### Answer:")
        st.markdown(f"<p style='color:blue;'>{response}</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px 0;
    }
    .footer img {
        vertical-align: middle;
        margin-left: 10px;
    }
    </style>
    <div class="footer">
        Developed by Bharathkumar M S
        <a href="https://www.linkedin.com/in/bharathkumar-m-s/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Linkedin_icon.svg" alt="LinkedIn" width="30" height="30">
        </a>
    </div>
    """, unsafe_allow_html=True
)

