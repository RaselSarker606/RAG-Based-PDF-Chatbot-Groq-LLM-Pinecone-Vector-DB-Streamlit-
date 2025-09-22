# ==============================
import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# HuggingFace Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# PDF Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ==========================================
load_dotenv()

st.title("📄 RAG Chatbot using GroqAPI + Pinecone")

# ======================== Session State ========================
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# ======================== File Upload ========================
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# ======================== Setup Pinecone ========================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "vectorgroqapixxx"


if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,   # depends on embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",       # or "gcp"
            region="us-east-1" # check from Pinecone console
        )
    )

# Connect to index
index = pc.Index(index_name)

# ======================== Embeddings ========================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ======================== Vectorstore Build ========================
@st.cache_resource
def get_vectorstore(uploaded_file):
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Upload docs to Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=index_name
        )
        return vectorstore
    return None

# ======================== Chat Input ========================
if uploaded_file is not None:
    vectorstore = get_vectorstore(uploaded_file)

    prompt = st.chat_input("Ask something about your PDF...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})

        # System prompt
        groq_sys_prompt = ChatPromptTemplate.from_template("""
            You are very smart at everything, you always give the best,
            the most accurate and most precise answer. Answer the following Question: {user_prompt}.
            Start the answer directly. No small talk please
        """)

        # Groq LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"   # Use supported Groq model
        )

        try:
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"]

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: [{str(e)}]")

else:
    st.info("⬆️ Please upload a PDF file to start chatting.")
