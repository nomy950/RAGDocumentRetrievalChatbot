import os
import streamlit as st
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PythonLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import tempfile
import openai
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup

# --- Load environment variables from .env file ---
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".py", ".js", ".json", ".yaml", ".yml"]

# --- Utility Functions ---
def get_files_in_folder(folder_path: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return files

def load_single_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".txt":
            return TextLoader(file_path).load()
        elif ext == ".md":
            return UnstructuredMarkdownLoader(file_path).load()
        elif ext == ".pdf":
            return PyPDFLoader(file_path).load()
        elif ext == ".py":
            return PythonLoader(file_path).load()
        elif ext in [".js", ".json", ".yaml", ".yml"]:
            return UnstructuredFileLoader(file_path).load()
        else:
            return []
    except Exception as e:
        st.warning(f"Failed to load {file_path}: {e}")
        return []

def load_documents(folder_path: str) -> List[Any]:
    files = get_files_in_folder(folder_path)
    docs = []
    for file in files:
        docs.extend(load_single_document(file))
    return docs

def chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []
    for doc in documents:
        chunked_docs.extend(splitter.split_documents([doc]))
    return chunked_docs

def create_vectorstore(docs: List[Any], openai_api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(docs, embeddings)

def get_qa_chain(vectorstore: FAISS, openai_api_key: str) -> RetrievalQA:
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # Define a system prompt
    system_prompt = (
        "You are an expert in analyzing documents. "
        "Answer user questions with exact and to-the-point answers, "
        "but elaborate using your prior knowledge if it helps clarify the answer. "
        "If the answer is not found in the provided documents, reply: 'I couldn't find the answer in the documents.'"
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            system_prompt + "\n\n"
            "Context from documents:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def load_url_content(url: str) -> List[Any]:
    """Fetch and parse the main text content from a webpage URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Get visible text
        text = "\n".join([t.strip() for t in soup.stripped_strings if t])
        # Create a dummy Document object for compatibility
        from langchain_core.documents import Document
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        st.sidebar.error(f"Failed to load URL: {e}")
        return []

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Document Chat", page_icon="📄", layout="wide")
st.title("📄 RAG Document Chat App")

# --- Sidebar ---
st.sidebar.header("Configuration")
# Remove OpenAI API Key textbox; always use .env
openai_api_key = DEFAULT_OPENAI_API_KEY

# Folder selection using Streamlit's directory picker (experimental) or file_uploader for multiple files
folder_picker = st.sidebar.text_input("Documents Folder Path", value=st.session_state.get("doc_folder", ""))
# If running locally, allow user to pick a folder using st.sidebar.file_uploader (workaround: user uploads a file from the folder, we use its parent)
uploaded_file = st.sidebar.file_uploader("Or select a file from your folder (to auto-detect folder)", type=None)
if uploaded_file is not None:
    # Save uploaded file to a temp location to get its path
    temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    detected_folder = os.path.dirname(temp_file_path)
    folder_picker = detected_folder
    st.sidebar.info(f"Detected folder: {detected_folder}")

# URL input
url_input = st.sidebar.text_input("Or paste a website URL to load its content", value=st.session_state.get("url_input", ""))
load_url_btn = st.sidebar.button("Load URL")

doc_folder = folder_picker
load_btn = st.sidebar.button("Load Documents")
clear_chat_btn = st.sidebar.button("Clear Chat History")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "doc_folder" not in st.session_state:
    st.session_state["doc_folder"] = ""
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

# --- Load Documents ---
if load_btn:
    if not openai_api_key:
        st.sidebar.error("OpenAI API key not found in .env file.")
    elif not doc_folder or not os.path.isdir(doc_folder):
        st.sidebar.error("Please enter a valid folder path.")
    else:
        st.session_state["openai_api_key"] = openai_api_key
        st.session_state["doc_folder"] = doc_folder
        st.session_state["url_input"] = ""
        with st.spinner("Loading and processing documents from folder..."):
            try:
                docs = load_documents(doc_folder)
                if not docs:
                    st.sidebar.error("No supported documents found in the folder.")
                else:
                    chunked_docs = chunk_documents(docs)
                    vectorstore = create_vectorstore(chunked_docs, openai_api_key)
                    qa_chain = get_qa_chain(vectorstore, openai_api_key)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["qa_chain"] = qa_chain
                    st.success(f"Loaded {len(docs)} documents, {len(chunked_docs)} chunks from folder.")
            except Exception as e:
                st.sidebar.error(f"Error loading documents: {e}")

# --- Load Content from URL ---
if load_url_btn:
    if not openai_api_key:
        st.sidebar.error("OpenAI API key not found in .env file.")
    elif not url_input or not url_input.startswith("http"):
        st.sidebar.error("Please enter a valid URL.")
    else:
        st.session_state["openai_api_key"] = openai_api_key
        st.session_state["url_input"] = url_input
        st.session_state["doc_folder"] = ""
        with st.spinner("Fetching and processing webpage content..."):
            try:
                docs = load_url_content(url_input)
                if not docs:
                    st.sidebar.error("No content could be loaded from the URL.")
                else:
                    chunked_docs = chunk_documents(docs)
                    vectorstore = create_vectorstore(chunked_docs, openai_api_key)
                    qa_chain = get_qa_chain(vectorstore, openai_api_key)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["qa_chain"] = qa_chain
                    st.success(f"Loaded and processed content from URL.")
            except Exception as e:
                st.sidebar.error(f"Error loading URL: {e}")

# --- Clear Chat ---
if clear_chat_btn:
    st.session_state["chat_history"] = []

# --- Main Chat Area ---
st.markdown("---")
if st.session_state["qa_chain"] is None:
    st.info("Please configure and load your documents or a website to start chatting.")
else:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question about your documents or the website:", height=80)
        submitted = st.form_submit_button("Send")
    if submitted and user_input:
        with st.spinner("Generating answer..."):
            try:
                result = st.session_state["qa_chain"].invoke({"query": user_input})
                answer = result["result"]
                sources = result.get("source_documents", [])
                st.session_state["chat_history"].append({
                    "question": user_input,
                    "answer": answer,
                    "sources": sources
                })
            except openai.error.AuthenticationError:
                st.error("Invalid OpenAI API key.")
            except Exception as e:
                st.error(f"Error during QA: {e}")

    # Display chat history
    for chat in st.session_state["chat_history"][::-1]:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Answer:** {chat['answer']}")
        if chat["sources"]:
            st.markdown("**Sources:**")
            for i, src in enumerate(chat["sources"]):
                meta = src.metadata
                fname = meta.get("source", "Unknown")
                page = meta.get("page", None)
                if page is not None:
                    st.markdown(f"- `{fname}` (page {page+1})")
                else:
                    st.markdown(f"- `{fname}`")
        st.markdown("---")

# --- Footer ---
st.markdown("<small>Built with Streamlit, LangChain, OpenAI, and FAISS. | [GitHub](https://github.com/)</small>", unsafe_allow_html=True) 