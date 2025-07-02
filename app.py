import os
import json
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
import time

# --- Load environment variables from .env file ---
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

UPLOAD_DIR = "uploaded_files"
KB_JSON_PATH = "knowledge_base.json"
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".py", ".js", ".json", ".yaml", ".yml"]

# --- Ensure upload directory and knowledge base JSON exist ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(KB_JSON_PATH):
    with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# --- Persistent Knowledge Base Functions ---
def load_persistent_kb():
    if os.path.exists(KB_JSON_PATH):
        with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_persistent_kb(kb):
    with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

# --- Session State Initialization (must be before any use) ---
if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = load_persistent_kb()
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
st.title('RAG Based Document QA')
# Remove the default Streamlit title and divider for a cleaner look
# st.title("📄 RAG Document Chat App")
# st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Knowledge Base Management")

# --- Upload Files Section ---
st.sidebar.subheader("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Add files to your knowledge base:",
    type=[ext[1:] for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)
if uploaded_files:
    kb = st.session_state["knowledge_base"]
    new_files = []
    for uploaded_file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if not any(entry["name"] == uploaded_file.name for entry in kb):
            kb.append({"name": uploaded_file.name, "path": save_path, "type": "file"})
            new_files.append(save_path)
    save_persistent_kb(kb)
    st.session_state["knowledge_base"] = kb
    # Immediately load all KB files into vectorstore and enable chat
    if kb:
        all_docs = []
        for entry in kb:
            all_docs.extend(load_single_document(entry["path"]))
        if all_docs:
            chunked_docs = chunk_documents(all_docs)
            vectorstore = create_vectorstore(chunked_docs, DEFAULT_OPENAI_API_KEY)
            qa_chain = get_qa_chain(vectorstore, DEFAULT_OPENAI_API_KEY)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["qa_chain"] = qa_chain
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s). They will persist across refreshes and chat is now enabled.")
    st.rerun()

# --- Display Uploaded Files Section ---
st.sidebar.subheader("Knowledge Base Files")
kb = st.session_state["knowledge_base"]
if not kb:
    st.sidebar.info("No files uploaded yet.")
else:
    for i, entry in enumerate(kb):
        st.sidebar.write(f"{entry['name']}")
        if st.sidebar.button(f"Delete {entry['name']}", key=f"del_{i}"):
            if os.path.exists(entry["path"]):
                os.remove(entry["path"])
            kb.pop(i)
            save_persistent_kb(kb)
            st.session_state["knowledge_base"] = kb
            st.rerun()

# --- Web URL Loader Section ---
st.sidebar.subheader("Add Webpage URL")
url_input = st.sidebar.text_input("Paste a website URL to load its content:", value=st.session_state.get("url_input", ""))
load_url_btn = st.sidebar.button("Load URL")
if load_url_btn:
    if not url_input or not url_input.startswith("http"):
        st.sidebar.error("Please enter a valid URL.")
    else:
        st.session_state["url_input"] = url_input
        st.session_state["doc_folder"] = ""
        with st.spinner("Fetching and processing webpage content..."):
            try:
                docs = load_url_content(url_input)
                if not docs:
                    st.sidebar.error("No content could be loaded from the URL.")
                else:
                    chunked_docs = chunk_documents(docs)
                    vectorstore = create_vectorstore(chunked_docs, DEFAULT_OPENAI_API_KEY)
                    qa_chain = get_qa_chain(vectorstore, DEFAULT_OPENAI_API_KEY)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["qa_chain"] = qa_chain
                    st.sidebar.success(f"Loaded and processed content from URL.")
            except Exception as e:
                st.sidebar.error(f"Error loading URL: {e}")
        st.rerun()

# --- Chat Management Section ---
st.sidebar.subheader("Chat Management")
if st.sidebar.button("Clear Chat History"):
    st.session_state["chat_history"] = []

# --- Main Chat Area ---
# Remove fixed CSS and custom divs for chat input and answer section
# Just use Streamlit layout for stacking

# Chat history area (user questions)
for chat in st.session_state["chat_history"][::-1]:
    st.markdown(f"**You:** {chat['question']}")

# Place the chat form in the main Streamlit flow so submission always works
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Ask a question:", height=80)
    submitted = st.form_submit_button("Send")
if submitted and user_input:
    with st.spinner("Generating answer..."):
        if st.session_state["qa_chain"] is not None:
            try:
                result = st.session_state["qa_chain"].invoke({"query": user_input})
                answer = result["result"]
                sources = result.get("source_documents", [])
                # --- Streaming effect ---
                chat_placeholder = st.empty()
                streamed = ""
                for token in answer.split():
                    streamed += token + " "
                    chat_placeholder.markdown(
                        f'<div style="background:#f1f5f9;color:#1e293b;padding:1em;border-radius:8px;margin-bottom:0.5em;">'
                        f'<b>🤖 Assistant:</b> {streamed.strip()}<span style="color:#3b82f6;font-weight:600;animation:blink 1s steps(2, start) infinite;">|</span></div>'
                        '<style>@keyframes blink { to { visibility: hidden; } }</style>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.04)
                chat_placeholder.markdown(
                    f'<div style="background:#f1f5f9;color:#1e293b;padding:1em;border-radius:8px;margin-bottom:0.5em;">'
                    f'<b>🤖 Assistant:</b> {answer}</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                answer = f"Error: {e}"
                sources = []
            finally:
                st.session_state["chat_history"].append({
                    "question": user_input,
                    "answer": answer,
                    "sources": sources
                })
                st.rerun()
        else:
            answer = "Please upload files or load a URL to enable document Q&A."
            sources = []
            st.session_state["chat_history"].append({
                "question": user_input,
                "answer": answer,
                "sources": sources
            })
            st.rerun()

# --- Dedicated Answer Section (scrollable) ---
# Use a Streamlit container with a max height and scroll if needed
st.markdown("<hr style='margin:1.5em 0;'>", unsafe_allow_html=True)
st.markdown("<b>Assistant Answers</b>", unsafe_allow_html=True)
with st.container():
    st.markdown(
        """
        <style>
        .answer-scroll {
            max-height: 220px;
            overflow-y: auto;
            padding-right: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="answer-scroll">', unsafe_allow_html=True)
    for chat in st.session_state["chat_history"][::-1]:
        st.markdown(
            f'<div style="background:#f1f5f9;color:#1e293b;padding:1em;border-radius:8px;margin-bottom:0.5em;">'
            f'<b>🤖 Assistant:</b> {chat["answer"]}'
            + (
                f'<div style="background:#e8eef6;margin-top:0.7em;margin-left:1.5em;padding:0.7em 1em;border-radius:7px;"><span style="color:#3b82f6;font-weight:600;">Sources:</span><ul style="margin:0.3em 0 0 1.2em;padding:0;">' +
                ''.join(
                    f'<li style="color:#64748b;font-size:0.97em;">'
                    f'{src.metadata.get("source", "Unknown")}'
                    + (f' <span style="color:#94a3b8;">(page {src.metadata.get("page")+1})</span>' if src.metadata.get("page") is not None else "")
                    + '</li>'
                    for src in chat["sources"]
                ) + '</ul></div>' if chat["sources"] else ''
            ) +
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<small>Built with Streamlit, LangChain, OpenAI, and FAISS. | [GitHub](https://github.com/)</small>", unsafe_allow_html=True)

# --- Auto-load vectorstore and QA chain if files are present and not loaded ---
if st.session_state["qa_chain"] is None and st.session_state["knowledge_base"]:
    all_docs = []
    for entry in st.session_state["knowledge_base"]:
        all_docs.extend(load_single_document(entry["path"]))
    if all_docs:
        chunked_docs = chunk_documents(all_docs)
        vectorstore = create_vectorstore(chunked_docs, DEFAULT_OPENAI_API_KEY)
        qa_chain = get_qa_chain(vectorstore, DEFAULT_OPENAI_API_KEY)
        st.session_state["vectorstore"] = vectorstore
        st.session_state["qa_chain"] = qa_chain 