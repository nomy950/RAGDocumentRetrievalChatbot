# RAG Document Chat App

A simple, production-ready Retrieval-Augmented Generation (RAG) chat application that allows you to ask questions about documents in a local folder or any website. Built with Streamlit, LangChain, OpenAI, and FAISS.

---

## 🚀 Features
- **Load documents** from a local folder (supports `.txt`, `.md`, `.pdf`, `.py`, `.js`, `.json`, `.yaml`, `.yml`)
- **Paste a website URL** to load and chat about webpage content
- **Automatic chunking** for better retrieval
- **Semantic search** using OpenAI embeddings and FAISS vector store
- **Chat interface** to ask questions about your documents or a website
- **Source references** for every answer
- **Chat history** with clear/reset option
- **Sidebar UI** for folder selection, URL input, and document loading
- **System prompt**: Answers are expert, concise, and elaborate when needed; if info is missing, the app will say so
- **Error handling** for missing API keys, folders, URLs, or documents
- **No API key in UI**: Key is loaded from `.env` for security

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd personalChatbot
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📂 Usage Guide

1. **Select your documents folder**
   - Use the sidebar to paste a folder path, or
   - Upload any file from your folder (the app will auto-detect the folder)
2. **Or paste a website URL**
   - Enter a valid URL in the sidebar and click "Load URL"
   - The app will fetch and process the webpage content
3. **Click "Load Documents" or "Load URL"**
   - The app will process and embed your documents or webpage
4. **Ask questions in the chat interface**
   - The app will answer using your documents, the website, and its prior knowledge
   - Source files or URLs for each answer are shown
5. **Clear chat history** anytime with the sidebar button

### Supported File Types
- `.txt`, `.md`, `.pdf`, `.py`, `.js`, `.json`, `.yaml`, `.yml`
- Any public website URL (HTML content only)

---

## 🧠 How It Works
- **Document Loading:** Loads and chunks all supported files in the selected folder
- **Webpage Loading:** Fetches and parses visible text from the provided URL
- **Embeddings:** Uses OpenAI API to create embeddings for each chunk
- **Vector Store:** Stores embeddings in a local FAISS database
- **Retrieval:** Finds the most relevant chunks for each question
- **LLM QA Chain:** Uses a system prompt to answer questions, referencing the retrieved chunks and elaborating if needed
- **UI:** Built with Streamlit for easy interaction

---

## ⚙️ Configuration
- **API Key:** Only loaded from `.env` for security (not shown in UI)
- **Folder Selection:** Paste a path or upload a file from your folder
- **URL Input:** Paste a website URL in the sidebar
- **Chunking:** Default chunk size is 1000 characters with 200 overlap

---

## 🐞 Troubleshooting
- **No API key found:** Make sure `.env` exists and contains `OPENAI_API_KEY`
- **No documents found:** Check your folder path and file types
- **No content from URL:** Make sure the URL is correct and the site is accessible
- **Deprecation warnings:** Make sure all dependencies are up to date (`pip install -r requirements.txt`)
- **Other errors:** Check the Streamlit sidebar for error messages

---

## 🙏 Credits
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📄 License
MIT License (add your license here if needed) 