# 🧠 Chat With Documents – Local AI-Powered Chatbot

Turn your static files into dynamic conversations.

> Upload. Ask. Get Answers.  
> All locally. All private. All smart.

## 🚀 Overview

This project is a local AI-powered document assistant that allows users to chat with any file (PDF, DOCX, TXT, MD). Built with open-source tools and privacy-first principles, it enables natural language interaction with your documents — no cloud, no API keys required.

## 🛠️ Tech Stack

Each component plays a critical role in bringing your documents to life:

- **🧪 Streamlit**: Provides an intuitive and interactive frontend for file uploads and chat.
- **🐍 Python**: Orchestrates the core logic including file parsing, query handling, and display.
- **🧱 LangChain**: Manages chaining of prompts and memory with LLM context handling.
- **🦙 Ollama + LLaMA 3**: Powers local inference using open-source language models.
- **🧠 FAISS**: Performs efficient vector similarity search for semantic retrieval.
- **🐘 PostgreSQL + SQLAlchemy**: Stores chat history and user session data for persistent conversations.

## 🎯 Features

- ✅ Upload and chat with files: PDF, DOCX, TXT, MD
- ✅ Natural language understanding and responses
- ✅ Semantic retrieval from file content
- ✅ Source citation in replies
- ✅ Stores and retrieves past conversations
- ✅ Works fully offline using Ollama

## 📂 Supported File Types

- `.pdf`
- `.docx`
- `.txt`
- `.md`

## ⚙️ How It Works

1. Upload a document via the Streamlit interface.
2. The document is chunked and converted into embeddings.
3. Embeddings are stored in a FAISS vector index.
4. Your query is embedded and matched against relevant chunks.
5. Ollama's LLaMA 3 processes the context and generates a response.
6. The conversation is stored in a PostgreSQL database via SQLAlchemy.

## 📸 UI Preview

> Coming soon! A sleek and simple interface for document interaction.

## 🔐 Roadmap

- [ ] Add user authentication system
- [ ] Enable cloud file storage support
- [ ] Chat across multiple files
- [ ] Deploy to Streamlit Cloud or HuggingFace Spaces

## 🧑‍💻 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/document-chatbot.git
cd document-chatbot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
