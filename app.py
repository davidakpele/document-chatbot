import asyncio
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import Ollama
from db import QAHistory, SessionLocal
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
import streamlit as st
import torch
import warnings
from db import init_db
import os

# Fix for PyTorch warnings and performance
torch.classes.__path__ = []
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

load_dotenv()


def get_text_from_documents(docs):
    text = ""
    for file in docs:
        name = file.name.lower()
        try:
            if name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

            elif name.endswith(".txt") or name.endswith(".md"):
                text += file.read().decode("utf-8")

            elif name.endswith(".docx"):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"

            else:
                st.warning(f"Unsupported file format: {name}")
        except Exception as e:
            st.warning(f"Could not read {file.name}: {str(e)}")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = Ollama(
        model="llama3",
        temperature=0.5,
        timeout=300,
        num_ctx=2048,
        num_gpu=1 if torch.cuda.is_available() else 0,
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        chain_type="stuff",
        verbose=False
    )


def handle_userinput(user_question):
    if "conversation" in st.session_state:
        try:
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]

            # Display and save Q&A
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    question = message.content
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    answer = message.content
                    # Save to DB
                    db = SessionLocal()
                    qa = QAHistory(question=question, answer=answer)
                    db.add(qa)
                    db.commit()
                    db.close()

            # Show sources
            if "source_documents" in response:
                with st.expander("Source Documents"):
                    for doc in response["source_documents"]:
                        st.write(f"**Page {doc.metadata.get('page', 'N/A')}**:")
                        st.write(doc.page_content[:500] + "...")
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")


def main():
    if os.name == 'nt':
        os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
        os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    st.set_page_config(
        page_title="Chat with Documents",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents :books:")
    user_question = st.chat_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "md"]
        )

        if st.button("Process", type="primary") and pdf_docs:
            with st.spinner("Processing your documents..."):
                try:
                    raw_text = get_text_from_documents(pdf_docs)

                    if not raw_text:
                        st.error(
                            "No text could be extracted. Try different files.")
                        return

                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                        st.error(
                            "No valid text chunks were created. Try different files.")
                        return

                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)

                    st.success(
                        "Processing complete! You can now ask questions.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

        with st.expander("System Information"):
            st.write(f"PyTorch version: {torch.__version__}")
            st.write(
                f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            st.write(f"Ollama model: llama3")

        # Save uploaded files
        saved_files = []
        for uploaded_file in pdf_docs:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            saved_files.append(file_path)

        # ✅ Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, message in enumerate(st.session_state.chat_history):
                role = "User" if i % 2 == 0 else "Bot"
                st.markdown(f"**{role}:** {message.content}")


if __name__ == '__main__':
    try:
        init_db()
        asyncio.set_event_loop(asyncio.new_event_loop())
        main()
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            pass
        else:
            raise
