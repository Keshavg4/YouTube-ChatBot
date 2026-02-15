import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ============================
# Load Environment Variables
# ============================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎥 YouTube Video Q&A Chatbot")
st.write("Ask any question from the YouTube video transcript")

# ============================
# YouTube Transcript Loader
# ============================
@st.cache_resource
def load_vectorstore(video_id):

    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id)
    transcript_data = " ".join([item.text for item in transcript_list])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_data])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# ============================
# RAG Pipeline
# ============================
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def build_chain(vector_store):

    retriever = vector_store.as_retriever(search_type="similarity", search_kargs={"k":4})

    llm = ChatGroq(model="qwen/qwen3-32b")

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context","question"]
    )

    parallelchain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    return parallelchain | prompt | llm | parser

# ============================
# UI Inputs
# ============================
video_id = st.text_input("📺 Enter YouTube Video ID", value="Gfr50f6ZBvo")
question = st.text_input("❓ Ask your question")

# ============================
# Main Execution
# ============================
if st.button("🚀 Ask"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):

            try:
                vector_store = load_vectorstore(video_id)
                chain = build_chain(vector_store)

                answer = chain.invoke(question)

                st.success("Answer:")
                st.write(answer)

            except TranscriptsDisabled:
                st.error("No transcript available for this video.")
            except Exception as e:
                st.error(f"Error: {e}")
