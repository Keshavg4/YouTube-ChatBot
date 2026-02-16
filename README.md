🎥 YouTube Video Q&A Chatbot using RAG & GenAI

An AI-powered YouTube Video Question–Answering Chatbot built using Retrieval Augmented Generation (RAG) architecture. This system enables users to ask natural language questions from any YouTube video simply by providing its video ID. The application automatically extracts the video transcript, processes and stores it using vector embeddings, retrieves the most relevant context, and generates accurate, context-grounded answers using a powerful Large Language Model (LLM). The entire solution is deployed using Streamlit, delivering a smooth and interactive user experience.

This project demonstrates end-to-end implementation of RAG pipelines, efficient semantic search using FAISS vector databases, and seamless integration of LangChain + Groq LLM, making it a complete real-world GenAI application.

🚀 Key Features

🔹 Ask real-time questions from any YouTube video

🔹 Fully implemented RAG pipeline for accurate and grounded responses

🔹 Automatic YouTube transcript extraction

🔹 Smart text chunking and embedding generation

🔹 High-performance vector search using FAISS

🔹 Fast inference using Groq LLM (Qwen-32B model)

🔹 Clean and interactive Streamlit UI

🔹 Live cloud deployment

🧠 Project Architecture & Flow

User Input
User enters the YouTube video ID and a question through the Streamlit interface.

Transcript Loading
The transcript is fetched using the YouTube Transcript API.

Text Chunking
The transcript is split into overlapping chunks using RecursiveCharacterTextSplitter for better semantic retrieval.

Embedding Generation
Each chunk is converted into vector embeddings using HuggingFace Sentence Transformers.

Vector Storage
The embeddings are stored in a FAISS vector database for fast similarity-based search.

Context Retrieval (RAG)
The most relevant chunks are retrieved based on the user query.

LLM Answer Generation
Retrieved context is passed to Groq LLM (Qwen-32B) through LangChain, generating an accurate and grounded response.

Final Output
The answer is displayed instantly on the Streamlit UI.

🛠️ Tech Stack

Programming Language: Python

Frontend: Streamlit

LLM Framework: LangChain

Embedding Model: HuggingFace – all-MiniLM-L6-v2

Vector Database: FAISS

LLM Provider: Groq (Qwen-32B)

Transcript API: YouTube Transcript API

Deployment: Streamlit Cloud

🔁 LangChain RAG Pipeline Overview

The system uses a parallel runnable chain architecture where:

Retriever fetches the most relevant transcript chunks

Context is dynamically injected into a custom prompt template

The LLM generates answers strictly based on the retrieved context

This ensures:

High factual correctness

Reduced hallucinations

Fast response generation

▶️ How To Run Locally
1️⃣ Clone Repository
git clone <your-github-repo-link>
cd <repo-folder>

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

4️⃣ Run Streamlit App
streamlit run app.py

🌐 Live Deployment

🎯 Real-World Use Cases

📚 AI-powered video learning assistant

🧑‍🎓 Smart educational content exploration

🎯 Quick knowledge extraction from long-form videos

🔍 Semantic video content search

🤖 GenAI-based interactive learning chatbot

📌 Future Enhancements

Multi-video knowledge base

Timestamp-based answer highlighting

Multilingual transcript support

Chat history and memory

PDF & document upload support

