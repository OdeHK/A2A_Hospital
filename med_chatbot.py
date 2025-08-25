import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from embedder import EmbeddingGenerator
from vector_db import VectorDB
from retriever import Retriever
from generator import Generator
from config import PDF_PATH, OUTPUT_LOG, OUTPUT_JSON_CLEAN

# Hàm khởi tạo các đối tượng (cache để tránh lặp lại)
@st.cache_resource
def init_components():
    print("[LOG] Khởi tạo các module...")
    print("[LOG] Đang khởi tạo DataLoader...")
    data_preprocessor = DataPreprocessor()
    print("[LOG] Đang khởi tạo EmbeddingGenerator...")
    embedder = EmbeddingGenerator()
    print("[LOG] Đang khởi tạo VectorDB...")
    vector_db = VectorDB()
    print("[LOG] Đang khởi tạo Retriever...")
    retriever = Retriever(vector_db, device="cpu")
    print("[LOG] Đang khởi tạo Generator...")
    generator = Generator(embedder, retriever)
    print("[LOG] Khởi tạo hoàn tất.")
    return data_preprocessor, embedder, vector_db, retriever, generator

# Gọi hàm khởi tạo
data_preprocessor, embedder, vector_db, retriever, generator = init_components()

# Giao diện chat
st.title("Medical RAG Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Nhập câu hỏi: "):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    print("[LOG] Đang xử lý câu hỏi người dùng...")
    processed_query = embedder.preprocess_and_tokenize(query)

    print("[LOG] Đang tạo embedding cho câu hỏi...")
    query_embedding = embedder.embed_query(processed_query)

    print("[LOG] Đang truy xuất và rerank kết quả...")
    retrieve_results = retriever.retrieve(query_embedding)
    documents = [result.payload["content"] for result in retrieve_results]
    ranked_results = retriever.rerank(query, documents)
    print("\nKết quả sau khi rerank (top 1):")
    context = "\n".join([f"{i+1}. {content}" for i, (_, content) in enumerate(ranked_results)])
    print(context)
    print("\n")
    print("[LOG] Đang sinh câu trả lời...")
    response = generator.generate(query, context)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    print("[LOG] Hoàn tất phản hồi.")