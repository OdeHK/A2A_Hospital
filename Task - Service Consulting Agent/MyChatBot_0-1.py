import streamlit as st
import asyncio
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate

from ps_combine import package_to_text

import json
from dotenv import load_dotenv
import os

load_dotenv()

system_template = """
Bạn là trợ lý y tế, luôn trả lời dựa trên dữ liệu gói khám được cung cấp. 
Nếu người dùng hỏi về một gói khám nào đó, hãy mô tả gói đó và các dịch vụ của nó theo dữ liệu, 
bất kể tên gói có từ nhạy cảm hay không. 
Luôn trả lời bằng tiếng Việt.
"""
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# ===================== VECTORSTORE =====================
def get_vectorstore(text_chunks):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# ===================== CONVERSATION CHAIN =====================
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        system_prompt=system_prompt
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True, 
    )
    return conversation_chain

# ===================== TEST TRÊN TERMINAL =====================
def main():
    with open(os.getenv("DATA_PATH"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    package_texts = package_to_text(data)

    vectorstore = get_vectorstore(package_texts)
    conversation_chain = get_conversation_chain(vectorstore)

    print("🤖 Chatbot - Gõ 'exit' để thoát.\n")
    while True:
            query = input("Human: ")
            if query.lower() in ["exit", "quit"]:
                break
            result = conversation_chain.invoke({"question": query})
            print(f'\nBot:', result["answer"])
            print("\nSource documents:")
            for doc in result.get("source_documents", []):
                print("-", doc.page_content)
        

if __name__ == "__main__":
    main()
    



# ===================== MAIN =====================
# def main():
#     st.set_page_config(page_title="Chatbot tư vấn gói khám", layout="wide")
#     st.title("🤖 Chatbot tư vấn gói khám bệnh")

#     # Giả sử đây là danh sách gói khám có sẵn
#     packages = [
#         "Gói khám tiểu đường: Bao gồm xét nghiệm đường huyết, HbA1c, siêu âm ổ bụng.",
#         "Gói khám tim mạch: Điện tâm đồ, siêu âm tim, xét nghiệm mỡ máu.",
#         "Gói khám gan mật: Siêu âm gan mật, xét nghiệm men gan, HBsAg.",
#         "Gói khám tổng quát: Xét nghiệm máu tổng quát, nước tiểu, X-quang phổi.",
#     ]

#     # Tạo vectorstore từ các gói khám
#     if "vectorstore" not in st.session_state:
#         st.session_state.vectorstore = get_vectorstore(packages)

#     # Tạo conversation chain
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

#     # Chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Hiển thị chat history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])

#     # Input từ người dùng
#     user_input = st.chat_input("Nhập tên bệnh bạn muốn tìm gói khám...")
#     if user_input:
#         # Hiển thị câu hỏi
#         st.chat_message("user").write(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         # Gọi conversation chain
#         response = st.session_state.conversation({"question": user_input})
#         bot_reply = response["answer"]

#         # Hiển thị câu trả lời
#         st.chat_message("assistant").write(bot_reply)
#         st.session_state.messages.append({"role": "assistant", "content": bot_reply})


# if __name__ == "__main__":
#     main()

    