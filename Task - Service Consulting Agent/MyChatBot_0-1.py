import streamlit as st
import asyncio
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import PromptTemplate

from ps_combine import package_to_text
import json
from dotenv import load_dotenv
import os

load_dotenv()


system_template = """
Bạn là trợ lý y tế, luôn trả lời dựa trên dữ liệu gói khám được cung cấp 1 cách tự nhiên hoặc theo ngữ cảnh, đa dạng, có thể thay đổi trong mỗi lần trả lời nhưng vẫn giữ đúng format yêu cầu.
Luôn trả lời bằng tiếng Việt.
Không dùng markdown
Khi người dùng nhập tình trạng sức khỏe hoặc mô tả bản thân. hãy phân tích và trả lời format liệt kế gói khám:

+ Gói khám <Tên gói>: <Thông tin về gói khám (mô tả, tổng giá)>

Nếu thông tin rõ ràng đầy đủ thì tư vấn gói phù hợp nhất và liệt kê thêm các gói liên quan. Không thì cứ gợi ý các gói khám liên quan.
Gợi ý và xem xét:
 Gói khám cần thông tin về giới tính, độ tuổi thì hỏi 
 Có thể hỏi thêm về: Mục đích & tình trạng, Mức độ mong muốn, Ngân sách để gợi ý chính xác hơn

Nếu hỏi về gói khám cụ thể trả lời các thông tin sau: <tên gói> <mô tả> <dịch vụ> <giá> <tổng giá>
"""

SYS_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=system_template + """

Dữ liệu liên quan:
{context}

Lịch sử trò chuyện:
{chat_history}

Câu hỏi:
{question}

Trả lời:
"""
)

llm = ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

# VECTORSTORE ___________________________
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

# CONVERSATION CHAIN ___________________________
def get_conversation_chain(vectorstore):
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": SYS_PROMPT},
        return_source_documents=True, 
    )
    return conversation_chain


def main():
    with open(os.getenv("DATA_PATH"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    package_texts = package_to_text(data)
    vectorstore = get_vectorstore(package_texts)
    conversation_chain = get_conversation_chain(vectorstore)

# STREAMLIT__________________________________________________________
    st.set_page_config(page_title='DEMO',page_icon="",layout="centered")
    st.title("Demo")
    st.markdown("Chatbot tư vấn gói khám")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for chat in st.session_state.messages:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])

    if prompt := st.chat_input("input....."):
        st.session_state.messages.append({'role':"user","content": prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        response = conversation_chain.invoke({"question": prompt})

        st.session_state.messages.append({'role':"chatbot","content": response["answer"]})
        with st.chat_message('chatbot'):
            st.markdown(response["answer"])

if __name__ == "__main__":
    main()
    



